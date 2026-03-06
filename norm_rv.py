#!/usr/bin/env python3
"""
================================================================================
RV ERROR NORMALIZATION & CALIBRATION — v8
Following Tsantaki et al. (2021) "Survey of Surveys I"  (A&A 659, A95)
================================================================================
CRITICAL FIXES vs v7  (see detailed notes below each phase):

  [FIX-1]  Phase 0: Extract stellar parameters (Teff, logg, FeH, Gmag, SNR)
           from external survey CSVs so they are available downstream.

  [FIX-2]  Phase 3: Propagate stellar parameters to CSV-matched stars.
           — If the CSV row has its own params, use those.
           — Otherwise inherit params from the FITS entry for the same
             spatial group (typically the GAIA or DESI entry).
           Without this fix, ALL external-survey stars (APOGEE, GALAH, GES,
           RAVE) had NaN params, making Phase 6 Gaia calibration non-functional
           (only DESI/SDSS contributed, with near-zero coefficients).

  [FIX-3]  Phase 4 DUP: Quality controls.
           — Skip GAIA entirely for DUP (paper Table 4 uses TCH for Gaia).
             The FITS "duplicates" for Gaia are DR2/DR3 median RVs, not
             independent transit measurements → normMAD ≈ 0 is spurious.
           — Require error ratio < 5 between pair members to avoid mixing
             different-pipeline entries (CSV vs A95).
           — Require minimum combined error > 0.01 km/s.
           — 5σ clip on normalized diffs before computing MAD.

  [FIX-4]  Phase 5 TCH: Sanity checks on combined factors.
           — DUP factors outside [0.1, 20] treated as unreliable.
           — TCH factors outside [0.1, 20] treated as unreliable.
           — If DESI×GAIA pairwise σ ≈ 0 (because DESI rows in FITS have
             Gaia-derived RVs), exclude that pair.

  [FIX-5]  Phase 6 Gaia calibration: Paper §5.1 exclusion rules restored.
           — Eq.5 (Gmag): exclude LAMOST (low resolution).
           — Eq.6 ([Fe/H]): exclude LAMOST + RAVE (opposite trend).
           — Eq.7 (Teff): APOGEE only (paper §5.1.3).
           Now all surveys with valid params contribute, producing non-trivial
           calibration polynomials.

  [FIX-6]  Phase 9 plots: Paper-quality figures for Fig 6, 7, 8, 9/10/11, 13.
           Before/after calibration panels match paper Figs. 9-12.

METHODOLOGY SUMMARY (paper):
  1. Cross-match surveys with Gaia (spatial + source_id)
  2. Normalize RV errors:
     - DUP method (repeated measurements): APOGEE, RAVE, LAMOST
     - TCH method (three-cornered hat):   Gaia, GALAH
     - GES: average of DUP and TCH
  3. Gaia internal calibration:
     - Eq.5: ΔRV vs G mag  (2nd-order polynomial)
     - Eq.6: ΔRV vs [Fe/H] (linear)
     - Eq.7: ΔRV vs Teff   (2nd-order polynomial, APOGEE only)
  4. Survey calibration (Eq.8): multivariate correction to calibrated Gaia frame
  5. Merge into unified catalogue with weighted average RVs
================================================================================
"""

import os, sys, gc, time, pickle, warnings, argparse
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial import cKDTree
from astropy.io import fits

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# =============================================================================
# SURVEYS
# =============================================================================
VALID_SURVEYS = {'GAIA', 'DESI', 'APOGEE', 'GALAH', 'GES', 'RAVE',
                 'LAMOST', 'SDSS'}
DUP_SURVEYS   = {'DESI', 'APOGEE', 'GALAH', 'GES', 'RAVE', 'LAMOST', 'SDSS'}
#  ^^^ FIX-3: GAIA removed from DUP_SURVEYS — paper uses TCH for Gaia
MIN_DUP_PAIRS_SIGNIFICANT = 200

# Paper Table 4 logic for combining DUP / TCH
DUP_ONLY_SURVEYS = {'APOGEE', 'RAVE', 'LAMOST'}
TCH_ONLY_SURVEYS = {'GAIA', 'GALAH'}
AVG_BOTH_SURVEYS = {'GES'}

# Paper §5.1 exclusion rules for Gaia calibration
EQ5_GMAG_EXCLUDE = {'LAMOST'}                    # low resolution
EQ6_FEH_EXCLUDE  = {'LAMOST', 'RAVE'}            # LAMOST: low-res; RAVE: opposite trend
EQ7_TEFF_SURVEYS = {'APOGEE'}                    # only APOGEE (paper §5.1.3)

# Sanity bounds for normalization factors
MIN_PLAUSIBLE_FACTOR = 0.1
MAX_PLAUSIBLE_FACTOR = 20.0

# =============================================================================
# COLORS
# =============================================================================
COLORS = {
    'GAIA':   '#1f77b4',
    'APOGEE': '#2ca02c',
    'LAMOST': '#d62728',
    'RAVE':   '#ff7f0e',
    'GALAH':  '#9467bd',
    'GES':    '#8c564b',
    'DESI':   '#e377c2',
    'SDSS':   '#17becf',
}
def _c(s): return COLORS.get(s, '#333333')

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    input_fits: str = ""
    output_dir: str = "./rv_norm_output_v8"
    checkpoint_dir: str = "./rv_norm_ckpt_v8"

    apogee_csv: str = "./astro_data/APOGEE_DR17/APOGEE_DR17_merged_rv_parallax.csv"
    galah_csv:  str = "./astro_data/GALAH_DR3/GALAH_DR3_merged_rv_parallax.csv"
    ges_csv:    str = "./astro_data/GES_DR5/GES_DR5_merged_rv_parallax.csv"
    rave_csv:   str = "./astro_data/RAVE_DR6/RAVE_DR6_merged_rv_parallax.csv"
    a95_dir:    str = "./astro_data/A95_cds"

    ra_col:     str = "RA_all"
    dec_col:    str = "DEC_all"
    survey_col: str = "Survey"
    code_col:   str = "Code"
    sid_cols:   List[str] = field(default_factory=lambda: [
        "source_id","Source_ID","Gaia_Source_ID","gaia_source_id"])

    rv_columns: List[Tuple[str,str]] = field(default_factory=lambda: [
        ("radial_velocity_1","radial_velocity_error_1"),
        ("radial_velocity_2","radial_velocity_error_2"),
        ("RV","e_RV"),("VRAD","VRAD_ERR"),
        ("dr2_radial_velocity","dr2_radial_velocity_error"),
    ])

    param_columns: Dict[str,List[str]] = field(default_factory=lambda: {
        'Gmag': ['Gmag','phot_g_mean_mag','GMAG','G'],
        'Teff': ['Teff','TEFF','Teff_x','teff'],
        'logg': ['logg','LOGG','logg_x','log_g'],
        'FeH':  ['[Fe/H]','FEH','feh','M_H','Fe_H'],
        'SNR':  ['RVSS/N','RVS/N','rvss_snr','rv_snr','snr','SNR','S_N'],
    })

    tolerance_arcsec:   float = 1.0
    healpix_nside:      int   = 32
    chunk_size:         int   = 3_000_000
    n_workers:          int   = 1
    min_pairs_dup:      int   = 50
    min_stars_tch:      int   = 20
    min_stars_zp:       int   = 30
    max_obs_per_star:   int   = 30
    max_pairs_per_star: int   = 200
    min_bin_count:      int   = 10

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# =============================================================================
# A95 FIXED-WIDTH SPECS  (unchanged from v7)
# =============================================================================
A95_SPECS = {
    'APOGEE': {'gz': 'apogee.dat.gz', 'dat': 'apogee.dat',
               'sid': (27, 46), 'ra': (47, 71), 'dec': (72, 94),
               'rv': (95, 104), 'erv': (105, 113), 'srv': (114, 121)},
    'GALAH':  {'gz': 'galah.dat.gz', 'dat': 'galah.dat',
               'sid': (25, 44), 'ra': (45, 68), 'dec': (69, 92),
               'rv': (93, 101), 'erv': (102, 107), 'srv': (107, 108)},
    'GES':    {'gz': 'ges.dat.gz', 'dat': 'ges.dat',
               'sid': (25, 44), 'ra': (45, 67), 'dec': (68, 91),
               'rv': (92, 100), 'erv': (101, 108), 'srv': (109, 114)},
    'RAVE':   {'gz': 'rave.dat.gz', 'dat': 'rave.dat',
               'sid': (25, 44), 'ra': (45, 68), 'dec': (69, 91),
               'rv': (92, 100), 'erv': (101, 107), 'srv': (108, 115)},
    'LAMOST': {'gz': 'lamost.dat.gz', 'dat': 'lamost.dat',
               'sid': (28, 47), 'ra': (48, 70), 'dec': (71, 94),
               'rv': (95, 103), 'erv': (104, 110), 'srv': (111, 118)},
}

# =============================================================================
# UTILITIES  (unchanged from v7)
# =============================================================================
def get_mem_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except:
        try:
            with open(f'/proc/{os.getpid()}/status') as f:
                for l in f:
                    if l.startswith('VmRSS:'):
                        return int(l.split()[1])/1024/1024
        except:
            return -1.0

def log(msg):
    ts  = time.strftime('%H:%M:%S')
    mem = get_mem_gb()
    m   = f" [{mem:.1f}GB]" if mem > 0 else ""
    print(f"[{ts}]{m} {msg}", flush=True)

def resolve_survey(survey_val, code_val):
    s = str(survey_val).strip().upper() if survey_val is not None else ''
    c = str(code_val).strip().upper()   if code_val   is not None else ''
    for x in ('NAN','NONE',''):
        if s == x: s = ''
        if c == x: c = ''
    blob = s+'|'+c
    if c.startswith('D33') or c.startswith('D125') or 'GAIA' in blob: return 'GAIA'
    if 'DESI'   in blob: return 'DESI'
    if 'APOGEE' in blob: return 'APOGEE'
    if 'LAMOST' in blob: return 'LAMOST'
    if 'GALAH'  in blob: return 'GALAH'
    if 'RAVE'   in blob and 'RAVEL' not in blob: return 'RAVE'
    if c == 'GES' or s == 'GES': return 'GES'
    if 'SDSS' in blob or 'BOSS' in blob or 'SEGUE' in blob: return 'SDSS'
    return None

def is_exact_duplicate(rv, err, tol=1e-6):
    return np.abs(np.abs(rv)-np.abs(err)) < tol

def find_col(avail, possible):
    for n in possible:
        if n in avail: return n
    return None


def open_maybe_gzip(path: Path):
    if path.suffix.lower() == '.gz':
        import gzip
        return gzip.open(path, 'rb')
    return path.open('rb')


def _to_float_field(line: bytes, sl: Tuple[int, int]) -> float:
    tok = line[sl[0]:sl[1]].strip()
    if not tok:
        return np.nan
    try:
        return float(tok)
    except Exception:
        return np.nan


def _to_int_field(line: bytes, sl: Tuple[int, int]) -> float:
    tok = line[sl[0]:sl[1]].strip()
    if not tok:
        return np.nan
    try:
        return float(int(tok))
    except Exception:
        return np.nan


def _resolve_a95_file(a95_dir: str, spec: Dict[str, str]) -> Optional[Path]:
    pdir = Path(a95_dir)
    cand = [pdir / spec['gz'], pdir / spec['dat']]
    for p in cand:
        if p.exists():
            return p
    return None


def _dedup_by_exact_rv_err(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    rv_vals = pd.to_numeric(df['rv'], errors='coerce').values
    err_vals = pd.to_numeric(df['e_rv'], errors='coerce').values
    valid = np.isfinite(rv_vals) & np.isfinite(err_vals) & (err_vals > 0)
    vi = np.where(valid)[0]
    seen = {}
    keep = np.ones(len(vi), dtype=bool)
    for pos, oi in enumerate(vi):
        key = (round(float(rv_vals[oi]), 6), round(float(err_vals[oi]), 6))
        if key in seen:
            keep[pos] = False
        else:
            seen[key] = oi
    vi = vi[keep]
    if len(vi) == 0:
        return df.iloc[0:0].copy()
    return df.iloc[vi].copy().reset_index(drop=True)


def _load_a95_survey_table(survey: str, path: Path, spec: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    sids, ras, decs, rvs, ervs, srvs = [], [], [], [], [], []
    with open_maybe_gzip(path) as handle:
        for line in handle:
            rv = _to_float_field(line, spec['rv'])
            erv = _to_float_field(line, spec['erv'])
            if not (np.isfinite(rv) and np.isfinite(erv) and erv > 0):
                continue
            sid = _to_int_field(line, spec['sid'])
            ra = _to_float_field(line, spec['ra'])
            dec = _to_float_field(line, spec['dec'])
            srv = _to_float_field(line, spec['srv'])
            sids.append(sid); ras.append(ra); decs.append(dec)
            rvs.append(rv); ervs.append(erv); srvs.append(srv)

    if not rvs:
        return pd.DataFrame(columns=[
            'source_id', 'ra', 'dec', 'rv', 'e_rv', 'survey',
            'parallax', 'parallax_error', 's_rvcor',
            'csv_Teff', 'csv_logg', 'csv_FeH', 'csv_Gmag', 'csv_SNR'
        ])

    n = len(rvs)
    out = pd.DataFrame({
        'source_id': np.array(sids, dtype=np.float64),
        'ra': np.array(ras, dtype=np.float64),
        'dec': np.array(decs, dtype=np.float64),
        'rv': np.array(rvs, dtype=np.float64),
        'e_rv': np.array(ervs, dtype=np.float64),
        'survey': survey,
        'parallax': np.full(n, np.nan, dtype=np.float64),
        'parallax_error': np.full(n, np.nan, dtype=np.float64),
        's_rvcor': np.array(srvs, dtype=np.float64),
        # A95 files don't carry stellar params
        'csv_Teff': np.full(n, np.nan, dtype=np.float64),
        'csv_logg': np.full(n, np.nan, dtype=np.float64),
        'csv_FeH':  np.full(n, np.nan, dtype=np.float64),
        'csv_Gmag': np.full(n, np.nan, dtype=np.float64),
        'csv_SNR':  np.full(n, np.nan, dtype=np.float64),
    })
    return _dedup_by_exact_rv_err(out)


def save_ckpt(path, data):
    with open(path,'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  Checkpoint saved: {path}")

def load_ckpt(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def bin_stat(x, y, n_bins=40, p_lo=1, p_hi=99, min_count=10):
    """Return (bin_centres, medians, mads) clipping to percentile range."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_count*2:
        return np.array([]), np.array([]), np.array([])
    x, y = x[ok], y[ok]
    lo, hi = np.percentile(x, p_lo), np.percentile(x, p_hi)
    be = np.linspace(lo, hi, n_bins+1)
    bc, meds, mads = [], [], []
    for j in range(n_bins):
        m = (x >= be[j]) & (x < be[j+1])
        if m.sum() < min_count:
            continue
        yy = y[m]
        med = float(np.median(yy))
        mad = float(np.median(np.abs(yy - med)))
        bc.append(0.5*(be[j]+be[j+1]))
        meds.append(med)
        mads.append(mad)
    return np.array(bc), np.array(meds), np.array(mads)


# =============================================================================
# PHASE 0: LOAD EXTERNAL SURVEY CSVs
# =============================================================================
# FIX-1: Now extracts stellar parameters (Teff, logg, FeH, Gmag, SNR) from CSVs
# =============================================================================
def phase0_load_csvs(cfg):
    ckpt = Path(cfg.checkpoint_dir)/"phase0_v8b.pkl"
    cfg.a95_s_rvcor_factors = {}
    if ckpt.exists():
        log("Phase 0: Loading checkpoint")
        d = load_ckpt(ckpt)
        if isinstance(d, dict) and 'tables' in d:
            tables = d.get('tables', {})
            cfg.a95_s_rvcor_factors = d.get('a95_s_rvcor_factors', {})
        else:
            tables = d
        for s, df in tables.items():
            log(f"  {s:<12s}: {len(df):>8,} rows")
            # Check if params are present
            has_params = 'csv_Teff' in df.columns and df['csv_Teff'].notna().any()
            log(f"    has stellar params: {has_params}")
        return tables

    log("Phase 0: Loading external survey CSVs + stellar params...")
    specs = {
        'APOGEE': (cfg.apogee_csv,'HRV','e_HRV'),
        'GALAH':  (cfg.galah_csv,'RVgalah','e_RVgalah'),
        'GES':    (cfg.ges_csv,'RV','RVprec'),
        'RAVE':   (cfg.rave_csv,'HRV','e_HRV'),
    }
    fallback_rv = {
        'APOGEE': [('RV','e_RV')],
        'RAVE':   [('cHRV','e_HRV')],
        'GALAH':  [('RVsmev2','e_RVsmev2'),('RVobst','e_RVobst')],
        'GES':    [],
    }

    # ── FIX-1: Parameter column search lists per survey ───────────────────
    param_search = {
        'Teff': ['Teff','TEFF','teff','Teff_x','teff_gspphot'],
        'logg': ['logg','LOGG','log_g','logg_x','logg_gspphot'],
        'FeH':  ['[Fe/H]','FEH','feh','M_H','Fe_H','mh_gspphot','met_N_K'],
        'Gmag': ['Gmag','phot_g_mean_mag','G','GMAG','gmag'],
        'SNR':  ['snr','SNR','S_N','snr_c2_iraf','rv_snr','RVSS/N'],
    }

    result = {}
    for surv,(path,rv_col,err_col) in specs.items():
        if not os.path.exists(path):
            log(f"  {surv}: CSV not found at {path}, skipping"); continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            log(f"  {surv}: failed — {e}"); continue

        cols = set(df.columns)
        sid_col = find_col(cols,['source_id','Source_ID','gaia_source_id'])
        df['_sid'] = pd.to_numeric(df[sid_col],errors='coerce').astype('Int64') if sid_col else pd.NA
        ra_col  = find_col(cols,['ra_deg','RA','ra','RAdeg'])
        dec_col = find_col(cols,['dec_deg','DEC','dec','DECdeg'])

        if rv_col not in cols:
            found = False
            for fb_rv,fb_err in fallback_rv.get(surv,[]):
                if fb_rv in cols and fb_err in cols:
                    rv_col,err_col = fb_rv,fb_err; found=True; break
            if not found:
                log(f"    {surv}: no usable RV column, skipping"); continue

        rv_vals  = pd.to_numeric(df[rv_col],  errors='coerce').values
        err_vals = pd.to_numeric(df[err_col], errors='coerce').values if err_col in cols else np.full(len(df),np.nan)
        if surv=='RAVE' and 'cHRV' in cols:
            crv=pd.to_numeric(df['cHRV'],errors='coerce').values; ok=np.isfinite(crv); rv_vals[ok]=crv[ok]
        ra_vals  = pd.to_numeric(df[ra_col], errors='coerce').values  if ra_col  else np.full(len(df),np.nan)
        dec_vals = pd.to_numeric(df[dec_col],errors='coerce').values  if dec_col else np.full(len(df),np.nan)
        plx_col  = find_col(cols,['parallax','Parallax','parallax_mas','plx','PLX','Plx'])
        plxe_col = find_col(cols,['parallax_error','e_parallax','parallax_err','plx_err','e_plx'])
        plx_vals = pd.to_numeric(df[plx_col], errors='coerce').values  if plx_col  else np.full(len(df),np.nan)
        plxe_vals= pd.to_numeric(df[plxe_col],errors='coerce').values  if plxe_col else np.full(len(df),np.nan)

        # ── FIX-1: Extract stellar parameters from CSV ────────────────────
        csv_params = {}
        for pname, search_names in param_search.items():
            pcol = find_col(cols, search_names)
            if pcol:
                csv_params[pname] = pd.to_numeric(df[pcol], errors='coerce').values.astype(np.float64)
                n_valid = np.sum(np.isfinite(csv_params[pname]))
                log(f"    {surv}: found param {pname} in column '{pcol}' ({n_valid:,} valid)")
            else:
                csv_params[pname] = np.full(len(df), np.nan, dtype=np.float64)

        raw = pd.DataFrame({
            'source_id': pd.to_numeric(df['_sid'], errors='coerce').values.astype(np.float64),
            'ra': ra_vals.astype(np.float64),
            'dec': dec_vals.astype(np.float64),
            'rv': rv_vals.astype(np.float64),
            'e_rv': err_vals.astype(np.float64),
            'survey': surv,
            'parallax': plx_vals.astype(np.float64),
            'parallax_error': plxe_vals.astype(np.float64),
            's_rvcor': np.full(len(df), np.nan, dtype=np.float64),
            'csv_Teff': csv_params['Teff'],
            'csv_logg': csv_params['logg'],
            'csv_FeH':  csv_params['FeH'],
            'csv_Gmag': csv_params['Gmag'],
            'csv_SNR':  csv_params['SNR'],
        })
        out = _dedup_by_exact_rv_err(raw)
        result[surv] = out
        log(f"  {surv}: {len(out):,} rows loaded")

    # ── A95 tables ─────────────────────────────────────────────────────────
    log("Phase 0b: Loading A95 survey tables (.dat/.dat.gz)...")
    a95_factors = {}
    for surv, spec in A95_SPECS.items():
        path = _resolve_a95_file(cfg.a95_dir, spec)
        if path is None:
            log(f"  {surv}: A95 file not found in {cfg.a95_dir}, skipping"); continue
        try:
            df_a95 = _load_a95_survey_table(surv, path, spec)
        except Exception as e:
            log(f"  {surv}: failed to read A95 file ({e})"); continue
        if len(df_a95) == 0:
            log(f"  {surv}: A95 loaded 0 valid rows"); continue

        ratio = (df_a95['s_rvcor'].values.astype(np.float64)
                 / df_a95['e_rv'].values.astype(np.float64))
        ok_ratio = np.isfinite(ratio) & (ratio > 0)
        if np.any(ok_ratio):
            a95_factors[surv] = float(np.median(ratio[ok_ratio]))

        if surv in result:
            merged = pd.concat([result[surv], df_a95], ignore_index=True, sort=False)
            merged = _dedup_by_exact_rv_err(merged)
            result[surv] = merged
            log(f"  {surv}: +A95 {len(df_a95):,} rows → merged {len(merged):,}")
        else:
            result[surv] = df_a95
            log(f"  {surv}: A95 only {len(df_a95):,} rows")

    cfg.a95_s_rvcor_factors = a95_factors
    if a95_factors:
        log("  A95 s_RVcor/e_RVcor priors:")
        for s, f in sorted(a95_factors.items()):
            log(f"    {s:<12s}: {f:.3f}")

    save_ckpt(ckpt, {'tables': result, 'a95_s_rvcor_factors': a95_factors})
    return result


# =============================================================================
# PHASE 1: EXTRACT FROM FITS  (unchanged from v7)
# =============================================================================
def phase1_extract(cfg):
    ckpt = Path(cfg.checkpoint_dir)/"phase1_v8.pkl"
    if ckpt.exists():
        log("Phase 1: Loading checkpoint")
        d = load_ckpt(ckpt)
        log(f"  {d['n_valid']:,} valid / {d['total_rows']:,} total")
        surv_arr = np.array(d['surveys'])
        for sn in sorted(VALID_SURVEYS):
            n = np.sum(surv_arr==sn)
            if n>0: log(f"    {sn:<16s}: {n:>10,}")
        return d

    log("Phase 1: Extracting one RV per row from FITS...")
    hdu   = fits.open(cfg.input_fits, memmap=True)
    fdata = hdu[1].data
    acols = {col.name for col in hdu[1].columns}
    total = len(fdata)

    valid_rv = [(v,e) for v,e in cfg.rv_columns if v in acols and e in acols]
    param_col_map = {pn: find_col(acols,possible) for pn,possible in cfg.param_columns.items()}
    sid_col  = find_col(acols, cfg.sid_cols)
    has_surv = cfg.survey_col in acols
    has_code = cfg.code_col   in acols
    plx_col  = find_col(acols,['parallax','Parallax','parallax_1','plx','PLX'])
    plxe_col = find_col(acols,['parallax_error','parallax_error_1','e_parallax','plx_err'])
    log(f"  Parallax col: {plx_col}, error col: {plxe_col}")
    log(f"  Param cols: {param_col_map}")

    all_ra,all_dec,all_rv,all_err,all_surveys,all_row_idx,all_sid = [],[],[],[],[],[],[]
    all_plx,all_plxe = [],[]
    all_params = {p:[] for p in cfg.param_columns}

    for start in tqdm(range(0,total,cfg.chunk_size),
                      total=(total+cfg.chunk_size-1)//cfg.chunk_size, desc="Phase1"):
        end  = min(start+cfg.chunk_size, total)
        clen = end-start
        ra_c  = np.array(fdata[cfg.ra_col][start:end],  dtype=np.float64)
        dec_c = np.array(fdata[cfg.dec_col][start:end], dtype=np.float64)
        surv_c= np.array(fdata[cfg.survey_col][start:end]).astype(str) if has_surv else np.full(clen,'')
        code_c= np.array(fdata[cfg.code_col][start:end]).astype(str)  if has_code else np.full(clen,'')
        sid_c = np.array(fdata[sid_col][start:end],dtype=np.int64)    if sid_col  else np.zeros(clen,dtype=np.int64)

        rv_arrs = {}
        for v,e in valid_rv:
            rv_arrs[v]=np.array(fdata[v][start:end],dtype=np.float64)
            rv_arrs[e]=np.array(fdata[e][start:end],dtype=np.float64)
        param_arrs = {}
        for pn,col in param_col_map.items():
            try:    param_arrs[pn]=np.array(fdata[col][start:end],dtype=np.float64) if col else np.full(clen,np.nan)
            except: param_arrs[pn]=np.full(clen,np.nan)
        try:    plx_c = np.array(fdata[plx_col][start:end],dtype=np.float64)  if plx_col  else np.full(clen,np.nan)
        except: plx_c = np.full(clen,np.nan)
        try:    plxe_c= np.array(fdata[plxe_col][start:end],dtype=np.float64) if plxe_col else np.full(clen,np.nan)
        except: plxe_c= np.full(clen,np.nan)

        all_rvs_s  = np.full((clen,len(valid_rv)),np.nan)
        all_errs_s = np.full((clen,len(valid_rv)),np.nan)
        for j,(v,e) in enumerate(valid_rv):
            vj = np.isfinite(rv_arrs[v])&np.isfinite(rv_arrs[e])&(rv_arrs[e]>0)
            all_rvs_s[vj,j]=rv_arrs[v][vj]; all_errs_s[vj,j]=rv_arrs[e][vj]
        wts   = np.where(np.isfinite(all_errs_s)&(all_errs_s>0),1.0/all_errs_s**2,0.0)
        w_sum = np.nansum(wts,axis=1); has_any=(w_sum>0)
        best_rv=np.full(clen,np.nan); best_err=np.full(clen,np.nan)
        best_rv[has_any] =np.nansum(all_rvs_s[has_any]*wts[has_any],axis=1)/w_sum[has_any]
        best_err[has_any]=1.0/np.sqrt(w_sum[has_any])
        no_err_mask=~has_any&np.any(np.isfinite(all_rvs_s),axis=1)
        if np.any(no_err_mask): best_rv[no_err_mask]=np.nanmean(all_rvs_s[no_err_mask],axis=1)

        resolved=[resolve_survey(surv_c[i],code_c[i]) for i in range(clen)]
        valid_mask=(np.isfinite(best_rv)&np.isfinite(ra_c)&np.isfinite(dec_c)
                    &np.array([r is not None for r in resolved]))
        vi=np.where(valid_mask)[0]
        all_ra.append(ra_c[vi]); all_dec.append(dec_c[vi])
        all_rv.append(best_rv[vi]); all_err.append(best_err[vi])
        all_row_idx.append(np.arange(start,end,dtype=np.int64)[vi])
        all_sid.append(sid_c[vi])
        for i in vi: all_surveys.append(resolved[i])
        for pn in cfg.param_columns: all_params[pn].append(param_arrs[pn][vi])
        all_plx.append(plx_c[vi]); all_plxe.append(plxe_c[vi])
        del ra_c,dec_c,surv_c,code_c,rv_arrs,param_arrs,all_rvs_s,all_errs_s,wts,plx_c,plxe_c
        gc.collect()
    hdu.close()

    ra=np.concatenate(all_ra); dec=np.concatenate(all_dec)
    rv=np.concatenate(all_rv); err=np.concatenate(all_err)
    row_idx=np.concatenate(all_row_idx); sid=np.concatenate(all_sid)
    plx=np.concatenate(all_plx); plxe=np.concatenate(all_plxe)
    params={p:np.concatenate(all_params[p]) for p in cfg.param_columns}
    surveys=all_surveys

    # within-survey exact dedup
    surveys_arr=np.array(surveys); keep_mask=np.ones(len(ra),dtype=bool)
    for surv in sorted(VALID_SURVEYS):
        si=np.where(surveys_arr==surv)[0]
        if len(si)<2: continue
        rv_s=rv[si]; err_s=err[si]; seen={}; disc=[]
        for pos,gi in enumerate(si):
            if not np.isfinite(rv_s[pos]) or not np.isfinite(err_s[pos]): continue
            key=(round(float(rv_s[pos]),6),round(float(err_s[pos]),6))
            if key in seen: disc.append(gi)
            else: seen[key]=gi
        if disc: keep_mask[disc]=False
    ra=ra[keep_mask]; dec=dec[keep_mask]; rv=rv[keep_mask]; err=err[keep_mask]
    sid=sid[keep_mask]; row_idx=row_idx[keep_mask]
    plx=plx[keep_mask]; plxe=plxe[keep_mask]
    surveys=[s for s,k in zip(surveys,keep_mask) if k]
    params={p:params[p][keep_mask] for p in params}
    n_valid=len(ra)

    log(f"  Final valid rows: {n_valid:,}")
    surveys_arr=np.array(surveys)
    for sn in sorted(VALID_SURVEYS):
        n=int(np.sum(surveys_arr==sn))
        if n>0: log(f"    {sn:<16s}: {n:>10,}")

    result={'ra':ra,'dec':dec,'rv':rv,'err':err,'surveys':surveys,
            'params':params,'row_idx':row_idx,'sid':sid,
            'plx':plx,'plxe':plxe,
            'n_valid':n_valid,'total_rows':total}
    save_ckpt(ckpt,result)
    gc.collect()
    return result


# =============================================================================
# PHASE 2: SPATIAL GROUPING  (unchanged)
# =============================================================================
def phase2_spatial(cfg, data):
    ckpt = Path(cfg.checkpoint_dir)/"phase2_v8.pkl"
    if ckpt.exists():
        log("Phase 2: Loading checkpoint")
        d=load_ckpt(ckpt)
        ug,gc_=np.unique(d['labels'],return_counts=True)
        log(f"  {len(ug):,} groups, {np.sum(gc_>1):,} multi-row, max={np.max(gc_)}")
        return d
    ra,dec=data['ra'],data['dec']; n=len(ra)
    log(f"\nPhase 2: Spatial matching on {n:,} rows...")
    if HAS_HEALPY:
        pix=hp.ang2pix(cfg.healpix_nside,np.radians(90-dec),np.radians(ra),nest=True)
    else:
        pix=(np.floor(ra).astype(np.int32)%360)*180+(np.floor(dec+90).astype(np.int32)%180)
    upix=np.unique(pix)
    parent=np.arange(n,dtype=np.int64); rank=np.zeros(n,dtype=np.int32)
    def find(x):
        r=x
        while parent[r]!=r: r=parent[r]
        while parent[x]!=r: parent[x],x=r,parent[x]
        return r
    def union(a,b):
        pa,pb=find(a),find(b)
        if pa==pb: return
        if rank[pa]<rank[pb]: pa,pb=pb,pa
        parent[pb]=pa
        if rank[pa]==rank[pb]: rank[pa]+=1
    tol=2*np.sin(np.radians(cfg.tolerance_arcsec/3600)/2)
    si=np.argsort(pix); sp=pix[si]
    pl=np.searchsorted(sp,upix,side='left'); pr=np.searchsorted(sp,upix,side='right')
    for pi in tqdm(range(len(upix)),desc="Phase2",unit="rgn"):
        l,r=pl[pi],pr[pi]
        if r-l<2: continue
        idx=si[l:r]; rr,dr=np.radians(ra[idx]),np.radians(dec[idx])
        xyz=np.column_stack([np.cos(dr)*np.cos(rr),np.cos(dr)*np.sin(rr),np.sin(dr)])
        pairs=cKDTree(xyz).query_pairs(r=tol,output_type='ndarray')
        if len(pairs)>0:
            for li,lj in pairs: union(idx[li],idx[lj])
    labels=np.array([find(i) for i in range(n)],dtype=np.int64)
    ug,gcounts=np.unique(labels,return_counts=True)
    log(f"  Groups: {len(ug):,}, multi: {np.sum(gcounts>1):,}, max: {np.max(gcounts)}")
    result={'labels':labels}
    save_ckpt(ckpt,result)
    del parent,rank,pix,si,sp; gc.collect()
    return result


# =============================================================================
# PHASE 2b: SOURCE_ID → GROUP_ID MAP  (unchanged)
# =============================================================================
def phase2b_sid_map(cfg, data, groups):
    ckpt=Path(cfg.checkpoint_dir)/"phase2b_v8.pkl"
    if ckpt.exists():
        log("Phase 2b: Loading checkpoint")
        d=load_ckpt(ckpt); log(f"  Map size: {len(d):,}"); return d
    log("\nPhase 2b: Building source_id → group_id map...")
    sids=data['sid']; labels=groups['labels']
    sid_map={}
    for i,(sid,gid) in enumerate(zip(sids,labels)):
        if sid!=0 and sid>0: sid_map[int(sid)]=int(gid)
    log(f"  Map size: {len(sid_map):,}")
    save_ckpt(ckpt,sid_map)
    return sid_map


# =============================================================================
# PHASE 3: BUILD PER-SURVEY STAR DATA
# =============================================================================
# FIX-2: Propagate params from (a) CSV columns, (b) FITS group entries
# =============================================================================
def phase3_build(cfg, data, groups, csv_data, sid_map):
    ckpt=Path(cfg.checkpoint_dir)/"phase3_v8b.pkl"
    if ckpt.exists():
        log("Phase 3: Loading checkpoint")
        d=load_ckpt(ckpt)
        try:
            for s in sorted(d.keys()):
                sample=next(iter(d[s].values()))
                if not isinstance(sample,dict) or 'rvs' not in sample:
                    raise ValueError("stale")
                n_multi=sum(1 for g in d[s].values() if len(g['rvs'])>=2)
                # Check param coverage
                n_has_teff=sum(1 for g in d[s].values()
                               if np.isfinite(g['params'].get('Teff',np.nan)))
                log(f"  {s:<16s}: {len(d[s]):>10,} stars, {n_multi:>8,} >=2obs, "
                    f"{n_has_teff:>8,} with Teff")
            return d
        except Exception as e:
            log(f"  Stale checkpoint ({e}), rebuilding...")
            ckpt.unlink(missing_ok=True)

    log("\nPhase 3: Building per-survey star data (with param propagation)...")
    gl=groups['labels']; rv=data['rv']; err=data['err']
    surveys=data['surveys']; params=data['params']; n=len(gl)
    fits_plx =data.get('plx',  np.full(n, np.nan))
    fits_plxe=data.get('plxe', np.full(n, np.nan))
    ss=defaultdict(dict)

    log("  Adding FITS rows...")
    for i in tqdm(range(n),desc="Phase3-FITS",mininterval=5.0):
        surv=surveys[i]; gid=int(gl[i])
        if surv not in VALID_SURVEYS: continue
        if gid not in ss[surv]:
            ss[surv][gid]={'rvs':[],'params':{p:float(params[p][i]) for p in params},
                           'plx':float(fits_plx[i]),'plxe':float(fits_plxe[i])}
        ss[surv][gid]['rvs'].append((float(rv[i]),float(err[i])))

    # ── Build FITS spatial KDTree ONCE ─────────────────────────────────────
    log("  Building FITS spatial KDTree...")
    ra_f=data['ra']; dec_f=data['dec']
    rr_f=np.radians(ra_f); dr_f=np.radians(dec_f)
    fits_xyz=np.column_stack([np.cos(dr_f)*np.cos(rr_f),
                              np.cos(dr_f)*np.sin(rr_f),
                              np.sin(dr_f)])
    fits_tree=cKDTree(fits_xyz)
    spatial_tol=2*np.sin(np.radians(cfg.tolerance_arcsec/3600)/2)

    def _plx_ok(plx_csv, plxe_csv, fits_idx_list):
        if not np.isfinite(plx_csv): return True
        for fi in fits_idx_list:
            plx_f = float(fits_plx[fi])
            if not np.isfinite(plx_f): return True
            plxe_f = float(fits_plxe[fi]) if np.isfinite(fits_plxe[fi]) else 0.5
            plxe_c = float(plxe_csv) if np.isfinite(plxe_csv) else 0.5
            sigma_comb = np.sqrt(plxe_f**2 + plxe_c**2)
            thr = max(3.0 * sigma_comb, 1.0)
            if abs(plx_csv - plx_f) < thr:
                return True
        return False

    # ── FIX-2: Helper to get best params for a group ──────────────────────
    def _get_group_params(gid):
        """Get stellar params from FITS entries for this group."""
        # Prefer GAIA params, then DESI, then any other FITS survey
        for prefer in ['GAIA', 'DESI', 'APOGEE', 'GALAH', 'RAVE', 'LAMOST', 'GES', 'SDSS']:
            if prefer in ss and gid in ss[prefer]:
                pr = ss[prefer][gid]['params']
                if any(np.isfinite(pr.get(p, np.nan)) for p in ['Teff','logg','FeH','Gmag']):
                    return dict(pr)
        return {p: np.nan for p in params}

    log("  Adding CSV rows (with parameter propagation)...")
    for surv, df in csv_data.items():
        n_sid=0; n_spatial=0; n_skip=0; n_plx_reject=0; n_params_from_csv=0; n_params_from_fits=0
        has_sid = df['source_id'].notna().any()
        has_plx = 'parallax' in df.columns and df['parallax'].notna().any()
        # ── FIX-1: Check for CSV stellar params ──────────────────────────
        has_csv_teff = 'csv_Teff' in df.columns and df['csv_Teff'].notna().any()

        ra_csv  = df['ra'].values.astype(np.float64)
        dec_csv = df['dec'].values.astype(np.float64)
        rv_csv  = df['rv'].values.astype(np.float64)
        erv_csv = df['e_rv'].values.astype(np.float64)
        sid_csv = df['source_id'].values if has_sid else np.full(len(df), np.nan)
        plx_csv = df['parallax'].values.astype(np.float64) if has_plx else np.full(len(df),np.nan)
        plxe_csv= df['parallax_error'].values.astype(np.float64) if has_plx and 'parallax_error' in df.columns else np.full(len(df),np.nan)

        # FIX-1: Load CSV stellar params
        csv_teff = df['csv_Teff'].values.astype(np.float64) if 'csv_Teff' in df.columns else np.full(len(df),np.nan)
        csv_logg = df['csv_logg'].values.astype(np.float64) if 'csv_logg' in df.columns else np.full(len(df),np.nan)
        csv_feh  = df['csv_FeH'].values.astype(np.float64)  if 'csv_FeH'  in df.columns else np.full(len(df),np.nan)
        csv_gmag = df['csv_Gmag'].values.astype(np.float64) if 'csv_Gmag' in df.columns else np.full(len(df),np.nan)
        csv_snr  = df['csv_SNR'].values.astype(np.float64)  if 'csv_SNR'  in df.columns else np.full(len(df),np.nan)

        valid_pos = np.isfinite(ra_csv) & np.isfinite(dec_csv)
        rr_csv  = np.radians(ra_csv[valid_pos])
        dr_csv  = np.radians(dec_csv[valid_pos])
        xyz_csv = np.column_stack([np.cos(dr_csv)*np.cos(rr_csv),
                                   np.cos(dr_csv)*np.sin(rr_csv),
                                   np.sin(dr_csv)])
        matches = fits_tree.query_ball_point(xyz_csv, r=spatial_tol)
        valid_idx_list = np.where(valid_pos)[0]

        for pos, orig_i in enumerate(valid_idx_list):
            rv_v  = rv_csv[orig_i];  err_v = erv_csv[orig_i]
            if not (np.isfinite(rv_v) and np.isfinite(err_v) and err_v > 0): continue
            if is_exact_duplicate(rv_v, err_v): continue

            gid = None
            if has_sid:
                sid_v = sid_csv[orig_i]
                if not pd.isna(sid_v):
                    sid_int = int(sid_v)
                    if sid_int in sid_map:
                        gid = sid_map[sid_int]; n_sid += 1

            if gid is None:
                cands = matches[pos]
                if not cands: n_skip += 1; continue
                if not _plx_ok(plx_csv[orig_i], plxe_csv[orig_i], cands):
                    n_plx_reject += 1; continue
                if len(cands) == 1:
                    gid = int(gl[cands[0]])
                else:
                    dists = np.linalg.norm(fits_xyz[cands] - xyz_csv[pos], axis=1)
                    gid   = int(gl[cands[int(np.argmin(dists))]])
                n_spatial += 1

            if gid is None: n_skip += 1; continue

            # ── FIX-2: Build params for this CSV star ─────────────────────
            if gid not in ss[surv]:
                # Try CSV params first
                csv_pr = {
                    'Teff': float(csv_teff[orig_i]),
                    'logg': float(csv_logg[orig_i]),
                    'FeH':  float(csv_feh[orig_i]),
                    'Gmag': float(csv_gmag[orig_i]),
                    'SNR':  float(csv_snr[orig_i]),
                }
                has_any_csv_param = any(np.isfinite(v) for v in csv_pr.values())

                if has_any_csv_param:
                    # Fill missing CSV params from FITS group
                    fits_pr = _get_group_params(gid)
                    for p in csv_pr:
                        if not np.isfinite(csv_pr[p]) and np.isfinite(fits_pr.get(p, np.nan)):
                            csv_pr[p] = fits_pr[p]
                    use_pr = csv_pr
                    n_params_from_csv += 1
                else:
                    # No CSV params at all — inherit from FITS group
                    use_pr = _get_group_params(gid)
                    n_params_from_fits += 1

                ss[surv][gid]={'rvs':[],'params':use_pr,
                               'plx':float(plx_csv[orig_i]),'plxe':float(plxe_csv[orig_i])}
            ss[surv][gid]['rvs'].append((rv_v, err_v))

        n_added = n_sid + n_spatial
        log(f"    {surv}: {n_added:,} added "
            f"(sid={n_sid:,} spatial={n_spatial:,} plx_reject={n_plx_reject:,} unmatched={n_skip:,})"
            f"\n      params: {n_params_from_csv:,} from CSV, {n_params_from_fits:,} from FITS")

    ss=dict(ss)
    log("\n  Per-survey unique star counts (with param coverage):")
    for s in sorted(ss.keys()):
        n_m=sum(1 for g in ss[s].values() if len(g['rvs'])>=2)
        n_teff=sum(1 for g in ss[s].values() if np.isfinite(g['params'].get('Teff',np.nan)))
        n_gmag=sum(1 for g in ss[s].values() if np.isfinite(g['params'].get('Gmag',np.nan)))
        log(f"    {s:<16s}: {len(ss[s]):>10,} stars, {n_m:>8,} >=2obs, "
            f"Teff:{n_teff:,} Gmag:{n_gmag:,}")
    save_ckpt(ckpt,ss)
    gc.collect()
    return ss


# =============================================================================
# HELPER: WEIGHTED BEST RV PER STAR
# =============================================================================
def _best_rv(survey_stars):
    out={}
    for surv,stars in survey_stars.items():
        best={}
        for gid,sdata in stars.items():
            obs=sdata['rvs']; pr=sdata['params']
            if len(obs)==1:
                best[gid]=(obs[0][0],obs[0][1],pr)
            else:
                v=np.array([x[0] for x in obs]); e=np.array([x[1] for x in obs])
                ok=np.isfinite(v)&np.isfinite(e)&(e>0)
                if np.any(ok):
                    w=1.0/e[ok]**2
                    best[gid]=(float(np.sum(v[ok]*w)/np.sum(w)),float(1.0/np.sqrt(np.sum(w))),pr)
                else:
                    best[gid]=(float(np.nanmean(v)),np.nan,pr)
        out[surv]=best
    return out


# =============================================================================
# PHASE 4: DUP METHOD
# =============================================================================
# FIX-3: Skip GAIA; quality controls on pairs
# =============================================================================
def phase4_dup(cfg, survey_stars):
    ckpt=Path(cfg.checkpoint_dir)/"phase4_v8.pkl"
    if ckpt.exists():
        log("Phase 4: Loading checkpoint"); return load_ckpt(ckpt)
    log("\nPhase 4: DUP method (with quality controls)...")
    results={}
    for surv in sorted(survey_stars.keys()):
        if surv not in DUP_SURVEYS:
            log(f"  {surv}: excluded from DUP (using TCH)")
            continue
        stars=survey_stars[surv]
        multi={g:d for g,d in stars.items() if len(d['rvs'])>=2}
        if len(multi)<5: log(f"  {surv}: {len(multi)} multi-obs, skip DUP"); continue
        norms,raws=[],[]
        n_err_ratio_skip=0
        for gid,sdata in tqdm(multi.items(),desc=f"DUP {surv}",leave=False):
            obs=sdata['rvs']
            if len(obs)>cfg.max_obs_per_star:
                obs=sorted(obs,key=lambda x:x[1])[:cfg.max_obs_per_star]
            np_s=0
            for (v1,e1),(v2,e2) in combinations(obs,2):
                if abs(v1-v2)<1e-6 and abs(e1-e2)<1e-6: continue
                # FIX-3: Quality control — reject pairs with very different errors
                # (likely from different pipelines: CSV vs A95)
                if e1 > 0 and e2 > 0:
                    err_ratio = max(e1,e2)/min(e1,e2)
                    if err_ratio > 5.0:
                        n_err_ratio_skip += 1
                        continue
                d=v1-v2; sc=np.sqrt(e1**2+e2**2)
                if sc > 0.01:  # FIX-3: minimum combined error
                    norms.append(d/sc); raws.append(d); np_s+=1
                    if np_s>=cfg.max_pairs_per_star: break
        if len(norms)<cfg.min_pairs_dup:
            log(f"  {surv}: {len(norms)} pairs (skipped {n_err_ratio_skip} err-ratio), skip DUP")
            continue
        nd=np.array(norms); rd=np.array(raws)
        # FIX-3: 5σ clip before computing MAD
        nd_c=nd[np.abs(nd)<50]; rd_c=rd[np.abs(rd)<500]
        mad_raw=float(np.median(np.abs(nd_c)))
        nf_raw=1.4826*mad_raw
        # 5σ clip
        clip_thr = 5.0 * nf_raw if nf_raw > 0 else 50.0
        nd_clipped = nd_c[np.abs(nd_c) < clip_thr]
        if len(nd_clipped) < cfg.min_pairs_dup:
            nd_clipped = nd_c
        mad=float(np.median(np.abs(nd_clipped)))
        nf=1.4826*mad
        mu,std=norm.fit(nd_clipped)

        results[surv]={
            'norm_diffs':nd,'raw_diffs':rd,'norm_factor':nf,'mad':mad,
            'n_pairs':len(nd),'n_stars':len(multi),'mean_norm':mu,'std_norm':std,
            'mean_raw':float(np.mean(rd_c)),'std_raw':float(np.std(rd_c)),
            'median_raw':float(np.median(rd_c)),
            'mad_raw':float(np.median(np.abs(rd_c-np.median(rd_c)))),
            'norm_mad':nf,'norm_std':std,
            'n_err_ratio_skip':n_err_ratio_skip,
        }
        log(f"  {surv:<12s}: {len(multi):>8,} stars {len(nd):>10,} pairs "
            f"(skip_err_ratio={n_err_ratio_skip:,}) | normMAD={nf:.3f}")
    save_ckpt(ckpt,results)
    return results


# =============================================================================
# PHASE 5: TCH METHOD
# =============================================================================
# FIX-4: Sanity checks; proper combined factor logic
# =============================================================================
def phase5_tch(cfg, survey_stars, dup_results):
    ckpt=Path(cfg.checkpoint_dir)/"phase5_v8b.pkl"
    if ckpt.exists():
        log("Phase 5: Loading checkpoint"); return load_ckpt(ckpt)
    log("\nPhase 5: TCH method...")
    sb=_best_rv(survey_stars); surveys=sorted(sb.keys())
    a95_fallback = getattr(cfg, 'a95_s_rvcor_factors', {}) or {}

    # Pairwise ZP
    pzp={}
    for si,sj in combinations(surveys,2):
        common=set(sb[si])&set(sb[sj])
        if len(common)<cfg.min_stars_tch: continue
        diffs=[sb[si][g][0]-sb[sj][g][0] for g in common
               if np.isfinite(sb[si][g][0]) and np.isfinite(sb[sj][g][0])]
        if len(diffs)>=cfg.min_stars_tch:
            pzp[(si,sj)]=float(np.median(diffs))

    # Pairwise variances
    pw={}
    for si,sj in combinations(surveys,2):
        common=set(sb[si])&set(sb[sj])
        if len(common)<cfg.min_stars_tch: continue
        zpc=pzp.get((si,sj),-pzp.get((sj,si),0.0) if (sj,si) in pzp else 0.0)
        diffs=np.array([(sb[si][g][0]-sb[sj][g][0])-zpc for g in common
                        if np.isfinite(sb[si][g][0]) and np.isfinite(sb[sj][g][0])])
        if len(diffs)<cfg.min_stars_tch: continue
        mad_raw=float(np.median(np.abs(diffs-np.median(diffs))))
        sig_rob=1.4826*mad_raw; clip_thr=5.0*sig_rob if sig_rob>0 else np.inf
        mask=np.abs(diffs-np.median(diffs))<=clip_thr
        d_clean=diffs[mask]
        if len(d_clean)<cfg.min_stars_tch: d_clean=diffs
        mad_c=float(np.median(np.abs(d_clean-np.median(d_clean))))
        var=(1.4826*mad_c)**2

        # FIX-4: Flag suspiciously low variances
        if var < 1e-6:
            log(f"  WARNING: {si}-{sj}: σ²≈0 ({var:.2e}), likely same RV source — skipping pair")
            continue

        pw[(si,sj)]={'var':var,'n':len(d_clean),'sigma':np.sqrt(var)}
        log(f"  {si}-{sj}: {len(d_clean):,} clean, σ_robust={np.sqrt(var):.3f} km/s")

    tch={'pairwise':pw,'survey_sigma':{},'norm_factors':{},'triplets_used':{}}
    for target in surveys:
        partners=[]
        for (si,sj) in pw:
            if si==target: partners.append(sj)
            elif sj==target: partners.append(si)
        ests=[]
        for p1,p2 in combinations(partners,2):
            keys=[tuple(sorted([target,p1])),tuple(sorted([target,p2])),tuple(sorted([p1,p2]))]
            if not all(k in pw for k in keys): continue
            s2=(pw[keys[0]]['var']+pw[keys[1]]['var']-pw[keys[2]]['var'])/2
            nm=min(pw[k]['n'] for k in keys); ests.append((s2,nm))
        if not ests: continue
        valid=[(s,n) for s,n in ests if s>0]
        sig=(np.sqrt(sum(s*n for s,n in valid)/sum(n for _,n in valid)) if valid else 0.0)
        errs=[d[1] for d in sb[target].values() if np.isfinite(d[1]) and d[1]>0]
        me=float(np.median(errs)) if errs else np.nan
        nf=sig/me if (np.isfinite(me) and me>0 and sig>0) else np.nan
        tch['survey_sigma'][target]=sig; tch['norm_factors'][target]=nf
        log(f"  TCH {target:<12s}: σ={sig:.3f} km/s  medErr={me:.3f}  f={nf:.3f}")

    # ── Combined factor — per paper Table 4 logic ──────────────────────────
    # FIX-4: Sanity checks on factor plausibility
    log("\n  === COMBINED NORMALIZATION FACTORS ===")
    log(f"  {'Survey':<14s}  {'DUP_f':>7s}  {'TCH_f':>7s}  {'A95_f':>7s}  {'Combined':>9s}  Method")
    log("  "+"-"*74)
    combined={}; reliability={}
    for surv in surveys:
        df_val=dup_results.get(surv,{}).get('norm_factor')
        df_n  =dup_results.get(surv,{}).get('n_pairs',0)
        tf_val=tch['norm_factors'].get(surv)
        a95_val=a95_fallback.get(surv)

        # FIX-4: Plausibility checks
        dup_ok=(df_val is not None and np.isfinite(df_val)
                and MIN_PLAUSIBLE_FACTOR < df_val < MAX_PLAUSIBLE_FACTOR
                and df_n>=MIN_DUP_PAIRS_SIGNIFICANT)
        tch_ok=(tf_val is not None and np.isfinite(tf_val)
                and MIN_PLAUSIBLE_FACTOR < tf_val < MAX_PLAUSIBLE_FACTOR)
        a95_ok=(a95_val is not None and np.isfinite(a95_val)
                and MIN_PLAUSIBLE_FACTOR < a95_val < MAX_PLAUSIBLE_FACTOR)

        if surv in DUP_ONLY_SURVEYS:
            if dup_ok: f,method=df_val,'DUP only'
            elif a95_ok: f,method=a95_val,'A95 fallback'
            elif tch_ok: f,method=tf_val,'TCH fallback (DUP implausible)'
            else: f,method=1.0,'default=1.0'
        elif surv in TCH_ONLY_SURVEYS:
            if tch_ok: f,method=tf_val,'TCH only'
            elif a95_ok: f,method=a95_val,'A95 fallback'
            else: f,method=1.0,'default=1.0'
        elif surv in AVG_BOTH_SURVEYS:
            if dup_ok and tch_ok: f,method=(df_val+tf_val)/2,'avg(DUP,TCH)'
            elif dup_ok: f,method=df_val,'DUP only'
            elif tch_ok: f,method=tf_val,'TCH only'
            elif a95_ok: f,method=a95_val,'A95 fallback'
            else: f,method=1.0,'default=1.0'
        else:  # DESI, SDSS, etc.
            if tch_ok: f,method=tf_val,'TCH preferred'
            elif dup_ok: f,method=df_val,'DUP fallback'
            elif a95_ok: f,method=a95_val,'A95 fallback'
            else: f,method=1.0,'default=1.0'

        combined[surv]=f; reliability[surv]=method
        ds=f"{df_val:.3f}" if df_val is not None and np.isfinite(df_val) else "  n/a "
        ts_=f"{tf_val:.3f}" if tf_val is not None and np.isfinite(tf_val) else "  n/a "
        as_=f"{a95_val:.3f}" if a95_val is not None and np.isfinite(a95_val) else "  n/a "
        log(f"  {surv:<14s}  {ds:>7s}  {ts_:>7s}  {as_:>7s}  {f:>9.3f}  {method}")

    tch['combined_factors']=combined; tch['pairwise_zp']=pzp
    tch['reliability']=reliability
    save_ckpt(ckpt,tch)
    return tch


# =============================================================================
# PHASE 6: GAIA INTERNAL CALIBRATION
# =============================================================================
# FIX-5: Restore paper §5.1 exclusion rules (Eq5/6/7)
# =============================================================================
def phase6_gaia_cal(cfg, survey_stars, tch_results):
    ckpt=Path(cfg.checkpoint_dir)/"phase6_v8b.pkl"
    if ckpt.exists():
        log("Phase 6: Loading checkpoint"); return load_ckpt(ckpt)

    log("\nPhase 6: Gaia internal calibration (paper §5.1 rules)...")
    sb=_best_rv(survey_stars)
    if 'GAIA' not in sb:
        log("  GAIA not found, skipping Phase 6")
        return {'coefficients':{},'per_survey':{},'zp_shifts':{}}

    gaia_data=sb['GAIA']
    results={'coefficients':{},'per_survey':{},'zp_shifts':{}}

    # FIX-5: Paper exclusion rules
    all_non_gaia = [s for s in sb if s != 'GAIA']
    eq5_surveys = [s for s in all_non_gaia if s not in EQ5_GMAG_EXCLUDE]
    eq6_surveys = [s for s in all_non_gaia if s not in EQ6_FEH_EXCLUDE]
    eq7_surveys = [s for s in all_non_gaia if s in EQ7_TEFF_SURVEYS]

    cal_specs=[
        ('Gmag', eq5_surveys, 2, 'Eq5'),
        ('FeH',  eq6_surveys, 1, 'Eq6'),
        ('Teff', eq7_surveys, 2, 'Eq7'),
    ]

    for cal_param, cal_surveys, degree, eq_name in cal_specs:
        log(f"\n  --- {eq_name}: ΔRV vs {cal_param} ---")
        log(f"      Surveys used: {cal_surveys}")
        per_surv_coeffs=[]
        zp_row={}

        for surv in cal_surveys:
            if surv not in sb: continue
            sd=sb[surv]
            common=set(gaia_data)&set(sd)
            if len(common)<cfg.min_stars_zp: continue

            # Collect ΔRV with parameter values
            all_drv, all_pval = [], []
            for g in common:
                rv_g, _, pr_g = gaia_data[g]
                rv_s, _, pr_s = sd[g]
                if not (np.isfinite(rv_g) and np.isfinite(rv_s)): continue
                pval = pr_s.get(cal_param, pr_g.get(cal_param, np.nan))
                if not np.isfinite(pval): continue
                all_drv.append(rv_g - rv_s)
                all_pval.append(pval)

            if len(all_drv) < cfg.min_stars_zp:
                log(f"    {surv}: {len(all_drv)} stars with {cal_param}, too few")
                continue

            all_drv = np.array(all_drv)
            all_pval = np.array(all_pval)

            # ZP shift = median(ΔRV) — paper §5.1
            zp_shift = float(np.median(all_drv))
            zp_row[surv] = zp_shift

            # ZP-subtracted ΔRV with weights
            xs, ys, ws = [], [], []
            for g in common:
                rv_g, err_g, pr_g = gaia_data[g]
                rv_s, err_s, pr_s = sd[g]
                if not (np.isfinite(rv_g) and np.isfinite(rv_s)): continue
                pval = pr_s.get(cal_param, pr_g.get(cal_param, np.nan))
                if not np.isfinite(pval): continue
                drv = (rv_g - rv_s) - zp_shift
                comb_err = (np.sqrt(err_g**2 + err_s**2)
                            if (np.isfinite(err_g) and np.isfinite(err_s)) else 1.0)
                comb_err = max(comb_err, 0.01)
                xs.append(pval); ys.append(drv); ws.append(1.0/comb_err**2)

            if len(xs) < 50: continue
            xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)

            results['per_survey'][f'{eq_name}_{surv}'] = {
                'xs':xs,'ys':ys,'ws':ws,'zp':zp_shift,'n':len(xs),'surv':surv,
                'param':cal_param,'eq_name':eq_name}

            # Bin data → fit polynomial to binned medians
            n_bins = min(40, max(15, len(xs)//300))
            lo, hi = np.percentile(xs, 1), np.percentile(xs, 99)
            be = np.linspace(lo, hi, n_bins+1)
            bc = 0.5*(be[:-1]+be[1:])
            bin_x, bin_y, bin_w = [], [], []
            for j in range(n_bins):
                m = (xs>=be[j]) & (xs<be[j+1])
                if m.sum() < cfg.min_bin_count: continue
                bin_x.append(bc[j])
                bin_y.append(float(np.median(ys[m])))
                bin_w.append(float(m.sum()))

            if len(bin_x) < degree+2: continue
            bx, by, bw = np.array(bin_x), np.array(bin_y), np.array(bin_w)
            ok = np.isfinite(bx) & np.isfinite(by)
            bx, by, bw = bx[ok], by[ok], bw[ok]
            coeffs = np.polyfit(bx, by, degree, w=np.sqrt(bw))
            per_surv_coeffs.append((coeffs, len(xs)))

            log(f"    {surv}: N={len(xs):,}  N_bins={len(bx)}"
                f"  ZP={zp_shift:+.3f}  coeffs={np.array2string(coeffs,precision=5)}")

        results['zp_shifts'][eq_name] = zp_row

        if not per_surv_coeffs:
            log(f"    {eq_name}: no surveys produced valid fits"); continue

        # Global = weighted mean of per-survey coefficients
        coeffs_arr = np.array([c for c,_ in per_surv_coeffs])
        weights_arr = np.array([n for _,n in per_surv_coeffs], dtype=float)
        global_coeffs = (coeffs_arr * weights_arr[:,None]).sum(axis=0) / weights_arr.sum()
        results['coefficients'][eq_name] = global_coeffs.tolist()
        log(f"    {eq_name} GLOBAL (weighted mean of {len(per_surv_coeffs)} surveys): "
            f"coeffs={np.array2string(global_coeffs,precision=6)}")

    save_ckpt(ckpt, results)
    return results


# =============================================================================
# PHASE 7: SURVEY CALIBRATION (Eq.8)  (unchanged logic from v7)
# =============================================================================
def phase7_survey_cal(cfg, survey_stars, gaia_cal):
    ckpt=Path(cfg.checkpoint_dir)/"phase7_v8.pkl"
    if ckpt.exists():
        log("Phase 7: Loading checkpoint"); return load_ckpt(ckpt)
    log("\nPhase 7: Survey calibration (Eq.8)...")
    sb=_best_rv(survey_stars)
    if 'GAIA' not in sb:
        log("  GAIA not found, skipping"); return {}

    gaia=sb['GAIA']; gaia_coeffs=gaia_cal.get('coefficients',{})

    def corrected_gaia_rv(gid):
        rv_g,err_g,pr=gaia[gid]; corr=0.0
        if 'Eq5' in gaia_coeffs:
            gmag=pr.get('Gmag',np.nan)
            if np.isfinite(gmag): corr+=np.polyval(gaia_coeffs['Eq5'],gmag)
        if 'Eq6' in gaia_coeffs:
            feh=pr.get('FeH',np.nan)
            if np.isfinite(feh): corr+=np.polyval(gaia_coeffs['Eq6'],feh)
        if 'Eq7' in gaia_coeffs:
            teff=pr.get('Teff',np.nan)
            if np.isfinite(teff): corr+=np.polyval(gaia_coeffs['Eq7'],teff)
        return rv_g-corr, err_g

    results={}
    for surv in sorted(sb.keys()):
        if surv=='GAIA': continue
        sd=sb[surv]; common=set(gaia)&set(sd)
        if len(common)<50: continue
        rows=[]
        for g in common:
            rv_g_c,err_g=corrected_gaia_rv(g); rv_s,err_s,pr_s=sd[g]
            if not (np.isfinite(rv_g_c) and np.isfinite(rv_s)): continue
            drv=rv_g_c-rv_s
            ce=(np.sqrt(err_g**2+err_s**2) if (np.isfinite(err_g) and np.isfinite(err_s)) else 1.0)
            ce=max(ce,0.01)
            rows.append({'drv':drv,'weight':1.0/ce**2,
                         'Teff':pr_s.get('Teff',np.nan),'logg':pr_s.get('logg',np.nan),
                         'FeH':pr_s.get('FeH',np.nan),'SNR':pr_s.get('SNR',np.nan),
                         'RV':rv_s,'Gmag':pr_s.get('Gmag',np.nan)})
        if len(rows)<50: continue
        df=pd.DataFrame(rows); log(f"  {surv}: {len(df):,} stars for Eq.8")

        splits=[]
        if surv=='LAMOST':
            lc=df[df['Teff']<6200].copy(); lh=df[df['Teff']>=6200].copy()
            if len(lc)>50: splits.append(('cool_Teff<6200K',lc))
            if len(lh)>50: splits.append(('hot_Teff>=6200K',lh))
        else:
            dw=df[df['logg']>3.5].copy(); gi=df[df['logg']<=3.5].copy()
            if len(dw)>50: splits.append(('dwarfs_logg>3.5',dw))
            if len(gi)>50: splits.append(('giants_logg<=3.5',gi))
        if not splits: splits=[('all',df)]

        surv_results={'diag':df,'fits':{}}
        for split_name,sdf in splits:
            features,feat_names=[],[]
            for fname,col,use_sq in [
                ('Teff2','Teff',True),('Teff','Teff',False),
                ('logg','logg',False),('FeH','FeH',False),
                ('SNR','SNR',False),('RV','RV',False)]:
                vals=sdf[col].values if col in sdf.columns else np.full(len(sdf),np.nan)
                if use_sq: vals=vals**2
                if np.sum(np.isfinite(vals))>len(sdf)*0.3:
                    features.append(np.where(np.isfinite(vals),vals,0))
                    feat_names.append(fname)
            if not features: continue
            X=np.column_stack(features+[np.ones(len(sdf))]); feat_names.append('intercept')
            y=sdf['drv'].values; w=sdf['weight'].values
            ok=np.isfinite(y)&np.isfinite(w)&(w>0)
            X,y,w=X[ok],y[ok],w[ok]
            if len(y)<20: continue
            try:
                XtWX=(X.T*(w[None,:]))@X; XtWy=X.T@(w*y); coeffs=np.linalg.solve(XtWX,XtWy)
                resid=y-X@coeffs; chi2=np.sum(w*resid**2)/(len(y)-len(coeffs))
                surv_results['fits'][split_name]={
                    'coeffs':coeffs.tolist(),'feat_names':feat_names,'n':len(y),
                    'chi2':chi2,'zp_before':float(np.median(y)),'zp_after':float(np.median(resid))}
                log(f"    {surv}/{split_name}: N={len(y):,}  chi2={chi2:.3f}"
                    f"  ZP: {np.median(y):+.3f} → {np.median(resid):+.3f} km/s")
            except Exception as e:
                log(f"    {surv}/{split_name}: fit failed: {e}")
        results[surv]=surv_results

    save_ckpt(ckpt,results)
    return results


# =============================================================================
# PHASE 8: NORMALIZED ERRORS
# =============================================================================
def phase8_norm_errors(cfg, survey_stars, tch_results):
    ckpt=Path(cfg.checkpoint_dir)/"phase8_v8.pkl"
    if ckpt.exists():
        log("Phase 8: Loading checkpoint"); return load_ckpt(ckpt)
    log("\nPhase 8: Normalized RV errors (per unique star)...")
    cf=tch_results.get('combined_factors',{}); result={}
    for surv in sorted(survey_stars.keys()):
        f=cf.get(surv,1.0); errs=[]
        for gid,sdata in survey_stars[surv].items():
            obs=sdata['rvs']; es=np.array([e for _,e in obs])
            ok=np.isfinite(es)&(es>0)
            if not np.any(ok): continue
            w_err=(float(es[ok][0]) if np.sum(ok)==1
                   else float(1.0/np.sqrt(np.sum(1.0/es[ok]**2))))
            errs.append(w_err*f)
        if errs:
            ea=np.array(errs)
            result[surv]={'errors':ea,'factor':f,
                          'median':float(np.median(ea)),'n':len(ea)}
            log(f"  {surv:<14s}: factor={f:.3f}  median_norm_err={np.median(ea):.3f}  N={len(ea):,}")
    save_ckpt(ckpt,result)
    return result


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
SIX_PARAMS=[('Gmag','G mag'),('RV','RV (km/s)'),('Teff','T$_{eff}$ (K)'),
            ('logg','log g (dex)'),('FeH','[Fe/H] (dex)'),('SNR','S/N')]
PARAM_TO_EQ={'Gmag':'Eq5','FeH':'Eq6','Teff':'Eq7'}


# =============================================================================
# PHASE 9: ALL PLOTS + CSV  (FIX-6: paper-quality figures)
# =============================================================================
def phase9_plots(cfg, dup_results, tch_results, gaia_cal, survey_cal,
                 nerr, survey_stars):
    log("\nPhase 9: Generating plots and tables...")
    out=Path(cfg.output_dir)

    # ── Fig 6: DUP normalized ΔRV/σ distributions (paper Fig 6) ───────────
    log("  Fig 6: DUP normalized differences...")
    fig,ax=plt.subplots(figsize=(10,6))
    bins=np.linspace(-15,15,160); bc=0.5*(bins[:-1]+bins[1:])
    xg=np.linspace(-15,15,500)
    ax.plot(xg,norm.pdf(xg,0,1),'k--',lw=2,alpha=0.5,label='N(0,1)')
    csv1={'bin_center':bc}
    for s in sorted(dup_results.keys()):
        dr=dup_results[s]; nd=dr['norm_diffs']
        ndp=nd[(nd>=-15)&(nd<=15)]
        if len(ndp)<50: continue
        c,_=np.histogram(ndp,bins=bins,density=True)
        csv1[s]=c
        ax.hist(bins[:-1],bins,weights=c,histtype='step',lw=2,color=_c(s),alpha=0.85,
                label=f"{s}  (normMAD={dr['norm_mad']:.2f}, N={len(ndp):,})")
    ax.set_xlim(-15,15)
    ax.set_xlabel(r'$\Delta$RV / $\sqrt{\sigma_1^2+\sigma_2^2}$',fontsize=13)
    ax.set_ylabel('Normalized density',fontsize=13)
    ax.set_title('DUP Method: Normalized RV Differences (Paper Fig. 6)',fontsize=12)
    ax.legend(fontsize=9,loc='upper right'); ax.grid(True,alpha=0.25,ls='--')
    fig.tight_layout()
    for e in ['png','pdf']: fig.savefig(out/f'fig6.{e}',dpi=300)
    plt.close(fig)
    pd.DataFrame(csv1).to_csv(out/'fig6_data.csv',index=False)

    # ── Fig 7: ΔRV histograms before/after ZP correction ─────────────────
    log("  Fig 7: ΔRV histograms before/after ZP...")
    sb=_best_rv(survey_stars)
    if 'GAIA' in sb:
        gaia_d=sb['GAIA']
        fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=False)
        bins7=np.linspace(-20,20,160)
        stats_rows=[]
        for ax,after in zip(axes,[False,True]):
            ax.axvline(0,color='k',ls=':',lw=1,alpha=0.4)
            for surv in sorted(sb.keys()):
                if surv=='GAIA': continue
                common=set(gaia_d)&set(sb[surv])
                drvs=np.array([gaia_d[g][0]-sb[surv][g][0] for g in common
                               if np.isfinite(gaia_d[g][0]) and np.isfinite(sb[surv][g][0])])
                if len(drvs)<30: continue
                zp=float(np.median(drvs))
                d_plot=drvs-zp if after else drvs
                d_plot=d_plot[(d_plot>=-20)&(d_plot<=20)]
                if len(d_plot)<10: continue
                c,_=np.histogram(d_plot,bins=bins7,density=True)
                ax.step(bins7[:-1],c,where='post',color=_c(surv),lw=1.8,alpha=0.85,
                        label=f"{surv}  med={np.median(d_plot):+.2f}  MAD={np.median(np.abs(d_plot-np.median(d_plot))):.2f}")
                if not after:
                    stats_rows.append({'Survey':surv,'N':len(drvs),'Median_ZP':zp,
                        'MAD':float(np.median(np.abs(drvs-zp)))})
            title='Before ZP correction' if not after else 'After ZP correction'
            ax.set_xlabel(r'$\Delta$RV = RV$_{\rm Gaia}$ − RV$_{\rm survey}$ (km/s)',fontsize=11)
            ax.set_ylabel('Normalized density',fontsize=11)
            ax.set_title(title,fontsize=11); ax.set_xlim(-20,20)
            ax.legend(fontsize=7,loc='upper right'); ax.grid(True,alpha=0.25,ls='--')
        fig.suptitle(r'$\Delta$RV Histograms (Paper Fig. 7)',fontsize=12)
        fig.tight_layout()
        for e in ['png','pdf']: fig.savefig(out/f'fig7.{e}',dpi=300)
        plt.close(fig)
        if stats_rows: pd.DataFrame(stats_rows).to_csv(out/'fig7_zp_stats.csv',index=False)
        log("    fig7 saved.")

    # ── Fig 8: ΔRV vs 6 parameters — all surveys (paper Fig 8) ───────────
    log("  Fig 8: ΔRV vs 6 params (all surveys)...")
    if 'GAIA' in sb:
        gaia_d=sb['GAIA']
        coeffs_all=gaia_cal.get('coefficients',{})
        per_surv=gaia_cal.get('per_survey',{})
        fig,axes=plt.subplots(2,3,figsize=(21,12)); axes=axes.flatten()
        for pi,(pname,xlabel) in enumerate(SIX_PARAMS):
            ax=axes[pi]
            eq_name=PARAM_TO_EQ.get(pname)
            global_coeffs=coeffs_all.get(eq_name) if eq_name else None
            for surv in sorted(sb.keys()):
                if surv=='GAIA': continue
                common=set(gaia_d)&set(sb[surv])
                if len(common)<30: continue
                xs,ys=[],[]
                for g in common:
                    rv_g,_,pr_g=gaia_d[g]; rv_s,_,pr_s=sb[surv][g]
                    if not (np.isfinite(rv_g) and np.isfinite(rv_s)): continue
                    pval=pr_s.get(pname,pr_g.get(pname,np.nan))
                    if pname=='RV': pval=rv_s
                    if not np.isfinite(pval): continue
                    xs.append(pval); ys.append(rv_g-rv_s)
                if len(xs)<20: continue
                xs,ys=np.array(xs),np.array(ys)
                bc,meds,mads=bin_stat(xs,ys,n_bins=35,min_count=cfg.min_bin_count)
                if len(bc)==0: continue
                ax.plot(bc,meds,'-',color=_c(surv),lw=1.8,alpha=0.9,label=surv)
                ax.fill_between(bc,meds-mads,meds+mads,color=_c(surv),alpha=0.1)
            if global_coeffs is not None:
                all_xs_flat=[]
                for surv in sorted(sb.keys()):
                    if surv=='GAIA': continue
                    key=f"{eq_name}_{surv}" if eq_name else None
                    if key and key in per_surv: all_xs_flat.append(per_surv[key]['xs'])
                if all_xs_flat:
                    all_xs_flat=np.concatenate(all_xs_flat)
                    x_fit=np.linspace(np.percentile(all_xs_flat,1),np.percentile(all_xs_flat,99),300)
                    ax.plot(x_fit,np.polyval(global_coeffs,x_fit),'k--',lw=2.2,alpha=0.8,
                            label='Global fit',zorder=10)
            ax.axhline(0,color='gray',ls=':',alpha=0.5)
            ax.set_xlabel(xlabel,fontsize=11)
            ax.set_ylabel(r'$\Delta$RV (km/s)',fontsize=10)
            ax.set_title(f'ΔRV vs {pname}',fontsize=11); ax.set_ylim(-15,15)
            ax.legend(fontsize=7,loc='upper right',ncol=2); ax.grid(True,alpha=0.25,ls='--')
        fig.suptitle(r'$\Delta$RV = RV$_{\rm Gaia}$ − RV$_{\rm survey}$ vs parameters (Paper Fig. 8)',fontsize=12)
        fig.tight_layout()
        for e in ['png','pdf']: fig.savefig(out/f'fig8_all.{e}',dpi=300)
        plt.close(fig)
        log("    fig8_all saved.")

    # ── Figs 9-11: Gaia calibration before/after per equation ─────────────
    coeffs_all=gaia_cal.get('coefficients',{})
    per_surv=gaia_cal.get('per_survey',{})
    for eq_name,coeffs in coeffs_all.items():
        keys=[k for k in per_surv if k.startswith(eq_name)]
        if not keys: continue
        param=per_surv[keys[0]]['param']
        log(f"  Gaia cal before/after: {eq_name} ({param})...")
        fig,axes=plt.subplots(1,2,figsize=(16,6))
        for ax,show_after in zip(axes,[False,True]):
            for k in keys:
                sv=per_surv[k]; sn=sv['surv']
                xs_=sv['xs']; ys_=sv['ys']
                bc,meds,mads=bin_stat(xs_,ys_,n_bins=30,min_count=cfg.min_bin_count)
                if len(bc)==0: continue
                if show_after:
                    ys_c=ys_-np.polyval(coeffs,xs_)
                    bc2,meds2,mads2=bin_stat(xs_,ys_c,n_bins=30,min_count=cfg.min_bin_count)
                    if len(bc2)==0: continue
                    ax.plot(bc2,meds2,'o-',ms=4,lw=1.5,color=_c(sn),alpha=0.85,label=sn)
                    ax.fill_between(bc2,meds2-mads2,meds2+mads2,color=_c(sn),alpha=0.1)
                else:
                    ax.plot(bc,meds,'o-',ms=4,lw=1.5,color=_c(sn),alpha=0.85,label=sn)
                    ax.fill_between(bc,meds-mads,meds+mads,color=_c(sn),alpha=0.1)
            if not show_after:
                all_xs=[per_surv[k]['xs'] for k in keys if len(per_surv[k]['xs'])>0]
                if all_xs:
                    axx=np.concatenate(all_xs)
                    xf=np.linspace(np.percentile(axx,1),np.percentile(axx,99),300)
                    ax.plot(xf,np.polyval(coeffs,xf),'k--',lw=2.2,alpha=0.8,
                            label='Global fit',zorder=10)
            ax.axhline(0,color='gray',ls=':',alpha=0.5)
            ax.set_xlabel(param,fontsize=12)
            ax.set_ylabel(r'$\Delta$RV (km/s)',fontsize=11)
            t='After calibration' if show_after else 'Before calibration'
            ax.set_title(f'{eq_name} ({param}): {t}',fontsize=12)
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3,ls='--')
        fig.suptitle(f'Gaia Calibration {eq_name}: ΔRV vs {param} (Paper Figs. 9-11)',fontsize=13)
        fig.tight_layout()
        for e in ['png','pdf']: fig.savefig(out/f'gaia_cal_{param}.{e}',dpi=300)
        plt.close(fig)

    # ── Fig 12: Calibrated ΔRV vs 6 params (paper Fig 12) ────────────────
    log("  Fig 12: Calibrated ΔRV vs 6 params...")
    param_list=['Teff','logg','FeH','SNR','RV','Gmag']
    all_survey_bins={p:{} for p in param_list}
    for surv,sres in survey_cal.items():
        if 'diag' not in sres: continue
        df=sres['diag']
        for pname in param_list:
            if pname not in df.columns or df[pname].isna().all(): continue
            x=df[pname].values; y=df['drv'].values
            ok=np.isfinite(x)&np.isfinite(y)
            if np.sum(ok)<20: continue
            bc,meds,_=bin_stat(x[ok],y[ok],n_bins=30,min_count=cfg.min_bin_count)
            if len(bc)>0: all_survey_bins[pname][surv]=(bc,meds)

    fig,axes=plt.subplots(2,3,figsize=(21,12)); axes=axes.flatten()
    for pi,pname in enumerate(param_list):
        ax=axes[pi]
        sdata={s:v for s,v in all_survey_bins[pname].items() if len(v[0])>0}
        for surv,(bc,med) in sdata.items():
            ax.plot(bc,med,'o-',ms=4,lw=1.8,color=_c(surv),alpha=0.85,label=surv)
        ax.axhline(0,color='gray',ls='--',lw=1,alpha=0.6)
        ax.set_xlabel(pname,fontsize=11)
        ax.set_ylabel(r'$\Delta$RV (km/s)',fontsize=10)
        ax.set_title(f'Calibrated ΔRV vs {pname}',fontsize=11)
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3,ls='--'); ax.set_ylim(-5,5)
    fig.suptitle('Survey Calibration: Calibrated ΔRV vs parameters (Paper Fig. 12)',fontsize=13)
    fig.tight_layout()
    for e in ['png','pdf']: fig.savefig(out/f'fig12_calibrated.{e}',dpi=300)
    plt.close(fig)

    # ── Fig 13: Normalized RV error distributions ─────────────────────────
    log("  Fig 13: Normalized RV errors...")
    fig,ax=plt.subplots(figsize=(10,6))
    FIXED_BINS=np.linspace(0.0,4.0,101)
    fixed_bc=0.5*(FIXED_BINS[:-1]+FIXED_BINS[1:])
    csv4={'bin_center':fixed_bc}
    for s in sorted(nerr.keys()):
        ea=nerr[s]['errors']; ep=ea[(ea>=0)&(ea<=4)]
        if len(ep)<20: continue
        c,_=np.histogram(ep,bins=FIXED_BINS,density=True)
        csv4[s]=c
        ax.hist(FIXED_BINS[:-1],FIXED_BINS,weights=c,histtype='step',lw=2,
                color=_c(s),alpha=0.85,
                label=(f"{s}  (med={nerr[s]['median']:.2f} km/s, "
                       f"N={nerr[s]['n']:,}, f={nerr[s]['factor']:.3f})"))
    ax.set_xlim(0,4)
    ax.set_xlabel('Normalized RV error (km/s)',fontsize=13)
    ax.set_ylabel('Density',fontsize=13)
    ax.set_title('Normalized RV Error Distributions (Paper Fig. 13)',fontsize=12)
    ax.legend(fontsize=9,loc='upper right'); ax.grid(True,alpha=0.3,ls='--')
    fig.tight_layout()
    for e in ['png','pdf']: fig.savefig(out/f'fig13.{e}',dpi=300)
    plt.close(fig)
    if len(csv4)>1: pd.DataFrame(csv4).to_csv(out/'fig13_data.csv',index=False)

    # ── CSV tables ─────────────────────────────────────────────────────────
    log("  Writing CSV tables...")

    # Table 3: DUP statistics
    rows=[{'Survey':s,'N_stars':d['n_stars'],'N_pairs':d['n_pairs'],
           'Mean_raw':d['mean_raw'],'Std_raw':d['std_raw'],
           'Median_raw':d['median_raw'],'MAD_raw':d['mad_raw'],
           'NormMAD':d['norm_mad'],'NormStd':d['norm_std']}
          for s,d in sorted(dup_results.items())]
    if rows: pd.DataFrame(rows).to_csv(out/'table3_dup.csv',index=False)

    # Table 4: Combined normalization factors
    cf=tch_results.get('combined_factors',{})
    rel=tch_results.get('reliability',{})
    rows=[{'Survey':s,'Combined_Factor':cf[s],
           'DUP_factor':dup_results.get(s,{}).get('norm_factor',np.nan),
           'TCH_factor':tch_results.get('norm_factors',{}).get(s,np.nan),
           'Method':rel.get(s,'')}
          for s in sorted(cf.keys())]
    if rows: pd.DataFrame(rows).to_csv(out/'table4_norm_factors.csv',index=False)

    # Table 5: ΔRV statistics (before and after)
    if 'GAIA' in sb:
        gaia_d=sb['GAIA']
        t5_rows=[]
        for surv in sorted(sb.keys()):
            if surv=='GAIA': continue
            common=set(gaia_d)&set(sb[surv])
            drvs=np.array([gaia_d[g][0]-sb[surv][g][0] for g in common
                           if np.isfinite(gaia_d[g][0]) and np.isfinite(sb[surv][g][0])])
            if len(drvs)<10: continue
            t5_rows.append({'Survey':surv,'N':len(drvs),
                           'Mean_DRV':float(np.mean(drvs)),
                           'Median_DRV':float(np.median(drvs)),
                           'Sigma':float(np.std(drvs)),
                           'MAD':float(np.median(np.abs(drvs-np.median(drvs))))})
        if t5_rows: pd.DataFrame(t5_rows).to_csv(out/'table5_drv_stats.csv',index=False)

    # Gaia calibration coefficients
    gc_=gaia_cal.get('coefficients',{})
    if gc_:
        pd.DataFrame([{'Equation':k,'Coefficients':str(v)} for k,v in gc_.items()]
                     ).to_csv(out/'gaia_cal_coefficients.csv',index=False)

    # ZP shifts per equation per survey
    zp_all=gaia_cal.get('zp_shifts',{})
    if zp_all:
        zp_rows=[]
        for eq,zdict in zp_all.items():
            for surv,zp in zdict.items():
                zp_rows.append({'Equation':eq,'Survey':surv,'ZP_shift_km_s':zp})
        pd.DataFrame(zp_rows).to_csv(out/'gaia_cal_zp_shifts.csv',index=False)

    # Survey calibration coefficients
    rows=[]
    for surv,sres in survey_cal.items():
        for split,fdata in sres.get('fits',{}).items():
            rows.append({'Survey':surv,'Split':split,
                         'Features':str(fdata['feat_names']),
                         'Coefficients':str(fdata['coeffs']),'N':fdata['n'],
                         'Chi2':fdata['chi2'],'ZP_before_km_s':fdata['zp_before'],
                         'ZP_after_km_s':fdata['zp_after']})
    if rows: pd.DataFrame(rows).to_csv(out/'survey_cal_coefficients.csv',index=False)

    # Summary per survey
    rows=[{'Survey':s,'N_unique_stars':nerr.get(s,{}).get('n',0),
           'Norm_factor':nerr.get(s,{}).get('factor',np.nan),
           'Median_norm_err_km_s':nerr.get(s,{}).get('median',np.nan)}
          for s in sorted(VALID_SURVEYS)]
    pd.DataFrame(rows).to_csv(out/'summary_unique_stars.csv',index=False)
    log("  All outputs saved.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    p=argparse.ArgumentParser(description='RV Normalization v8 (fixed)')
    p.add_argument('input_fits')
    p.add_argument('--output-dir','-o',default='./rv_norm_output_v8')
    p.add_argument('--checkpoint-dir',default='./rv_norm_ckpt_v8')
    p.add_argument('--chunk-size','-c',type=int,default=3_000_000)
    p.add_argument('--tolerance',type=float,default=1.0)
    p.add_argument('--nside',type=int,default=32)
    p.add_argument('--apogee-csv',default='./astro_data/APOGEE_DR17/APOGEE_DR17_merged_rv_parallax.csv')
    p.add_argument('--galah-csv', default='./astro_data/GALAH_DR3/GALAH_DR3_merged_rv_parallax.csv')
    p.add_argument('--ges-csv',   default='./astro_data/GES_DR5/GES_DR5_merged_rv_parallax.csv')
    p.add_argument('--rave-csv',  default='./astro_data/RAVE_DR6/RAVE_DR6_merged_rv_parallax.csv')
    p.add_argument('--a95-dir',   default='./astro_data/A95_cds')
    p.add_argument('--clean',action='store_true',help='Delete checkpoints and restart')
    args=p.parse_args()

    if args.clean:
        import shutil
        if os.path.exists(args.checkpoint_dir):
            shutil.rmtree(args.checkpoint_dir)
            print(f"Cleaned {args.checkpoint_dir}")

    cfg=Config(
        input_fits=args.input_fits,output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,chunk_size=args.chunk_size,
        tolerance_arcsec=args.tolerance,healpix_nside=args.nside,
        apogee_csv=args.apogee_csv,galah_csv=args.galah_csv,
        ges_csv=args.ges_csv,rave_csv=args.rave_csv,a95_dir=args.a95_dir,
    )

    t0=time.time()
    log("="*72)
    log("RV NORMALIZATION v8 — Survey of Surveys I (FIXED)")
    log("="*72)
    log(f"Input FITS : {cfg.input_fits}")
    log(f"Output     : {cfg.output_dir}")
    log(f"Surveys    : {', '.join(sorted(VALID_SURVEYS))}")
    log("FIXES: param propagation, DUP quality ctrl, paper exclusion rules")

    csv_data     = phase0_load_csvs(cfg)
    data         = phase1_extract(cfg)
    groups       = phase2_spatial(cfg, data)
    sid_map      = phase2b_sid_map(cfg, data, groups)
    survey_stars = phase3_build(cfg, data, groups, csv_data, sid_map)
    del data, groups, sid_map, csv_data; gc.collect()

    dup_results  = phase4_dup(cfg, survey_stars)
    tch_results  = phase5_tch(cfg, survey_stars, dup_results)
    gaia_cal     = phase6_gaia_cal(cfg, survey_stars, tch_results)
    survey_cal   = phase7_survey_cal(cfg, survey_stars, gaia_cal)
    norm_errors  = phase8_norm_errors(cfg, survey_stars, tch_results)

    phase9_plots(cfg, dup_results, tch_results, gaia_cal, survey_cal,
                 norm_errors, survey_stars)

    log(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    log(f"Outputs in: {cfg.output_dir}")


if __name__=='__main__':
    main()