#!/usr/bin/env python3
"""
================================================================================
RV ERROR NORMALIZATION & CALIBRATION — v9.3
Following Tsantaki et al. (2021) "Survey of Surveys I"  (A&A 659, A95)
================================================================================
CHANGES vs v8  (all cumulative):

  [NEW-1]  Phase 6: Bevington (1969) covariance propagation.
           np.polyfit(..., cov=True) gives the chi²-scaled covariance matrix.
           Per-star fit uncertainty: δΔRV = sqrt(grad^T @ Cov @ grad).
           Stored in results['covariances'][eq_name].

  [NEW-2]  Phase 7 corrected_gaia_rv_with_err():
           δRV_calib = sqrt( δΔRV² + (err_Gaia × f_TCH)² )
           Rebuilt identically in Phase 10 (closures not picklable).

  [NEW-3]  Phase 10: Merged catalogue — weighted-mean RV, δRV, σRV (Eq.9).

  [NEW-4]  Fig. 16 + Fig. 17 added.

CHANGES vs v9 (v9.1):

  [v9.1-A] _poly_err(): 5 km/s nan-safety ceiling for near-singular matrices.
  [v9.1-B] corrected_gaia_rv_with_err(): skips nan _poly_err contributions.
  [v9.1-C] Fig. 13 redesigned — 2 panels (histogram + cumulative), stats CSV.
  [v9.1-D] Fig. 17 redesigned — 3 panels, data-driven peaks, per-survey panel.

CHANGES vs v9.2 (v9.3):

  [v9.3-A] No merged catalogue file is written.
           phase10_merge() renamed to _compute_rv_stats_for_plots().
           δRV and σRV are computed in-memory purely for Fig 16/17 plotting.
           No phase10_v9.pkl checkpoint, no merged_catalogue_summary.csv.
           The computation (Bevington Gaia errors, Eq.8 corrections, Eq.9
           σRV) is identical to v9.2 — only the output destination changes.

CHANGES vs v9.1 (v9.2):

  [v9.2-A] Phase 6 ALL-SURVEYS CHANGE (KEY):
           ΔRV calibration now uses ALL non-Gaia surveys for each parameter
           equation (Eq5=Gmag, Eq6=FeH, Eq7=Teff).
           Previously, survey subsets were used:
             Eq5: all except LAMOST
             Eq6: all except LAMOST + RAVE
             Eq7: APOGEE only
           Now every non-Gaia survey contributes to every equation, subject
           only to the automatic N ≥ cfg.min_stars_zp check.  Surveys with
           insufficient coverage of a given parameter drop out naturally.
           This is more data-driven and avoids manual curation.
           Checkpoint renamed to phase6_v9_allsurveys.pkl to force rebuild.

  [v9.2-B] Phase 6 diagnostics: all median/percentile computations over
           δΔRV arrays are now nan-safe (np.nanmedian / finite filter).

  [v9.2-C] Global constants EQ5_GMAG_EXCLUDE / EQ6_FEH_EXCLUDE /
           EQ7_TEFF_SURVEYS removed; replaced with EQ_ALL_NON_GAIA sentinel.

DESIGN NOTE — Fig. 13 vs Fig. 17  (answer to "will Fig. 13 change?"):
  Fig. 13 = per-survey  pipeline_σ × DUP/TCH_factor   (Phase 8)
            NOT affected by Phase 6 Bevington, NOT affected by the
            all-surveys change.  Phase 8 reads only Phase 5 combined
            factors and raw pipeline errors.

  Fig. 17 = δRV and σRV from the merged catalogue   (Phase 10)
            Gaia entries: δRV_calib = sqrt(δΔRV² + δRV²_Gaia_norm)
                          ← δΔRV now comes from a richer polynomial fit
                             because more surveys constrain it (v9.2-A)
            Non-Gaia single-survey: δRV = σ_pipe × f_DUP/TCH
            Multi-survey: δRV = 1/sqrt(Σ w_i)
            ∴ Fig. 17 Gaia δRV values will slightly change with v9.2-A;
              Fig. 13 will NOT change.

METHODOLOGY SUMMARY (paper):
  1. Cross-match surveys with Gaia (spatial + source_id)
  2. Normalize RV errors:
     - DUP method (repeated measurements): APOGEE, RAVE, LAMOST
     - TCH method (three-cornered hat):   Gaia, GALAH
     - GES: average of DUP and TCH
  3. Gaia internal calibration (Phase 6):
     - Eq.5: ΔRV vs G mag  (2nd-order polynomial, all non-Gaia surveys)
     - Eq.6: ΔRV vs [Fe/H] (linear,               all non-Gaia surveys)
     - Eq.7: ΔRV vs Teff   (2nd-order polynomial,  all non-Gaia surveys)
     - All with Bevington covariance → δΔRV per star
  4. Survey calibration (Eq.8): multivariate correction to calibrated Gaia
  5. Merge into unified catalogue:  weighted mean RV, δRV, σRV
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
MIN_DUP_PAIRS_SIGNIFICANT = 200

DUP_ONLY_SURVEYS = {'APOGEE', 'RAVE', 'LAMOST'}
TCH_ONLY_SURVEYS = {'GAIA', 'GALAH'}
AVG_BOTH_SURVEYS = {'GES'}

# Phase 6 Gaia calibration: ALL non-Gaia surveys are used for every
# parameter fit (Gmag, FeH, Teff).  Each survey contributes only the
# stars for which that parameter is available and finite; surveys that
# lack coverage in a given parameter are silently skipped (N < min_stars_zp).
# The global coefficient is the weighted mean over all participating
# per-survey fits, with weights = number of common Gaia–survey stars.
#
# Historical note: the original paper used per-parameter survey subsets
# (LAMOST excluded from Eq5, LAMOST+RAVE excluded from Eq6, only APOGEE
# for Eq7).  The all-surveys approach is more data-driven and avoids
# manual curation; a survey with poor [Fe/H] coverage will simply
# contribute 0 stars and drop out automatically.
EQ_ALL_NON_GAIA = True    # sentinel — kept for documentation / grep convenience

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
    output_dir: str = "./rv_norm_output_v9"
    checkpoint_dir: str = "./rv_norm_ckpt_v9"

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
# A95 FIXED-WIDTH SPECS
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
# UTILITIES
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
    if not tok: return np.nan
    try: return float(tok)
    except: return np.nan

def _to_int_field(line: bytes, sl: Tuple[int, int]) -> float:
    tok = line[sl[0]:sl[1]].strip()
    if not tok: return np.nan
    try: return float(int(tok))
    except: return np.nan

def _resolve_a95_file(a95_dir: str, spec: Dict[str, str]) -> Optional[Path]:
    pdir = Path(a95_dir)
    for p in [pdir / spec['gz'], pdir / spec['dat']]:
        if p.exists(): return p
    return None

def _dedup_by_exact_rv_err(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0: return df
    rv_vals  = pd.to_numeric(df['rv'],   errors='coerce').values
    err_vals = pd.to_numeric(df['e_rv'], errors='coerce').values
    valid = np.isfinite(rv_vals) & np.isfinite(err_vals) & (err_vals > 0)
    vi = np.where(valid)[0]
    seen = {}; keep = np.ones(len(vi), dtype=bool)
    for pos, oi in enumerate(vi):
        key = (round(float(rv_vals[oi]), 6), round(float(err_vals[oi]), 6))
        if key in seen: keep[pos] = False
        else: seen[key] = oi
    vi = vi[keep]
    if len(vi) == 0: return df.iloc[0:0].copy()
    return df.iloc[vi].copy().reset_index(drop=True)

def _load_a95_survey_table(survey, path, spec):
    sids, ras, decs, rvs, ervs, srvs = [], [], [], [], [], []
    with open_maybe_gzip(path) as handle:
        for line in handle:
            rv = _to_float_field(line, spec['rv'])
            erv = _to_float_field(line, spec['erv'])
            if not (np.isfinite(rv) and np.isfinite(erv) and erv > 0): continue
            sid = _to_int_field(line, spec['sid'])
            ra  = _to_float_field(line, spec['ra'])
            dec = _to_float_field(line, spec['dec'])
            srv = _to_float_field(line, spec['srv'])
            sids.append(sid); ras.append(ra); decs.append(dec)
            rvs.append(rv); ervs.append(erv); srvs.append(srv)
    if not rvs:
        return pd.DataFrame(columns=[
            'source_id','ra','dec','rv','e_rv','survey',
            'parallax','parallax_error','s_rvcor',
            'csv_Teff','csv_logg','csv_FeH','csv_Gmag','csv_SNR'])
    n = len(rvs)
    out = pd.DataFrame({
        'source_id': np.array(sids, dtype=np.float64),
        'ra': np.array(ras, dtype=np.float64),
        'dec': np.array(decs, dtype=np.float64),
        'rv': np.array(rvs, dtype=np.float64),
        'e_rv': np.array(ervs, dtype=np.float64),
        'survey': survey,
        'parallax': np.full(n, np.nan), 'parallax_error': np.full(n, np.nan),
        's_rvcor': np.array(srvs, dtype=np.float64),
        'csv_Teff': np.full(n, np.nan), 'csv_logg': np.full(n, np.nan),
        'csv_FeH':  np.full(n, np.nan), 'csv_Gmag': np.full(n, np.nan),
        'csv_SNR':  np.full(n, np.nan),
    })
    return _dedup_by_exact_rv_err(out)

def save_ckpt(path, data):
    with open(path,'wb') as f: pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  Checkpoint saved: {path}")

def load_ckpt(path):
    with open(path,'rb') as f: return pickle.load(f)

def bin_stat(x, y, n_bins=40, p_lo=1, p_hi=99, min_count=10):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_count*2: return np.array([]), np.array([]), np.array([])
    x, y = x[ok], y[ok]
    lo, hi = np.percentile(x, p_lo), np.percentile(x, p_hi)
    be = np.linspace(lo, hi, n_bins+1)
    bc, meds, mads = [], [], []
    for j in range(n_bins):
        m = (x >= be[j]) & (x < be[j+1])
        if m.sum() < min_count: continue
        yy = y[m]
        med = float(np.median(yy))
        mad = float(np.median(np.abs(yy - med)))
        bc.append(0.5*(be[j]+be[j+1])); meds.append(med); mads.append(mad)
    return np.array(bc), np.array(meds), np.array(mads)


# =============================================================================
# BEVINGTON COVARIANCE HELPER  [NEW-1]
# =============================================================================
def _poly_err(coeffs_list, cov_matrix, x_val):
    """
    Bevington (1969) §8.5 error propagation for a polynomial evaluated at x_val.

    For  p(x) = Σ_i  coeffs[i] · x^(n−i)  (numpy highest-degree-first convention)
    the gradient of p with respect to each coefficient is:
        ∂p/∂coeffs[i] = x^(n−i)

    The propagated 1-σ uncertainty is then:
        δp = sqrt( grad^T · Cov · grad )

    where Cov is the chi²-scaled covariance matrix returned by
    np.polyfit(..., cov=True).  This is identical to the formula in
    Tsantaki et al. (2021) §5.1.

    Parameters
    ----------
    coeffs_list : list or 1-D array
        Polynomial coefficients, highest degree first (numpy convention).
    cov_matrix  : 2-D array or None
        Covariance matrix from np.polyfit(cov=True).
        If None the function returns 0.0 (no uncertainty known).
    x_val       : float
        Parameter value at which to evaluate the propagated uncertainty.

    Returns
    -------
    float
        1-σ uncertainty on p(x_val) in km/s.
        Returns 0.0 if cov_matrix is None.
        Returns nan if the quadratic form is negative or non-finite
        (signals a poorly-constrained fit; caller should fall back to
        the raw measurement error).

    Notes
    -----
    A sanity ceiling of 5.0 km/s is applied: if the propagated uncertainty
    exceeds this (typically from a near-singular covariance matrix), the
    function returns nan so the caller can exclude the star or fall back
    gracefully.  5 km/s is ~25 × the typical Gaia RVS error and would
    never be a meaningful calibration uncertainty.
    """
    if cov_matrix is None:
        return 0.0
    try:
        n    = len(coeffs_list) - 1          # polynomial degree
        grad = np.array([float(x_val)**(n - i) for i in range(n + 1)])
        cov  = np.asarray(cov_matrix, dtype=np.float64)
        var  = float(grad @ cov @ grad)
        if not np.isfinite(var) or var < 0.0:
            return np.nan                    # non-finite / non-PSD covariance
        result = float(np.sqrt(var))
        return result if result <= 5.0 else np.nan   # sanity ceiling
    except Exception:
        return 0.0


# =============================================================================
# PHASE 0: LOAD EXTERNAL SURVEY CSVs
# =============================================================================
def phase0_load_csvs(cfg):
    ckpt = Path(cfg.checkpoint_dir)/"phase0_v9.pkl"
    cfg.a95_s_rvcor_factors = {}
    if ckpt.exists():
        log("Phase 0: Loading checkpoint")
        d = load_ckpt(ckpt)
        tables = d.get('tables', d) if isinstance(d, dict) and 'tables' in d else d
        cfg.a95_s_rvcor_factors = d.get('a95_s_rvcor_factors', {}) if isinstance(d, dict) else {}
        for s, df in tables.items():
            has_params = 'csv_Teff' in df.columns and df['csv_Teff'].notna().any()
            log(f"  {s:<12s}: {len(df):>8,} rows  (has params: {has_params})")
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
        try: df = pd.read_csv(path, low_memory=False)
        except Exception as e: log(f"  {surv}: failed — {e}"); continue

        cols = set(df.columns)
        sid_col = find_col(cols,['source_id','Source_ID','gaia_source_id'])
        df['_sid'] = pd.to_numeric(df[sid_col],errors='coerce').astype('Int64') if sid_col else pd.NA
        ra_col_  = find_col(cols,['ra_deg','RA','ra','RAdeg'])
        dec_col_ = find_col(cols,['dec_deg','DEC','dec','DECdeg'])

        if rv_col not in cols:
            found = False
            for fb_rv,fb_err in fallback_rv.get(surv,[]):
                if fb_rv in cols and fb_err in cols:
                    rv_col,err_col = fb_rv,fb_err; found=True; break
            if not found: log(f"    {surv}: no usable RV column, skipping"); continue

        rv_vals  = pd.to_numeric(df[rv_col],  errors='coerce').values
        err_vals = pd.to_numeric(df[err_col], errors='coerce').values if err_col in cols else np.full(len(df),np.nan)
        if surv=='RAVE' and 'cHRV' in cols:
            crv=pd.to_numeric(df['cHRV'],errors='coerce').values; ok=np.isfinite(crv); rv_vals[ok]=crv[ok]
        ra_vals  = pd.to_numeric(df[ra_col_],  errors='coerce').values if ra_col_  else np.full(len(df),np.nan)
        dec_vals = pd.to_numeric(df[dec_col_], errors='coerce').values if dec_col_ else np.full(len(df),np.nan)
        plx_col  = find_col(cols,['parallax','Parallax','parallax_mas','plx','PLX','Plx'])
        plxe_col = find_col(cols,['parallax_error','e_parallax','parallax_err','plx_err','e_plx'])
        plx_vals = pd.to_numeric(df[plx_col],  errors='coerce').values if plx_col  else np.full(len(df),np.nan)
        plxe_vals= pd.to_numeric(df[plxe_col], errors='coerce').values if plxe_col else np.full(len(df),np.nan)

        csv_params = {}
        for pname, search_names in param_search.items():
            pcol = find_col(cols, search_names)
            if pcol:
                csv_params[pname] = pd.to_numeric(df[pcol], errors='coerce').values.astype(np.float64)
                n_valid = np.sum(np.isfinite(csv_params[pname]))
                log(f"    {surv}: found param {pname} in '{pcol}' ({n_valid:,} valid)")
            else:
                csv_params[pname] = np.full(len(df), np.nan, dtype=np.float64)

        raw = pd.DataFrame({
            'source_id': pd.to_numeric(df['_sid'], errors='coerce').values.astype(np.float64),
            'ra': ra_vals.astype(np.float64), 'dec': dec_vals.astype(np.float64),
            'rv': rv_vals.astype(np.float64), 'e_rv': err_vals.astype(np.float64),
            'survey': surv,
            'parallax': plx_vals.astype(np.float64), 'parallax_error': plxe_vals.astype(np.float64),
            's_rvcor': np.full(len(df), np.nan, dtype=np.float64),
            'csv_Teff': csv_params['Teff'], 'csv_logg': csv_params['logg'],
            'csv_FeH':  csv_params['FeH'],  'csv_Gmag': csv_params['Gmag'],
            'csv_SNR':  csv_params['SNR'],
        })
        result[surv] = _dedup_by_exact_rv_err(raw)
        log(f"  {surv}: {len(result[surv]):,} rows loaded")

    # A95 tables
    log("Phase 0b: Loading A95 survey tables...")
    a95_factors = {}
    for surv, spec in A95_SPECS.items():
        path = _resolve_a95_file(cfg.a95_dir, spec)
        if path is None: log(f"  {surv}: A95 file not found, skipping"); continue
        try: df_a95 = _load_a95_survey_table(surv, path, spec)
        except Exception as e: log(f"  {surv}: A95 failed ({e})"); continue
        if len(df_a95) == 0: continue

        ratio = df_a95['s_rvcor'].values.astype(np.float64) / df_a95['e_rv'].values.astype(np.float64)
        ok_ratio = np.isfinite(ratio) & (ratio > 0)
        if np.any(ok_ratio): a95_factors[surv] = float(np.median(ratio[ok_ratio]))

        if surv in result:
            merged = _dedup_by_exact_rv_err(
                pd.concat([result[surv], df_a95], ignore_index=True, sort=False))
            result[surv] = merged
            log(f"  {surv}: +A95 {len(df_a95):,} → merged {len(merged):,}")
        else:
            result[surv] = df_a95
            log(f"  {surv}: A95 only {len(df_a95):,} rows")

    cfg.a95_s_rvcor_factors = a95_factors
    if a95_factors:
        log("  A95 s_RVcor/e_RVcor priors:")
        for s, f in sorted(a95_factors.items()): log(f"    {s:<12s}: {f:.3f}")

    save_ckpt(ckpt, {'tables': result, 'a95_s_rvcor_factors': a95_factors})
    return result


# =============================================================================
# PHASE 1: EXTRACT FROM FITS
# =============================================================================
def phase1_extract(cfg):
    ckpt = Path(cfg.checkpoint_dir)/"phase1_v9.pkl"
    if ckpt.exists():
        log("Phase 1: Loading checkpoint")
        d = load_ckpt(ckpt); log(f"  {d['n_valid']:,} valid / {d['total_rows']:,} total")
        surv_arr = np.array(d['surveys'])
        for sn in sorted(VALID_SURVEYS):
            n = np.sum(surv_arr==sn)
            if n>0: log(f"    {sn:<16s}: {n:>10,}")
        return d

    log("Phase 1: Extracting RVs from FITS...")
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
            'plx':plx,'plxe':plxe,'n_valid':n_valid,'total_rows':total}
    save_ckpt(ckpt,result)
    gc.collect()
    return result


# =============================================================================
# PHASE 2: SPATIAL GROUPING
# =============================================================================
def phase2_spatial(cfg, data):
    ckpt = Path(cfg.checkpoint_dir)/"phase2_v9.pkl"
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
# PHASE 2b: SOURCE_ID → GROUP_ID MAP
# =============================================================================
def phase2b_sid_map(cfg, data, groups):
    ckpt=Path(cfg.checkpoint_dir)/"phase2b_v9.pkl"
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
def phase3_build(cfg, data, groups, csv_data, sid_map):
    ckpt=Path(cfg.checkpoint_dir)/"phase3_v9.pkl"
    if ckpt.exists():
        log("Phase 3: Loading checkpoint")
        d=load_ckpt(ckpt)
        try:
            for s in sorted(d.keys()):
                sample=next(iter(d[s].values()))
                if not isinstance(sample,dict) or 'rvs' not in sample: raise ValueError("stale")
                n_multi=sum(1 for g in d[s].values() if len(g['rvs'])>=2)
                n_has_teff=sum(1 for g in d[s].values()
                               if np.isfinite(g['params'].get('Teff',np.nan)))
                log(f"  {s:<16s}: {len(d[s]):>10,} stars, {n_multi:>8,} >=2obs, "
                    f"{n_has_teff:>8,} with Teff")
            return d
        except Exception as e:
            log(f"  Stale checkpoint ({e}), rebuilding...")
            ckpt.unlink(missing_ok=True)

    log("\nPhase 3: Building per-survey star data...")
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
            if abs(plx_csv - plx_f) < thr: return True
        return False

    def _get_group_params(gid):
        for prefer in ['GAIA','DESI','APOGEE','GALAH','RAVE','LAMOST','GES','SDSS']:
            if prefer in ss and gid in ss[prefer]:
                pr = ss[prefer][gid]['params']
                if any(np.isfinite(pr.get(p, np.nan)) for p in ['Teff','logg','FeH','Gmag']):
                    return dict(pr)
        return {p: np.nan for p in params}

    log("  Adding CSV rows...")
    for surv, df in csv_data.items():
        n_sid=0; n_spatial=0; n_skip=0; n_plx_reject=0; n_params_csv=0; n_params_fits=0
        has_sid = df['source_id'].notna().any()
        has_plx = 'parallax' in df.columns and df['parallax'].notna().any()

        ra_csv  = df['ra'].values.astype(np.float64)
        dec_csv = df['dec'].values.astype(np.float64)
        rv_csv  = df['rv'].values.astype(np.float64)
        erv_csv = df['e_rv'].values.astype(np.float64)
        sid_csv = df['source_id'].values if has_sid else np.full(len(df), np.nan)
        plx_csv = df['parallax'].values.astype(np.float64) if has_plx else np.full(len(df),np.nan)
        plxe_csv= df['parallax_error'].values.astype(np.float64) if has_plx and 'parallax_error' in df.columns else np.full(len(df),np.nan)
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
            rv_v  = rv_csv[orig_i]; err_v = erv_csv[orig_i]
            if not (np.isfinite(rv_v) and np.isfinite(err_v) and err_v > 0): continue
            if is_exact_duplicate(rv_v, err_v): continue

            gid = None
            if has_sid:
                sid_v = sid_csv[orig_i]
                if not pd.isna(sid_v):
                    sid_int = int(sid_v)
                    if sid_int in sid_map: gid = sid_map[sid_int]; n_sid += 1

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

            if gid not in ss[surv]:
                csv_pr = {
                    'Teff': float(csv_teff[orig_i]), 'logg': float(csv_logg[orig_i]),
                    'FeH':  float(csv_feh[orig_i]),  'Gmag': float(csv_gmag[orig_i]),
                    'SNR':  float(csv_snr[orig_i]),
                }
                has_any_csv_param = any(np.isfinite(v) for v in csv_pr.values())
                if has_any_csv_param:
                    fits_pr = _get_group_params(gid)
                    for p in csv_pr:
                        if not np.isfinite(csv_pr[p]) and np.isfinite(fits_pr.get(p, np.nan)):
                            csv_pr[p] = fits_pr[p]
                    use_pr = csv_pr; n_params_csv += 1
                else:
                    use_pr = _get_group_params(gid); n_params_fits += 1
                ss[surv][gid]={'rvs':[],'params':use_pr,
                               'plx':float(plx_csv[orig_i]),'plxe':float(plxe_csv[orig_i])}
            ss[surv][gid]['rvs'].append((rv_v, err_v))

        log(f"    {surv}: added {n_sid+n_spatial:,} "
            f"(sid={n_sid:,} spatial={n_spatial:,} plx_reject={n_plx_reject:,} "
            f"unmatched={n_skip:,})\n"
            f"      params: {n_params_csv:,} from CSV, {n_params_fits:,} from FITS")

    ss=dict(ss)
    log("\n  Per-survey star counts:")
    for s in sorted(ss.keys()):
        n_m=sum(1 for g in ss[s].values() if len(g['rvs'])>=2)
        n_teff=sum(1 for g in ss[s].values() if np.isfinite(g['params'].get('Teff',np.nan)))
        n_gmag=sum(1 for g in ss[s].values() if np.isfinite(g['params'].get('Gmag',np.nan)))
        log(f"    {s:<16s}: {len(ss[s]):>10,} stars, {n_m:>8,} >=2obs, "
            f"Teff:{n_teff:,} Gmag:{n_gmag:,}")
    save_ckpt(ckpt,ss); gc.collect()
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
def phase4_dup(cfg, survey_stars):
    ckpt=Path(cfg.checkpoint_dir)/"phase4_v9.pkl"
    if ckpt.exists():
        log("Phase 4: Loading checkpoint"); return load_ckpt(ckpt)
    log("\nPhase 4: DUP method...")
    results={}
    for surv in sorted(survey_stars.keys()):
        if surv not in DUP_SURVEYS:
            log(f"  {surv}: excluded from DUP (uses TCH)"); continue
        stars=survey_stars[surv]
        multi={g:d for g,d in stars.items() if len(d['rvs'])>=2}
        if len(multi)<5: log(f"  {surv}: {len(multi)} multi-obs, skip"); continue
        norms,raws=[],[]
        n_err_ratio_skip=0
        for gid,sdata in tqdm(multi.items(),desc=f"DUP {surv}",leave=False):
            obs=sdata['rvs']
            if len(obs)>cfg.max_obs_per_star:
                obs=sorted(obs,key=lambda x:x[1])[:cfg.max_obs_per_star]
            np_s=0
            for (v1,e1),(v2,e2) in combinations(obs,2):
                if abs(v1-v2)<1e-6 and abs(e1-e2)<1e-6: continue
                if e1 > 0 and e2 > 0:
                    if max(e1,e2)/min(e1,e2) > 5.0:
                        n_err_ratio_skip += 1; continue
                d=v1-v2; sc=np.sqrt(e1**2+e2**2)
                if sc > 0.01:
                    norms.append(d/sc); raws.append(d); np_s+=1
                    if np_s>=cfg.max_pairs_per_star: break
        if len(norms)<cfg.min_pairs_dup:
            log(f"  {surv}: {len(norms)} pairs (skip_err_ratio={n_err_ratio_skip}), skip"); continue
        nd=np.array(norms); rd=np.array(raws)
        nd_c=nd[np.abs(nd)<50]; rd_c=rd[np.abs(rd)<500]
        mad_raw=float(np.median(np.abs(nd_c)))
        nf_raw=1.4826*mad_raw
        clip_thr = 5.0 * nf_raw if nf_raw > 0 else 50.0
        nd_clipped = nd_c[np.abs(nd_c) < clip_thr]
        if len(nd_clipped) < cfg.min_pairs_dup: nd_clipped = nd_c
        mad=float(np.median(np.abs(nd_clipped)))
        nf=1.4826*mad
        mu,std=norm.fit(nd_clipped)
        results[surv]={
            'norm_diffs':nd,'raw_diffs':rd,'norm_factor':nf,'mad':mad,
            'n_pairs':len(nd),'n_stars':len(multi),'mean_norm':mu,'std_norm':std,
            'mean_raw':float(np.mean(rd_c)),'std_raw':float(np.std(rd_c)),
            'median_raw':float(np.median(rd_c)),
            'mad_raw':float(np.median(np.abs(rd_c-np.median(rd_c)))),
            'norm_mad':nf,'norm_std':std,'n_err_ratio_skip':n_err_ratio_skip,
        }
        log(f"  {surv:<12s}: {len(multi):>8,} stars {len(nd):>10,} pairs "
            f"(skip_err_ratio={n_err_ratio_skip}) | normMAD={nf:.3f}")
    save_ckpt(ckpt,results)
    return results


# =============================================================================
# PHASE 5: TCH METHOD
# =============================================================================
def phase5_tch(cfg, survey_stars, dup_results):
    ckpt=Path(cfg.checkpoint_dir)/"phase5_v9.pkl"
    if ckpt.exists():
        log("Phase 5: Loading checkpoint"); return load_ckpt(ckpt)
    log("\nPhase 5: TCH method...")
    sb=_best_rv(survey_stars); surveys=sorted(sb.keys())
    a95_fallback = getattr(cfg, 'a95_s_rvcor_factors', {}) or {}

    pzp={}
    for si,sj in combinations(surveys,2):
        common=set(sb[si])&set(sb[sj])
        if len(common)<cfg.min_stars_tch: continue
        diffs=[sb[si][g][0]-sb[sj][g][0] for g in common
               if np.isfinite(sb[si][g][0]) and np.isfinite(sb[sj][g][0])]
        if len(diffs)>=cfg.min_stars_tch: pzp[(si,sj)]=float(np.median(diffs))

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
        if var < 1e-6:
            log(f"  WARNING: {si}-{sj}: σ²≈0 ({var:.2e}), skipping pair"); continue
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

    log("\n  === COMBINED NORMALIZATION FACTORS ===")
    log(f"  {'Survey':<14s}  {'DUP_f':>7s}  {'TCH_f':>7s}  {'A95_f':>7s}  {'Combined':>9s}  Method")
    log("  "+"-"*74)
    combined={}; reliability={}
    for surv in surveys:
        df_val=dup_results.get(surv,{}).get('norm_factor')
        df_n  =dup_results.get(surv,{}).get('n_pairs',0)
        tf_val=tch['norm_factors'].get(surv)
        a95_val=a95_fallback.get(surv)
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
            elif tch_ok: f,method=tf_val,'TCH fallback'
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
        else:
            if tch_ok: f,method=tf_val,'TCH preferred'
            elif dup_ok: f,method=df_val,'DUP fallback'
            elif a95_ok: f,method=a95_val,'A95 fallback'
            else: f,method=1.0,'default=1.0'

        combined[surv]=f; reliability[surv]=method
        ds=f"{df_val:.3f}" if df_val is not None and np.isfinite(df_val) else "  n/a "
        ts_=f"{tf_val:.3f}" if tf_val is not None and np.isfinite(tf_val) else "  n/a "
        as_=f"{a95_val:.3f}" if a95_val is not None and np.isfinite(a95_val) else "  n/a "
        log(f"  {surv:<14s}  {ds:>7s}  {ts_:>7s}  {as_:>7s}  {f:>9.3f}  {method}")

    tch['combined_factors']=combined; tch['pairwise_zp']=pzp; tch['reliability']=reliability
    save_ckpt(ckpt,tch)
    return tch


# =============================================================================
# PHASE 6: GAIA INTERNAL CALIBRATION  [NEW-1: Bevington covariance]
# =============================================================================
def phase6_gaia_cal(cfg, survey_stars, tch_results):
    """
    Phase 6: Gaia internal calibration with Bevington covariance propagation.

    For each of the three calibration equations (Eq5/Eq6/Eq7), ALL non-Gaia
    surveys that have sufficient common stars with Gaia AND valid measurements
    of the relevant parameter (Gmag, FeH, Teff) contribute to the fit.
    Surveys with no coverage in a given parameter fall below cfg.min_stars_zp
    and are silently skipped — no manual exclusion lists needed.

    Fitting procedure per survey s, per equation eq:
      1. Collect all Gaia–survey common stars with finite ΔRV = RV_Gaia−RV_s
         and finite parameter value.
      2. Subtract per-survey zero-point (median ΔRV) to centre the fit.
      3. Bin into N_bins median bins for robustness.
      4. Fit polynomial with np.polyfit(cov=True) → Bevington covariance matrix.
      5. Diagnose median δΔRV = sqrt(grad^T Cov grad) across parameter range.

    Global fit = weighted mean over all per-survey coefficients and covariances
    (weights = number of common Gaia–survey stars).

    Returns dict with keys:
      'coefficients'  : {eq_name: [c0, c1, ...]} (highest degree first)
      'covariances'   : {eq_name: [[...]] or None}
      'per_survey'    : {eq_name_surveyname: {xs, ys, ws, zp, n, surv, param, eq_name}}
      'zp_shifts'     : {eq_name: {survey: float}}
    """
    # Checkpoint name encodes the all-surveys approach so stale phase6_v9
    # checkpoints (which used per-parameter subsets) are NOT reused.
    ckpt = Path(cfg.checkpoint_dir) / "phase6_v9_allsurveys.pkl"
    if ckpt.exists():
        log("Phase 6: Loading checkpoint (all-surveys Bevington)")
        d = load_ckpt(ckpt)
        if 'covariances' not in d:
            log("  Missing covariances — rebuilding...")
            ckpt.unlink(missing_ok=True)
        else:
            for eq, cov in d['covariances'].items():
                log(f"  Covariance stored for {eq}: shape {np.array(cov).shape}")
            return d

    log("\nPhase 6: Gaia internal calibration — ALL non-Gaia surveys (Bevington covariance)...")
    sb = _best_rv(survey_stars)
    if 'GAIA' not in sb:
        log("  GAIA not found, skipping Phase 6")
        return {'coefficients': {}, 'covariances': {}, 'per_survey': {}, 'zp_shifts': {}}

    gaia_data = sb['GAIA']
    results   = {'coefficients': {}, 'covariances': {}, 'per_survey': {}, 'zp_shifts': {}}

    # ALL non-Gaia surveys for every equation.
    # Surveys that have too few stars with a given parameter will drop out
    # automatically when the N < cfg.min_stars_zp check is reached.
    all_non_gaia = sorted(s for s in sb if s != 'GAIA')
    log(f"  Non-Gaia surveys in data: {all_non_gaia}")

    cal_specs = [
        ('Gmag', all_non_gaia, 2, 'Eq5'),   # ΔRV vs G magnitude   (2nd order)
        ('FeH',  all_non_gaia, 1, 'Eq6'),   # ΔRV vs [Fe/H]        (linear)
        ('Teff', all_non_gaia, 2, 'Eq7'),   # ΔRV vs Teff          (2nd order)
    ]

    for cal_param, cal_surveys, degree, eq_name in cal_specs:
        log(f"\n  --- {eq_name}: ΔRV vs {cal_param} (degree={degree}) ---")
        log(f"      Surveys: {cal_surveys}")
        per_surv_coeffs=[]; per_surv_covs=[]; zp_row={}

        for surv in cal_surveys:
            if surv not in sb: continue
            sd=sb[surv]
            common=set(gaia_data)&set(sd)
            if len(common)<cfg.min_stars_zp: continue

            all_drv, all_pval = [], []
            for g in common:
                rv_g, _, pr_g = gaia_data[g]
                rv_s, _, pr_s = sd[g]
                if not (np.isfinite(rv_g) and np.isfinite(rv_s)): continue
                pval = pr_s.get(cal_param, pr_g.get(cal_param, np.nan))
                if not np.isfinite(pval): continue
                all_drv.append(rv_g - rv_s); all_pval.append(pval)

            if len(all_drv) < cfg.min_stars_zp:
                log(f"    {surv}: {len(all_drv)} stars with {cal_param}, too few"); continue

            zp_shift = float(np.median(all_drv))
            zp_row[surv] = zp_shift

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
                'xs':xs,'ys':ys,'ws':ws,'zp':zp_shift,'n':len(xs),
                'surv':surv,'param':cal_param,'eq_name':eq_name}

            # Bin data → fit polynomial
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

            # ── Bevington covariance from np.polyfit(cov=True) ────────────
            # cov='unscaled' gives raw (X^T W X)^{-1}
            # cov=True       rescales by chi²/(N-dof) — this is what Bevington uses
            try:
                coeffs, cov_matrix = np.polyfit(bx, by, degree, w=np.sqrt(bw), cov=True)
                # Sanity: reject infinite/NaN covariance
                if not np.all(np.isfinite(cov_matrix)):
                    log(f"    {surv}/{eq_name}: covariance has NaN/Inf — using unscaled")
                    coeffs, cov_matrix = np.polyfit(bx, by, degree, w=np.sqrt(bw), cov='unscaled')
                if not np.all(np.isfinite(cov_matrix)):
                    coeffs = np.polyfit(bx, by, degree, w=np.sqrt(bw))
                    cov_matrix = None
            except Exception as e:
                log(f"    {surv}/{eq_name}: polyfit with cov failed ({e}), using no cov")
                coeffs = np.polyfit(bx, by, degree, w=np.sqrt(bw))
                cov_matrix = None

            per_surv_coeffs.append((coeffs, len(xs)))
            per_surv_covs.append((cov_matrix, len(xs)))

            # Diagnose: median δΔRV across the parameter range
            if cov_matrix is not None:
                x_sample = np.linspace(bx.min(), bx.max(), 20)
                delta_drv_vals = np.array([_poly_err(coeffs, cov_matrix, xv) for xv in x_sample])
                valid_d = delta_drv_vals[np.isfinite(delta_drv_vals)]
                med_d   = float(np.median(valid_d)) if len(valid_d) > 0 else np.nan
                log(f"    {surv}: N={len(xs):,}  N_bins={len(bx)}"
                    f"  ZP={zp_shift:+.3f}"
                    f"  median_δΔRV={med_d:.4f} km/s"
                    f"  coeffs={np.array2string(coeffs, precision=5)}")
            else:
                log(f"    {surv}: N={len(xs):,}  ZP={zp_shift:+.3f}"
                    f"  (no cov)  coeffs={np.array2string(coeffs, precision=5)}")

        results['zp_shifts'][eq_name] = zp_row

        if not per_surv_coeffs:
            log(f"    {eq_name}: no surveys produced valid fits"); continue

        # Global = weighted mean of per-survey coefficients
        coeffs_arr  = np.array([c for c,_ in per_surv_coeffs])
        weights_arr = np.array([n for _,n in per_surv_coeffs], dtype=float)
        global_coeffs = (coeffs_arr * weights_arr[:,None]).sum(axis=0) / weights_arr.sum()
        results['coefficients'][eq_name] = global_coeffs.tolist()

        # Global covariance = weighted mean of per-survey covariances
        # (approximation: independent surveys, so weighted mean is appropriate)
        valid_covs = [(cov, n) for (cov, n) in per_surv_covs if cov is not None]
        if valid_covs:
            cov_arrays  = np.array([c for c,_ in valid_covs])
            cov_weights = np.array([n for _,n in valid_covs], dtype=float)
            global_cov  = (cov_arrays * cov_weights[:,None,None]).sum(axis=0) / cov_weights.sum()
            results['covariances'][eq_name] = global_cov.tolist()
            # Diagnose global δΔRV
            x_sample = np.percentile(
                np.concatenate([results['per_survey'][k]['xs']
                                for k in results['per_survey'] if k.startswith(eq_name)]),
                [5, 25, 50, 75, 95])
            delta_drv_global = np.array([_poly_err(global_coeffs, global_cov, xv) for xv in x_sample])
            valid_g = delta_drv_global[np.isfinite(delta_drv_global)]
            log(f"    {eq_name} GLOBAL: N_surveys={len(per_surv_coeffs)}"
                f"  coeffs={np.array2string(global_coeffs, precision=6)}")
            log(f"    {eq_name} GLOBAL: δΔRV at [5,25,50,75,95]%ile = "
                f"{[f'{v:.4f}' if np.isfinite(v) else 'nan' for v in delta_drv_global]} km/s"
                f"  median={float(np.median(valid_g)):.4f}" if len(valid_g)>0 else "  (all nan)")
        else:
            results['covariances'][eq_name] = None
            log(f"    {eq_name} GLOBAL: coeffs={np.array2string(global_coeffs,precision=6)} "
                f"(no covariance available)")

    save_ckpt(ckpt, results)
    return results


# =============================================================================
# PHASE 7: SURVEY CALIBRATION (Eq.8)  [NEW-2: δRV_calib in corrected_gaia_rv]
# =============================================================================
def phase7_survey_cal(cfg, survey_stars, gaia_cal, tch_results):
    ckpt=Path(cfg.checkpoint_dir)/"phase7_v9.pkl"
    if ckpt.exists():
        log("Phase 7: Loading checkpoint"); return load_ckpt(ckpt)
    log("\nPhase 7: Survey calibration (Eq.8) with Bevington-corrected Gaia errors...")
    sb=_best_rv(survey_stars)
    if 'GAIA' not in sb:
        log("  GAIA not found, skipping"); return {}

    gaia_data    = sb['GAIA']
    gaia_coeffs  = gaia_cal.get('coefficients', {})
    gaia_cov     = gaia_cal.get('covariances',  {})
    cf           = tch_results.get('combined_factors', {})
    f_gaia       = cf.get('GAIA', 1.5)   # TCH normalization factor for Gaia

    # ------------------------------------------------------------------
    # corrected_gaia_rv_with_err:
    #   Returns (rv_corrected, δRV_calib)
    #   δRV_calib = sqrt(δΔRV² + (err_g × f_gaia)²)
    #
    #   where δΔRV = sqrt(sum over active equations of _poly_err²)
    #   This is the Bevington §5.1 formula from the paper:
    #       δRV_calib = sqrt(δΔRV² + δRV²_Gaia)
    # ------------------------------------------------------------------
    def corrected_gaia_rv_with_err(gid):
        """
        Apply the three Gaia internal-calibration corrections (Eq5/Eq6/Eq7)
        to a single star and return the calibrated RV together with the
        fully-propagated Bevington uncertainty.

        Calibration model
        -----------------
        RV_cal = RV_Gaia - [ΔRV(Gmag) + ΔRV(FeH) + ΔRV(Teff)]
                                  Eq5           Eq6        Eq7

        Combined calibration uncertainty (Bevington §5.1, paper §5.1)
        ---------------------------------------------------------------
        δΔRV = sqrt( Σ_eq  δΔRV_eq² )          (quadrature over active eqs)

        Final error (δRV_calib) propagated to the merged catalogue
        -----------------------------------------------------------
        δRV_calib = sqrt( δΔRV² + (err_Gaia × f_Gaia)² )
                           ↑ fit        ↑ measurement error (TCH-normalized)

        Notes
        -----
        - _poly_err returns nan when the covariance is poorly conditioned
          (diagonal variance > 25 km²/s²).  In that case the contribution
          of that equation to δΔRV is treated as 0 (conservative: the fit
          uncertainty is already large so the δRV_calib is dominated by
          the Gaia measurement error).
        - If err_Gaia is missing and no correction is applicable the
          function returns (nan, nan) and the star is excluded from the
          merged catalogue.
        """
        rv_g, err_g, pr = gaia_data[gid]
        corr          = 0.0
        delta_drv_sq  = 0.0

        for eq_name, param in [('Eq5','Gmag'), ('Eq6','FeH'), ('Eq7','Teff')]:
            if eq_name not in gaia_coeffs: continue
            pval = pr.get(param, np.nan)
            if not np.isfinite(pval): continue
            coeffs = gaia_coeffs[eq_name]
            corr  += np.polyval(coeffs, pval)
            cov    = gaia_cov.get(eq_name)
            if cov is not None:
                d = _poly_err(coeffs, cov, pval)
                if np.isfinite(d):               # skip nan from poorly-conditioned cov
                    delta_drv_sq += d ** 2

        # Gaia normalized error (from Phase-5 combined TCH/DUP factor)
        if np.isfinite(err_g) and err_g > 0:
            norm_err_g  = err_g * f_gaia
            # δRV_calib = sqrt(δΔRV² + δRV_Gaia_norm²)
            delta_calib = float(np.sqrt(delta_drv_sq + norm_err_g**2))
        else:
            # Pipeline error missing — propagate only the polynomial uncertainty
            delta_calib = float(np.sqrt(delta_drv_sq)) if delta_drv_sq > 0 else np.nan

        return rv_g - corr, delta_calib

    results={}
    for surv in sorted(sb.keys()):
        if surv=='GAIA': continue
        sd=sb[surv]; common=set(gaia_data)&set(sd)
        if len(common)<50: continue
        rows=[]
        for g in common:
            rv_g_c, err_g_c = corrected_gaia_rv_with_err(g)
            rv_s, err_s, pr_s = sd[g]
            if not (np.isfinite(rv_g_c) and np.isfinite(rv_s)): continue
            drv = rv_g_c - rv_s
            ce  = (np.sqrt(err_g_c**2 + err_s**2)
                   if (np.isfinite(err_g_c) and np.isfinite(err_s)) else 1.0)
            ce  = max(ce, 0.01)
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

        surv_results={'diag':df,'fits':{},'corrected_gaia_rv_fn': None}
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
                XtWX=(X.T*(w[None,:]))@X; XtWy=X.T@(w*y)
                coeffs=np.linalg.solve(XtWX,XtWy)
                resid=y-X@coeffs; chi2=np.sum(w*resid**2)/(len(y)-len(coeffs))
                surv_results['fits'][split_name]={
                    'coeffs':coeffs.tolist(),'feat_names':feat_names,'n':len(y),
                    'chi2':chi2,'zp_before':float(np.median(y)),
                    'zp_after':float(np.median(resid))}
                log(f"    {surv}/{split_name}: N={len(y):,}  chi2={chi2:.3f}"
                    f"  ZP: {np.median(y):+.3f} → {np.median(resid):+.3f} km/s")
            except Exception as e:
                log(f"    {surv}/{split_name}: fit failed: {e}")
        results[surv]=surv_results

    # Store the corrected_gaia_rv_with_err function in a way Phase 10 can reuse
    results['_gaia_correction_fn'] = corrected_gaia_rv_with_err
    save_ckpt(ckpt, {k:v for k,v in results.items() if k != '_gaia_correction_fn'})
    results['_gaia_correction_fn'] = corrected_gaia_rv_with_err
    return results


# =============================================================================
# PHASE 8: NORMALIZED ERRORS (per survey, for Fig. 13)
# Note: These are NOT affected by the Bevington fix.
#       Fig. 13 = pipeline_error × DUP/TCH factor.
#       The Bevington δRV_calib only enters the merged catalogue (Phase 10).
# =============================================================================
def phase8_norm_errors(cfg, survey_stars, tch_results):
    ckpt=Path(cfg.checkpoint_dir)/"phase8_v9.pkl"
    if ckpt.exists():
        log("Phase 8: Loading checkpoint"); return load_ckpt(ckpt)
    log("\nPhase 8: Normalized RV errors (for Fig. 13)...")
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
# PHASE 10: COMPUTE δRV / σRV ARRAYS FOR PLOTTING  [NEW-3]
# NO catalogue file is written — values are computed in-memory for Fig 16/17.
# The computation uses the same Bevington-corrected Gaia errors as Phase 7.
# =============================================================================
def _compute_rv_stats_for_plots(cfg, survey_stars, gaia_cal, survey_cal, tch_results):
    """
    Compute per-group weighted-mean δRV and inter-survey σRV (Eq.9) for
    every spatial group that has at least one valid calibrated RV entry.

    Returns a plain dict  {group_id: {'delta_rv', 'sigma_rv', 'n_surveys',
                                       'surveys'}}  held in memory only.
    NO checkpoint is written.  Call this function once and pass the result
    directly to phase9_plots().

    δRV  = 1/sqrt(Σ w_i)          (weighted-mean precision)
    σRV  = sqrt(Σ w_i (RV_i − <RV>)² / ((N−1) Σ w_i))   [Eq. 9, N≥2 only]

    For Gaia entries:
        δRV_calib = sqrt( δΔRV² + (err_Gaia × f_TCH)² )
        where δΔRV comes from Bevington covariance propagation (Phase 6).
    For non-Gaia entries:
        δRV = err_pipeline × f_DUP/TCH
    """

    log("\nComputing δRV / σRV arrays for plots (no catalogue written)...")
    sb  = _best_rv(survey_stars)
    cf  = tch_results.get('combined_factors', {})
    gaia_coeffs = gaia_cal.get('coefficients', {})
    gaia_cov    = gaia_cal.get('covariances',  {})
    f_gaia      = cf.get('GAIA', 1.5)

    # Reuse corrected_gaia_rv from Phase 7 if available (not picklable, rebuild)
    def corrected_gaia_rv_with_err(gid, gaia_data):
        """
        Identical to Phase 7 corrected_gaia_rv_with_err.
        Rebuilt here because closures are not picklable.
        See Phase 7 docstring for full documentation.
        δRV_calib = sqrt( δΔRV² + (err_Gaia × f_Gaia)² )
        """
        rv_g, err_g, pr = gaia_data[gid]
        corr=0.0; delta_drv_sq=0.0
        for eq_name, param in [('Eq5','Gmag'), ('Eq6','FeH'), ('Eq7','Teff')]:
            if eq_name not in gaia_coeffs: continue
            pval = pr.get(param, np.nan)
            if not np.isfinite(pval): continue
            coeffs = gaia_coeffs[eq_name]
            corr  += np.polyval(coeffs, pval)
            cov    = gaia_cov.get(eq_name)
            if cov is not None:
                d = _poly_err(coeffs, cov, pval)
                if np.isfinite(d):
                    delta_drv_sq += d**2
        if np.isfinite(err_g) and err_g > 0:
            norm_err_g  = err_g * f_gaia
            delta_calib = float(np.sqrt(delta_drv_sq + norm_err_g**2))
        else:
            delta_calib = float(np.sqrt(delta_drv_sq)) if delta_drv_sq > 0 else np.nan
        return rv_g - corr, delta_calib

    # Build set of all spatial groups
    all_groups = set()
    for surv_data in sb.values(): all_groups.update(surv_data.keys())

    gaia_data = sb.get('GAIA', {})
    merged    = {}
    n_total   = len(all_groups)
    log(f"  Merging {n_total:,} spatial groups...")

    for gid in all_groups:
        entries = []   # list of (rv_calibrated, err_calibrated, survey_name)

        # ── Gaia ──
        if gid in gaia_data:
            rv_c, err_c = corrected_gaia_rv_with_err(gid, gaia_data)
            if np.isfinite(rv_c) and np.isfinite(err_c) and err_c > 0:
                entries.append((rv_c, err_c, 'GAIA'))

        # ── Other surveys ──
        for surv in sorted(sb.keys()):
            if surv == 'GAIA': continue
            if gid not in sb[surv]: continue
            rv_s, err_s, pr_s = sb[surv][gid]
            if not np.isfinite(rv_s): continue

            # Apply Eq.8 survey calibration
            rv_cal = rv_s
            if surv in survey_cal:
                sres = survey_cal[surv]
                logg = pr_s.get('logg', np.nan)
                teff = pr_s.get('Teff', np.nan)
                for split_name, fdata in sres.get('fits', {}).items():
                    in_split = False
                    sl = split_name.lower()
                    if 'all'    in sl: in_split = True
                    elif 'dwarf' in sl and np.isfinite(logg) and logg > 3.5:  in_split = True
                    elif 'giant' in sl and np.isfinite(logg) and logg <= 3.5: in_split = True
                    elif 'cool'  in sl and np.isfinite(teff) and teff < 6200: in_split = True
                    elif 'hot'   in sl and np.isfinite(teff) and teff >= 6200:in_split = True
                    if not in_split: continue
                    feat_vals = {
                        'Teff':      pr_s.get('Teff', 0.0),
                        'Teff2':     pr_s.get('Teff', 0.0)**2,
                        'logg':      pr_s.get('logg', 0.0),
                        'FeH':       pr_s.get('FeH',  0.0),
                        'SNR':       pr_s.get('SNR',  0.0),
                        'RV':        rv_s,
                        'intercept': 1.0,
                    }
                    correction = sum(
                        float(c) * feat_vals.get(f, 0.0)
                        for c, f in zip(fdata['coeffs'], fdata['feat_names'])
                        if np.isfinite(c)
                    )
                    # ΔRV = RV_Gaia_cal - RV_survey  →  RV_survey_cal = RV_survey + ΔRV_predicted
                    rv_cal = rv_s + correction
                    break

            # Normalized error for this survey
            f_surv = cf.get(surv, 1.0)
            if np.isfinite(err_s) and err_s > 0:
                err_cal = err_s * f_surv
            else:
                err_cal = np.nan

            if np.isfinite(rv_cal) and np.isfinite(err_cal) and err_cal > 0:
                entries.append((rv_cal, err_cal, surv))

        if not entries: continue

        rvs  = np.array([e[0] for e in entries])
        errs = np.array([e[1] for e in entries])
        ws   = 1.0 / errs**2

        rv_mean  = float(np.sum(ws * rvs) / np.sum(ws))
        delta_rv = float(1.0 / np.sqrt(np.sum(ws)))

        # Eq. 9 (paper): σ²_RV = Σ[w_i(rv_i - rv_mean)²] / ((N-1) × Σw_i)
        if len(entries) >= 2:
            sigma_rv_sq = float(
                np.sum(ws * (rvs - rv_mean)**2) / ((len(entries) - 1) * np.sum(ws)))
            sigma_rv = float(np.sqrt(max(sigma_rv_sq, 0.0)))
        else:
            sigma_rv = np.nan

        merged[gid] = {
            'rv':        rv_mean,
            'delta_rv':  delta_rv,
            'sigma_rv':  sigma_rv,
            'n_surveys': len(entries),
            'surveys':   [e[2] for e in entries],
        }

    log(f"  δRV/σRV computed: {len(merged):,} groups  (no file written)")
    n_multi = sum(1 for v in merged.values() if v['n_surveys'] >= 2)
    all_drv = np.array([v['delta_rv'] for v in merged.values() if np.isfinite(v['delta_rv'])])
    all_srv = np.array([v['sigma_rv'] for v in merged.values() if np.isfinite(v['sigma_rv'])])
    log(f"  Multi-survey stars: {n_multi:,}")
    log(f"  δRV: median={np.median(all_drv):.3f} km/s  p5={np.percentile(all_drv,5):.3f}"
        f"  p95={np.percentile(all_drv,95):.3f}")
    if len(all_srv) > 0:
        log(f"  σRV: median={np.median(all_srv):.3f} km/s  p5={np.percentile(all_srv,5):.3f}"
            f"  p95={np.percentile(all_srv,95):.3f}")
    return merged


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
SIX_PARAMS=[('Gmag','G mag'),('RV','RV (km/s)'),('Teff','T$_{eff}$ (K)'),
            ('logg','log g (dex)'),('FeH','[Fe/H] (dex)'),('SNR','S/N')]
PARAM_TO_EQ={'Gmag':'Eq5','FeH':'Eq6','Teff':'Eq7'}


# =============================================================================
# PHASE 9: ALL PLOTS + CSV
# =============================================================================
def phase9_plots(cfg, dup_results, tch_results, gaia_cal, survey_cal,
                 nerr, survey_stars, merged_catalogue):
    log("\nPhase 9: Generating plots and tables...")
    out=Path(cfg.output_dir)

    # ── Fig 6: DUP normalized ΔRV/σ distributions ─────────────────────────
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

    # ── Fig 7: ΔRV histograms before/after ZP ─────────────────────────────
    log("  Fig 7: ΔRV histograms before/after ZP...")
    sb=_best_rv(survey_stars)
    if 'GAIA' in sb:
        gaia_d=sb['GAIA']
        fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=False)
        bins7=np.linspace(-20,20,160); stats_rows=[]
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
            ax.set_xlabel(r'$\Delta$RV = RV$_{\rm Gaia}$ − RV$_{\rm survey}$ (km/s)',fontsize=11)
            ax.set_ylabel('Normalized density',fontsize=11)
            ax.set_title('Before ZP correction' if not after else 'After ZP correction',fontsize=11)
            ax.set_xlim(-20,20)
            ax.legend(fontsize=7,loc='upper right'); ax.grid(True,alpha=0.25,ls='--')
        fig.suptitle(r'$\Delta$RV Histograms (Paper Fig. 7)',fontsize=12)
        fig.tight_layout()
        for e in ['png','pdf']: fig.savefig(out/f'fig7.{e}',dpi=300)
        plt.close(fig)
        if stats_rows: pd.DataFrame(stats_rows).to_csv(out/'fig7_zp_stats.csv',index=False)

    # ── Fig 8: ΔRV vs 6 parameters ────────────────────────────────────────
    log("  Fig 8: ΔRV vs 6 params...")
    if 'GAIA' in sb:
        gaia_d=sb['GAIA']
        coeffs_all=gaia_cal.get('coefficients',{})
        per_surv=gaia_cal.get('per_survey',{})
        fig,axes=plt.subplots(2,3,figsize=(21,12)); axes=axes.flatten()
        for pi,(pname,xlabel) in enumerate(SIX_PARAMS):
            ax=axes[pi]
            eq_name=PARAM_TO_EQ.get(pname)
            global_coeffs=coeffs_all.get(eq_name) if eq_name else None
            global_cov=gaia_cal.get('covariances',{}).get(eq_name) if eq_name else None
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
                    y_fit=np.polyval(global_coeffs,x_fit)
                    ax.plot(x_fit,y_fit,'k--',lw=2.2,alpha=0.8,label='Global fit',zorder=10)
                    # Bevington confidence band  [NEW]
                    if global_cov is not None:
                        delta_fit=np.array([_poly_err(global_coeffs,global_cov,xv) for xv in x_fit])
                        ax.fill_between(x_fit,y_fit-delta_fit,y_fit+delta_fit,
                                        color='k',alpha=0.15,label='95% CI (Bevington)')
            ax.axhline(0,color='gray',ls=':',alpha=0.5)
            ax.set_xlabel(xlabel,fontsize=11); ax.set_ylabel(r'$\Delta$RV (km/s)',fontsize=10)
            ax.set_title(f'ΔRV vs {pname}',fontsize=11); ax.set_ylim(-15,15)
            ax.legend(fontsize=7,loc='upper right',ncol=2); ax.grid(True,alpha=0.25,ls='--')
        fig.suptitle(r'$\Delta$RV vs parameters (Paper Fig. 8) — shaded = 95% CI from Bevington covariance',fontsize=11)
        fig.tight_layout()
        for e in ['png','pdf']: fig.savefig(out/f'fig8_all.{e}',dpi=300)
        plt.close(fig)
        log("    fig8_all saved.")

    # ── Gaia cal before/after (Figs 9-11) ─────────────────────────────────
    coeffs_all=gaia_cal.get('coefficients',{})
    covar_all =gaia_cal.get('covariances', {})
    per_surv  =gaia_cal.get('per_survey',  {})
    for eq_name,coeffs in coeffs_all.items():
        keys=[k for k in per_surv if k.startswith(eq_name)]
        if not keys: continue
        param=per_surv[keys[0]]['param']
        global_cov=covar_all.get(eq_name)
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
                    yf=np.polyval(coeffs,xf)
                    ax.plot(xf,yf,'k--',lw=2.2,alpha=0.8,label='Global fit',zorder=10)
                    if global_cov is not None:
                        delta_f=np.array([_poly_err(coeffs,global_cov,xv) for xv in xf])
                        ax.fill_between(xf,yf-delta_f,yf+delta_f,
                                        color='k',alpha=0.15,label='95% CI')
            ax.axhline(0,color='gray',ls=':',alpha=0.5)
            ax.set_xlabel(param,fontsize=12); ax.set_ylabel(r'$\Delta$RV (km/s)',fontsize=11)
            ax.set_title(f'{eq_name}: {"After" if show_after else "Before"} calibration',fontsize=12)
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3,ls='--')
        fig.suptitle(f'Gaia Calibration {eq_name} (Paper Figs. 9-11) — shaded = Bevington 95% CI',fontsize=12)
        fig.tight_layout()
        for e in ['png','pdf']: fig.savefig(out/f'gaia_cal_{param}.{e}',dpi=300)
        plt.close(fig)

    # ── Fig 12: Calibrated ΔRV vs 6 params ────────────────────────────────
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
        for surv,(bc,med) in all_survey_bins[pname].items():
            ax.plot(bc,med,'o-',ms=4,lw=1.8,color=_c(surv),alpha=0.85,label=surv)
        ax.axhline(0,color='gray',ls='--',lw=1,alpha=0.6)
        ax.set_xlabel(pname,fontsize=11); ax.set_ylabel(r'$\Delta$RV (km/s)',fontsize=10)
        ax.set_title(f'Calibrated ΔRV vs {pname}',fontsize=11)
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3,ls='--'); ax.set_ylim(-5,5)
    fig.suptitle('Survey Calibration: Calibrated ΔRV vs parameters (Paper Fig. 12)',fontsize=13)
    fig.tight_layout()
    for e in ['png','pdf']: fig.savefig(out/f'fig12_calibrated.{e}',dpi=300)
    plt.close(fig)

    # ── Fig 13: Normalized RV error distributions (Phase 8, unchanged by Bevington)
    # ─────────────────────────────────────────────────────────────────────────────
    # ARCHITECTURE NOTE (paper §5.1):
    #   Fig. 13  =  pipeline_error × combined_factor(DUP/TCH)
    #              ← per-survey measurement precision; Phase 8 only
    #   Fig. 17  =  δRV from merged catalogue
    #              ← includes Bevington δΔRV for Gaia entries
    #              ← δRV_calib = sqrt(δΔRV² + δRV_Gaia_norm²)
    # The two quantities are conceptually different:
    #   Fig. 13: "how well does each survey measure an individual star?"
    #   Fig. 17: "how well do we know each star's RV in the final catalogue?"
    # ─────────────────────────────────────────────────────────────────────────────
    log("  Fig 13: Normalized RV errors (pipeline × factor + A95 s_RVcor)...")
    a95_srv = {}
    for surv, spec in A95_SPECS.items():
        path = _resolve_a95_file(cfg.a95_dir, spec)
        if path is None: continue
        try:
            srv_vals = []
            with open_maybe_gzip(path) as handle:
                for line in handle:
                    srv = _to_float_field(line, spec['srv'])
                    if np.isfinite(srv) and srv > 0: srv_vals.append(srv)
            if srv_vals:
                a95_srv[surv] = np.array(srv_vals, dtype=np.float64)
                log(f"    A95 s_RVcor loaded: {surv} → {len(srv_vals):,} (median={np.median(srv_vals):.3f})")
        except Exception as e:
            log(f"    A95 s_RVcor {surv}: failed ({e})")

    fig, axes13 = plt.subplots(1, 2, figsize=(18, 7))
    ax13_main, ax13_cum = axes13

    FIXED_BINS = np.linspace(0.0, 4.0, 101)
    fixed_bc   = 0.5*(FIXED_BINS[:-1] + FIXED_BINS[1:])
    csv4 = {'bin_center': fixed_bc}
    all_survs = sorted(set(list(nerr.keys()) + list(a95_srv.keys())))

    stats13_rows = []
    for s in all_survs:
        parts = []
        n_pipeline, n_a95 = 0, 0
        if s in nerr:
            ea = nerr[s]['errors']
            parts.append(ea[np.isfinite(ea) & (ea > 0)]); n_pipeline=len(parts[-1])
        if s in a95_srv:
            srv = a95_srv[s]
            parts.append(srv[np.isfinite(srv) & (srv > 0)]); n_a95=len(parts[-1])
        if not parts: continue
        combined = np.concatenate(parts)
        ep = combined[(combined >= 0) & (combined <= 4)]
        if len(ep) < 20: continue
        med_combined = float(np.median(ep))
        p16 = float(np.percentile(ep, 16))
        p84 = float(np.percentile(ep, 84))
        c, _ = np.histogram(ep, bins=FIXED_BINS, density=True)
        cum   = np.cumsum(c) * (FIXED_BINS[1] - FIXED_BINS[0])
        csv4[s] = c
        f_val = nerr[s]['factor'] if s in nerr else np.nan
        f_str = f"f={f_val:.2f}" if np.isfinite(f_val) else "f=n/a"
        ax13_main.hist(FIXED_BINS[:-1], FIXED_BINS, weights=c, histtype='step', lw=2,
                color=_c(s), alpha=0.85,
                label=f"{s}  med={med_combined:.2f} km/s  {f_str}")
        ax13_cum.plot(fixed_bc, cum, lw=2, color=_c(s), alpha=0.85,
                      label=f"{s}  med={med_combined:.2f}")
        stats13_rows.append({'Survey': s, 'N_pipeline': n_pipeline, 'N_a95': n_a95,
                             'Median_err_km_s': med_combined,
                             'P16_km_s': p16, 'P84_km_s': p84,
                             'Norm_factor': f_val})

    # Annotation box explaining the Bevington relationship
    note = ("Fig. 13 = pipeline σ × DUP/TCH factor\n"
            "NOT affected by Bevington calibration.\n"
            "Bevington δΔRV enters Fig. 17 δRV only.")
    ax13_main.text(0.97, 0.97, note, transform=ax13_main.transAxes,
                   fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.8))

    ax13_main.set_xlim(0, 4)
    ax13_main.set_xlabel('Normalized RV error  σ × f  (km/s)', fontsize=13)
    ax13_main.set_ylabel('Density', fontsize=13)
    ax13_main.set_title('Normalized RV Error Distributions  (Paper Fig. 13)\n'
                        'Per-survey pipeline errors × DUP/TCH normalization factor',
                        fontsize=11)
    ax13_main.legend(fontsize=8, loc='upper right')
    ax13_main.grid(True, alpha=0.3, ls='--')

    ax13_cum.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5)
    ax13_cum.set_xlim(0, 4); ax13_cum.set_ylim(0, 1)
    ax13_cum.set_xlabel('Normalized RV error  σ × f  (km/s)', fontsize=13)
    ax13_cum.set_ylabel('Cumulative fraction', fontsize=13)
    ax13_cum.set_title('Cumulative distributions', fontsize=11)
    ax13_cum.legend(fontsize=8, loc='lower right')
    ax13_cum.grid(True, alpha=0.3, ls='--')

    fig.suptitle('Paper Fig. 13 — normalized per-survey measurement errors\n'
                 '(DUP/TCH normalization only; Bevington polynomial uncertainty is in Fig. 17)',
                 fontsize=11)
    fig.tight_layout()
    for e in ['png','pdf']: fig.savefig(out/f'fig13.{e}', dpi=300)
    plt.close(fig)
    if len(csv4) > 1: pd.DataFrame(csv4).to_csv(out/'fig13_data.csv', index=False)
    if stats13_rows: pd.DataFrame(stats13_rows).to_csv(out/'fig13_stats.csv', index=False)
    log("    fig13 saved (main + cumulative panels).")

    # ── Fig 16: δRV vs stellar parameters (paper Fig. 16) ─────────────────
    log("  Fig 16: δRV vs stellar parameters (merged catalogue)...")
    if merged_catalogue:
        # Collect per-group params from survey_stars
        param_vals = {pn:[] for pn in ['Gmag','Teff','logg','FeH']}
        drv_vals   = []
        srv_vals   = []
        sb_all     = _best_rv(survey_stars)

        for gid, mdata in merged_catalogue.items():
            drv = mdata['delta_rv']
            srv = mdata['sigma_rv']
            if not np.isfinite(drv): continue
            # Get best params from any survey for this group
            best_pr = {}
            for surv in ['GAIA','APOGEE','GALAH','GES','RAVE','LAMOST','DESI','SDSS']:
                if surv not in sb_all or gid not in sb_all[surv]: continue
                pr = sb_all[surv][gid][2]
                for pn in param_vals:
                    if pn not in best_pr and np.isfinite(pr.get(pn,np.nan)):
                        best_pr[pn] = pr[pn]
            for pn in param_vals:
                param_vals[pn].append(best_pr.get(pn, np.nan))
            drv_vals.append(drv)
            srv_vals.append(srv)

        drv_arr = np.array(drv_vals)
        srv_arr = np.array(srv_vals)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for pi, pname in enumerate(['Gmag','Teff','logg','FeH']):
            ax = axes[pi]
            xv = np.array(param_vals[pname])
            ok = np.isfinite(xv) & np.isfinite(drv_arr)
            if ok.sum() > 100:
                bc, meds_d, _ = bin_stat(xv[ok], drv_arr[ok], n_bins=30, min_count=cfg.min_bin_count)
                ax.plot(bc, meds_d, 'b-o', ms=3, lw=2, alpha=0.9, label='δRV (weighted err)')
                # σRV for multi-survey only
                ok2 = ok & np.isfinite(srv_arr)
                if ok2.sum() > 50:
                    bc2, meds_s, _ = bin_stat(xv[ok2], srv_arr[ok2], n_bins=30, min_count=cfg.min_bin_count)
                    ax.plot(bc2, meds_s, 'r-s', ms=3, lw=2, alpha=0.9, label='σRV (inter-survey scatter)')
            ax.set_xlabel(pname, fontsize=12)
            ax.set_ylabel('Error (km/s)', fontsize=11)
            ax.set_title(f'RV errors vs {pname}', fontsize=12)
            ax.set_ylim(0, min(5, ax.get_ylim()[1]))
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3, ls='--')
        fig.suptitle('Merged Catalogue: δRV and σRV vs Parameters (Paper Fig. 16)', fontsize=13)
        fig.tight_layout()
        for e in ['png','pdf']: fig.savefig(out/f'fig16_drv_params.{e}', dpi=300)
        plt.close(fig)
        log("    fig16 saved.")

    # ── Fig 17: Distribution of δRV and σRV  [NEW-4]  ──────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Paper Fig. 17: "Distribution of errors (δRV and σRV) in SoS after the
    #                 homogenization process."
    #
    # δRV  = 1/sqrt(Σ w_i)  = weighted-mean precision of the merged catalogue
    #         For Gaia-only stars this INCLUDES the Bevington polynomial
    #         uncertainty: δRV_calib = sqrt(δΔRV² + δRV_Gaia_norm²)
    #         For other single-survey stars: δRV = σ_pipeline × f_DUP/TCH
    #
    # σRV  = inter-survey scatter (Eq. 9 of paper); defined only for N≥2 surveys
    #
    # Panel layout (3 columns):
    #   Left  : full δRV + σRV distributions (0–10 km/s)
    #   Centre: zoomed 0–4 km/s (paper-figure range) with fill
    #   Right : δRV distribution per survey colour-coded
    # ─────────────────────────────────────────────────────────────────────────
    log("  Fig 17: δRV and σRV distributions (paper Fig. 17)...")
    if merged_catalogue:
        all_drv_m = np.array([v['delta_rv'] for v in merged_catalogue.values()
                              if np.isfinite(v['delta_rv']) and v['delta_rv'] > 0])
        all_srv_m = np.array([v['sigma_rv'] for v in merged_catalogue.values()
                              if np.isfinite(v['sigma_rv']) and v['sigma_rv'] > 0])

        # ── per-survey δRV decomposition ──────────────────────────────────
        surv_drv: Dict[str, list] = defaultdict(list)
        for v in merged_catalogue.values():
            if not (np.isfinite(v['delta_rv']) and v['delta_rv'] > 0): continue
            # for multi-survey stars assign δRV to first survey in list
            first_surv = v['surveys'][0] if v['surveys'] else 'unknown'
            surv_drv[first_surv].append(v['delta_rv'])
        surv_drv = {s: np.array(a) for s, a in surv_drv.items() if len(a) >= 20}

        fig17, axes17 = plt.subplots(1, 3, figsize=(22, 7))
        BINS17  = np.linspace(0, 10, 201)
        BINS17b = np.linspace(0, 4,  201)
        bc17    = 0.5*(BINS17[:-1]  + BINS17[1:])
        bc17b   = 0.5*(BINS17b[:-1] + BINS17b[1:])

        # ── helper: compute mode of a histogram ───────────────────────────
        def _hist_mode(arr, bins):
            c, edges = np.histogram(arr[arr <= edges[-1]], bins=bins, density=True)
            bc_      = 0.5*(edges[:-1] + edges[1:])
            return float(bc_[np.argmax(c)]) if len(c) > 0 else float(np.median(arr))

        drv_mode = _hist_mode(all_drv_m, BINS17b)
        srv_mode = _hist_mode(all_srv_m, BINS17b) if len(all_srv_m) > 0 else np.nan

        drv_med  = float(np.median(all_drv_m))
        drv_p5   = float(np.percentile(all_drv_m, 5))
        drv_p95  = float(np.percentile(all_drv_m, 95))
        srv_med  = float(np.median(all_srv_m))  if len(all_srv_m)>0 else np.nan
        srv_p5   = float(np.percentile(all_srv_m, 5))  if len(all_srv_m)>0 else np.nan
        srv_p95  = float(np.percentile(all_srv_m, 95)) if len(all_srv_m)>0 else np.nan

        # ──────────────────────────────────────────────────────────────────
        # Panel 0: Full range 0–10 km/s
        # ──────────────────────────────────────────────────────────────────
        ax0 = axes17[0]
        c_drv, _ = np.histogram(all_drv_m, bins=BINS17, density=True)
        ax0.step(bc17, c_drv, where='mid', lw=2.5, color='tomato',
                 label=f'δRV  (N={len(all_drv_m):,})')
        if len(all_srv_m) > 0:
            c_srv, _ = np.histogram(all_srv_m, bins=BINS17, density=True)
            ax0.step(bc17, c_srv, where='mid', lw=2.5, color='steelblue',
                     label=f'σRV  (N={len(all_srv_m):,}, ≥2 surveys)')
        # Data-driven vertical markers at mode
        ax0.axvline(drv_mode, color='tomato', ls='--', lw=1.3, alpha=0.7,
                    label=f'δRV mode ≈ {drv_mode:.2f} km/s')
        if np.isfinite(srv_mode):
            ax0.axvline(srv_mode, color='steelblue', ls='--', lw=1.3, alpha=0.7,
                        label=f'σRV mode ≈ {srv_mode:.2f} km/s')
        stats_txt = (f"δRV:  med={drv_med:.3f}  [{drv_p5:.3f}, {drv_p95:.3f}] km/s\n"
                     f"σRV:  med={srv_med:.3f}  [{srv_p5:.3f}, {srv_p95:.3f}] km/s"
                     if np.isfinite(srv_med) else
                     f"δRV:  med={drv_med:.3f}  [{drv_p5:.3f}, {drv_p95:.3f}] km/s")
        ax0.text(0.97, 0.97, stats_txt, transform=ax0.transAxes,
                 fontsize=8.5, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.85))
        ax0.set_xlim(0, 10)
        ax0.set_xlabel('Error  (km/s)', fontsize=12)
        ax0.set_ylabel('Density', fontsize=12)
        ax0.set_title('Full range  0–10 km/s', fontsize=11)
        ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3, ls='--')

        # ──────────────────────────────────────────────────────────────────
        # Panel 1: Zoomed 0–4 km/s  (paper figure range) with fill
        # ──────────────────────────────────────────────────────────────────
        ax1 = axes17[1]
        c_drv2, _ = np.histogram(all_drv_m[all_drv_m <= 4], bins=BINS17b, density=True)
        ax1.fill_between(bc17b, c_drv2, step='mid', alpha=0.25, color='tomato')
        ax1.step(bc17b, c_drv2, where='mid', lw=2.5, color='tomato',
                 label=f'δRV  med={drv_med:.3f} km/s')
        if len(all_srv_m) > 0:
            c_srv2, _ = np.histogram(all_srv_m[all_srv_m <= 4], bins=BINS17b, density=True)
            ax1.fill_between(bc17b, c_srv2, step='mid', alpha=0.25, color='steelblue')
            ax1.step(bc17b, c_srv2, where='mid', lw=2.5, color='steelblue',
                     label=f'σRV  med={srv_med:.3f} km/s')
        # Median lines
        ax1.axvline(drv_med, color='tomato',   ls=':', lw=1.5, alpha=0.8)
        if np.isfinite(srv_med):
            ax1.axvline(srv_med, color='steelblue', ls=':', lw=1.5, alpha=0.8)

        # Annotation box explaining Bevington contribution
        note17 = ("δRV for Gaia entries includes\n"
                  "Bevington polynomial uncertainty:\n"
                  "δRV_calib = √(δΔRV² + δRV_Gaia²)\n"
                  "Non-Gaia: δRV = σ_pipe × f_DUP/TCH")
        ax1.text(0.97, 0.97, note17, transform=ax1.transAxes,
                 fontsize=7.5, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.4', fc='#e8f4f8', alpha=0.9))
        ax1.set_xlim(0, 4)
        ax1.set_xlabel('Error  (km/s)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Zoomed  0–4 km/s', fontsize=11)
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, ls='--')

        # ──────────────────────────────────────────────────────────────────
        # Panel 2: Per-survey δRV breakdown
        # ──────────────────────────────────────────────────────────────────
        ax2 = axes17[2]
        for surv in sorted(surv_drv.keys()):
            da = surv_drv[surv]
            da_plot = da[da <= 4]
            if len(da_plot) < 20: continue
            c_s, _ = np.histogram(da_plot, bins=BINS17b, density=True)
            med_s  = float(np.median(da))
            ax2.step(bc17b, c_s, where='mid', lw=1.8, color=_c(surv), alpha=0.85,
                     label=f"{surv}  med={med_s:.3f}  N={len(da):,}")
        ax2.set_xlim(0, 4)
        ax2.set_xlabel('δRV  (km/s)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('δRV per survey\n(assigned by first-survey in merged group)', fontsize=10)
        ax2.legend(fontsize=8, loc='upper right'); ax2.grid(True, alpha=0.3, ls='--')

        fig17.suptitle(
            'Distribution of δRV and σRV after Homogenization  (Paper Fig. 17)\n'
            'δRV = weighted-mean uncertainty  |  σRV = inter-survey scatter (Eq. 9)\n'
            'Gaia δRV includes Bevington calibration uncertainty  δRV_calib = √(δΔRV² + δRV²_Gaia)',
            fontsize=11)
        fig17.tight_layout()
        for e in ['png','pdf']: fig17.savefig(out/f'fig17_error_dist.{e}', dpi=300)
        plt.close(fig17)

        # Save combined CSV
        max_len = max(len(all_drv_m), len(all_srv_m)) if len(all_srv_m) > 0 else len(all_drv_m)
        drv_padded = np.pad(all_drv_m, (0, max_len - len(all_drv_m)),
                            constant_values=np.nan) if len(all_drv_m) < max_len else all_drv_m
        srv_padded = (np.pad(all_srv_m, (0, max_len - len(all_srv_m)),
                             constant_values=np.nan)
                      if len(all_srv_m) > 0 and len(all_srv_m) < max_len
                      else (all_srv_m if len(all_srv_m) > 0 else np.full(max_len, np.nan)))
        pd.DataFrame({'delta_rv': drv_padded, 'sigma_rv': srv_padded}).to_csv(
            out/'fig17_drv_srv.csv', index=False)

        # Stats summary
        stats17 = {
            'N_total': len(merged_catalogue),
            'N_delta_rv': len(all_drv_m),
            'N_sigma_rv': len(all_srv_m),
            'drv_mode_km_s':  drv_mode,
            'drv_median_km_s': drv_med,
            'drv_p5_km_s':    drv_p5,
            'drv_p95_km_s':   drv_p95,
            'srv_mode_km_s':  srv_mode,
            'srv_median_km_s': srv_med,
            'srv_p5_km_s':    srv_p5,
            'srv_p95_km_s':   srv_p95,
        }
        pd.DataFrame([stats17]).to_csv(out/'fig17_stats.csv', index=False)
        log(f"    fig17 saved — δRV mode={drv_mode:.3f} km/s, σRV mode={srv_mode:.3f} km/s")
        log(f"    δRV: med={drv_med:.3f} [{drv_p5:.3f}, {drv_p95:.3f}] km/s")
        if np.isfinite(srv_med):
            log(f"    σRV: med={srv_med:.3f} [{srv_p5:.3f}, {srv_p95:.3f}] km/s")
    else:
        log("  Fig 17: merged_catalogue empty, skipping.")

    # ── CSV tables ─────────────────────────────────────────────────────────
    log("  Writing CSV tables...")
    rows=[{'Survey':s,'N_stars':d['n_stars'],'N_pairs':d['n_pairs'],
           'Mean_raw':d['mean_raw'],'Std_raw':d['std_raw'],
           'Median_raw':d['median_raw'],'MAD_raw':d['mad_raw'],
           'NormMAD':d['norm_mad'],'NormStd':d['norm_std']}
          for s,d in sorted(dup_results.items())]
    if rows: pd.DataFrame(rows).to_csv(out/'table3_dup.csv',index=False)

    cf_=tch_results.get('combined_factors',{})
    rel=tch_results.get('reliability',{})
    rows=[{'Survey':s,'Combined_Factor':cf_[s],
           'DUP_factor':dup_results.get(s,{}).get('norm_factor',np.nan),
           'TCH_factor':tch_results.get('norm_factors',{}).get(s,np.nan),
           'Method':rel.get(s,'')}
          for s in sorted(cf_.keys())]
    if rows: pd.DataFrame(rows).to_csv(out/'table4_norm_factors.csv',index=False)

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
                           'Mean_DRV':float(np.mean(drvs)),'Median_DRV':float(np.median(drvs)),
                           'Sigma':float(np.std(drvs)),
                           'MAD':float(np.median(np.abs(drvs-np.median(drvs))))})
        if t5_rows: pd.DataFrame(t5_rows).to_csv(out/'table5_drv_stats.csv',index=False)

    gc_ = gaia_cal.get('coefficients',{})
    gcov_ = gaia_cal.get('covariances',{})
    if gc_:
        rows=[]
        for eq,coeffs in gc_.items():
            cov = gcov_.get(eq)
            x_sample = np.linspace(-3, 20, 10)  # generic range for diagnostic
            if cov is not None:
                delta_drv_med = float(np.median([_poly_err(coeffs, cov, x) for x in x_sample]))
            else:
                delta_drv_med = np.nan
            rows.append({'Equation':eq,'Coefficients':str(coeffs),
                         'Median_delta_DRV_km_s':delta_drv_med,
                         'Has_Bevington_Covariance': cov is not None})
        pd.DataFrame(rows).to_csv(out/'gaia_cal_coefficients.csv',index=False)

    zp_all=gaia_cal.get('zp_shifts',{})
    if zp_all:
        zp_rows=[]
        for eq,zdict in zp_all.items():
            for surv,zp in zdict.items():
                zp_rows.append({'Equation':eq,'Survey':surv,'ZP_shift_km_s':zp})
        pd.DataFrame(zp_rows).to_csv(out/'gaia_cal_zp_shifts.csv',index=False)

    rows=[]
    for surv,sres in survey_cal.items():
        for split,fdata in sres.get('fits',{}).items():
            rows.append({'Survey':surv,'Split':split,
                         'Features':str(fdata['feat_names']),
                         'Coefficients':str(fdata['coeffs']),'N':fdata['n'],
                         'Chi2':fdata['chi2'],'ZP_before_km_s':fdata['zp_before'],
                         'ZP_after_km_s':fdata['zp_after']})
    if rows: pd.DataFrame(rows).to_csv(out/'survey_cal_coefficients.csv',index=False)

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
    p=argparse.ArgumentParser(description='RV Normalization v9.2 (Bevington + all-surveys Phase 6 + merged catalogue)')
    p.add_argument('input_fits')
    p.add_argument('--output-dir','-o',default='./rv_norm_output_v9')
    p.add_argument('--checkpoint-dir',default='./rv_norm_ckpt_v9')
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
        input_fits=args.input_fits, output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir, chunk_size=args.chunk_size,
        tolerance_arcsec=args.tolerance, healpix_nside=args.nside,
        apogee_csv=args.apogee_csv, galah_csv=args.galah_csv,
        ges_csv=args.ges_csv, rave_csv=args.rave_csv, a95_dir=args.a95_dir,
    )

    t0=time.time()
    log("="*72)
    log("RV NORMALIZATION v9.2 — Survey of Surveys I")
    log("="*72)
    log(f"Input FITS : {cfg.input_fits}")
    log(f"Output     : {cfg.output_dir}")
    log(f"Surveys    : {', '.join(sorted(VALID_SURVEYS))}")
    log("v9.2: Bevington covariance (Phase 6), ALL non-Gaia surveys in Phase 6,")
    log("      δRV_calib (Phase 7+10), merged catalogue (Phase 10), Fig.13+16+17")

    csv_data     = phase0_load_csvs(cfg)
    data         = phase1_extract(cfg)
    groups       = phase2_spatial(cfg, data)
    sid_map      = phase2b_sid_map(cfg, data, groups)
    survey_stars = phase3_build(cfg, data, groups, csv_data, sid_map)
    del data, groups, sid_map, csv_data; gc.collect()

    dup_results  = phase4_dup(cfg, survey_stars)
    tch_results  = phase5_tch(cfg, survey_stars, dup_results)
    gaia_cal     = phase6_gaia_cal(cfg, survey_stars, tch_results)
    survey_cal   = phase7_survey_cal(cfg, survey_stars, gaia_cal, tch_results)
    norm_errors  = phase8_norm_errors(cfg, survey_stars, tch_results)

    log("\n" + "="*72)
    log("δRV / σRV computation for Fig 16 + 17 (in-memory, no catalogue written)")
    log("="*72)
    merged_cat   = _compute_rv_stats_for_plots(cfg, survey_stars, gaia_cal, survey_cal, tch_results)

    phase9_plots(cfg, dup_results, tch_results, gaia_cal, survey_cal,
                 norm_errors, survey_stars, merged_cat)

    log(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    log(f"Outputs in: {cfg.output_dir}")

    # Final summary
    log("\n" + "="*72)
    log("SUMMARY OF KEY OUTPUTS")
    log("="*72)
    log(f"  Fig 6:  DUP normalized ΔRV/σ distributions")
    log(f"  Fig 7:  ΔRV histograms before/after ZP correction")
    log(f"  Fig 8:  ΔRV vs 6 parameters (with Bevington 95% CI band on fit)")
    log(f"  Fig 9-11: Gaia calibration Eq5/6/7 before/after (with CI band)")
    log(f"  Fig 12: Calibrated ΔRV vs 6 parameters")
    log(f"  Fig 13: Normalized error distributions — 2 panels")
    log(f"          Left : pipeline σ × DUP/TCH factor  (per survey)")
    log(f"          Right: cumulative distributions")
    log(f"          NOTE : NOT changed by Bevington — those are per-survey measurement errors")
    log(f"  Fig 16: δRV and σRV vs stellar parameters (merged catalogue)")
    log(f"  Fig 17: δRV and σRV distributions — paper Fig. 17  (3 panels)")
    log(f"          Panel 0: full range 0–10 km/s, data-driven mode lines")
    log(f"          Panel 1: zoomed 0–4 km/s, Bevington annotation")
    log(f"          Panel 2: per-survey δRV breakdown")
    log(f"          Gaia δRV_calib = sqrt(δΔRV² + δRV_Gaia_norm²)  [Bevington]")
    log("="*72)


if __name__=='__main__':
    main()