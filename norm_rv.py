#!/usr/bin/env python3
"""
================================================================================
RV ERROR NORMALIZATION & CALIBRATION — v6
Following Tsantaki et al. (2021) "Survey of Surveys I"
================================================================================
FIXES vs v5:
  [1] SDSS-BOSS + SEGUE merged → single 'SDSS' survey (same telescope / R~2000).
  [2] Phase 3 checkpoint: stale-structure guard — auto-deletes & rebuilds if
      old format detected (fixes: TypeError 'int' not subscriptable).
  [3] DUP: all non-Gaia surveys attempted.  Factor accepted only if N_pairs >=
      MIN_DUP_PAIRS_SIGNIFICANT (200).  Combined = avg(DUP,TCH) when both
      reliable; graceful fallback otherwise.
  [4] Phase 6 Gaia cal: polynomial fitted to BINNED MEDIANS only, not to raw
      millions of scattered points. Fit/plot clipped to actual data range.
  [5] Gaia cal plots: before/after panels + per-survey overlay per parameter
      + single combined all-surveys figure.
  [6] Survey cal plots: ΔRV axis labelled explicitly; all-surveys overlay plots
      per calibration parameter.

WHAT DOES ΔRV MEAN?
  Phase 6 (Gaia cal, Eq.5–7):
      ΔRV = RV_Gaia − RV_survey   [ZP-shifted per survey before fitting]
      → calibrating Gaia RVs using ground-based surveys as reference
      → if positive at high G-mag: Gaia overestimates RV for faint stars
  Phase 7 (survey cal, Eq.8):
      ΔRV = RV_Gaia_corrected − RV_survey
      → calibrating each survey to the corrected Gaia reference frame
      → positive = survey underestimates RV relative to calibrated Gaia
  Example (DESI):
      ΔRV_DESI = RV_Gaia_corrected − RV_DESI
      A positive trend with Teff means DESI underestimates RV for hot stars

MAHALANOBIS DISTANCE (paper §3) — for duplicate verification:
  The paper uses MD² on (ΔGmag, ΔRV) pairs to confirm spatial "mates" as
  true duplicates (threshold: χ²_{n=2}(97.5%) = 7.38).  Our code uses spatial
  KDTree grouping at 1 arcsec which is equivalent for FITS input with Gaia
  source_ids already present.  Helper function verify_duplicates_md() is
  provided below if you want to apply MD filtering to a candidate list.
================================================================================
Surveys: GAIA, DESI, APOGEE, GALAH, GES, RAVE, LAMOST, SDSS
External CSV paths (configure below or via CLI):
  ./astro_data/RAVE_DR6/RAVE_DR6_merged_rv_parallax.csv
  ./astro_data/APOGEE_DR17/APOGEE_DR17_merged_rv_parallax.csv
  ./astro_data/GALAH_DR3/GALAH_DR3_merged_rv_parallax.csv
  ./astro_data/GES_DR5/GES_DR5_merged_rv_parallax.csv
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

warnings.filterwarnings('ignore')

# =============================================================================
# SURVEYS TO INCLUDE — HARD LIST (anything else is excluded)
# =============================================================================
VALID_SURVEYS = {'GAIA', 'DESI', 'APOGEE', 'GALAH', 'GES', 'RAVE',
                 'LAMOST', 'SDSS'}
# SDSS-BOSS and SEGUE are both SDSS-family low-resolution spectrographs — merged.

# All non-Gaia surveys attempt DUP.  The factor is accepted only when
# N_pairs >= MIN_DUP_PAIRS_SIGNIFICANT (set below).  Surveys with too few
# pairs fall back to TCH.  Gaia is excluded: its spatial "duplicates" in the
# FITS already have weighted-averaged RVs so ΔRV ≈ 0 by construction.
DUP_SURVEYS = {'DESI', 'APOGEE', 'GALAH', 'GES', 'RAVE', 'LAMOST', 'SDSS'}
MIN_DUP_PAIRS_SIGNIFICANT = 200   # below this → DUP factor flagged as unreliable

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    input_fits: str = ""
    output_dir: str = "./rv_norm_output_v5"
    checkpoint_dir: str = "./rv_norm_ckpt_v5"

    # External survey CSVs
    apogee_csv: str = "./astro_data/APOGEE_DR17/APOGEE_DR17_merged_rv_parallax.csv"
    galah_csv:  str = "./astro_data/GALAH_DR3/GALAH_DR3_merged_rv_parallax.csv"
    ges_csv:    str = "./astro_data/GES_DR5/GES_DR5_merged_rv_parallax.csv"
    rave_csv:   str = "./astro_data/RAVE_DR6/RAVE_DR6_merged_rv_parallax.csv"

    ra_col:     str = "RA_all"
    dec_col:    str = "DEC_all"
    survey_col: str = "Survey"
    code_col:   str = "Code"
    # source_id column in FITS (try these in order)
    sid_cols:   List[str] = field(default_factory=lambda: [
        "source_id", "Source_ID", "Gaia_Source_ID", "gaia_source_id"
    ])

    # RV column pairs in the FITS
    rv_columns: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("radial_velocity_1",   "radial_velocity_error_1"),
        ("radial_velocity_2",   "radial_velocity_error_2"),
        ("RV",                  "e_RV"),
        ("VRAD",                "VRAD_ERR"),
        ("dr2_radial_velocity", "dr2_radial_velocity_error"),
    ])

    # Stellar parameter columns
    param_columns: Dict[str, List[str]] = field(default_factory=lambda: {
        'Gmag': ['Gmag', 'phot_g_mean_mag', 'GMAG', 'G'],
        'Teff': ['Teff', 'TEFF', 'Teff_x', 'teff'],
        'logg': ['logg', 'LOGG', 'logg_x', 'log_g'],
        'FeH':  ['[Fe/H]', 'FEH', 'feh', 'M_H', 'Fe_H'],
        'SNR':  ['RVSS/N', 'RVS/N', 'rvss_snr', 'rv_snr', 'snr', 'SNR', 'S_N'],
    })

    tolerance_arcsec:  float = 1.0
    healpix_nside:     int   = 32
    chunk_size:        int   = 3_000_000
    n_workers:         int   = 1

    min_pairs_dup:     int   = 50
    min_stars_tch:     int   = 20
    min_stars_zp:      int   = 30
    max_obs_per_star:  int   = 30
    max_pairs_per_star:int   = 200
    min_bin_count:     int   = 10

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
                        return int(l.split()[1]) / 1024 / 1024
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
    blob = s + '|' + c
    if c.startswith('D33') or c.startswith('D125') or 'GAIA' in blob: return 'GAIA'
    if 'DESI'  in blob: return 'DESI'
    if 'APOGEE' in blob: return 'APOGEE'
    if 'LAMOST' in blob: return 'LAMOST'
    if 'GALAH'  in blob: return 'GALAH'
    if 'RAVE'   in blob and 'RAVEL' not in blob: return 'RAVE'
    if c == 'GES' or s == 'GES': return 'GES'
    # SDSS-BOSS and SEGUE share the same telescope + resolution class → merge as SDSS
    if 'SDSS' in blob or 'BOSS' in blob or 'SEGUE' in blob: return 'SDSS'
    # Everything else is excluded
    return None   # ← None means excluded (was 'DSOS','SOSI','UNKNOWN', etc.)

def is_exact_duplicate(rv, err, tol=1e-6):
    """Return True if |RV| == |e_RV| within tol — instrument artefact, discard."""
    return np.abs(np.abs(rv) - np.abs(err)) < tol


def verify_duplicates_md(delta_gmag_arr, delta_rv_arr, quantile=0.975):
    """
    Mahalanobis-distance duplicate verification (Tsantaki+2021 §3, Eq. 2).

    Given arrays of (ΔGmag, ΔRV) pairs between candidate duplicates, returns
    a boolean mask: True = confirmed duplicate (MD² below threshold).

    MD²(Xᵢ) = (Xᵢ − X̄)ᵀ · V̂⁻¹ · (Xᵢ − X̄)
    where Xᵢ = [ΔGmag_i, ΔRV_i], X̄ = mean vector, V̂ = covariance matrix.
    Threshold: χ²_{n=2}(q) — paper uses q=97.5% → threshold ≈ 7.38.

    Parameters
    ----------
    delta_gmag_arr : array-like, shape (N,)
        Paired G-mag differences between candidate mates.
    delta_rv_arr   : array-like, shape (N,)
        Paired RV differences [km/s] between candidate mates.
    quantile : float
        Chi² quantile for threshold. Default 0.975 (paper value).

    Returns
    -------
    is_dup  : np.ndarray of bool, shape (N,)
        True where MD² < threshold → confirmed duplicate.
    md2     : np.ndarray of float, shape (N,)
        Raw MD² values for each pair.
    threshold : float
        The chi² threshold used.

    Example
    -------
    >>> gmag_diffs = np.array([0.02, 0.05, 3.5])   # last one is a mismatch
    >>> rv_diffs   = np.array([0.1,  0.3, 15.0])
    >>> is_dup, md2, thr = verify_duplicates_md(gmag_diffs, rv_diffs)
    >>> print(is_dup)   # [True, True, False]
    """
    from scipy.stats import chi2 as _chi2
    X = np.column_stack([np.asarray(delta_gmag_arr, dtype=float),
                         np.asarray(delta_rv_arr,   dtype=float)])
    ok = np.all(np.isfinite(X), axis=1)
    md2 = np.full(len(X), np.inf)
    if ok.sum() < 3:
        return ok, md2, np.inf   # not enough data

    Xok   = X[ok]
    Xmean = Xok.mean(axis=0)
    Vcov  = np.cov(Xok.T)
    try:
        Vinv = np.linalg.inv(Vcov)
    except np.linalg.LinAlgError:
        return ok, md2, np.inf

    diff      = Xok - Xmean
    md2[ok]   = np.einsum('ij,jk,ik->i', diff, Vinv, diff)
    threshold = _chi2.ppf(quantile, df=2)
    is_dup    = md2 < threshold
    return is_dup, md2, threshold

def find_col(avail, possible):
    for n in possible:
        if n in avail: return n
    return None

def save_ckpt(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  Checkpoint saved: {path}")

def load_ckpt(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def adaptive_bins(n, lo=0.0, hi=4.0):
    """Return bin edges with adaptive count: sqrt(N) capped [30, 120]."""
    nb = int(np.clip(np.sqrt(n), 30, 120))
    return np.linspace(lo, hi, nb + 1)

# =============================================================================
# PHASE 0: LOAD EXTERNAL SURVEY CSVs
# =============================================================================
def phase0_load_csvs(cfg):
    """
    Load APOGEE, GALAH, GES, RAVE CSV files.
    Returns dict: survey_name → DataFrame with columns [source_id, ra_deg, dec_deg, rv, e_rv]
    source_id may be NaN for GES (no source_id in file → will use spatial match).

    Column mapping:
      APOGEE : HRV  (heliocentric RV), e_HRV
      GALAH  : RVgalah, e_RVgalah
      GES    : RV, RVprec (precision = error)
      RAVE   : HRV  (or cHRV if present), e_HRV
    """
    ckpt = Path(cfg.checkpoint_dir) / "phase0_v5.pkl"
    if ckpt.exists():
        log("Phase 0: Loading checkpoint")
        d = load_ckpt(ckpt)
        for s, df in d.items():
            log(f"  {s:<12s}: {len(df):>8,} rows  (source_id ok: "
                f"{df['source_id'].notna().sum():,})")
        return d

    log("Phase 0: Loading external survey CSVs...")

    specs = {
        'APOGEE': (cfg.apogee_csv, 'HRV',     'e_HRV'),
        'GALAH':  (cfg.galah_csv,  'RVgalah',  'e_RVgalah'),
        'GES':    (cfg.ges_csv,    'RV',        'RVprec'),
        'RAVE':   (cfg.rave_csv,   'HRV',       'e_HRV'),
    }
    # Fallback RV column if primary missing
    fallback_rv = {
        'APOGEE': [('RV', 'e_RV')],
        'RAVE':   [('cHRV', 'e_HRV')],
        'GALAH':  [('RVsmev2', 'e_RVsmev2'), ('RVobst', 'e_RVobst')],
        'GES':    [],
    }

    result = {}
    for surv, (path, rv_col, err_col) in specs.items():
        if not os.path.exists(path):
            log(f"  {surv}: CSV not found at {path}, skipping")
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            log(f"  {surv}: {len(df):,} rows, columns: {list(df.columns)}")
        except Exception as e:
            log(f"  {surv}: failed to load — {e}")
            continue

        cols = set(df.columns)

        # --- source_id ---
        sid_col = find_col(cols, ['source_id', 'Source_ID', 'gaia_source_id'])
        if sid_col:
            df['_sid'] = pd.to_numeric(df[sid_col], errors='coerce').astype('Int64')
        else:
            df['_sid'] = pd.NA
            log(f"    {surv}: no source_id column → will use spatial matching")

        # --- RA / Dec ---
        ra_col  = find_col(cols, ['ra_deg', 'RA', 'ra', 'RAdeg'])
        dec_col = find_col(cols, ['dec_deg', 'DEC', 'dec', 'DECdeg'])

        # --- RV primary ---
        if rv_col not in cols:
            # try fallbacks
            found = False
            for fb_rv, fb_err in fallback_rv.get(surv, []):
                if fb_rv in cols and fb_err in cols:
                    rv_col, err_col = fb_rv, fb_err
                    log(f"    {surv}: using fallback RV={rv_col}, e_RV={err_col}")
                    found = True
                    break
            if not found:
                log(f"    {surv}: no usable RV column found, skipping")
                continue

        rv_vals  = pd.to_numeric(df[rv_col],  errors='coerce').values
        err_vals = pd.to_numeric(df[err_col], errors='coerce').values if err_col in cols else np.full(len(df), np.nan)

        # For RAVE use cHRV (corrected) if available and e_HRV for error
        if surv == 'RAVE' and 'cHRV' in cols:
            crv = pd.to_numeric(df['cHRV'], errors='coerce').values
            ok  = np.isfinite(crv)
            rv_vals[ok] = crv[ok]
            log(f"    RAVE: replaced HRV with cHRV for {ok.sum():,} stars")

        ra_vals  = pd.to_numeric(df[ra_col],  errors='coerce').values if ra_col  else np.full(len(df), np.nan)
        dec_vals = pd.to_numeric(df[dec_col], errors='coerce').values if dec_col else np.full(len(df), np.nan)

        # --- Exact duplicate filter (|RV| == |e_RV|) ---
        excl = np.array([
            is_exact_duplicate(rv_vals[i], err_vals[i])
            if np.isfinite(rv_vals[i]) and np.isfinite(err_vals[i]) else False
            for i in range(len(rv_vals))
        ])
        if excl.any():
            log(f"    {surv}: discarding {excl.sum():,} exact-duplicate RV=e_RV rows")

        # --- Valid mask ---
        valid = (np.isfinite(rv_vals) & (err_vals > 0) & np.isfinite(err_vals)
                 & ~excl)
        log(f"    {surv}: {valid.sum():,} / {len(df):,} valid rows after |RV|==|e_RV| filter")

        # --- Within-survey exact (rv, err) deduplication for CSV rows ---
        # If two rows in this CSV have identical (rv_final, err_final) after
        # all filtering, keep only the first occurrence.
        valid_idx   = np.where(valid)[0]
        seen_csv    = {}
        dedup_keep  = np.ones(len(valid_idx), dtype=bool)
        n_csv_dedup = 0
        for pos, orig_i in enumerate(valid_idx):
            key = (round(float(rv_vals[orig_i]),  6),
                   round(float(err_vals[orig_i]), 6))
            if key in seen_csv:
                dedup_keep[pos] = False
                n_csv_dedup    += 1
            else:
                seen_csv[key] = orig_i
        if n_csv_dedup:
            log(f"    {surv}: {n_csv_dedup:,} exact (rv,err) duplicate CSV rows discarded")
        valid_idx_clean        = valid_idx[dedup_keep]
        valid_clean            = np.zeros(len(df), dtype=bool)
        valid_clean[valid_idx_clean] = True
        valid                  = valid_clean
        log(f"    {surv}: {valid.sum():,} rows retained after all filters")

        out = pd.DataFrame({
            'source_id': df['_sid'].values[valid],
            'ra':        ra_vals[valid],
            'dec':       dec_vals[valid],
            'rv':        rv_vals[valid].astype(np.float64),
            'e_rv':      err_vals[valid].astype(np.float64),
            'survey':    surv,
        })
        result[surv] = out

    save_ckpt(ckpt, result)
    return result


# =============================================================================
# PHASE 1: EXTRACT FROM FITS — ONE WEIGHTED-AVG RV PER ROW + STELLAR PARAMS
# =============================================================================
def phase1_extract(cfg):
    """
    For each FITS row:
      - Collect all valid (rv, err) pairs from rv_columns
      - Exact-duplicate filter: skip pair if |rv| == |err| within 1e-6
      - Compute weighted average → single (rv_final, err_final)
      - resolve_survey() → only keep VALID_SURVEYS (None → skip row)
      - Extract source_id, stellar params
    """
    ckpt = Path(cfg.checkpoint_dir) / "phase1_v5.pkl"
    if ckpt.exists():
        log("Phase 1: Loading checkpoint")
        d = load_ckpt(ckpt)
        log(f"  {d['n_valid']:,} valid rows / {d['total_rows']:,} total")
        surv_arr = np.array(d['surveys'])
        for sn in sorted(VALID_SURVEYS):
            n = np.sum(surv_arr == sn)
            if n > 0:
                log(f"    {sn:<16s}: {n:>10,}")
        return d

    log("Phase 1: Extracting one RV per row from FITS...")
    hdu   = fits.open(cfg.input_fits, memmap=True)
    fdata = hdu[1].data
    acols = {col.name for col in hdu[1].columns}
    total = len(fdata)
    log(f"  Total FITS rows: {total:,}")

    # Find RV column pairs
    valid_rv = [(v,e) for v,e in cfg.rv_columns if v in acols and e in acols]
    log(f"  RV pairs: {[(v,e) for v,e in valid_rv]}")

    # Stellar param columns
    param_col_map = {}
    for pname, possible in cfg.param_columns.items():
        found = find_col(acols, possible)
        param_col_map[pname] = found
        log(f"  {pname}: {found}")

    # source_id column
    sid_col = find_col(acols, cfg.sid_cols)
    log(f"  source_id column: {sid_col}")

    has_surv = cfg.survey_col in acols
    has_code = cfg.code_col  in acols

    all_ra=[]; all_dec=[]; all_rv=[]; all_err=[]; all_surveys=[]; all_row_idx=[]
    all_sid = []
    all_params = {p: [] for p in cfg.param_columns}

    n_excl_survey = 0
    n_excl_exact  = 0
    n_chunks = (total + cfg.chunk_size - 1) // cfg.chunk_size

    for start in tqdm(range(0, total, cfg.chunk_size),
                      total=n_chunks, desc="Phase1", unit="chunk"):
        end  = min(start + cfg.chunk_size, total)
        clen = end - start

        ra_c   = np.array(fdata[cfg.ra_col][start:end],  dtype=np.float64)
        dec_c  = np.array(fdata[cfg.dec_col][start:end], dtype=np.float64)
        surv_c = np.array(fdata[cfg.survey_col][start:end]).astype(str) if has_surv else np.full(clen,'')
        code_c = np.array(fdata[cfg.code_col][start:end]).astype(str)  if has_code else np.full(clen,'')
        sid_c  = np.array(fdata[sid_col][start:end], dtype=np.int64) if sid_col else np.zeros(clen, dtype=np.int64)

        # Load all RV arrays
        rv_arrs = {}
        for v, e in valid_rv:
            rv_arrs[v] = np.array(fdata[v][start:end], dtype=np.float64)
            rv_arrs[e] = np.array(fdata[e][start:end], dtype=np.float64)

        # Load stellar params
        param_arrs = {}
        for pname, col in param_col_map.items():
            if col is not None:
                try:    param_arrs[pname] = np.array(fdata[col][start:end], dtype=np.float64)
                except: param_arrs[pname] = np.full(clen, np.nan)
            else:
                param_arrs[pname] = np.full(clen, np.nan)

        # --- Per-row weighted average ---
        all_rvs_stack  = np.full((clen, len(valid_rv)), np.nan)
        all_errs_stack = np.full((clen, len(valid_rv)), np.nan)

        for j, (v, e) in enumerate(valid_rv):
            rv_v = rv_arrs[v]
            rv_e = rv_arrs[e]
            # Exact-duplicate filter per column
            exact = np.array([is_exact_duplicate(rv_v[i], rv_e[i])
                               if (np.isfinite(rv_v[i]) and np.isfinite(rv_e[i])) else False
                               for i in range(clen)])
            valid_j = np.isfinite(rv_v) & np.isfinite(rv_e) & (rv_e > 0) & ~exact
            all_rvs_stack[valid_j, j]  = rv_v[valid_j]
            all_errs_stack[valid_j, j] = rv_e[valid_j]

        weights = np.where(np.isfinite(all_errs_stack) & (all_errs_stack > 0),
                           1.0 / all_errs_stack**2, 0.0)
        w_sum   = np.nansum(weights, axis=1)
        has_any = w_sum > 0

        best_rv  = np.full(clen, np.nan)
        best_err = np.full(clen, np.nan)
        best_rv[has_any]  = np.nansum(all_rvs_stack[has_any] * weights[has_any], axis=1) / w_sum[has_any]
        best_err[has_any] = 1.0 / np.sqrt(w_sum[has_any])

        # Fallback: some RVs but all errs bad → simple mean (no error)
        has_rv_no_err = ~has_any & np.any(np.isfinite(all_rvs_stack), axis=1)
        if np.any(has_rv_no_err):
            best_rv[has_rv_no_err] = np.nanmean(all_rvs_stack[has_rv_no_err], axis=1)

        # Resolve survey — only keep VALID_SURVEYS
        resolved = [resolve_survey(surv_c[i], code_c[i]) for i in range(clen)]

        valid_mask = (
            np.isfinite(best_rv) & np.isfinite(ra_c) & np.isfinite(dec_c)
            & np.array([r is not None for r in resolved])
        )
        vi = np.where(valid_mask)[0]

        n_excl_survey += int(clen - np.sum(np.array([r is not None for r in resolved])))

        all_ra.append(ra_c[vi]); all_dec.append(dec_c[vi])
        all_rv.append(best_rv[vi]); all_err.append(best_err[vi])
        all_row_idx.append(np.arange(start, end, dtype=np.int64)[vi])
        all_sid.append(sid_c[vi])

        for i in vi:
            all_surveys.append(resolved[i])

        for pname in cfg.param_columns:
            all_params[pname].append(param_arrs[pname][vi])

        del ra_c, dec_c, surv_c, code_c, rv_arrs, param_arrs
        del all_rvs_stack, all_errs_stack, weights
        gc.collect()

    hdu.close()

    ra      = np.concatenate(all_ra)
    dec     = np.concatenate(all_dec)
    rv      = np.concatenate(all_rv)
    err     = np.concatenate(all_err)
    row_idx = np.concatenate(all_row_idx)
    sid     = np.concatenate(all_sid)
    params  = {p: np.concatenate(all_params[p]) for p in cfg.param_columns}
    surveys = all_surveys
    del all_ra, all_dec, all_rv, all_err, all_row_idx, all_params, all_surveys

    n_before_dedup = len(ra)
    log(f"\n  Rows after survey filter: {n_before_dedup:,} / {total:,}")
    log(f"  Excluded (non-survey): {n_excl_survey:,}")

    # -------------------------------------------------------------------------
    # WITHIN-SURVEY EXACT (rv, err) DEDUPLICATION
    # -------------------------------------------------------------------------
    # After the weighted average, if two rows from the SAME survey have
    # bit-for-bit identical (rv_final, err_final), they are the same
    # measurement entered twice in the catalog.  Keep the first occurrence,
    # discard all later ones.
    #
    # Why within-survey only?
    #   The same (rv, err) appearing in LAMOST AND GALAH is a genuine
    #   cross-survey match of the same star — those rows MUST be kept as
    #   independent measurements for DUP/TCH.
    #
    # Tolerance: round to 6 decimal places (= 1e-6 km/s precision).
    #   This catches exact floating-point copies without accidentally
    #   merging genuinely close but distinct measurements.
    # -------------------------------------------------------------------------
    log("\n  Within-survey exact (rv, err) deduplication...")
    surveys_arr = np.array(surveys)
    keep_mask   = np.ones(n_before_dedup, dtype=bool)
    n_total_discarded = 0

    for surv in sorted(VALID_SURVEYS):
        surv_idx = np.where(surveys_arr == surv)[0]
        if len(surv_idx) < 2:
            continue

        rv_s  = rv[surv_idx]
        err_s = err[surv_idx]

        seen    = {}   # key: (rv_rounded, err_rounded) → first index in surv_idx
        discard = []

        for pos, global_i in enumerate(surv_idx):
            # NaN errors mean the weighted avg had no valid error — keep those
            # rows as-is (they can't meaningfully match another row's NaN)
            if not np.isfinite(rv_s[pos]) or not np.isfinite(err_s[pos]):
                continue
            key = (round(float(rv_s[pos]),  6),
                   round(float(err_s[pos]), 6))
            if key in seen:
                discard.append(global_i)   # duplicate → discard
            else:
                seen[key] = global_i       # first occurrence → keep

        if discard:
            keep_mask[discard] = False
            n_total_discarded += len(discard)
            log(f"    {surv:<16s}: {len(discard):>8,} exact (rv,err) duplicate rows discarded")
        else:
            log(f"    {surv:<16s}: no exact (rv,err) duplicates found")

    # Apply mask to every array
    ra      = ra[keep_mask]
    dec     = dec[keep_mask]
    rv      = rv[keep_mask]
    err     = err[keep_mask]
    sid     = sid[keep_mask]
    row_idx = row_idx[keep_mask]
    surveys = [s for s, k in zip(surveys, keep_mask) if k]
    params  = {p: params[p][keep_mask] for p in params}

    n_valid = len(ra)
    log(f"\n  Total exact (rv,err) duplicates discarded: {n_total_discarded:,}")
    log(f"  Final valid rows after deduplication: {n_valid:,}")
    log(f"\n  Per-survey counts after deduplication:")
    surv_arr = np.array(surveys)
    for sn in sorted(VALID_SURVEYS):
        n = int(np.sum(surv_arr == sn))
        if n > 0:
            log(f"    {sn:<16s}: {n:>10,}")

    result = {
        'ra': ra, 'dec': dec, 'rv': rv, 'err': err,
        'surveys': surveys, 'params': params, 'row_idx': row_idx,
        'sid': sid,
        'n_valid': n_valid, 'total_rows': total,
    }
    save_ckpt(ckpt, result)
    gc.collect()
    return result


# =============================================================================
# PHASE 2: SPATIAL GROUPING
# =============================================================================
def phase2_spatial(cfg, data):
    ckpt = Path(cfg.checkpoint_dir) / "phase2_v5.pkl"
    if ckpt.exists():
        log("Phase 2: Loading checkpoint")
        d = load_ckpt(ckpt)
        ug, gc_ = np.unique(d['labels'], return_counts=True)
        log(f"  {len(ug):,} unique groups, {np.sum(gc_>1):,} multi-row, max={np.max(gc_)}")
        return d

    ra, dec = data['ra'], data['dec']
    n = len(ra)
    log(f"\nPhase 2: Spatial matching on {n:,} rows...")

    if HAS_HEALPY:
        pix = hp.ang2pix(cfg.healpix_nside, np.radians(90-dec), np.radians(ra), nest=True)
    else:
        pix = (np.floor(ra).astype(np.int32)%360)*180 + (np.floor(dec+90).astype(np.int32)%180)

    upix = np.unique(pix)
    log(f"  HEALPix regions: {len(upix):,}")

    parent = np.arange(n, dtype=np.int64)
    rank   = np.zeros(n, dtype=np.int32)

    def find(x):
        r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa == pb: return
        if rank[pa] < rank[pb]: pa, pb = pb, pa
        parent[pb] = pa
        if rank[pa] == rank[pb]: rank[pa] += 1

    tol = 2 * np.sin(np.radians(cfg.tolerance_arcsec / 3600) / 2)
    si  = np.argsort(pix); sp = pix[si]
    pl  = np.searchsorted(sp, upix, side='left')
    pr  = np.searchsorted(sp, upix, side='right')

    total_pairs = 0
    for pi in tqdm(range(len(upix)), desc="Phase2 spatial", unit="rgn"):
        l, r = pl[pi], pr[pi]
        if r - l < 2: continue
        idx = si[l:r]
        rr, dr = np.radians(ra[idx]), np.radians(dec[idx])
        xyz   = np.column_stack([np.cos(dr)*np.cos(rr), np.cos(dr)*np.sin(rr), np.sin(dr)])
        pairs = cKDTree(xyz).query_pairs(r=tol, output_type='ndarray')
        if len(pairs) > 0:
            for li, lj in pairs: union(idx[li], idx[lj])
            total_pairs += len(pairs)

    log(f"  Pairs: {total_pairs:,}")
    labels = np.array([find(i) for i in range(n)], dtype=np.int64)
    ug, gcounts = np.unique(labels, return_counts=True)
    log(f"  Groups: {len(ug):,}, multi: {np.sum(gcounts>1):,}, max: {np.max(gcounts)}")

    result = {'labels': labels}
    save_ckpt(ckpt, result)
    del parent, rank, pix, si, sp, ug, gcounts
    gc.collect()
    return result


# =============================================================================
# PHASE 2b: BUILD SOURCE_ID → GROUP_ID MAP (for CSV matching)
# =============================================================================
def phase2b_sid_map(cfg, data, groups):
    """
    Build source_id → group_id mapping from the FITS rows that have a valid source_id.
    Returns dict: source_id (int) → group_id (int)
    """
    ckpt = Path(cfg.checkpoint_dir) / "phase2b_v5.pkl"
    if ckpt.exists():
        log("Phase 2b: Loading checkpoint (source_id→group_id map)")
        d = load_ckpt(ckpt)
        log(f"  Map size: {len(d):,}")
        return d

    log("\nPhase 2b: Building source_id → group_id map...")
    sids   = data['sid']
    labels = groups['labels']

    sid_map = {}
    for i, (sid, gid) in enumerate(zip(sids, labels)):
        if sid != 0 and sid > 0:
            sid_map[int(sid)] = int(gid)

    log(f"  Map size: {len(sid_map):,}")
    save_ckpt(ckpt, sid_map)
    return sid_map


# =============================================================================
# PHASE 3: BUILD PER-SURVEY STAR DATA (FITS + External CSVs)
# =============================================================================
def phase3_build(cfg, data, groups, csv_data, sid_map):
    """
    survey_stars[survey][group_id] = {
        'rvs':    [(rv, err), ...],   # independent measurements
        'params': {Gmag, Teff, logg, FeH, SNR}
    }

    FITS rows → grouped by spatial labels.
    CSV rows  → matched via source_id → same group_id as Gaia row.
               If no source_id (GES) → spatial match using RA/Dec.
    """
    ckpt = Path(cfg.checkpoint_dir) / "phase3_v5.pkl"
    if ckpt.exists():
        log("Phase 3: Loading checkpoint")
        d = load_ckpt(ckpt)
        # ── Structure guard ──────────────────────────────────────────────────
        # Old checkpoints stored {survey: int} instead of {survey: {gid: {rvs,params}}}.
        # If any value is not a dict, the checkpoint is stale → delete and rebuild.
        try:
            for s in sorted(d.keys()):
                sample = next(iter(d[s].values()))
                if not isinstance(sample, dict) or 'rvs' not in sample:
                    raise ValueError("stale checkpoint structure")
                n_multi = sum(1 for g in d[s].values() if len(g['rvs']) >= 2)
                log(f"  {s:<16s}: {len(d[s]):>10,} unique stars, {n_multi:>8,} >=2 obs")
            return d
        except Exception as e:
            log(f"  Phase 3 checkpoint stale ({e}), rebuilding...")
            ckpt.unlink(missing_ok=True)

    log("\nPhase 3: Building per-survey star data...")
    gl      = groups['labels']
    rv      = data['rv']
    err     = data['err']
    surveys = data['surveys']
    params  = data['params']
    n       = len(gl)

    ss = defaultdict(dict)   # ss[survey][gid] = {'rvs':[], 'params':{}}

    # --- FITS rows ---
    log("  Adding FITS rows...")
    for i in tqdm(range(n), desc="Phase3-FITS", mininterval=5.0):
        surv = surveys[i]
        gid  = int(gl[i])
        if surv not in VALID_SURVEYS:
            continue
        if gid not in ss[surv]:
            ss[surv][gid] = {
                'rvs':    [],
                'params': {p: float(params[p][i]) for p in params}
            }
        ss[surv][gid]['rvs'].append((float(rv[i]), float(err[i])))

    # --- External CSV rows via source_id ---
    log("  Adding CSV rows via source_id / spatial match...")
    # Build spatial KD-tree from FITS for GES fallback
    fits_xyz = None
    fits_gids = None

    for surv, df in csv_data.items():
        n_added = 0; n_new_gid = 0; n_skip = 0
        has_sid = df['source_id'].notna().any()

        if has_sid:
            # source_id matching
            for _, row in df.iterrows():
                sid = row['source_id']
                if pd.isna(sid): continue
                sid = int(sid)
                if sid not in sid_map:
                    n_skip += 1
                    continue
                gid = sid_map[sid]
                rv_v  = float(row['rv'])
                err_v = float(row['e_rv'])
                if not (np.isfinite(rv_v) and np.isfinite(err_v) and err_v > 0):
                    continue
                if is_exact_duplicate(rv_v, err_v):
                    continue
                if gid not in ss[surv]:
                    ss[surv][gid] = {'rvs': [], 'params': {p: np.nan for p in params}}
                    n_new_gid += 1
                ss[surv][gid]['rvs'].append((rv_v, err_v))
                n_added += 1
        else:
            # Spatial matching (GES: no source_id)
            log(f"    {surv}: using spatial matching (no source_id)")
            if fits_xyz is None:
                log("      Building FITS spatial KDTree for GES matching...")
                ra_f   = data['ra']
                dec_f  = data['dec']
                rr_f   = np.radians(ra_f)
                dr_f   = np.radians(dec_f)
                fits_xyz  = np.column_stack([
                    np.cos(dr_f)*np.cos(rr_f),
                    np.cos(dr_f)*np.sin(rr_f),
                    np.sin(dr_f)
                ])
                fits_gids = gl
                fits_tree = cKDTree(fits_xyz)

            tol = 2 * np.sin(np.radians(cfg.tolerance_arcsec / 3600) / 2)
            for _, row in df.iterrows():
                ra_v  = float(row['ra'])  if np.isfinite(float(row['ra']))  else np.nan
                dec_v = float(row['dec']) if np.isfinite(float(row['dec'])) else np.nan
                if not (np.isfinite(ra_v) and np.isfinite(dec_v)): continue
                rr = np.radians(ra_v); dr = np.radians(dec_v)
                qxyz = np.array([np.cos(dr)*np.cos(rr), np.cos(dr)*np.sin(rr), np.sin(dr)])
                idx = fits_tree.query_ball_point(qxyz, r=tol)
                if not idx: n_skip += 1; continue
                gid   = int(fits_gids[idx[0]])
                rv_v  = float(row['rv'])
                err_v = float(row['e_rv'])
                if not (np.isfinite(rv_v) and np.isfinite(err_v) and err_v > 0): continue
                if is_exact_duplicate(rv_v, err_v): continue
                if gid not in ss[surv]:
                    ss[surv][gid] = {'rvs': [], 'params': {p: np.nan for p in params}}
                    n_new_gid += 1
                ss[surv][gid]['rvs'].append((rv_v, err_v))
                n_added += 1

        log(f"    {surv}: {n_added:,} measurements added, "
            f"{n_new_gid:,} new stars, {n_skip:,} unmatched")

    ss = dict(ss)
    log("\n  Per-survey unique star counts:")
    for s in sorted(ss.keys()):
        n_multi = sum(1 for g in ss[s].values() if len(g['rvs']) >= 2)
        log(f"    {s:<16s}: {len(ss[s]):>10,} unique stars, {n_multi:>8,} >=2 obs")

    save_ckpt(ckpt, ss)
    gc.collect()
    return ss


# =============================================================================
# HELPER: WEIGHTED BEST RV PER STAR PER SURVEY
# =============================================================================
def _best_rv(survey_stars):
    """Returns survey_best[survey][gid] = (rv, err, params_dict)"""
    out = {}
    for surv, stars in survey_stars.items():
        best = {}
        for gid, sdata in stars.items():
            obs = sdata['rvs']
            pr  = sdata['params']
            if len(obs) == 1:
                best[gid] = (obs[0][0], obs[0][1], pr)
            else:
                v = np.array([x[0] for x in obs])
                e = np.array([x[1] for x in obs])
                ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
                if np.any(ok):
                    w = 1.0 / e[ok]**2
                    best[gid] = (float(np.sum(v[ok]*w)/np.sum(w)),
                                 float(1.0/np.sqrt(np.sum(w))), pr)
                else:
                    best[gid] = (float(np.nanmean(v)), np.nan, pr)
        out[surv] = best
    return out


# =============================================================================
# PHASE 4: DUP METHOD (Sec 4.1)  — GAIA EXCLUDED
# =============================================================================
def phase4_dup(cfg, survey_stars):
    ckpt = Path(cfg.checkpoint_dir) / "phase4_v5.pkl"
    if ckpt.exists():
        log("Phase 4: Loading checkpoint")
        return load_ckpt(ckpt)

    log("\nPhase 4: DUP method (Sec 4.1) [GAIA excluded — spatial dups already averaged]...")
    results = {}

    for surv in sorted(survey_stars.keys()):
        # ← KEY FIX: skip GAIA
        if surv not in DUP_SURVEYS:
            log(f"  {surv:<12s}: skipped (not in DUP_SURVEYS)")
            continue

        stars = survey_stars[surv]
        # Only stars with >=2 INDEPENDENT measurements (from CSV or FITS multi-epoch)
        multi = {g: d for g, d in stars.items() if len(d['rvs']) >= 2}
        if len(multi) < 5:
            log(f"  {surv:<12s}: {len(multi)} multi-obs stars, skip DUP")
            continue

        norms, raws = [], []
        for gid, sdata in tqdm(multi.items(), desc=f"DUP {surv}",
                                leave=False, mininterval=2.0):
            obs = sdata['rvs']
            if len(obs) > cfg.max_obs_per_star:
                obs = sorted(obs, key=lambda x: x[1])[:cfg.max_obs_per_star]
            np_s = 0
            for (v1, e1), (v2, e2) in combinations(obs, 2):
                # Skip exact-duplicate pairs
                if abs(v1-v2) < 1e-6 and abs(e1-e2) < 1e-6: continue
                d  = v1 - v2
                sc = np.sqrt(e1**2 + e2**2)
                if sc > 0:
                    norms.append(d/sc); raws.append(d)
                    np_s += 1
                    if np_s >= cfg.max_pairs_per_star: break

        if len(norms) < cfg.min_pairs_dup:
            log(f"  {surv:<12s}: {len(norms)} pairs, skip DUP"); continue

        nd   = np.array(norms); rd = np.array(raws)
        nd_c = nd[np.abs(nd) < 50]
        mad  = float(np.median(np.abs(nd_c)))
        nf   = 1.4826 * mad          # ← normalization factor (MAD-based)
        mu, std = norm.fit(nd_c)
        rd_c    = rd[np.abs(rd) < 500]

        results[surv] = {
            'norm_diffs':  nd,
            'raw_diffs':   rd,
            'norm_factor': nf,
            'mad':         mad,
            'n_pairs':     len(nd),
            'n_stars':     len(multi),
            'mean_norm':   mu,
            'std_norm':    std,
            'mean_raw':    float(np.mean(rd_c)),
            'std_raw':     float(np.std(rd_c)),
            'median_raw':  float(np.median(rd_c)),
            'mad_raw':     float(np.median(np.abs(rd_c - np.median(rd_c)))),
            'norm_mad':    nf,
            'norm_std':    std,
        }
        log(f"  {surv:<12s}: {len(multi):>8,} stars {len(nd):>10,} pairs | "
            f"normMAD={nf:.3f}  normStd={std:.3f}")

    save_ckpt(ckpt, results)
    return results


# =============================================================================
# PHASE 5: TCH METHOD (Sec 4.2)
# =============================================================================
def phase5_tch(cfg, survey_stars, dup_results):
    ckpt = Path(cfg.checkpoint_dir) / "phase5_v5.pkl"
    if ckpt.exists():
        log("Phase 5: Loading checkpoint")
        return load_ckpt(ckpt)

    log("\nPhase 5: TCH method (Sec 4.2)...")
    sb      = _best_rv(survey_stars)
    surveys = sorted(sb.keys())

    # Simple pairwise ZP (median offset)
    pzp = {}
    for si, sj in combinations(surveys, 2):
        common = set(sb[si]) & set(sb[sj])
        if len(common) < cfg.min_stars_tch: continue
        diffs = [sb[si][g][0] - sb[sj][g][0] for g in common
                 if np.isfinite(sb[si][g][0]) and np.isfinite(sb[sj][g][0])]
        if len(diffs) >= cfg.min_stars_tch:
            pzp[(si, sj)] = float(np.median(diffs))

    # Pairwise variances (ZP-corrected)
    pw = {}
    for si, sj in combinations(surveys, 2):
        common = set(sb[si]) & set(sb[sj])
        if len(common) < cfg.min_stars_tch: continue
        zpc = pzp.get((si, sj), -pzp.get((sj, si), 0) if (sj, si) in pzp else 0)
        diffs = [(sb[si][g][0] - sb[sj][g][0]) - zpc for g in common
                 if np.isfinite(sb[si][g][0]) and np.isfinite(sb[sj][g][0])]
        if len(diffs) < cfg.min_stars_tch: continue
        d   = np.array(diffs)
        var = float(np.sum(d**2) / len(d))
        pw[(si, sj)] = {'var': var, 'n': len(d), 'sigma': np.sqrt(var)}
        log(f"  {si}-{sj}: {len(d):,} common, σ={np.sqrt(var):.3f} km/s")

    tch = {'pairwise': pw, 'survey_sigma': {}, 'norm_factors': {},
           'triplets_used': {}}

    for target in surveys:
        partners = []
        for (si, sj) in pw:
            if si == target: partners.append(sj)
            elif sj == target: partners.append(si)
        ests = []
        for p1, p2 in combinations(partners, 2):
            keys = [tuple(sorted([target, p1])),
                    tuple(sorted([target, p2])),
                    tuple(sorted([p1, p2]))]
            if not all(k in pw for k in keys): continue
            s2 = (pw[keys[0]]['var'] + pw[keys[1]]['var'] - pw[keys[2]]['var']) / 2
            nm = min(pw[k]['n'] for k in keys)
            ests.append((s2, nm))

        if not ests: continue
        valid = [(s, n) for s, n in ests if s > 0]
        sig   = (np.sqrt(sum(s*n for s,n in valid) / sum(n for _,n in valid))
                 if valid else 0.0)
        errs  = [d[1] for d in sb[target].values()
                 if np.isfinite(d[1]) and d[1] > 0]
        me    = float(np.median(errs)) if errs else np.nan
        nf    = sig / me if (np.isfinite(me) and me > 0 and sig > 0) else np.nan

        tch['survey_sigma'][target]  = sig
        tch['norm_factors'][target]  = nf
        log(f"  TCH {target:<12s}: σ={sig:.3f} km/s  medErr={me:.3f}  f={nf:.3f}")

    # Combined factors
    # ── Strategy ────────────────────────────────────────────────────────────
    # 1. If DUP N_pairs >= MIN_DUP_PAIRS_SIGNIFICANT  → DUP is reliable
    # 2. If TCH also available → use average of both (most robust)
    # 3. Only DUP available and reliable → use DUP
    # 4. Only TCH available → use TCH
    # 5. DUP unreliable (too few pairs) and no TCH → use TCH if available, else 1.0
    # This keeps all surveys in the analysis while distinguishing reliability.
    log("\n  === COMBINED NORMALIZATION FACTORS ===")
    combined = {}
    reliability = {}
    for surv in surveys:
        df_val  = dup_results.get(surv, {}).get('norm_factor')
        df_n    = dup_results.get(surv, {}).get('n_pairs', 0)
        tf_val  = tch['norm_factors'].get(surv)
        dup_ok  = (df_val is not None and np.isfinite(df_val)
                   and df_n >= MIN_DUP_PAIRS_SIGNIFICANT)
        tch_ok  = (tf_val is not None and np.isfinite(tf_val) and tf_val > 0)

        if dup_ok and tch_ok:
            f      = (df_val + tf_val) / 2.0
            method = f'avg(DUP={df_val:.3f}, TCH={tf_val:.3f})'
        elif dup_ok:
            f      = df_val
            method = f'DUP={df_val:.3f} (N_pairs={df_n:,})'
        elif tch_ok:
            f      = tf_val
            method = f'TCH={tf_val:.3f}'
        else:
            f      = 1.0
            method = 'default=1.0 (insufficient data)'

        combined[surv]     = f
        reliability[surv]  = method
        log(f"    {surv:<14s}: factor = {f:.3f}  [{method}]")

    tch['combined_factors'] = combined
    tch['pairwise_zp']      = pzp

    save_ckpt(ckpt, tch)
    return tch


# =============================================================================
# PHASE 6: GAIA INTERNAL CALIBRATION (Sec 5.1, Eq. 5-7)
# =============================================================================
def phase6_gaia_cal(cfg, survey_stars, tch_results):
    ckpt = Path(cfg.checkpoint_dir) / "phase6_v5.pkl"
    if ckpt.exists():
        log("Phase 6: Loading checkpoint")
        return load_ckpt(ckpt)

    log("\nPhase 6: Gaia internal calibration (Sec 5.1)...")
    sb = _best_rv(survey_stars)

    if 'GAIA' not in sb:
        log("  GAIA not found, skipping Phase 6")
        return {'coefficients': {}, 'corrections': {}, 'diag_data': {}}

    gaia_data = sb['GAIA']

    def wls_fit(x, y, w, degree=2):
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x, y, w = x[valid], y[valid], w[valid]
        if len(x) < 10: return None, None
        coeffs    = np.polyfit(x, y, degree, w=np.sqrt(w))
        residuals = y - np.polyval(coeffs, x)
        return coeffs, {'x': x, 'y': y, 'resid': residuals, 'n': len(x)}

    results = {'coefficients': {}, 'per_survey': {}, 'diag_data': {}}

    for cal_param, cal_surveys, degree, eq_name in [
        ('Gmag', ['APOGEE', 'GALAH', 'GES', 'RAVE'], 2, 'Eq5'),
        ('FeH',  ['APOGEE', 'GALAH', 'GES'],         1, 'Eq6'),
        ('Teff', ['APOGEE'],                          2, 'Eq7'),
    ]:
        log(f"\n  --- {eq_name}: ΔRV vs {cal_param} ---")
        # We bin per survey then pool the bins for a single global fit.
        # This prevents surveys with millions of stars from dominating the
        # polynomial shape vs surveys with only thousands.
        all_bin_x, all_bin_y, all_bin_w = [], [], []

        for surv in cal_surveys:
            if surv not in sb: continue
            sd     = sb[surv]
            common = set(gaia_data) & set(sd)
            if len(common) < cfg.min_stars_zp: continue

            drvs   = [gaia_data[g][0] - sd[g][0] for g in common
                      if np.isfinite(gaia_data[g][0]) and np.isfinite(sd[g][0])]
            zp_shift = np.median(drvs) if drvs else 0.0

            xs, ys, ws = [], [], []
            for g in common:
                rv_g, err_g, pr_g = gaia_data[g]
                rv_s, err_s, pr_s = sd[g]
                if not (np.isfinite(rv_g) and np.isfinite(rv_s)): continue
                pval = pr_s.get(cal_param, pr_g.get(cal_param, np.nan))
                if not np.isfinite(pval): continue
                drv      = (rv_g - rv_s) - zp_shift
                comb_err = (np.sqrt(err_g**2 + err_s**2)
                            if (np.isfinite(err_g) and np.isfinite(err_s)) else 1.0)
                comb_err = max(comb_err, 0.01)
                xs.append(pval); ys.append(drv); ws.append(1.0/comb_err**2)

            if len(xs) < 50: continue
            xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)

            # ── KEY FIX: bin the data, fit to medians only ─────────────────
            n_bins = min(40, max(15, len(xs) // 300))
            lo, hi = np.percentile(xs, 1), np.percentile(xs, 99)
            be     = np.linspace(lo, hi, n_bins + 1)
            bc     = 0.5 * (be[:-1] + be[1:])
            for j in range(n_bins):
                m = (xs >= be[j]) & (xs < be[j+1])
                if m.sum() < cfg.min_bin_count: continue
                all_bin_x.append(bc[j])
                all_bin_y.append(float(np.median(ys[m])))
                all_bin_w.append(float(m.sum()))   # weight = N per bin

            # Per-survey raw storage for diagnostics (kept for plot overlay)
            results['per_survey'][f'{eq_name}_{surv}'] = {
                'xs': xs, 'ys': ys, 'zp': zp_shift,
                'n': len(xs), 'surv': surv
            }
            log(f"    {surv}: N_raw={len(xs):,}  N_bins={n_bins}  "
                f"ZP_shift={zp_shift:+.3f} km/s")

        if len(all_bin_x) >= degree + 2:
            bx = np.array(all_bin_x)
            by = np.array(all_bin_y)
            bw = np.array(all_bin_w)
            valid = np.isfinite(bx) & np.isfinite(by)
            bx, by, bw = bx[valid], by[valid], bw[valid]
            # WLS fit to binned medians — polynomial anchored to where data exists
            coeffs_g = np.polyfit(bx, by, degree, w=np.sqrt(bw))
            results['coefficients'][eq_name] = coeffs_g.tolist()
            log(f"    GLOBAL (bins): N_bins={len(bx)}  "
                f"coeffs={np.array2string(coeffs_g, precision=6)}")
            results['diag_data'][cal_param] = {
                'bin_x':   bx,   'bin_y':   by,   'bin_w':   bw,
                'coeffs':  coeffs_g,
                'eq_name': eq_name,
            }

    save_ckpt(ckpt, results)
    return results


# =============================================================================
# PHASE 7: SURVEY CALIBRATION (Sec 5.2, Eq. 8)
# =============================================================================
def phase7_survey_cal(cfg, survey_stars, gaia_cal):
    """
    Fit ΔRV = f(Teff², Teff, logg, [Fe/H], S/N, RV) per survey.
    Splits: dwarfs / giants for most surveys; cool / hot only for LAMOST.
    NOTE: Dwarf/giant/hot/cold splits are ONLY here in Phase 7 (not in DUP/TCH).
    """
    ckpt = Path(cfg.checkpoint_dir) / "phase7_v5.pkl"
    if ckpt.exists():
        log("Phase 7: Loading checkpoint")
        return load_ckpt(ckpt)

    log("\nPhase 7: Survey calibration (Sec 5.2, Eq. 8)...")
    sb = _best_rv(survey_stars)

    if 'GAIA' not in sb:
        log("  GAIA not found, skipping Phase 7")
        return {}

    gaia       = sb['GAIA']
    gaia_coeffs = gaia_cal.get('coefficients', {})

    def corrected_gaia_rv(gid):
        rv_g, err_g, pr = gaia[gid]
        corr = 0.0
        if 'Eq5' in gaia_coeffs:
            gmag = pr.get('Gmag', np.nan)
            if np.isfinite(gmag): corr += np.polyval(gaia_coeffs['Eq5'], gmag)
        if 'Eq6' in gaia_coeffs:
            feh = pr.get('FeH', np.nan)
            if np.isfinite(feh): corr += np.polyval(gaia_coeffs['Eq6'], feh)
        if 'Eq7' in gaia_coeffs:
            teff = pr.get('Teff', np.nan)
            if np.isfinite(teff): corr += np.polyval(gaia_coeffs['Eq7'], teff)
        return rv_g - corr, err_g

    results = {}

    for surv in sorted(sb.keys()):
        if surv == 'GAIA': continue
        sd     = sb[surv]
        common = set(gaia) & set(sd)
        if len(common) < 50:
            log(f"  {surv}: {len(common)} common stars, skip"); continue

        rows = []
        for g in common:
            rv_g_corr, err_g = corrected_gaia_rv(g)
            rv_s, err_s, pr_s = sd[g]
            if not (np.isfinite(rv_g_corr) and np.isfinite(rv_s)): continue
            drv      = rv_g_corr - rv_s
            comb_err = (np.sqrt(err_g**2 + err_s**2)
                        if (np.isfinite(err_g) and np.isfinite(err_s)) else 1.0)
            comb_err = max(comb_err, 0.01)
            rows.append({
                'drv':   drv,
                'weight': 1.0/comb_err**2,
                'Teff':  pr_s.get('Teff',  np.nan),
                'logg':  pr_s.get('logg',  np.nan),
                'FeH':   pr_s.get('FeH',   np.nan),
                'SNR':   pr_s.get('SNR',   np.nan),
                'RV':    rv_s,
                'Gmag':  pr_s.get('Gmag',  np.nan),
            })

        if len(rows) < 50:
            log(f"  {surv}: {len(rows)} valid rows, skip"); continue

        df = pd.DataFrame(rows)
        log(f"  {surv}: {len(df):,} stars for Eq.8")

        # Splits — dwarfs/giants for most; cool/hot ONLY for LAMOST
        splits = []
        if surv == 'LAMOST':
            lam_cool = df[df['Teff'] < 6200].copy()
            lam_hot  = df[df['Teff'] >= 6200].copy()
            if len(lam_cool) > 50: splits.append(('cool_Teff<6200K', lam_cool))
            if len(lam_hot)  > 50: splits.append(('hot_Teff>=6200K', lam_hot))
        else:
            dwarfs = df[df['logg'] > 3.5].copy()
            giants = df[df['logg'] <= 3.5].copy()
            if len(dwarfs) > 50: splits.append(('dwarfs_logg>3.5',  dwarfs))
            if len(giants) > 50: splits.append(('giants_logg<=3.5', giants))

        if not splits: splits = [('all', df)]

        surv_results = {'diag': df, 'fits': {}}

        for split_name, sdf in splits:
            features, feat_names = [], []
            for fname, col, use_sq in [
                ('Teff2','Teff',True), ('Teff','Teff',False),
                ('logg','logg',False), ('FeH','FeH',False),
                ('SNR','SNR',False), ('RV','RV',False)
            ]:
                vals = sdf[col].values if col in sdf.columns else np.full(len(sdf), np.nan)
                if use_sq: vals = vals**2
                if np.sum(np.isfinite(vals)) > len(sdf)*0.3:
                    features.append(np.where(np.isfinite(vals), vals, 0))
                    feat_names.append(fname)

            if not features: continue
            X = np.column_stack(features + [np.ones(len(sdf))])
            feat_names.append('intercept')
            y  = sdf['drv'].values
            w  = sdf['weight'].values
            ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
            X, y, w = X[ok], y[ok], w[ok]
            if len(y) < 20: continue

            try:
                W      = np.diag(w)
                coeffs = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
                resid  = y - X @ coeffs
                chi2   = np.sum(w * resid**2) / (len(y) - len(coeffs))
                surv_results['fits'][split_name] = {
                    'coeffs':     coeffs.tolist(),
                    'feat_names': feat_names,
                    'n':          len(y),
                    'chi2':       chi2,
                    'zp_before':  float(np.median(y)),
                    'zp_after':   float(np.median(resid)),
                }
                log(f"    {surv}/{split_name}: N={len(y):,}  chi2={chi2:.3f}  "
                    f"ZP: {np.median(y):+.3f} → {np.median(resid):+.3f} km/s")
            except Exception as e:
                log(f"    {surv}/{split_name}: fit failed: {e}")

        results[surv] = surv_results

    save_ckpt(ckpt, results)
    return results


# =============================================================================
# PHASE 8: NORMALIZED ERRORS PER UNIQUE STAR (Fig 13)
# =============================================================================
def phase8_norm_errors(cfg, survey_stars, tch_results):
    """
    For each survey, for each UNIQUE STAR (group_id):
      1. Compute weighted-average error across all that star's measurements.
      2. Multiply by the survey's combined normalization factor.
    This gives the normalized error per unique star — not per raw FITS row.

    WHY this is correct:
      - The normalization factor (from DUP/TCH) corrects for systematic over/under-
        estimation of the reported errors by the survey pipeline.
      - We plot per unique star so Gaia shows ~32M, not 38.9M.
      - ZP correction affects the RV, not the error. The normalization factor
        indirectly encodes systematic ZP-related scatter if ZP correlates with error.
    """
    ckpt = Path(cfg.checkpoint_dir) / "phase8_v5.pkl"
    if ckpt.exists():
        log("Phase 8: Loading checkpoint")
        return load_ckpt(ckpt)

    log("\nPhase 8: Normalized RV errors (per unique star)...")
    cf     = tch_results.get('combined_factors', {})
    result = {}

    for surv in sorted(survey_stars.keys()):
        f    = cf.get(surv, 1.0)
        errs = []
        for gid, sdata in survey_stars[surv].items():
            obs = sdata['rvs']
            # Weighted-average error for THIS unique star
            es  = np.array([e for _, e in obs])
            ok  = np.isfinite(es) & (es > 0)
            if not np.any(ok): continue
            if np.sum(ok) == 1:
                w_err = float(es[ok][0])
            else:
                w = 1.0 / es[ok]**2
                w_err = float(1.0 / np.sqrt(np.sum(w)))   # combined error
            errs.append(w_err * f)   # normalized error = combined_err × norm_factor

        if errs:
            ea = np.array(errs)
            result[surv] = {
                'errors': ea,
                'factor': f,
                'median': float(np.median(ea)),
                'n':      len(ea),           # ← unique stars, not rows
            }
            log(f"  {surv:<14s}: factor={f:.3f}  "
                f"median_norm_err={np.median(ea):.3f} km/s  "
                f"N_unique={len(ea):,}")

    save_ckpt(ckpt, result)
    return result


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
    'SDSS':   '#17becf',   # SDSS-BOSS + SEGUE merged
}
def _c(s): return COLORS.get(s, '#333333')


# =============================================================================
# PHASE 9: PLOTS + CSV
# =============================================================================
def phase9_plots(cfg, dup_results, tch_results, gaia_cal, survey_cal, nerr):
    log("\nPhase 9: Generating plots and tables...")
    out = Path(cfg.output_dir)

    # -----------------------------------------------------------------------
    # Fig 6: DUP Normalized ΔRV/σ distributions
    # -----------------------------------------------------------------------
    log("  Fig 6: DUP normalized differences...")
    fig, ax = plt.subplots(figsize=(12, 7))
    bins = np.linspace(-8, 8, 120)
    bc   = 0.5 * (bins[:-1] + bins[1:])
    xg   = np.linspace(-8, 8, 500)
    ax.plot(xg, norm.pdf(xg, 0, 1), 'k--', lw=2, alpha=0.5, label='N(0,1)')
    csv1 = {'bin_center': bc}

    for s in sorted(dup_results.keys()):
        dr  = dup_results[s]
        nd  = dr['norm_diffs']
        ndp = nd[(nd >= -8) & (nd <= 8)]
        if len(ndp) < 50: continue
        c, _ = np.histogram(ndp, bins=bins, density=True)
        csv1[s] = c
        ax.hist(bins[:-1], bins, weights=c, histtype='step', lw=2,
                color=_c(s), alpha=0.85,
                label=f"{s}  (N={len(ndp):,}, normMAD={dr['norm_mad']:.2f})")

    ax.set_xlim(-8, 8)
    ax.set_xlabel(r'$\Delta$RV / $\sqrt{\sigma_1^2+\sigma_2^2}$', fontsize=14)
    ax.set_ylabel('Normalized density', fontsize=14)
    ax.set_title('DUP Method: Normalized RV Differences [Sec 4.1, Fig 6]', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')
    fig.tight_layout()
    for e in ['png', 'pdf']: fig.savefig(out/f'fig6.{e}', dpi=300)
    plt.close(fig)
    pd.DataFrame(csv1).to_csv(out/'fig6_data.csv', index=False)

    # -----------------------------------------------------------------------
    # Fig 13: Normalized RV error per unique star — ADAPTIVE BINS
    # -----------------------------------------------------------------------
    log("  Fig 13: Normalized RV errors (per unique star, adaptive bins)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    csv4 = {}

    for s in sorted(nerr.keys()):
        ea = nerr[s]['errors']
        ep = ea[(ea >= 0) & (ea <= 4)]
        if len(ep) < 20: continue

        # Adaptive bins: sqrt(N) capped [30, 120] → NO spikes for small N
        be  = adaptive_bins(len(ep), lo=0.0, hi=4.0)
        bce = 0.5 * (be[:-1] + be[1:])
        c, _ = np.histogram(ep, bins=be, density=True)

        if 'bin_center' not in csv4:
            csv4['bin_center'] = bce   # use first survey's bin centres (approx)
        csv4[s] = c if len(c) == len(csv4.get('bin_center',c)) else c

        ax.hist(be[:-1], be, weights=c, histtype='step', lw=2,
                color=_c(s), alpha=0.85,
                label=(f"{s}  (med={nerr[s]['median']:.2f} km/s, "
                       f"N={nerr[s]['n']:,} unique stars, "
                       f"f={nerr[s]['factor']:.3f})"))

    ax.set_xlim(0, 4)
    ax.set_xlabel('Normalized RV error  [original err × norm_factor]  (km/s)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(
        'Normalized RV Error Distributions [Fig 13]\n'
        '(per unique star, weighted-avg error × DUP/TCH normalization factor)',
        fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, ls='--')
    fig.tight_layout()
    for e in ['png', 'pdf']: fig.savefig(out/f'fig13.{e}', dpi=300)
    plt.close(fig)
    if csv4: pd.DataFrame(csv4).to_csv(out/'fig13_data.csv', index=False)

    # -----------------------------------------------------------------------
    # Gaia calibration diagnostic plots (ΔRV vs Gmag, [Fe/H], Teff)
    # ΔRV = RV_Gaia − RV_survey  (calibrating Gaia to ground-based frame)
    # Fit is to binned medians only — polynomial is NOT extrapolated beyond data.
    # -----------------------------------------------------------------------
    diag = gaia_cal.get('diag_data', {})
    per_surv = gaia_cal.get('per_survey', {})

    for pname, pdata in diag.items():
        log(f"  Gaia cal plot: ΔRV vs {pname}...")
        coeffs  = pdata['coeffs']
        bx      = pdata['bin_x']
        by      = pdata['bin_y']
        bw      = pdata['bin_w']
        corrected_bins = by - np.polyval(coeffs, bx)

        # ── Individual survey medians for overlay ──────────────────────────
        eq_name = pdata.get('eq_name', '')
        surv_curves = {}   # surv → (bin_x, bin_med_before, bin_med_after)
        for key, sv in per_surv.items():
            if not key.startswith(eq_name): continue
            surv_name = sv.get('surv', key.replace(eq_name+'_',''))
            xs_ = sv.get('xs'); ys_ = sv.get('ys')
            if xs_ is None or len(xs_) < 20: continue
            zp_ = sv.get('zp', 0.0)
            n_b = min(30, max(10, len(xs_) // 200))
            lo_, hi_ = np.percentile(xs_, 2), np.percentile(xs_, 98)
            be_ = np.linspace(lo_, hi_, n_b + 1)
            bc_ = 0.5*(be_[:-1]+be_[1:])
            med_b, med_a = [], []
            for j in range(n_b):
                m_ = (xs_ >= be_[j]) & (xs_ < be_[j+1])
                if m_.sum() < cfg.min_bin_count:
                    med_b.append(np.nan); med_a.append(np.nan); continue
                y_b = float(np.median(ys_[m_]))
                y_a = float(np.median((ys_ - np.polyval(coeffs, xs_))[m_]))
                med_b.append(y_b); med_a.append(y_a)
            surv_curves[surv_name] = (bc_, np.array(med_b), np.array(med_a))

        # ── Figure: per-survey overlay (before only) + global fit ─────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax_i, (ax, show_after) in enumerate(zip(axes, [False, True])):
            for surv_name, (bc_, mb_, ma_) in surv_curves.items():
                vals = ma_ if show_after else mb_
                ok_  = np.isfinite(vals)
                ax.plot(bc_[ok_], vals[ok_], 'o-', ms=4, lw=1.5,
                        color=_c(surv_name), alpha=0.85, label=surv_name)

            # Global binned medians
            ok_b = np.isfinite(by)
            ok_a = np.isfinite(corrected_bins)
            if show_after:
                ax.plot(bx[ok_a], corrected_bins[ok_a], 'k^-', ms=6, lw=2,
                        label='Global (after)', zorder=5)
            else:
                ax.plot(bx[ok_b], by[ok_b], 'ks-', ms=6, lw=2,
                        label='Global (before)', zorder=5)
                # Fit line — ONLY over data range
                x_fit = np.linspace(bx.min(), bx.max(), 300)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), 'k--', lw=2,
                        alpha=0.6, label='Polynomial fit')

            ax.axhline(0, color='gray', ls=':', alpha=0.5)
            ax.set_xlabel(pname, fontsize=13)
            title_state = 'After calibration' if show_after else 'Before calibration'
            ax.set_ylabel(r'$\Delta$RV = RV$_{\rm Gaia}$ − RV$_{\rm survey}$  (km/s)',
                          fontsize=12)
            ax.set_title(f'Gaia cal: ΔRV vs {pname} — {title_state}', fontsize=12)
            # Clip x to data range only
            ax.set_xlim(bx.min() - 0.05*(bx.max()-bx.min()),
                        bx.max() + 0.05*(bx.max()-bx.min()))
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3, ls='--')

        fig.suptitle(
            f'Gaia Internal Calibration: ΔRV vs {pname}\n'
            r'ΔRV = RV$_{\rm Gaia}$ − RV$_{\rm survey}$  '
            '(fit to binned medians only, clipped to data range)',
            fontsize=12)
        fig.tight_layout()
        for e in ['png', 'pdf']: fig.savefig(out/f'gaia_cal_{pname}.{e}', dpi=300)
        plt.close(fig)

    # ── Multi-survey overlay: all Gaia calibration params in one figure ────
    if diag:
        pnames = list(diag.keys())
        ncols  = min(3, len(pnames))
        nrows  = (len(pnames) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
        axes = np.array(axes).flatten() if len(pnames) > 1 else [axes]
        for pi, pname_ in enumerate(pnames):
            ax = axes[pi]
            pdata_ = diag[pname_]
            eq_name_ = pdata_.get('eq_name', '')
            coeffs_  = pdata_['coeffs']
            bx_      = pdata_['bin_x']
            by_      = pdata_['bin_y']
            ok_      = np.isfinite(by_)
            ax.plot(bx_[ok_], by_[ok_], 'ks-', ms=7, lw=2, label='Global before')
            x_fit_ = np.linspace(bx_.min(), bx_.max(), 300)
            ax.plot(x_fit_, np.polyval(coeffs_, x_fit_), 'k--', lw=2, alpha=0.6,
                    label='Polynomial fit')
            for key_, sv_ in per_surv.items():
                if not key_.startswith(eq_name_): continue
                sn_ = sv_.get('surv', '')
                xs__ = sv_.get('xs'); ys__ = sv_.get('ys')
                if xs__ is None or len(xs__) < 20: continue
                n_b_ = min(25, max(8, len(xs__) // 200))
                be__ = np.linspace(np.percentile(xs__,2), np.percentile(xs__,98), n_b_+1)
                bc__ = 0.5*(be__[:-1]+be__[1:])
                mb__ = [np.median(ys__[(xs__>=be__[j])&(xs__<be__[j+1])])
                        if np.sum((xs__>=be__[j])&(xs__<be__[j+1]))>=cfg.min_bin_count
                        else np.nan for j in range(n_b_)]
                mb__ = np.array(mb__); ok__ = np.isfinite(mb__)
                ax.plot(bc__[ok__], mb__[ok__], 'o-', ms=4, lw=1.5,
                        color=_c(sn_), alpha=0.8, label=sn_)
            ax.axhline(0, color='gray', ls=':', alpha=0.5)
            ax.set_xlabel(pname_, fontsize=12)
            ax.set_ylabel(r'$\Delta$RV = RV$_{\rm Gaia}$ − RV$_{\rm survey}$  (km/s)',
                          fontsize=11)
            ax.set_xlim(bx_.min()-0.05*(bx_.max()-bx_.min()),
                        bx_.max()+0.05*(bx_.max()-bx_.min()))
            ax.set_title(f'ΔRV vs {pname_}', fontsize=12)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, ls='--')
        for ax in axes[len(pnames):]: ax.set_visible(False)
        fig.suptitle(
            'Gaia Calibration: All parameters — survey-by-survey binned medians\n'
            r'ΔRV = RV$_{\rm Gaia}$ − RV$_{\rm survey}$',
            fontsize=13)
        fig.tight_layout()
        for e in ['png', 'pdf']: fig.savefig(out/f'gaia_cal_all_surveys.{e}', dpi=300)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Survey calibration 6-panel plots (one per survey)
    # ΔRV = RV_Gaia_corrected − RV_survey  (calibrating survey to Gaia frame)
    # -----------------------------------------------------------------------
    param_list = ['Teff', 'logg', 'FeH', 'SNR', 'RV', 'Gmag']
    # Collect per-param binned medians for the all-surveys overlay
    all_survey_bins = {p: {} for p in param_list}

    for surv, sres in survey_cal.items():
        if 'diag' not in sres: continue
        df   = sres['diag']
        log(f"  Survey cal 6-panel: {surv}...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for pi, pname in enumerate(param_list):
            ax = axes[pi]
            if pname not in df.columns or df[pname].isna().all():
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes); ax.set_title(pname); continue
            x = df[pname].values; y = df['drv'].values
            ok = np.isfinite(x) & np.isfinite(y)
            if np.sum(ok) < 20:
                ax.text(0.5, 0.5, 'Too few', ha='center', va='center',
                        transform=ax.transAxes); ax.set_title(pname); continue
            x, y = x[ok], y[ok]
            ax.scatter(x, y, s=1, alpha=0.1, color=_c(surv), rasterized=True)
            nbins = min(30, max(10, len(x)//200))
            be    = np.linspace(np.percentile(x,2), np.percentile(x,98), nbins+1)
            bc    = 0.5*(be[:-1]+be[1:])
            meds  = [np.median(y[(x>=be[j])&(x<be[j+1])])
                     if np.sum((x>=be[j])&(x<be[j+1])) >= cfg.min_bin_count
                     else np.nan for j in range(nbins)]
            meds  = np.array(meds); okm = np.isfinite(meds)
            ax.plot(bc[okm], meds[okm], 'ko-', ms=5, lw=1.5, label='Median ΔRV')
            ax.axhline(0, color='gray', ls=':', alpha=0.5)
            ax.set_xlabel(pname, fontsize=11)
            ax.set_ylabel(r'$\Delta$RV = RV$_{\rm Gaia,\,corr}$ − RV$_{\rm survey}$  (km/s)',
                          fontsize=10)
            ax.set_ylim(-15, 15); ax.grid(True, alpha=0.3, ls='--'); ax.legend(fontsize=8)
            # Store for overlay
            all_survey_bins[pname][surv] = (bc[okm], meds[okm])

        fig.suptitle(
            f'Survey Calibration: {surv}\n'
            r'$\Delta$RV = RV$_{\rm Gaia,\,corrected}$ − RV$_{\rm ' + surv + r'}$'
            '\n(positive = survey underestimates RV relative to Gaia)',
            fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        for e in ['png', 'pdf']: fig.savefig(out/f'survey_cal_{surv}.{e}', dpi=300)
        plt.close(fig)

    # ── All-surveys overlay: one figure per calibration parameter ─────────
    log("  All-surveys calibration overlay plots...")
    for pname in param_list:
        sdata = {s: v for s, v in all_survey_bins[pname].items() if len(v[0]) > 0}
        if not sdata: continue
        fig, ax = plt.subplots(figsize=(12, 6))
        for surv, (bc_, med_) in sdata.items():
            ax.plot(bc_, med_, 'o-', ms=5, lw=1.8,
                    color=_c(surv), alpha=0.85, label=surv)
        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
        ax.set_xlabel(pname, fontsize=13)
        ax.set_ylabel(r'$\Delta$RV = RV$_{\rm Gaia,\,corr}$ − RV$_{\rm survey}$  (km/s)',
                      fontsize=12)
        ax.set_title(
            f'All-Survey Calibration Overlay: ΔRV vs {pname}\n'
            r'Each curve = median $\Delta$RV per bin for that survey',
            fontsize=12)
        ax.legend(fontsize=10, loc='best'); ax.grid(True, alpha=0.3, ls='--')
        ax.set_ylim(-10, 10)
        fig.tight_layout()
        for e in ['png', 'pdf']:
            fig.savefig(out/f'survey_cal_overlay_{pname}.{e}', dpi=300)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------------
    log("  Writing CSV tables...")

    # Table 3: DUP results
    rows = []
    for s, dr in sorted(dup_results.items()):
        rows.append({'Survey': s, 'N_stars': dr['n_stars'], 'N_pairs': dr['n_pairs'],
                     'Mean_raw': dr['mean_raw'], 'Std_raw': dr['std_raw'],
                     'Median_raw': dr['median_raw'], 'MAD_raw': dr['mad_raw'],
                     'NormMAD': dr['norm_mad'], 'NormStd': dr['norm_std']})
    if rows: pd.DataFrame(rows).to_csv(out/'table3_dup.csv', index=False)

    # Table 4: Combined normalization factors
    cf = tch_results.get('combined_factors', {})
    rows = [{'Survey': s, 'Combined_Factor': cf[s],
             'DUP_factor': dup_results.get(s, {}).get('norm_factor', np.nan),
             'TCH_factor': tch_results.get('norm_factors', {}).get(s, np.nan)}
            for s in sorted(cf.keys())]
    if rows: pd.DataFrame(rows).to_csv(out/'table4_norm_factors.csv', index=False)

    # Gaia calibration coefficients
    gc_ = gaia_cal.get('coefficients', {})
    if gc_:
        rows = [{'Equation': k, 'Coefficients': str(v)} for k, v in gc_.items()]
        pd.DataFrame(rows).to_csv(out/'gaia_cal_coefficients.csv', index=False)

    # Survey calibration coefficients
    rows = []
    for surv, sres in survey_cal.items():
        for split, fdata in sres.get('fits', {}).items():
            rows.append({'Survey': surv, 'Split': split,
                         'Features': str(fdata['feat_names']),
                         'Coefficients': str(fdata['coeffs']),
                         'N': fdata['n'], 'Chi2': fdata['chi2'],
                         'ZP_before_km_s': fdata['zp_before'],
                         'ZP_after_km_s':  fdata['zp_after']})
    if rows: pd.DataFrame(rows).to_csv(out/'survey_cal_coefficients.csv', index=False)

    # Summary: N unique stars per survey + normalized error stats
    rows = []
    for s in sorted(VALID_SURVEYS):
        nr = nerr.get(s, {})
        rows.append({
            'Survey':        s,
            'N_unique_stars': nr.get('n', 0),
            'Norm_factor':    nr.get('factor', np.nan),
            'Median_norm_err_km_s': nr.get('median', np.nan),
        })
    pd.DataFrame(rows).to_csv(out/'summary_unique_stars.csv', index=False)

    log("  All outputs saved.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(description='RV Normalization v5')
    p.add_argument('input_fits')
    p.add_argument('--output-dir', '-o', default='./rv_norm_output_v5')
    p.add_argument('--checkpoint-dir', default='./rv_norm_ckpt_v5')
    p.add_argument('--chunk-size', '-c', type=int, default=3_000_000)
    p.add_argument('--tolerance', type=float, default=1.0)
    p.add_argument('--nside', type=int, default=32)
    p.add_argument('--apogee-csv', default='./astro_data/APOGEE_DR17/APOGEE_DR17_merged_rv_parallax.csv')
    p.add_argument('--galah-csv',  default='./astro_data/GALAH_DR3/GALAH_DR3_merged_rv_parallax.csv')
    p.add_argument('--ges-csv',    default='./astro_data/GES_DR5/GES_DR5_merged_rv_parallax.csv')
    p.add_argument('--rave-csv',   default='./astro_data/RAVE_DR6/RAVE_DR6_merged_rv_parallax.csv')
    p.add_argument('--clean', action='store_true',
                   help='Delete checkpoints and restart from scratch')
    args = p.parse_args()

    if args.clean:
        import shutil
        if os.path.exists(args.checkpoint_dir):
            shutil.rmtree(args.checkpoint_dir)
            print(f"Cleaned {args.checkpoint_dir}")

    cfg = Config(
        input_fits     = args.input_fits,
        output_dir     = args.output_dir,
        checkpoint_dir = args.checkpoint_dir,
        chunk_size     = args.chunk_size,
        tolerance_arcsec = args.tolerance,
        healpix_nside  = args.nside,
        apogee_csv     = args.apogee_csv,
        galah_csv      = args.galah_csv,
        ges_csv        = args.ges_csv,
        rave_csv       = args.rave_csv,
    )

    t0 = time.time()
    log("=" * 72)
    log("RV NORMALIZATION v5 — Survey of Surveys I")
    log("=" * 72)
    log(f"Input FITS : {cfg.input_fits}")
    log(f"Output     : {cfg.output_dir}")
    log(f"Surveys    : {', '.join(sorted(VALID_SURVEYS))}")
    log(f"DUP surveys: {', '.join(sorted(DUP_SURVEYS))}  [GAIA excluded from DUP]")

    # Phase 0: Load external survey CSVs
    csv_data = phase0_load_csvs(cfg)

    # Phase 1: Extract one RV/star from FITS
    data = phase1_extract(cfg)

    # Phase 2: Spatial grouping
    groups = phase2_spatial(cfg, data)

    # Phase 2b: source_id → group_id map
    sid_map = phase2b_sid_map(cfg, data, groups)

    # Phase 3: Build per-survey star data (FITS + CSVs)
    survey_stars = phase3_build(cfg, data, groups, csv_data, sid_map)
    del data, groups, sid_map, csv_data
    gc.collect()

    # Phase 4: DUP normalization (GAIA excluded)
    dup_results = phase4_dup(cfg, survey_stars)

    # Phase 5: TCH normalization
    tch_results = phase5_tch(cfg, survey_stars, dup_results)

    # Phase 6: Gaia internal calibration (Eq. 5-7)
    gaia_cal = phase6_gaia_cal(cfg, survey_stars, tch_results)

    # Phase 7: Survey calibration (Eq. 8, dwarfs/giants/cool/hot)
    survey_cal = phase7_survey_cal(cfg, survey_stars, gaia_cal)

    # Phase 8: Normalized errors per unique star
    norm_errors = phase8_norm_errors(cfg, survey_stars, tch_results)

    # Phase 9: All plots + CSV tables
    phase9_plots(cfg, dup_results, tch_results, gaia_cal, survey_cal, norm_errors)

    log(f"\nDONE in {(time.time()-t0)/60:.1f} min")
    log(f"Outputs in: {cfg.output_dir}")


if __name__ == '__main__':
    main()