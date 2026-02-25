"""
Adaptive Membership Analysis V11 — Full Phase-Space + Publication Plots
=======================================================================
V11 additions over V10:
  - SGR stream: full 7-term Bayesian EM (same rigor as GC/OC/DW)
  - SGR finer bins: 5 kpc width, 15–80 kpc
  - Blue parallax distance overlay (member → master fallback)
  - Gray opaque background for PM/Sky (all pre-match members)
  - 4 separate summary series (RA/Dec, PM, Dist, RV)
  - GRAND_SUMMARY: 4 columns × N rows, objects with ≥10 green matches
  - Median + MAD bands for green and blue histograms
  - Reference bands on highest zorder
  - Distance gap bridging: literature + parallax fallback
"""

import os, sys, json, time, logging, argparse, warnings, glob
import gc as gcmod
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import (multivariate_normal, gaussian_kde,
                         median_abs_deviation, ks_2samp, kstest, norm)
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Astropy ────────────────────────────────────────────────────────────────
try:
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

# ── HDBSCAN ────────────────────────────────────────────────────────────────
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file=None, level=logging.INFO):
    logger = logging.getLogger('AdaptiveMembership')
    logger.setLevel(level)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MASTER_CATALOG   = None
    GC_MEMBERS_FILE  = None;  OC_MEMBERS_FILE  = None
    SGR_MEMBERS_FILE = None;  DWG_MEMBERS_FILE = None
    GC_DIST_FILE     = None
    OUTPUT_DIR       = './outputs'
    CHECKPOINT_DIR   = './checkpoints'

    # ── Master catalog column mapping ───────────────────────────────────────
    MASTER_COLS = {
        'ra':           'RA_final',
        'dec':          'DEC_final',
        'pmra':         'pmra_final',
        'pmdec':        'pmdec_final',
        'pmra_err':     'pmra_err_final',
        'pmdec_err':    'pmdec_err_final',
        'dist':         'distance_final',
        'dist_err':     'distance_err_final',
        'rv':           'Weighted_Avg_final',
        'rv_err':       'Weighted_Avg_err_final',
        'parallax':     'parallax_final',
        'parallax_err': 'parallax_err_final',
        'params_est':   'stellar_params_est',
        'params_err':   'stellar_params_err',
        'gmag':         'Gmag',
        'bpmag':        'BPmag',
        'rpmag':        'RPmag',
        'bp_rp':        'BP-RP',
        'bp_g':         'BP-G',
        'g_rp':         'G-RP',
        'logg':         'logg',
        'feh':          'feh',
        'teff':         'Teff',
        'ag':           'AG',
        'ebprp':        'E(BP-RP)',
        'a0':           'A0',
        'ruwe':         'RUWE',
        'gofal':        'gofAL',
        'chi2al':       'chi2AL',
        'rv_sn':        'RVS/N',
        'rv_chi2':      'RVchi2',
        'rv_gof':       'RVgof',
        'rvnper':       'RVNper',
        'pm_corr':      'pmRApmDEcor',
        'gal_l':        'l',
        'gal_b':        'b',
        'pgal':         'PGal',
        'pqso':         'PQSO',
        'pss':          'PSS',
        'grvsm':        'GRVSmag',
        'catwise_w1':   'catwise_w1',
        'catwise_w2':   'catwise_w2',
        'logg_lo':      'b_logg_x',
        'logg_hi':      'B_logg_xa',
        'feh_lo':       'b_[Fe/H]_x',
        'feh_hi':       'B_[Fe/H]_xa',
        'teff_lo':      'b_Teff_x',
        'teff_hi':      'B_Teff_xa',
        'ag_lo':        'b_AG_x',
        'ag_hi':        'B_AG_xa',
        'ebprp_lo':     'b_E(BP-RP)_x',
        'ebprp_hi':     'B_E(BP-RP)_xa',
    }
    ALT_MASTER_COLS = {
        'rv':     ['Weighted_Avg_final', 'ZP_final', 'RV_final',
                   'Weighted_Avg', 'radial_velocity', 'RV'],
        'rv_err': ['Weighted_Avg_err_final', 'ZP_err_final', 'RV_err_final',
                   'radial_velocity_error', 'RV_err'],
    }

    # ── Member catalog column mappings ──────────────────────────────────────
    GC_MEM_COLS = {
        'key': 'source', 'ra': 'ra', 'dec': 'dec',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmra_error', 'pmdec_err': 'pmdec_error',
        'pmra_pmdec_corr': 'pmra_pmdec_corr',
        'parallax': 'parallax',
        'membership_prob': 'membership_probability',
        'rv': 'RV_weighted_avg', 'rv_err': 'e_RV_weighted_avg',
        'gmag': 'g_mag', 'bp_rp': 'bp_rp',
    }
    GC_DIST_COLS = {
        'name':          'Name',
        'lit_dist':      'Lit. dist. (kpc)',
        'lit_dist_err':  'Lit. dist. Err+',
        'mean_dist':     'Mean distance (kpc)',
        'mean_dist_err': 'Mean distance Err+',
    }
    OC_MEM_COLS = {
        'key': 'Cluster', 'ra': 'RAdeg', 'dec': 'DEdeg',
        'pmra': 'pmRA', 'pmdec': 'pmDE',
        'pmra_err': 'e_pmRA', 'pmdec_err': 'e_pmDE',
        'pmra_pmdec_corr': 'pmRApmDEcor',
        'parallax': 'Plx', 'membership_prob': 'Proba',
        'rv': 'RV', 'rv_err': 'e_RV',
        'gmag': 'Gmag', 'bp_rp': 'BP-RP',
        'plx_err': 'e_Plx',
    }
    # ── V11: SGR columns expanded for full phase-space EM ───────────────────
    SGR_MEM_COLS = {
        'key': None,
        'ra': 'ra', 'dec': 'dec',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmraerr', 'pmdec_err': 'pmdecerr',
        'parallax': 'parallax', 'parallax_err': 'plxerr',
        'dist': 'dist', 'dist_err': 'disterr',
        'rv': 'vlos', 'rv_err': 'vloserr',
        'feh': 'FeH', 'feh_err': 'FeHerr',
        'gmag': 'g_mag', 'bp_rp': 'bp_rp',
        'j_mag': 'j_mag', 'h_mag': 'h_mag', 'k_mag': 'k_mag',
        'ebv': 'ebv',
        'lambda_sgr': 'Lambda', 'beta_sgr': 'Beta',
        'src': 'src',
    }
    DWG_MEM_COLS = {
        'key': 'name',
        'ra': 'ra_x', 'dec': 'dec_x',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmra_error', 'pmdec_err': 'pmdec_error',
        'distance': 'distance', 'distance_err': 'distance_error',
        'distance_modulus': 'distance_modulus',
        'distance_modulus_err': 'distance_modulus_error',
        'rhalf': 'rhalf',
        'rv': 'RV_km_s', 'rv_err': 'e_RV_km_s',
        'rv_ref': 'vlos_systemic', 'rv_ref_err': 'vlos_systemic_error',
        'rv_sigma': 'vlos_sigma', 'rv_sigma_err': 'vlos_sigma_error',
        'ellipticity': 'ellipticity',
        'position_angle': 'position_angle',
        'rcore': 'rcore', 'rking': 'rking',
        'king_c': 'king_concentration',
        'metallicity': 'metallicity',
        'metallicity_err': 'metallicity_error',
        'metallicity_sigma': 'metallicity_sigma',
        'edr3_pmra': 'edr3_pmra', 'edr3_pmdec': 'edr3_pmdec',
        'edr3_pmra_err': 'edr3_pmra_error',
        'edr3_pmdec_err': 'edr3_pmdec_error',
        'gmax': 'Gmax', 'rmax_deg': 'Rmax',
    }

    # ── Analysis parameters ─────────────────────────────────────────────────
    CROSSMATCH_RADIUS_ARCSEC = 1.0
    GMM_N_COMPONENTS  = 2;  GMM_MAX_ITER = 300
    GMM_N_INIT        = 10; GMM_RANDOM_STATE = 42
    DBSCAN_EPS_CLEANUP      = 0.3;  DBSCAN_MIN_SAMPLES_CLEANUP = 3
    DBSCAN_EPS_OC           = 0.25; DBSCAN_MIN_SAMPLES_OC      = 5
    DBSCAN_EPS_STREAM       = 0.4;  DBSCAN_MIN_SAMPLES_STREAM  = 3
    P_MEM_HIGH              = 0.8;  P_MEM_LOW                  = 0.2
    MIN_STARS_FOR_ANALYSIS  = 10;   MIN_SUMMARY_MATCH          = 5
    # V11: SGR finer bins — 5 kpc width, 15–80 kpc range
    SGR_BIN_START_KPC       = 15.0; SGR_BIN_WIDTH_KPC          = 5.0
    SGR_BIN_END_KPC         = 80.0
    SGR_MIN_STARS_PER_BIN   = 5

    P_MEM_PLOT_THRESHOLD = 0.5

    # ── Epoch propagation ───────────────────────────────────────────────────
    EPOCH_DELTA = 16.0
    EPOCH_FROM  = 2000.0
    EPOCH_TO    = 2016.0

    # ── Quality filter thresholds ───────────────────────────────────────────
    RUWE_GOOD_THRESHOLD      = 1.4
    RUWE_PM_ERR_INFLATE      = 3.0
    RV_SN_MIN                = 3.0
    RV_ERR_INFLATE_BAD_SN    = 5.0
    GOFAL_THRESHOLD          = 3.0

    # ── EM parameters (universal) ───────────────────────────────────────────
    EM_MAX_ITER              = 60
    EM_CONVERGENCE_TOL       = 1e-5
    COV_REGULARIZE           = 1e-6
    ETA_INIT                 = 0.3
    BIC_TEST_COMPONENTS      = [1, 2, 3]

    # ── Parallax foreground cut ─────────────────────────────────────────────
    PLX_FOREGROUND_SIGMA     = 3.0
    PLX_FOREGROUND_THRESHOLD = 0.10

    # ── Intrinsic dispersion ────────────────────────────────────────────────
    SIGMA_INT_INIT           = 10.0
    SIGMA_INT_FLOOR          = 0.5
    FIELD_RV_SIGMA_INIT      = 80.0

    # ── Spatial prior ──────────────────────────────────────────────────────
    FIELD_INIT_RHALF_MULT    = 3.0
    RHALF_FALLBACK_DEG       = 0.5

    # ── CMD likelihood ─────────────────────────────────────────────────────
    CMD_KDE_BANDWIDTH        = 0.15
    CMD_FIELD_ANNULUS_MULT   = 5.0
    CMD_MIN_MEMBERS_FOR_KDE  = 15

    # ── Spectroscopic likelihood ───────────────────────────────────────────
    LOGG_GIANT_THRESHOLD     = 3.5
    LOGG_GIANT_SIGMA         = 0.5
    FEH_FIELD_MEAN           = -1.0
    FEH_FIELD_SIGMA          = 0.8
    FEH_MEMBER_SIGMA_DEFAULT = 0.3

    # ── Distance likelihood ────────────────────────────────────────────────
    DIST_FIELD_SIGMA_FRAC    = 0.5
    DIST_MEMBER_SIGMA_FLOOR  = 1.5

    # ── Control field ──────────────────────────────────────────────────────
    CONTROL_FIELD_INNER_MULT = 5.0
    CONTROL_FIELD_OUTER_MULT = 10.0

    # ── Summary plot parameters ────────────────────────────────────────────
    SUMMARY_ROWS_PER_PAGE    = 5
    SUMMARY_COLS_PER_PAGE    = 4
    SUMMARY_MIN_HIGH_PMEM    = 10
    SUMMARY_MIN_PER_TYPE     = 5
    SUMMARY_KS_ALPHA         = 0.05
    # V11: Grand summary minimum green matches
    GRAND_MIN_GREEN          = 10

    # ── Plot aesthetics ────────────────────────────────────────────────────
    CMAP_PMEM   = 'RdYlGn'
    PLOT_DPI    = 150
    SAVE_FORMAT = 'png'
    COL_MASTER  = '#DC143C'       # Red — all matched
    COL_MEMBER  = '#000080'       # Navy blue — parallax distance
    COL_HIGHMEM = '#006400'       # Dark green — high P_mem
    COL_GRAY_BG = '#C0C0C0'      # Gray — pre-match member background
    GRAY_BG_ALPHA = 1.0           # Fully opaque
    GRAY_BG_SIZE  = 6             # Small marker size for background


cfg = Config()


# ============================================================================
# UTILITIES
# ============================================================================

def normalize_name(name):
    import re
    if not isinstance(name, str):
        name = str(name)
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())

normalize_cluster_name = normalize_name


def _safe_float(row, col):
    if col is None:
        return None
    v = row.get(col)
    if v is None or pd.isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_col(df, colname, dtype=float):
    """Safely extract a column as numpy array, returning NaN array if missing."""
    if colname and colname in df.columns:
        return pd.to_numeric(df[colname], errors='coerce').values.astype(dtype)
    return np.full(len(df), np.nan, dtype=dtype)


def angular_separation_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """Great-circle angular separation (degrees) via Haversine formula."""
    ra1  = np.radians(np.asarray(ra1_deg,  dtype=float))
    dec1 = np.radians(np.asarray(dec1_deg, dtype=float))
    ra2  = np.radians(float(ra2_deg))
    dec2 = np.radians(float(dec2_deg))
    dra  = ra2 - ra1
    ddec = dec2 - dec1
    a    = (np.sin(ddec / 2.0) ** 2
            + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2)
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


# ============================================================================
# EPOCH PROPAGATION
# ============================================================================

def propagate_coords_to_gaia_epoch(ra, dec, pmra, pmdec, logger,
                                    epoch_delta=None):
    if epoch_delta is None:
        epoch_delta = cfg.EPOCH_DELTA
    ra    = np.asarray(ra,    dtype=float)
    dec   = np.asarray(dec,   dtype=float)
    pmra  = np.asarray(pmra,  dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    ra_out  = ra.copy()
    dec_out = dec.copy()
    pos_ok   = np.isfinite(ra)    & np.isfinite(dec)
    prop_ok  = pos_ok & np.isfinite(pmra) & np.isfinite(pmdec)
    fallback = pos_ok & ~prop_ok
    n_prop = int(np.sum(prop_ok))
    n_fb   = int(np.sum(fallback))
    if n_prop == 0:
        logger.warning("      [epoch] No stars with finite PM — kept at J2000")
        return ra_out, dec_out, 0, n_fb
    if HAS_ASTROPY:
        try:
            t_from = Time(f'J{cfg.EPOCH_FROM:.1f}')
            t_to   = Time(f'J{cfg.EPOCH_TO:.1f}')
            c = SkyCoord(
                ra=ra[prop_ok] * u.deg, dec=dec[prop_ok] * u.deg,
                pm_ra_cosdec=pmra[prop_ok] * u.mas / u.yr,
                pm_dec=pmdec[prop_ok] * u.mas / u.yr,
                frame='icrs', obstime=t_from)
            c_prop = c.apply_space_motion(new_obstime=t_to)
            ra_out[prop_ok]  = c_prop.ra.deg
            dec_out[prop_ok] = c_prop.dec.deg
            logger.info(f"      [epoch] astropy: {n_prop} propagated | "
                        f"{n_fb} fallback")
            return ra_out, dec_out, n_prop, n_fb
        except Exception as e:
            logger.warning(f"      [epoch] astropy failed ({e}); linear fallback")
    dec_rad = np.radians(dec[prop_ok])
    cos_dec = np.cos(dec_rad)
    cos_dec = np.where(np.abs(cos_dec) < 1e-10, 1e-10, cos_dec)
    ra_out[prop_ok]  += pmra[prop_ok]  * epoch_delta / (cos_dec * 3.6e6)
    dec_out[prop_ok] += pmdec[prop_ok] * epoch_delta / 3.6e6
    logger.info(f"      [epoch] linear: {n_prop} propagated | {n_fb} fallback")
    return ra_out, dec_out, n_prop, n_fb


def _get_query_coords(cdf, cols, logger, epoch_mode):
    ra  = pd.to_numeric(cdf[cols['ra']],  errors='coerce').values
    dec = pd.to_numeric(cdf[cols['dec']], errors='coerce').values
    if epoch_mode != '2000':
        return ra, dec
    pmra_col  = cols.get('pmra')
    pmdec_col = cols.get('pmdec')
    pmra  = (_safe_col(cdf, pmra_col)  if pmra_col  else np.full(len(ra), np.nan))
    pmdec = (_safe_col(cdf, pmdec_col) if pmdec_col else np.full(len(ra), np.nan))
    ra_prop, dec_prop, _, _ = propagate_coords_to_gaia_epoch(
        ra, dec, pmra, pmdec, logger)
    return ra_prop, dec_prop


# ============================================================================
# APJ STYLE
# ============================================================================

def set_paper_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family':  'serif',
        'font.serif':   ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size':    14,
        'axes.labelsize':   16, 'axes.titlesize':   18,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'axes.linewidth': 1.8, 'axes.edgecolor': 'black',
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'xtick.major.size': 7, 'ytick.major.size': 7,
        'xtick.minor.size': 4, 'ytick.minor.size': 4,
        'xtick.major.width': 1.4, 'ytick.major.width': 1.4,
        'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True,  'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'legend.fontsize':   11, 'legend.framealpha': 0.85,
        'legend.edgecolor': 'black', 'legend.fancybox': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.bbox': 'tight',
        'figure.dpi': 150, 'savefig.dpi': 300, 'text.usetex': False,
    })


# ============================================================================
# MASTER CATALOG
# ============================================================================

class MasterCatalog:
    def __init__(self, logger):
        self.logger    = logger
        self.df        = None
        self.tree      = None
        self.coords_3d = None
        self.max_chord = None
        self.nrows     = 0

    EXTRA_COLS = [
        'Gmag', 'BPmag', 'RPmag', 'BP-RP', 'BP-G', 'G-RP',
        'logg', 'feh', 'Teff',
        'AG', 'E(BP-RP)', 'A0',
        'b_logg_x', 'B_logg_xa', 'b_[Fe/H]_x', 'B_[Fe/H]_xa',
        'b_Teff_x', 'B_Teff_xa',
        'b_AG_x', 'B_AG_xa', 'b_E(BP-RP)_x', 'B_E(BP-RP)_xa',
        'RUWE', 'gofAL', 'chi2AL',
        'RVS/N', 'RVchi2', 'RVgof', 'RVNper',
        'pmRApmDEcor',
        'l', 'b',
        'PGal', 'PQSO', 'PSS',
        'GRVSmag',
        'catwise_w1', 'catwise_w2',
    ]

    def _find_col(self, key, available_cols):
        primary = cfg.MASTER_COLS.get(key)
        if primary and primary in available_cols:
            return primary
        for alt in cfg.ALT_MASTER_COLS.get(key, []):
            if alt in available_cols:
                return alt
        return None

    def load(self, filepath, checkpoint_dir=None):
        from astropy.io import fits as afits
        self.logger.info("=" * 70)
        self.logger.info("LOADING MASTER CATALOG (V11)")
        self.logger.info("=" * 70)

        tree_cp = (os.path.join(checkpoint_dir, 'master_tree_v10.npz')
                   if checkpoint_dir else None)
        data_cp = (os.path.join(checkpoint_dir, 'master_data_v10.parquet')
                   if checkpoint_dir else None)

        if (tree_cp and os.path.exists(tree_cp) and
                data_cp and os.path.exists(data_cp)):
            self.logger.info("Loading from checkpoint...")
            try:
                self.df        = pd.read_parquet(data_cp)
                self.coords_3d = np.load(tree_cp)['coords']
                self.tree      = cKDTree(self.coords_3d)
                self.nrows     = len(self.df)
                self._compute_max_chord()
                n_rv  = (self.df['best_rv'].notna().sum()
                         if 'best_rv' in self.df.columns else 0)
                n_cmd = (self.df['Gmag'].notna().sum()
                         if 'Gmag' in self.df.columns else 0)
                n_feh = (self.df['feh'].notna().sum()
                         if 'feh' in self.df.columns else 0)
                self.logger.info(f"Checkpoint: {self.nrows:,} rows | "
                                 f"{n_rv:,} RV | {n_cmd:,} CMD | {n_feh:,} [Fe/H]")
                return True
            except Exception as e:
                self.logger.warning(f"Checkpoint failed: {e}")

        files = self._resolve_files(filepath)
        if not files:
            self.logger.error(f"No FITS files found for: {filepath}")
            return False

        self.logger.info(f"Loading {len(files)} file(s)...")
        t0       = time.time()
        all_dfs  = []

        for fi, fpath in enumerate(files):
            self.logger.info(f"  [{fi+1}/{len(files)}] {os.path.basename(fpath)}")
            try:
                with afits.open(fpath, memmap=True) as hdul:
                    data_hdu = None
                    for hdu in hdul:
                        if hasattr(hdu, 'columns') and hdu.columns is not None:
                            data_hdu = hdu
                            break
                    if data_hdu is None:
                        self.logger.warning("    No data table — skipping")
                        continue

                    col_names = [c.name for c in data_hdu.columns]
                    nchunk    = data_hdu.data.shape[0]
                    chunk     = {}

                    ra_col  = cfg.MASTER_COLS['ra']
                    dec_col = cfg.MASTER_COLS['dec']
                    if ra_col not in col_names or dec_col not in col_names:
                        self.logger.warning(f"    Missing RA/Dec — skipping")
                        continue
                    chunk[ra_col]  = np.array(data_hdu.data[ra_col],  dtype=np.float64)
                    chunk[dec_col] = np.array(data_hdu.data[dec_col], dtype=np.float64)

                    for key in ['pmra', 'pmdec', 'pmra_err', 'pmdec_err']:
                        col = cfg.MASTER_COLS.get(key)
                        if col and col in col_names:
                            chunk[col] = np.array(data_hdu.data[col], dtype=np.float64)

                    dist_col = cfg.MASTER_COLS['dist']
                    chunk['best_dist'] = (np.array(data_hdu.data[dist_col], dtype=np.float64)
                                          if dist_col in col_names
                                          else np.full(nchunk, np.nan))
                    dist_err_col = cfg.MASTER_COLS.get('dist_err')
                    chunk['best_dist_err'] = (
                        np.array(data_hdu.data[dist_err_col], dtype=np.float64)
                        if dist_err_col and dist_err_col in col_names
                        else np.full(nchunk, np.nan))

                    rv_col     = self._find_col('rv',     col_names)
                    rv_err_col = self._find_col('rv_err', col_names)
                    chunk['best_rv'] = (np.array(data_hdu.data[rv_col], dtype=np.float64)
                                        if rv_col else np.full(nchunk, np.nan))
                    chunk['best_rv_err'] = (np.array(data_hdu.data[rv_err_col], dtype=np.float64)
                                            if rv_err_col else np.full(nchunk, np.nan))
                    if fi == 0 and rv_col:
                        self.logger.info(f"    RV from: {rv_col}")

                    plx_col     = cfg.MASTER_COLS.get('parallax',     'parallax_final')
                    plx_err_col = cfg.MASTER_COLS.get('parallax_err', 'parallax_err_final')
                    if plx_col in col_names:
                        chunk['plx_from_params'] = np.array(
                            data_hdu.data[plx_col], dtype=np.float64)
                    else:
                        pe_col = cfg.MASTER_COLS.get('params_est')
                        if pe_col and pe_col in col_names:
                            try:
                                chunk['plx_from_params'] = (
                                    data_hdu.data[pe_col][:, 4].copy().astype(np.float64))
                            except Exception:
                                chunk['plx_from_params'] = np.full(nchunk, np.nan)
                        else:
                            chunk['plx_from_params'] = np.full(nchunk, np.nan)
                    if plx_err_col in col_names:
                        chunk['plx_err_from_params'] = np.array(
                            data_hdu.data[plx_err_col], dtype=np.float64)
                    else:
                        pe_err_col = cfg.MASTER_COLS.get('params_err')
                        if pe_err_col and pe_err_col in col_names:
                            try:
                                chunk['plx_err_from_params'] = (
                                    data_hdu.data[pe_err_col][:, 4].copy().astype(np.float64))
                            except Exception:
                                chunk['plx_err_from_params'] = np.full(nchunk, np.nan)
                        else:
                            chunk['plx_err_from_params'] = np.full(nchunk, np.nan)

                    for ecol in self.EXTRA_COLS:
                        if ecol in col_names:
                            try:
                                chunk[ecol] = np.array(
                                    data_hdu.data[ecol], dtype=np.float64)
                            except (ValueError, TypeError):
                                chunk[ecol] = np.full(nchunk, np.nan)

                    cdf = pd.DataFrame(chunk)
                    cdf = cdf.dropna(subset=[ra_col, dec_col])
                    all_dfs.append(cdf)
                    self.logger.info(f"    -> {len(cdf):,} rows")

            except Exception as e:
                self.logger.warning(f"    [ERROR] {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_dfs:
            self.logger.error("No data loaded!")
            return False

        self.df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs
        gcmod.collect()
        self.nrows = len(self.df)

        n_rv   = np.sum(np.isfinite(self.df['best_rv'].values))
        n_dist = np.sum(np.isfinite(self.df['best_dist'].values))
        n_cmd  = np.sum(np.isfinite(self.df.get('Gmag', pd.Series(dtype=float)).values))
        n_feh  = np.sum(np.isfinite(self.df.get('feh', pd.Series(dtype=float)).values))
        n_logg = np.sum(np.isfinite(self.df.get('logg', pd.Series(dtype=float)).values))
        n_ruwe = np.sum(np.isfinite(self.df.get('RUWE', pd.Series(dtype=float)).values))
        self.logger.info(f"  Combined: {self.nrows:,} rows")
        self.logger.info(f"    RV: {n_rv:,} | dist: {n_dist:,} | "
                         f"CMD: {n_cmd:,} | [Fe/H]: {n_feh:,} | "
                         f"logg: {n_logg:,} | RUWE: {n_ruwe:,}")

        self._build_kdtree()

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.df.to_parquet(data_cp)
            np.savez_compressed(tree_cp, coords=self.coords_3d)
            self.logger.info("  Checkpoint saved")

        self.logger.info(f"Master loaded in {time.time()-t0:.1f}s")
        gcmod.collect()
        return True

    def _resolve_files(self, filepath):
        if '*' in filepath or '?' in filepath:
            return sorted(glob.glob(filepath))
        if os.path.isdir(filepath):
            pattern = os.path.join(filepath, 'Entire_catalogue_chunk*.fits')
            found   = sorted(glob.glob(pattern))
            if found:
                return found
            return sorted(glob.glob(os.path.join(filepath, '*.fits')))
        if os.path.exists(filepath):
            return [filepath]
        return []

    def _build_kdtree(self):
        ra  = np.radians(self.df[cfg.MASTER_COLS['ra']].values)
        dec = np.radians(self.df[cfg.MASTER_COLS['dec']].values)
        self.coords_3d = np.column_stack([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)])
        self.tree = cKDTree(self.coords_3d)
        self._compute_max_chord()

    def _compute_max_chord(self):
        self.max_chord = 2 * np.sin(
            np.radians(cfg.CROSSMATCH_RADIUS_ARCSEC / 3600.0) / 2)

    def query(self, ra, dec):
        rr = np.radians(ra)
        dr = np.radians(dec)
        c  = np.column_stack([
            np.cos(dr) * np.cos(rr),
            np.cos(dr) * np.sin(rr),
            np.sin(dr)])
        d, i = self.tree.query(c, k=1, distance_upper_bound=self.max_chord)
        v = np.isfinite(d)
        return i[v], np.where(v)[0], np.degrees(2 * np.arcsin(d[v] / 2)) * 3600

    def get_matched_data(self, master_idx):
        return self.df.iloc[master_idx].reset_index(drop=True)


# ============================================================================
# QUALITY FILTERS
# ============================================================================

def apply_quality_flags(mdf, logger):
    """Apply quality-based error inflation to matched DataFrame."""
    n = len(mdf)
    pmra_err  = mdf.get(f"{cfg.MASTER_COLS['pmra_err']}_master",
                         pd.Series(np.full(n, 1.0))).values.astype(float)
    pmdec_err = mdf.get(f"{cfg.MASTER_COLS['pmdec_err']}_master",
                         pd.Series(np.full(n, 1.0))).values.astype(float)
    ruwe      = mdf.get('RUWE_master',
                         pd.Series(np.full(n, np.nan))).values.astype(float)
    bad_ruwe  = np.isfinite(ruwe) & (ruwe > cfg.RUWE_GOOD_THRESHOLD)
    n_bad_ruwe = int(np.sum(bad_ruwe))
    inflate_pm = np.where(bad_ruwe, cfg.RUWE_PM_ERR_INFLATE, 1.0)
    mdf['pmra_err_adj']  = pmra_err  * inflate_pm
    mdf['pmdec_err_adj'] = pmdec_err * inflate_pm
    rv_err    = mdf.get('best_rv_err_master',
                         pd.Series(np.full(n, 5.0))).values.astype(float)
    rv_err    = np.where(np.isfinite(rv_err) & (rv_err > 0), rv_err, 5.0)
    rv_sn     = mdf.get('RVS/N_master',
                         pd.Series(np.full(n, np.nan))).values.astype(float)
    bad_rv_sn = np.isfinite(rv_sn) & (rv_sn < cfg.RV_SN_MIN)
    n_bad_rv  = int(np.sum(bad_rv_sn))
    inflate_rv = np.where(bad_rv_sn, cfg.RV_ERR_INFLATE_BAD_SN, 1.0)
    mdf['rv_err_adj'] = rv_err * inflate_rv
    if n_bad_ruwe > 0 or n_bad_rv > 0:
        logger.info(f"      [quality] RUWE>{cfg.RUWE_GOOD_THRESHOLD}: "
                    f"{n_bad_ruwe} (PM err x{cfg.RUWE_PM_ERR_INFLATE}) | "
                    f"RV S/N<{cfg.RV_SN_MIN}: {n_bad_rv} (RV err inflated)")
    return mdf


# ============================================================================
# SPATIAL PROFILES
# ============================================================================

def elliptical_plummer_pdf(ra_deg, dec_deg, ra0, dec0,
                            rhalf_deg, ellipticity=0.0,
                            position_angle_deg=0.0, r_max_deg=None):
    """Normalised elliptical Plummer surface-density profile."""
    ra   = np.asarray(ra_deg,  dtype=float)
    dec  = np.asarray(dec_deg, dtype=float)
    ra0_r  = np.radians(float(ra0))
    dec0_r = np.radians(float(dec0))
    ra_r   = np.radians(ra)
    dec_r  = np.radians(dec)
    dra    = ra_r - ra0_r
    cos_d  = np.cos(dec_r);  sin_d  = np.sin(dec_r)
    cos_d0 = np.cos(dec0_r); sin_d0 = np.sin(dec0_r)
    denom  = sin_d0 * sin_d + cos_d0 * cos_d * np.cos(dra)
    denom  = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    xi     = np.degrees(cos_d * np.sin(dra) / denom)
    eta    = np.degrees((cos_d0 * sin_d - sin_d0 * cos_d * np.cos(dra)) / denom)
    eps = float(np.clip(ellipticity, 0.0, 0.95))
    q   = 1.0 - eps
    theta = np.radians(float(position_angle_deg))
    ct, st = np.cos(theta), np.sin(theta)
    u = xi * ct + eta * st
    v = -xi * st + eta * ct
    r_ell = np.sqrt((u / q) ** 2 + v ** 2)
    a = float(rhalf_deg) / np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)
    if r_max_deg is None or not np.isfinite(r_max_deg) or r_max_deg <= 0:
        r_max_deg = max(np.nanmax(r_ell), 5.0 * rhalf_deg)
    R_max = float(r_max_deg)
    norm = (a ** 2 + R_max ** 2) / (np.pi * a ** 2 * R_max ** 2 * q)
    pdf  = norm / (1.0 + (r_ell / a) ** 2) ** 2
    pdf[r_ell > R_max] = 0.0
    return pdf, r_ell


def uniform_field_pdf(r_max_deg, q=1.0):
    return 1.0 / (np.pi * r_max_deg ** 2 * q)


# ============================================================================
# CMD LIKELIHOOD
# ============================================================================

def build_cmd_template(gmag_members, bprp_members, dist_mod,
                        ag_members=None, ebprp_members=None):
    """Build empirical CMD template as a 2D KDE from known member photometry."""
    g     = np.asarray(gmag_members, dtype=float)
    color = np.asarray(bprp_members, dtype=float)
    if ag_members is not None:
        ag = np.asarray(ag_members, dtype=float)
        ok_ag = np.isfinite(ag)
        g = np.where(ok_ag, g - ag, g)
    if ebprp_members is not None:
        eb = np.asarray(ebprp_members, dtype=float)
        ok_eb = np.isfinite(eb)
        color = np.where(ok_eb, color - eb, color)
    if dist_mod is not None and np.isfinite(dist_mod):
        M_G = g - float(dist_mod)
    else:
        M_G = g
    ok = np.isfinite(M_G) & np.isfinite(color)
    if np.sum(ok) < cfg.CMD_MIN_MEMBERS_FOR_KDE:
        return None
    data = np.vstack([M_G[ok], color[ok]])
    try:
        return gaussian_kde(data, bw_method=cfg.CMD_KDE_BANDWIDTH)
    except Exception:
        return None


def cmd_likelihood(gmag, bprp, dist_mod, ag=None, ebprp=None,
                    member_kde=None, field_kde=None):
    """Evaluate CMD likelihood for member and field models."""
    n     = len(gmag)
    L_mem = np.ones(n)
    L_fld = np.ones(n)
    g     = np.asarray(gmag, dtype=float)
    color = np.asarray(bprp, dtype=float)
    if ag is not None:
        ag_arr = np.asarray(ag, dtype=float)
        ok_ag  = np.isfinite(ag_arr)
        g      = np.where(ok_ag, g - ag_arr, g)
    if ebprp is not None:
        eb_arr = np.asarray(ebprp, dtype=float)
        ok_eb  = np.isfinite(eb_arr)
        color  = np.where(ok_eb, color - eb_arr, color)
    if dist_mod is not None and np.isfinite(dist_mod):
        M_G = g - float(dist_mod)
    else:
        M_G = g
    ok = np.isfinite(M_G) & np.isfinite(color)
    if not np.any(ok):
        return L_mem, L_fld
    points = np.vstack([M_G[ok], color[ok]])
    if member_kde is not None:
        try:
            L_mem[ok] = np.clip(member_kde(points), 1e-300, None)
        except Exception:
            pass
    if field_kde is not None:
        try:
            L_fld[ok] = np.clip(field_kde(points), 1e-300, None)
        except Exception:
            pass
    else:
        L_fld[ok] = 1e-2
    return L_mem, L_fld


# ============================================================================
# SPECTROSCOPIC LIKELIHOODS
# ============================================================================

def logg_likelihood(logg_arr, obj_type, logg_threshold=None):
    n = len(logg_arr)
    L_mem = np.ones(n);  L_fld = np.ones(n)
    if obj_type in ['OC', 'SGR', 'STREAM']:
        return L_mem, L_fld
    if logg_threshold is None:
        logg_threshold = cfg.LOGG_GIANT_THRESHOLD
    lg = np.asarray(logg_arr, dtype=float)
    ok = np.isfinite(lg)
    if not np.any(ok):
        return L_mem, L_fld
    sigma_g = cfg.LOGG_GIANT_SIGMA
    excess  = np.clip(lg[ok] - logg_threshold, 0, None)
    L_mem[ok] = np.exp(-0.5 * (excess / sigma_g) ** 2)
    L_fld[ok] = 0.3 + 0.7 * np.exp(-0.5 * ((lg[ok] - 4.2) / 0.8) ** 2)
    return L_mem, L_fld


def feh_likelihood(feh_arr, feh_sys, feh_sys_sigma=None, feh_member_sigma=None):
    n = len(feh_arr)
    L_mem = np.ones(n);  L_fld = np.ones(n)
    if feh_sys is None or not np.isfinite(feh_sys):
        return L_mem, L_fld
    feh = np.asarray(feh_arr, dtype=float)
    ok  = np.isfinite(feh)
    if not np.any(ok):
        return L_mem, L_fld
    if feh_member_sigma is None:
        feh_member_sigma = cfg.FEH_MEMBER_SIGMA_DEFAULT
    if feh_sys_sigma is not None and np.isfinite(feh_sys_sigma) and feh_sys_sigma > 0:
        sig_mem = max(feh_sys_sigma, 0.05)
    else:
        sig_mem = feh_member_sigma
    L_mem[ok] = norm.pdf(feh[ok], loc=feh_sys, scale=sig_mem)
    L_fld[ok] = norm.pdf(feh[ok], loc=cfg.FEH_FIELD_MEAN,
                         scale=cfg.FEH_FIELD_SIGMA)
    return L_mem, L_fld


def distance_likelihood(dist_arr, dist_err_arr, ref_dist, ref_dist_err):
    n = len(dist_arr)
    L_mem = np.ones(n);  L_fld = np.ones(n)
    if ref_dist is None or not np.isfinite(ref_dist) or ref_dist <= 0:
        return L_mem, L_fld
    d   = np.asarray(dist_arr, dtype=float)
    de  = np.asarray(dist_err_arr, dtype=float)
    # Default per-star error: 30% of ref_dist (Gaia distances scatter this much)
    de  = np.where(np.isfinite(de) & (de > 0), de, ref_dist * 0.3)
    ok  = np.isfinite(d) & (d > 0)
    if not np.any(ok):
        return L_mem, L_fld
    # Reference distance uncertainty: use provided, or fallback to
    # max(20% of ref_dist, FLOOR). This prevents overly tight priors
    # when Harris errors are absent or very small.
    dist_floor = max(ref_dist * 0.20, cfg.DIST_MEMBER_SIGMA_FLOOR)
    if (ref_dist_err is not None and np.isfinite(ref_dist_err)
            and ref_dist_err > 0):
        rde = max(float(ref_dist_err), dist_floor)
    else:
        rde = dist_floor
    sig_mem = np.sqrt(rde ** 2 + de[ok] ** 2)
    sig_mem = np.clip(sig_mem, cfg.DIST_MEMBER_SIGMA_FLOOR, None)
    L_mem[ok] = norm.pdf(d[ok], loc=ref_dist, scale=sig_mem)
    # Field: broad distribution centred on solar neighbourhood
    field_center = 8.0
    field_sigma  = max(ref_dist * cfg.DIST_FIELD_SIGMA_FRAC, 15.0)
    L_fld[ok] = norm.pdf(d[ok], loc=field_center, scale=field_sigma)
    L_fld[ok] = np.clip(L_fld[ok], 1e-10, None)
    return L_mem, L_fld


# ============================================================================
# PM LIKELIHOOD (VECTORISED)
# ============================================================================

def pm_likelihood_vectorised(pmra, pmdec, mu, Sigma,
                              pmra_err, pmdec_err, pm_corr=None):
    """Vectorised bivariate Gaussian PM likelihood with per-star errors."""
    n = len(pmra)
    L = np.ones(n)
    pmra_v   = np.asarray(pmra,     dtype=float)
    pmdec_v  = np.asarray(pmdec,    dtype=float)
    pe       = np.asarray(pmra_err, dtype=float)
    pde      = np.asarray(pmdec_err, dtype=float)
    mu_v     = np.asarray(mu, dtype=float)
    S        = np.asarray(Sigma, dtype=float)
    ok = np.isfinite(pmra_v) & np.isfinite(pmdec_v)
    if not np.any(ok):
        return L
    idx = np.where(ok)[0]
    dx  = pmra_v[idx]  - mu_v[0]
    dy  = pmdec_v[idx] - mu_v[1]
    pe_ok  = np.where(np.isfinite(pe[idx]),  pe[idx],  0.0)
    pde_ok = np.where(np.isfinite(pde[idx]), pde[idx], 0.0)
    if pm_corr is not None:
        corr = np.asarray(pm_corr, dtype=float)
        rho  = np.where(np.isfinite(corr[idx]), corr[idx], 0.0)
    else:
        rho = np.zeros(len(idx))
    s11 = S[0, 0] + pe_ok ** 2 + cfg.COV_REGULARIZE
    s22 = S[1, 1] + pde_ok ** 2 + cfg.COV_REGULARIZE
    s12 = S[0, 1] + rho * pe_ok * pde_ok
    det = s11 * s22 - s12 ** 2
    det = np.clip(det, 1e-20, None)
    inv11 = s22 / det
    inv22 = s11 / det
    inv12 = -s12 / det
    maha  = dx ** 2 * inv11 + 2.0 * dx * dy * inv12 + dy ** 2 * inv22
    L[idx] = np.exp(-0.5 * maha) / (2.0 * np.pi * np.sqrt(det))
    L[idx] = np.clip(L[idx], 1e-300, None)
    return L


# ============================================================================
# RV LIKELIHOOD WITH MLE sigma_int
# ============================================================================

def rv_likelihood(rv, rv_err, mu, sigma):
    rv    = np.asarray(rv,     dtype=float)
    rv_e  = np.asarray(rv_err, dtype=float)
    n     = len(rv)
    L     = np.ones(n)
    ok    = np.isfinite(rv) & np.isfinite(rv_e)
    if not np.any(ok):
        return L
    sig_tot = np.sqrt(float(sigma) ** 2 + rv_e[ok] ** 2)
    L[ok]   = norm.pdf(rv[ok], loc=float(mu), scale=sig_tot)
    L[ok]   = np.clip(L[ok], 1e-300, None)
    return L


def mle_sigma_int(rv, rv_err, weights, v_sys, sigma_floor=None):
    """MLE estimate of intrinsic RV dispersion via Brent's method."""
    if sigma_floor is None:
        sigma_floor = cfg.SIGMA_INT_FLOOR
    rv_v  = np.asarray(rv, dtype=float)
    rv_e  = np.asarray(rv_err, dtype=float)
    w     = np.asarray(weights, dtype=float)
    ok    = np.isfinite(rv_v) & np.isfinite(rv_e) & (w > 1e-6)
    if np.sum(ok) < 3:
        return cfg.SIGMA_INT_INIT
    rv_ok = rv_v[ok]; re_ok = rv_e[ok]; w_ok = w[ok]
    def neg_loglik(sigma_int):
        s2 = sigma_int ** 2 + re_ok ** 2
        ll = -0.5 * w_ok * (np.log(2 * np.pi * s2) + (rv_ok - v_sys) ** 2 / s2)
        return -np.sum(ll)
    try:
        result = minimize_scalar(neg_loglik, bounds=(sigma_floor, 200.0),
                                  method='bounded')
        return max(result.x, sigma_floor)
    except Exception:
        W = np.sum(w_ok)
        var_tot = np.sum(w_ok * (rv_ok - v_sys) ** 2) / W
        var_meas = np.sum(w_ok * re_ok ** 2) / W
        return max(np.sqrt(max(var_tot - var_meas, 0)), sigma_floor)


# ============================================================================
# WEIGHTED 2D COVARIANCE
# ============================================================================

def _weighted_cov_2d(x, y, weights):
    W = np.sum(weights)
    if W <= 0 or np.sum(weights > 0) < 3:
        return (np.nanmean(x), np.nanmean(y),
                np.eye(2) * (cfg.COV_REGULARIZE + 0.1))
    mx = np.sum(weights * x) / W
    my = np.sum(weights * y) / W
    dx = x - mx;  dy = y - my
    Sxx = np.sum(weights * dx * dx) / W
    Syy = np.sum(weights * dy * dy) / W
    Sxy = np.sum(weights * dx * dy) / W
    C   = np.array([[Sxx, Sxy], [Sxy, Syy]])
    C  += np.eye(2) * cfg.COV_REGULARIZE
    return mx, my, C


# ============================================================================
# UNIVERSAL BAYESIAN EM (all object types)
# ============================================================================

def algorithm_bayesian_em(
        pmra, pmdec, cp, cd,
        pmra_err=None, pmdec_err=None, pm_corr=None,
        ra_deg=None, dec_deg=None,
        ra_center=None, dec_center=None,
        rhalf_deg=None, ellipticity=0.0, position_angle=0.0,
        rv_obs=None, rv_err_obs=None,
        rv_sys_prior=None, sigma_int_prior=None,
        dist_obs=None, dist_err_obs=None,
        ref_dist=None, ref_dist_err=None,
        gmag_obs=None, bprp_obs=None, ag_obs=None, ebprp_obs=None,
        dist_modulus=None,
        member_cmd_kde=None, field_cmd_kde=None,
        logg_obs=None, feh_obs=None,
        feh_sys=None, feh_sys_sigma=None,
        obj_type='GC', ms=10, logger=None):
    """Universal 7-term Bayesian EM membership for ALL object types."""
    n     = len(pmra)
    pmra  = np.asarray(pmra,  dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    ok_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    if np.sum(ok_pm) < ms:
        return np.full(n, 0.5), {
            'status': 'insufficient_data', 'algorithm': 'BayesianEM-V11'}

    def _ensure(arr):
        return np.asarray(arr, dtype=float) if arr is not None else np.full(n, np.nan)

    ra_deg     = _ensure(ra_deg);     dec_deg    = _ensure(dec_deg)
    rv_obs     = _ensure(rv_obs);     rv_err_obs = _ensure(rv_err_obs)
    dist_obs   = _ensure(dist_obs);   dist_err_obs = _ensure(dist_err_obs)
    gmag_obs   = _ensure(gmag_obs);   bprp_obs   = _ensure(bprp_obs)
    ag_obs     = _ensure(ag_obs);     ebprp_obs  = _ensure(ebprp_obs)
    logg_obs   = _ensure(logg_obs);   feh_obs    = _ensure(feh_obs)
    pmra_err   = _ensure(pmra_err);   pmdec_err  = _ensure(pmdec_err)
    if pm_corr is not None:
        pm_corr = np.asarray(pm_corr, dtype=float)
    if sigma_int_prior is None:
        sigma_int_prior = cfg.SIGMA_INT_INIT

    # Spatial setup
    use_spatial = (ra_center is not None and dec_center is not None
                   and np.isfinite(ra_center) and np.isfinite(dec_center))
    if rhalf_deg is None or not np.isfinite(rhalf_deg) or rhalf_deg <= 0:
        rhalf_deg = cfg.RHALF_FALLBACK_DEG
    r_ell = np.full(n, np.nan)
    r_max = 5.0 * rhalf_deg
    if use_spatial:
        _, r_ell = elliptical_plummer_pdf(
            ra_deg, dec_deg, ra_center, dec_center,
            rhalf_deg, ellipticity, position_angle)
        r_max_cand = np.nanmax(r_ell)
        if np.isfinite(r_max_cand) and r_max_cand > 0:
            r_max = r_max_cand

    ok_rv = np.isfinite(rv_obs) & np.isfinite(rv_err_obs)
    n_rv  = int(np.sum(ok_rv))
    n_dist = int(np.sum(np.isfinite(dist_obs) & (dist_obs > 0)))
    n_cmd  = int(np.sum(np.isfinite(gmag_obs) & np.isfinite(bprp_obs)))
    n_feh  = int(np.sum(np.isfinite(feh_obs)))
    n_logg = int(np.sum(np.isfinite(logg_obs)))

    # INITIALISATION
    mu_mem  = np.array([cp, cd])
    if use_spatial and np.any(np.isfinite(r_ell)):
        inner = ok_pm & (r_ell < 2.0 * rhalf_deg)
        if np.sum(inner) >= 5:
            _, _, Sig_mem = _weighted_cov_2d(pmra[inner], pmdec[inner],
                                              np.ones(np.sum(inner)))
        else:
            Sig_mem = np.eye(2) * 0.5
    else:
        Sig_mem = np.eye(2) * 0.5

    if use_spatial and np.any(np.isfinite(r_ell)):
        outer = ok_pm & (r_ell > cfg.FIELD_INIT_RHALF_MULT * rhalf_deg)
        if np.sum(outer) < ms:
            outer = ok_pm
    else:
        outer = ok_pm
    if np.sum(outer) >= 3:
        mx_f, my_f, Sig_field = _weighted_cov_2d(
            pmra[outer], pmdec[outer], np.ones(np.sum(outer)))
        mu_field = np.array([mx_f, my_f])
    else:
        mu_field  = np.array([0.0, 0.0])
        Sig_field = np.eye(2) * 4.0

    rv_sys    = rv_sys_prior if (rv_sys_prior is not None
                                  and np.isfinite(rv_sys_prior)) else (
                    float(np.median(rv_obs[ok_rv])) if n_rv >= 5 else 0.0)
    sigma_int = float(sigma_int_prior)
    if n_rv >= 5:
        rv_field_mu  = float(np.median(rv_obs[ok_rv]))
        rv_field_sig = max(
            float(median_abs_deviation(rv_obs[ok_rv], nan_policy='omit') * 1.4826),
            cfg.FIELD_RV_SIGMA_INIT)
    else:
        rv_field_mu  = 0.0
        rv_field_sig = cfg.FIELD_RV_SIGMA_INIT
    eta = cfg.ETA_INIT

    # Precompute static likelihoods
    L_dist_mem, L_dist_fld = distance_likelihood(
        dist_obs, dist_err_obs, ref_dist, ref_dist_err)
    if member_cmd_kde is not None:
        L_cmd_mem, L_cmd_fld = cmd_likelihood(
            gmag_obs, bprp_obs, dist_modulus, ag_obs, ebprp_obs,
            member_cmd_kde, field_cmd_kde)
    else:
        L_cmd_mem = np.ones(n);  L_cmd_fld = np.ones(n)
    L_feh_mem, L_feh_fld = feh_likelihood(feh_obs, feh_sys, feh_sys_sigma)
    L_logg_mem, L_logg_fld = logg_likelihood(logg_obs, obj_type)
    static_mem = L_dist_mem * L_cmd_mem * L_feh_mem * L_logg_mem
    static_fld = L_dist_fld * L_cmd_fld * L_feh_fld * L_logg_fld

    # EM LOOP
    P         = np.full(n, eta)
    logL_prev = -np.inf
    q_ell     = max(1.0 - ellipticity, 0.05)

    for em_iter in range(cfg.EM_MAX_ITER):
        # E-STEP
        if use_spatial and np.any(np.isfinite(r_ell)):
            pdf_mem, _ = elliptical_plummer_pdf(
                ra_deg, dec_deg, ra_center, dec_center,
                rhalf_deg, ellipticity, position_angle, r_max)
            pdf_field  = np.where(np.isfinite(r_ell),
                                  uniform_field_pdf(r_max, q_ell), 1.0)
            pdf_mem   = np.clip(pdf_mem,   1e-300, None)
            pdf_field = np.clip(pdf_field, 1e-300, None)
        else:
            pdf_mem   = np.ones(n);  pdf_field = np.ones(n)
        L_pm_mem = pm_likelihood_vectorised(
            pmra, pmdec, mu_mem, Sig_mem, pmra_err, pmdec_err, pm_corr)
        L_pm_fld = pm_likelihood_vectorised(
            pmra, pmdec, mu_field, Sig_field, pmra_err, pmdec_err, pm_corr)
        L_rv_mem = rv_likelihood(rv_obs, rv_err_obs, rv_sys, sigma_int)
        L_rv_fld = rv_likelihood(rv_obs, rv_err_obs, rv_field_mu, rv_field_sig)
        num = eta * pdf_mem * L_pm_mem * L_rv_mem * static_mem
        den = num + (1.0 - eta) * pdf_field * L_pm_fld * L_rv_fld * static_fld
        zero_den = (den <= 0) | ~np.isfinite(den)
        den[zero_den] = 1e-300
        P_new = np.where(ok_pm, num / den, np.nan)
        P_new = np.clip(P_new, 1e-6, 1.0 - 1e-6)
        logL = np.nansum(np.log(np.clip(den[ok_pm], 1e-300, None)))

        # M-STEP
        P_valid = np.where(np.isfinite(P_new), P_new, 0.0)
        eta = float(np.clip(np.mean(P_valid[ok_pm]), 0.01, 0.99))
        w_mem = P_valid.copy();  w_mem[~ok_pm] = 0.0
        if np.sum(w_mem) >= 3:
            mx_m, my_m, Sig_mem_new = _weighted_cov_2d(pmra, pmdec, w_mem)
            mu_mem  = np.array([mx_m, my_m])
            Sig_mem = Sig_mem_new
        dist_from_ref = np.linalg.norm(mu_mem - np.array([cp, cd]))
        if dist_from_ref > 3.0:
            mu_mem = 0.5 * mu_mem + 0.5 * np.array([cp, cd])
        w_fld = (1.0 - P_valid);  w_fld[~ok_pm] = 0.0
        if np.sum(w_fld) >= 3:
            mx_f, my_f, Sig_fld_new = _weighted_cov_2d(pmra, pmdec, w_fld)
            mu_field  = np.array([mx_f, my_f])
            Sig_field = Sig_fld_new
        if n_rv >= 5:
            w_rv = P_valid[ok_rv];  W_rv = np.sum(w_rv)
            rv_ok_v = rv_obs[ok_rv];  re_ok_v = rv_err_obs[ok_rv]
            if W_rv > 0:
                sig_total = np.sqrt(sigma_int ** 2 + re_ok_v ** 2)
                iv = 1.0 / (sig_total ** 2 + 1e-10)
                rv_sys = float(np.sum(w_rv * iv * rv_ok_v) / (np.sum(w_rv * iv) + 1e-10))
                sigma_int = mle_sigma_int(rv_ok_v, re_ok_v, w_rv, rv_sys)
            w_rv_f = (1.0 - P_valid)[ok_rv];  W_rv_f = np.sum(w_rv_f)
            if W_rv_f > 0:
                rv_field_mu  = float(np.sum(w_rv_f * rv_ok_v) / W_rv_f)
                var_f = np.sum(w_rv_f * (rv_ok_v - rv_field_mu) ** 2) / W_rv_f
                rv_field_sig = max(float(np.sqrt(var_f + 1e-4)),
                                   cfg.FIELD_RV_SIGMA_INIT / 2.0)
        dlogL = abs(logL - logL_prev) / (abs(logL_prev) + 1e-10)
        if em_iter > 2 and dlogL < cfg.EM_CONVERGENCE_TOL:
            break
        logL_prev = logL;  P = P_new

    # Final E-step
    L_pm_mem = pm_likelihood_vectorised(
        pmra, pmdec, mu_mem, Sig_mem, pmra_err, pmdec_err, pm_corr)
    L_pm_fld = pm_likelihood_vectorised(
        pmra, pmdec, mu_field, Sig_field, pmra_err, pmdec_err, pm_corr)
    L_rv_mem = rv_likelihood(rv_obs, rv_err_obs, rv_sys, sigma_int)
    L_rv_fld = rv_likelihood(rv_obs, rv_err_obs, rv_field_mu, rv_field_sig)
    if use_spatial and np.any(np.isfinite(r_ell)):
        pdf_mem, _ = elliptical_plummer_pdf(
            ra_deg, dec_deg, ra_center, dec_center,
            rhalf_deg, ellipticity, position_angle, r_max)
        pdf_field = np.where(np.isfinite(r_ell),
                             uniform_field_pdf(r_max, q_ell), 1.0)
        pdf_mem   = np.clip(pdf_mem,   1e-300, None)
        pdf_field = np.clip(pdf_field, 1e-300, None)
    else:
        pdf_mem   = np.ones(n);  pdf_field = np.ones(n)
    num   = eta * pdf_mem * L_pm_mem * L_rv_mem * static_mem
    den   = num + (1.0 - eta) * pdf_field * L_pm_fld * L_rv_fld * static_fld
    den   = np.where((den <= 0) | ~np.isfinite(den), 1e-300, den)
    P_fin = np.where(ok_pm, np.clip(num / den, 0.0, 1.0), np.nan)

    n_valid = int(np.sum(ok_pm))
    n_params_2comp = 2 * 5 + 1
    bic_2 = -2.0 * logL_prev + n_params_2comp * np.log(max(n_valid, 1))

    info = {
        'status':          'success',
        'algorithm':       'BayesianEM-V11',
        'obj_type':        obj_type,
        'n_em_iterations': em_iter + 1,
        'eta':             float(eta),
        'mu_cluster':      mu_mem.tolist(),
        'Sigma_cluster':   Sig_mem.tolist(),
        'mu_field':        mu_field.tolist(),
        'Sigma_field':     Sig_field.tolist(),
        'pm_dispersion':   float(np.sqrt(np.trace(Sig_mem))),
        'center_pmra':     cp,
        'center_pmdec':    cd,
        'v_sys_recovered': float(rv_sys),
        'sigma_int_km_s':  float(sigma_int),
        'rv_field_mu':     float(rv_field_mu),
        'rv_field_sigma':  float(rv_field_sig),
        'rhalf_deg_used':  float(rhalf_deg),
        'ellipticity':     float(ellipticity),
        'r_max_deg':       float(r_max),
        'n_rv_used':       n_rv,
        'n_dist_used':     n_dist,
        'n_cmd_used':      n_cmd,
        'n_feh_used':      n_feh,
        'n_logg_used':     n_logg,
        'use_spatial':     bool(use_spatial),
        'logL_final':      float(logL_prev),
        'BIC':             float(bic_2),
        'ref_dist':        ref_dist,
        'ref_dist_err':    ref_dist_err,
        'feh_sys':         feh_sys,
    }
    return P_fin, info


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def compute_diagnostics(matched_df, P_mem, ref_dist, ref_rv,
                         obj_type, algo_info, logger=None):
    diag = {'quality_flag': 'GOOD'}
    P = np.asarray(P_mem, dtype=float)
    hi = np.isfinite(P) & (P > cfg.P_MEM_PLOT_THRESHOLD)
    n_hi = int(np.sum(hi))
    diag['n_high_pmem'] = n_hi
    if n_hi < cfg.SUMMARY_MIN_HIGH_PMEM:
        diag['quality_flag'] = 'LOW_N'

    if ref_dist is not None and np.isfinite(ref_dist) and ref_dist > 0:
        hi_dist = matched_df.loc[hi, 'best_dist'].values if 'best_dist' in matched_df.columns else np.array([])
        hi_dist = hi_dist[np.isfinite(hi_dist) & (hi_dist > 0)]
        if len(hi_dist) >= 5:
            ref_err = algo_info.get('ref_dist_err', ref_dist * 0.15)
            if ref_err is None or not np.isfinite(ref_err) or ref_err <= 0:
                ref_err = ref_dist * 0.15
            stat, pval = kstest(hi_dist, 'norm',
                                args=(ref_dist, max(ref_err, 0.5)))
            diag['ks_dist_stat']  = float(stat)
            diag['ks_dist_pval']  = float(pval)
            diag['dist_median']   = float(np.median(hi_dist))
            diag['dist_mad']      = float(median_abs_deviation(hi_dist))
            if pval < cfg.SUMMARY_KS_ALPHA:
                diag['quality_flag'] = 'KS_DIST_FAIL'
        else:
            diag['ks_dist_pval'] = np.nan

    if ref_rv is not None and np.isfinite(ref_rv):
        hi_rv = matched_df.loc[hi, 'best_rv'].values if 'best_rv' in matched_df.columns else np.array([])
        hi_rv = hi_rv[np.isfinite(hi_rv)]
        if len(hi_rv) >= 5:
            sig_int = algo_info.get('sigma_int_km_s', 20.0)
            if sig_int is None or not np.isfinite(sig_int):
                sig_int = 20.0
            stat, pval = kstest(hi_rv, 'norm',
                                args=(ref_rv, max(sig_int, 5.0)))
            diag['ks_rv_stat'] = float(stat)
            diag['ks_rv_pval'] = float(pval)
            diag['rv_median']  = float(np.median(hi_rv))
            diag['rv_mad']     = float(median_abs_deviation(hi_rv))
        else:
            diag['ks_rv_pval'] = np.nan

    if algo_info.get('use_spatial', False):
        eta_v = algo_info.get('eta', 0.5)
        diag['contamination_est'] = 1.0 - eta_v
    else:
        diag['contamination_est'] = np.nan
    return diag


# ============================================================================
# CONTROL FIELD
# ============================================================================

def estimate_control_field_contamination(
        master, ra_center, dec_center, rhalf_deg,
        inner_mult=None, outer_mult=None, logger=None):
    if inner_mult is None: inner_mult = cfg.CONTROL_FIELD_INNER_MULT
    if outer_mult is None: outer_mult = cfg.CONTROL_FIELD_OUTER_MULT
    r_inner = inner_mult * rhalf_deg
    r_outer = outer_mult * rhalf_deg
    cos_dec = max(np.cos(np.radians(dec_center)), 0.1)
    ra_lo   = ra_center  - r_outer / cos_dec
    ra_hi   = ra_center  + r_outer / cos_dec
    dec_lo  = dec_center - r_outer
    dec_hi  = dec_center + r_outer
    mdf = master.df
    ra_col  = cfg.MASTER_COLS['ra'];  dec_col = cfg.MASTER_COLS['dec']
    box_mask = ((mdf[ra_col]  >= ra_lo)  & (mdf[ra_col]  <= ra_hi) &
                (mdf[dec_col] >= dec_lo) & (mdf[dec_col] <= dec_hi))
    box_df = mdf[box_mask]
    if len(box_df) == 0: return None
    seps = angular_separation_deg(
        box_df[ra_col].values, box_df[dec_col].values,
        ra_center, dec_center)
    annulus_mask = (seps >= r_inner) & (seps <= r_outer)
    n_annulus = int(np.sum(annulus_mask))
    area_annulus = np.pi * (r_outer ** 2 - r_inner ** 2)
    if area_annulus > 0 and n_annulus > 0:
        density = n_annulus / area_annulus
        if logger:
            logger.info(f"      [control] annulus {r_inner:.3f}-{r_outer:.3f} deg: "
                        f"{n_annulus} stars, density={density:.1f} /deg2")
        return density
    return None


# ============================================================================
# LEGACY ALGORITHMS (fallback)
# ============================================================================

def algorithm_dbscan(pmra, pmdec, cp, cd, eps=0.25, minsamp=5,
                     use_h=False, ms=10):
    """DBSCAN-based membership (fallback for OC when EM fails)."""
    n  = len(pmra)
    vm = ~(np.isnan(pmra) | np.isnan(pmdec))
    nv = int(np.sum(vm))
    if nv < ms:
        return np.full(n, 0.5), {'status': 'insufficient_data', 'algorithm': 'DBSCAN'}
    X  = np.column_stack([pmra[vm], pmdec[vm]])
    sc = StandardScaler();  Xs = sc.fit_transform(X)
    rs = sc.transform([[cp, cd]])[0]
    try:
        if use_h and HAS_HDBSCAN:
            cl = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
            labels = cl.fit_predict(Xs); an = 'HDBSCAN'
        else:
            cl = DBSCAN(eps=eps, min_samples=minsamp)
            labels = cl.fit_predict(Xs); an = 'DBSCAN'
        ul = set(labels) - {-1}
        if len(ul) == 0:
            d = np.linalg.norm(Xs - rs, axis=1)
            P = np.exp(-d ** 2 / 2)
            Pm = np.full(n, np.nan); Pm[vm] = P
            return Pm, {'status': 'no_cluster_found', 'algorithm': an,
                        'center_pmra': cp, 'center_pmdec': cd}
        bc = min(ul, key=lambda l: np.linalg.norm(Xs[labels == l].mean(axis=0) - rs))
        cm = labels == bc;  P = np.zeros(nv)
        if use_h and HAS_HDBSCAN and hasattr(cl, 'probabilities_'):
            P[cm] = cl.probabilities_[cm]
        else:
            P[cm] = 1.0
        cc = Xs[cm].mean(axis=0);  cs = max(Xs[cm].std(axis=0).mean(), 0.1)
        nc2 = ~cm
        if np.any(nc2):
            d = np.linalg.norm(Xs[nc2] - cc, axis=1)
            P[nc2] = np.exp(-d ** 2 / (2 * cs ** 2)) * 0.3
        Pm = np.full(n, np.nan); Pm[vm] = P
        cpm = X[cm]
        mu  = cpm.mean(axis=0) if len(cpm) > 0 else np.array([cp, cd])
        Sig = np.cov(cpm.T) if len(cpm) > 2 else np.eye(2) * 0.1
        return Pm, {'status': 'success', 'algorithm': an,
                    'n_cluster_members': int(np.sum(cm)),
                    'mu_cluster': mu.tolist(), 'Sigma_cluster': Sig.tolist(),
                    'pm_dispersion': float(np.sqrt(np.trace(Sig))) if len(cpm) > 2 else np.nan,
                    'eta': float(np.sum(cm) / nv),
                    'center_pmra': cp, 'center_pmdec': cd}
    except Exception as e:
        return np.full(n, 0.5), {'status': f'error: {e}', 'algorithm': 'DBSCAN'}


def algorithm_stream_dbscan(pmra, pmdec, cp, cd, eps=0.4, minsamp=3, ms=10):
    """Stream-adapted DBSCAN for SGR (fallback only)."""
    n  = len(pmra)
    vm = ~(np.isnan(pmra) | np.isnan(pmdec))
    nv = int(np.sum(vm))
    if nv < ms:
        return np.full(n, 0.5), {'status': 'insufficient_data', 'algorithm': 'Stream-DBSCAN'}
    X  = np.column_stack([pmra[vm], pmdec[vm]])
    sc = StandardScaler();  Xs = sc.fit_transform(X)
    rs = sc.transform([[cp, cd]])[0]
    try:
        db = DBSCAN(eps=eps, min_samples=minsamp)
        labels = db.fit_predict(Xs)
        ul = set(labels) - {-1}
        if len(ul) == 0:
            d = np.linalg.norm(Xs - rs, axis=1)
            P = np.exp(-d ** 2 / (2 * eps ** 2))
            Pm = np.full(n, np.nan); Pm[vm] = P
            return Pm, {'status': 'no_cluster_found', 'algorithm': 'Stream-DBSCAN',
                        'center_pmra': cp, 'center_pmdec': cd}
        bc = min(ul, key=lambda l: np.min(np.linalg.norm(Xs[labels == l] - rs, axis=1)))
        cm = labels == bc;  cp2 = Xs[cm];  P = np.zeros(nv)
        if len(cp2) > 0:
            tr = cKDTree(cp2)
            for idx in np.where(cm)[0]:
                d, _ = tr.query(Xs[idx], k=min(3, len(cp2)))
                P[idx] = np.clip(1.0 - np.mean(d[1:] if len(d) > 1 else [0]) / eps, 0.5, 1.0)
        nc2 = ~cm
        if np.any(nc2):
            for idx in np.where(nc2)[0]:
                d = np.min(np.linalg.norm(cp2 - Xs[idx], axis=1))
                P[idx] = 0.3 * np.exp(-d ** 2 / (2 * eps ** 2))
        Pm = np.full(n, np.nan); Pm[vm] = P
        cpm = X[cm]
        mu  = cpm.mean(axis=0) if len(cpm) > 0 else np.array([cp, cd])
        Sig = np.cov(cpm.T) if len(cpm) > 2 else np.eye(2) * 0.1
        return Pm, {'status': 'success', 'algorithm': 'Stream-DBSCAN',
                    'n_stream_members': int(np.sum(cm)),
                    'mu_cluster': mu.tolist(), 'Sigma_cluster': Sig.tolist(),
                    'pm_dispersion': float(np.sqrt(np.trace(Sig))) if len(cpm) > 2 else np.nan,
                    'eta': float(np.sum(cm) / nv),
                    'center_pmra': cp, 'center_pmdec': cd}
    except Exception as e:
        return np.full(n, 0.5), {'status': f'error: {e}', 'algorithm': 'Stream-DBSCAN'}


# ============================================================================
# STANDARD MATCH COLUMNS + CMD BUILDER
# ============================================================================

def _standard_match_columns(mdf):
    mdf['pmra']      = mdf.get(f"{cfg.MASTER_COLS['pmra']}_master", pd.Series(dtype=float))
    mdf['pmdec']     = mdf.get(f"{cfg.MASTER_COLS['pmdec']}_master", pd.Series(dtype=float))
    mdf['ra']        = mdf.get(f"{cfg.MASTER_COLS['ra']}_master", pd.Series(dtype=float))
    mdf['dec']       = mdf.get(f"{cfg.MASTER_COLS['dec']}_master", pd.Series(dtype=float))
    mdf['best_dist'] = mdf.get('best_dist_master', pd.Series(dtype=float))
    mdf['best_rv']   = mdf.get('best_rv_master', pd.Series(dtype=float))
    return mdf


def _build_cmd_kde_from_members(member_df, cols, dist_mod):
    gmag_col = cols.get('gmag')
    bprp_col = cols.get('bp_rp')
    if gmag_col is None or bprp_col is None:
        return None
    if gmag_col not in member_df.columns or bprp_col not in member_df.columns:
        return None
    g    = pd.to_numeric(member_df[gmag_col], errors='coerce').values
    bprp = pd.to_numeric(member_df[bprp_col], errors='coerce').values
    return build_cmd_template(g, bprp, dist_mod)


def _build_field_cmd_kde(matched_df, r_ell=None, rhalf_deg=None):
    gmag = matched_df.get('Gmag_master', pd.Series(dtype=float)).values
    bprp = matched_df.get('BP-RP_master', pd.Series(dtype=float)).values
    ag   = matched_df.get('AG_master', pd.Series(dtype=float)).values
    eb   = matched_df.get('E(BP-RP)_master', pd.Series(dtype=float)).values
    if r_ell is not None and rhalf_deg is not None:
        outer = r_ell > cfg.CMD_FIELD_ANNULUS_MULT * rhalf_deg
        if np.sum(outer) >= cfg.CMD_MIN_MEMBERS_FOR_KDE:
            return build_cmd_template(gmag[outer], bprp[outer], None, ag[outer], eb[outer])
    ok = np.isfinite(gmag) & np.isfinite(bprp)
    if np.sum(ok) >= cfg.CMD_MIN_MEMBERS_FOR_KDE:
        return build_cmd_template(gmag, bprp, None, ag, eb)
    return None


# ============================================================================
# V11: PARALLAX DISTANCE COMPUTATION
# ============================================================================

def compute_parallax_distance(matched_df, member_df, cols):
    """
    Compute parallax-based distance for matched member stars.
    Strategy: member catalog parallax first, then master catalog fallback.
    Returns array of 1/parallax distances (kpc) for matched stars.
    """
    n = len(matched_df)
    plx_dist = np.full(n, np.nan)

    # 1st priority: member catalog parallax
    plx_col = cols.get('parallax')
    if plx_col and plx_col in member_df.columns:
        plx_mem = pd.to_numeric(member_df[plx_col], errors='coerce').values
        # Match indices — member_df should already be aligned
        if len(plx_mem) == n:
            good = np.isfinite(plx_mem) & (plx_mem > 0)
            plx_dist[good] = 1.0 / plx_mem[good]

    # 2nd priority: master catalog parallax (fallback for stars without member plx)
    still_nan = ~np.isfinite(plx_dist)
    plx_master = matched_df.get('plx_from_params_master',
                                 pd.Series(np.full(n, np.nan))).values.astype(float)
    good_m = still_nan & np.isfinite(plx_master) & (plx_master > 0)
    plx_dist[good_m] = 1.0 / plx_master[good_m]

    return plx_dist


# ============================================================================
# REFERENCE DISTANCES (V11: enhanced with parallax fallback)
# ============================================================================

def load_gc_reference_distances(filepath, logger):
    if not filepath or not os.path.exists(filepath):
        logger.warning("GC_dist.csv not found")
        return {}
    logger.info(f"Loading GC reference distances from {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    gc_dists = {}

    def empty(val):
        if pd.isna(val): return True
        s = str(val).strip()
        return s in ['', '-', '\u2013', '\u2014', '\u2212', 'nan', 'NaN', 'N/A', 'n/a'] or len(s) == 0

    nl = nm = 0
    for _, row in df.iterrows():
        name = str(row[cfg.GC_DIST_COLS['name']]).strip()
        dist = err = np.nan
        src  = None
        try:
            v = row[cfg.GC_DIST_COLS['lit_dist']]
            if not empty(v):
                dist = float(v)
                ev = row[cfg.GC_DIST_COLS['lit_dist_err']]
                err = float(str(ev).replace('+', '').strip()) if not empty(ev) else np.nan
                src = 'lit'
        except Exception: pass
        if not np.isfinite(dist):
            try:
                v = row[cfg.GC_DIST_COLS['mean_dist']]
                if not empty(v):
                    dist = float(v)
                    ev = row[cfg.GC_DIST_COLS['mean_dist_err']]
                    err = float(str(ev).replace('+', '').strip()) if not empty(ev) else np.nan
                    src = 'mean'
            except Exception: pass
        if np.isfinite(dist):
            if src == 'lit': nl += 1
            else: nm += 1
            nn = normalize_name(name)
            # V11: store under multiple normalisations for better matching
            for key in [nn, name, name.lower(), name.upper(),
                        name.replace(' ', ''), name.replace('_', '')]:
                gc_dists[key] = (dist, err)
            # Also store common aliases (e.g. NGC 104 = 47 Tuc)
            gc_dists[normalize_name(name.replace(' ', ''))] = (dist, err)
    logger.info(f"  Loaded {nl} lit + {nm} mean = {nl+nm} total")
    return gc_dists


def _get_ref_dist_with_fallback(name, gc_dists, member_df, cols, logger):
    """
    V11: Get reference distance with fallback chain:
    1. Literature/Harris catalog lookup
    2. Member catalog median parallax → 1/plx
    """
    rd = rde = None
    nn = normalize_name(name)

    # 1. Literature lookup (multiple normalisations)
    for nt in [nn, name, name.lower(), name.upper(),
               name.replace(' ', ''), name.replace('_', ''),
               normalize_name(name.replace(' ', ''))]:
        if nt in gc_dists:
            rd, rde = gc_dists[nt]
            break

    # 2. Parallax fallback if no literature distance
    if rd is None or not np.isfinite(rd):
        plx_col = cols.get('parallax')
        if plx_col and plx_col in member_df.columns:
            plx = pd.to_numeric(member_df[plx_col], errors='coerce').values
            vp  = plx[np.isfinite(plx) & (plx > 0) & (plx < 100)]
            if len(vp) >= 3:
                med_plx = float(np.median(vp))
                mad_plx = float(median_abs_deviation(vp, nan_policy='omit'))
                if med_plx > 0:
                    rd  = 1.0 / med_plx
                    rde = rd * (mad_plx / med_plx) if mad_plx > 0 else rd * 0.15
                    logger.info(f"      [dist] Parallax fallback: d={rd:.2f}+/-{rde:.2f} kpc "
                                f"(from {len(vp)} stars, med_plx={med_plx:.4f} mas)")

    return rd, rde


def _get_member_rv(member_df, cols):
    rv_col     = cols.get('rv')
    rv_err_col = cols.get('rv_err')
    rv = rv_err = None
    if rv_col and rv_col in member_df.columns:
        rv = pd.to_numeric(member_df[rv_col], errors='coerce').values
    else:
        for alt in ['RV_weighted_avg', 'RV_km_s', 'vlos', 'radial_velocity', 'RV']:
            if alt in member_df.columns:
                rv = pd.to_numeric(member_df[alt], errors='coerce').values
                break
    if rv_err_col and rv_err_col in member_df.columns:
        rv_err = pd.to_numeric(member_df[rv_err_col], errors='coerce').values
    return rv, rv_err


def _compute_rv_reference(member_rv_arr):
    if member_rv_arr is None: return None, None
    rv = np.asarray(member_rv_arr, dtype=float)
    rv = rv[np.isfinite(rv)]
    if len(rv) < 3: return None, None
    return float(np.median(rv)), float(median_abs_deviation(rv, nan_policy='omit'))


# ============================================================================
# PROCESS GLOBULAR CLUSTERS
# ============================================================================

def process_gc_members(master, gc_dists, logger,
                        epoch_mode='2016', epoch_output_dir=None):
    if not cfg.GC_MEMBERS_FILE or not os.path.exists(cfg.GC_MEMBERS_FILE):
        return []
    logger.info(f"\n{'_'*50}\nProcessing GLOBULAR CLUSTERS [{epoch_mode}] (V11)\n{'_'*50}")
    df   = pd.read_csv(cfg.GC_MEMBERS_FILE)
    cols = cfg.GC_MEM_COLS
    kc   = cols['key']
    clusters = sorted(df[kc].unique()) if kc in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters")
    results = []

    for i, cn in enumerate(clusters):
        logger.info(f"\n  [{i+1}/{len(clusters)}] {cn}  [{epoch_mode}]")
        cdf = df[df[kc] == cn].copy()
        if len(cdf) < cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra  = pd.to_numeric(cdf[cols['ra']],  errors='coerce').values
        dec = pd.to_numeric(cdf[cols['dec']], errors='coerce').values
        vm  = np.isfinite(ra) & np.isfinite(dec)
        cdf = cdf[vm].reset_index(drop=True); ra = ra[vm]; dec = dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra_query, dec_query = _get_query_coords(cdf, cols, logger, epoch_mode)
        c_pmra  = pd.to_numeric(cdf[cols['pmra']],  errors='coerce').median()
        c_pmdec = pd.to_numeric(cdf[cols['pmdec']], errors='coerce').median()

        # V11: enhanced distance lookup with parallax fallback
        rd, rde = _get_ref_dist_with_fallback(cn, gc_dists, cdf, cols, logger)
        dist_mod = 5.0 * np.log10(max(rd, 0.01) * 1000) - 5.0 if rd and np.isfinite(rd) else None

        algo = {'status': 'no_matches', 'algorithm': 'None',
                'center_pmra': c_pmra, 'center_pmdec': c_pmdec}
        mdf = None
        master_dist_arr = member_dist_arr = np.array([])
        master_rv_arr = member_rv_arr = np.array([])
        plx_dist_arr = np.array([])
        ref_rv = ref_rv_err = None; diag = {}

        midx, memidx, seps = master.query(ra_query, dec_query)
        logger.info(f"    Members: {len(ra)} | Matched: {len(midx)} "
                    f"({100*len(midx)/max(len(ra),1):.1f}%)")

        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm  = cdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx)
            mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1)
            mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            mdf = apply_quality_flags(mdf, logger)

            # V11: parallax distance
            plx_dist_arr = compute_parallax_distance(mdf, mm, cols)
            mdf['plx_dist'] = plx_dist_arr

            member_cmd = _build_cmd_kde_from_members(cdf, cols, dist_mod)
            field_cmd  = _build_field_cmd_kde(mdf)
            ra_c = float(np.median(ra_query))
            dec_c = float(np.median(dec_query))
            mrv, _ = _get_member_rv(mm, cols)
            member_rv_arr = mrv if mrv is not None else np.array([])
            ref_rv, ref_rv_err = _compute_rv_reference(member_rv_arr)

            P, algo = algorithm_bayesian_em(
                pmra=mdf['pmra'].values, pmdec=mdf['pmdec'].values,
                cp=c_pmra, cd=c_pmdec,
                pmra_err=mdf['pmra_err_adj'].values,
                pmdec_err=mdf['pmdec_err_adj'].values,
                pm_corr=mdf.get('pmRApmDEcor_master', pd.Series(dtype=float)).values,
                ra_deg=mdf['ra'].values, dec_deg=mdf['dec'].values,
                ra_center=ra_c, dec_center=dec_c,
                rhalf_deg=None,
                rv_obs=mdf['best_rv'].values,
                rv_err_obs=mdf['rv_err_adj'].values,
                rv_sys_prior=ref_rv, sigma_int_prior=cfg.SIGMA_INT_INIT,
                dist_obs=mdf['best_dist'].values,
                dist_err_obs=mdf.get('best_dist_err_master', pd.Series(dtype=float)).values,
                ref_dist=rd, ref_dist_err=rde,
                gmag_obs=mdf.get('Gmag_master', pd.Series(dtype=float)).values,
                bprp_obs=mdf.get('BP-RP_master', pd.Series(dtype=float)).values,
                ag_obs=mdf.get('AG_master', pd.Series(dtype=float)).values,
                ebprp_obs=mdf.get('E(BP-RP)_master', pd.Series(dtype=float)).values,
                dist_modulus=dist_mod,
                member_cmd_kde=member_cmd, field_cmd_kde=field_cmd,
                logg_obs=mdf.get('logg_master', pd.Series(dtype=float)).values,
                feh_obs=mdf.get('feh_master', pd.Series(dtype=float)).values,
                feh_sys=None, obj_type='GC',
                ms=cfg.MIN_STARS_FOR_ANALYSIS, logger=logger)

            mdf['P_mem'] = P
            master_dist_arr = mdf['best_dist'].values
            plx_arr = _safe_col(mm, cols.get('parallax'))
            member_dist_arr = np.where(plx_arr > 0, 1.0 / plx_arr, np.nan)
            master_rv_arr = mdf['best_rv'].values
            diag = compute_diagnostics(mdf, P, rd, ref_rv, 'GC', algo, logger)
            n_hi = int(np.sum(np.where(np.isfinite(P), P, 0) > cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"    V11 EM: {algo.get('n_em_iterations',0)} iters | "
                        f"eta={algo.get('eta',0):.3f} | "
                        f"n(P>{cfg.P_MEM_PLOT_THRESHOLD})={n_hi}")

        results.append({
            'cluster_name': cn, 'obj_type': 'GC',
            'member_df': cdf, 'matched_df': mdf,
            'algo_info': algo, 'mem_cols': cols,
            'n_members': len(cdf), 'n_matched': len(midx),
            'ref_dist': rd, 'ref_dist_err': rde,
            'ref_rv': ref_rv, 'ref_rv_err': ref_rv_err,
            'master_dist': master_dist_arr, 'member_dist': member_dist_arr,
            'master_rv': master_rv_arr, 'member_rv': member_rv_arr,
            'plx_dist': plx_dist_arr,
            'diagnostics': diag,
        })
        gcmod.collect()
    return results


# ============================================================================
# PROCESS OPEN CLUSTERS
# ============================================================================

def process_oc_members(master, logger, epoch_mode='2016', epoch_output_dir=None):
    if not cfg.OC_MEMBERS_FILE or not os.path.exists(cfg.OC_MEMBERS_FILE):
        return []
    logger.info(f"\n{'_'*50}\nProcessing OPEN CLUSTERS [{epoch_mode}] (V11)\n{'_'*50}")
    df   = pd.read_csv(cfg.OC_MEMBERS_FILE)
    cols = cfg.OC_MEM_COLS
    kc   = cols['key']
    clusters = sorted(df[kc].unique()) if kc in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters")
    results = []

    for i, cn in enumerate(clusters):
        logger.info(f"\n  [{i+1}/{len(clusters)}] {cn}  [{epoch_mode}]")
        cdf = df[df[kc] == cn].copy()
        if len(cdf) < cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra  = pd.to_numeric(cdf[cols['ra']],  errors='coerce').values
        dec = pd.to_numeric(cdf[cols['dec']], errors='coerce').values
        vm  = np.isfinite(ra) & np.isfinite(dec)
        cdf = cdf[vm].reset_index(drop=True); ra = ra[vm]; dec = dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue
        plx = pd.to_numeric(cdf[cols['parallax']], errors='coerce').values
        vp  = plx[plx > 0]
        rd  = float(np.mean(1.0 / vp)) if len(vp) > 0 else None
        rde = float(np.std(1.0 / vp))  if len(vp) > 0 else None
        dist_mod = 5.0 * np.log10(max(rd, 0.01) * 1000) - 5.0 if rd and np.isfinite(rd) else None
        ra_query, dec_query = _get_query_coords(cdf, cols, logger, epoch_mode)
        c_pmra  = pd.to_numeric(cdf[cols['pmra']],  errors='coerce').median()
        c_pmdec = pd.to_numeric(cdf[cols['pmdec']], errors='coerce').median()

        algo = {'status': 'no_matches', 'algorithm': 'None',
                'center_pmra': c_pmra, 'center_pmdec': c_pmdec}
        mdf = None
        master_dist_arr = member_dist_arr = np.array([])
        master_rv_arr = member_rv_arr = np.array([])
        plx_dist_arr = np.array([])
        ref_rv = ref_rv_err = None; diag = {}

        midx, memidx, seps = master.query(ra_query, dec_query)
        logger.info(f"    Members: {len(ra)} | Matched: {len(midx)} "
                    f"({100*len(midx)/max(len(ra),1):.1f}%)")

        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm  = cdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx)
            mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1)
            mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            mdf = apply_quality_flags(mdf, logger)

            plx_dist_arr = compute_parallax_distance(mdf, mm, cols)
            mdf['plx_dist'] = plx_dist_arr

            member_cmd = _build_cmd_kde_from_members(cdf, cols, dist_mod)
            field_cmd  = _build_field_cmd_kde(mdf)
            mrv, _ = _get_member_rv(mm, cols)
            member_rv_arr = mrv if mrv is not None else np.array([])
            ref_rv, ref_rv_err = _compute_rv_reference(member_rv_arr)

            P, algo = algorithm_bayesian_em(
                pmra=mdf['pmra'].values, pmdec=mdf['pmdec'].values,
                cp=c_pmra, cd=c_pmdec,
                pmra_err=mdf['pmra_err_adj'].values,
                pmdec_err=mdf['pmdec_err_adj'].values,
                pm_corr=mdf.get('pmRApmDEcor_master', pd.Series(dtype=float)).values,
                rv_obs=mdf['best_rv'].values,
                rv_err_obs=mdf['rv_err_adj'].values,
                rv_sys_prior=ref_rv,
                dist_obs=mdf['best_dist'].values,
                dist_err_obs=mdf.get('best_dist_err_master', pd.Series(dtype=float)).values,
                ref_dist=rd, ref_dist_err=rde,
                gmag_obs=mdf.get('Gmag_master', pd.Series(dtype=float)).values,
                bprp_obs=mdf.get('BP-RP_master', pd.Series(dtype=float)).values,
                ag_obs=mdf.get('AG_master', pd.Series(dtype=float)).values,
                ebprp_obs=mdf.get('E(BP-RP)_master', pd.Series(dtype=float)).values,
                dist_modulus=dist_mod,
                member_cmd_kde=member_cmd, field_cmd_kde=field_cmd,
                feh_obs=mdf.get('feh_master', pd.Series(dtype=float)).values,
                obj_type='OC', ms=cfg.MIN_STARS_FOR_ANALYSIS, logger=logger)

            mdf['P_mem'] = P
            master_dist_arr = mdf['best_dist'].values
            member_dist_arr = np.where(
                pd.to_numeric(mm[cols['parallax']], errors='coerce').values > 0,
                1.0 / pd.to_numeric(mm[cols['parallax']], errors='coerce').values, np.nan)
            master_rv_arr = mdf['best_rv'].values
            diag = compute_diagnostics(mdf, P, rd, ref_rv, 'OC', algo, logger)
            n_hi = int(np.sum(np.where(np.isfinite(P), P, 0) > cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"    V11 EM: {algo.get('n_em_iterations',0)} iters | "
                        f"eta={algo.get('eta',0):.3f} | n(P>{cfg.P_MEM_PLOT_THRESHOLD})={n_hi}")

        results.append({
            'cluster_name': cn, 'obj_type': 'OC',
            'member_df': cdf, 'matched_df': mdf,
            'algo_info': algo, 'mem_cols': cols,
            'n_members': len(cdf), 'n_matched': len(midx),
            'ref_dist': rd, 'ref_dist_err': rde,
            'ref_rv': ref_rv, 'ref_rv_err': ref_rv_err,
            'master_dist': master_dist_arr, 'member_dist': member_dist_arr,
            'master_rv': master_rv_arr, 'member_rv': member_rv_arr,
            'plx_dist': plx_dist_arr,
            'diagnostics': diag,
        })
        gcmod.collect()
    return results


# ============================================================================
# V11: PROCESS SGR STREAM — FULL BAYESIAN EM (same rigor as GC/OC/DW)
# ============================================================================

def process_sgr_members(master, logger, epoch_mode='2016', epoch_output_dir=None):
    """
    V11: SGR stream processing with full 7-term Bayesian EM.
    Uses dist, vlos, FeH, CMD from member catalog.
    5 kpc bins from 15-80 kpc. Stream DBSCAN as fallback only.
    """
    if not cfg.SGR_MEMBERS_FILE or not os.path.exists(cfg.SGR_MEMBERS_FILE):
        return []
    logger.info(f"\n{'_'*50}")
    logger.info(f"Processing SGR STREAM [{epoch_mode}] (V11 full phase-space EM)")
    logger.info(f"{'_'*50}")
    df   = pd.read_csv(cfg.SGR_MEMBERS_FILE)
    cols = cfg.SGR_MEM_COLS

    # Check distance column
    dc = cols.get('dist', 'dist')
    if dc not in df.columns:
        logger.error(f"  SGR distance column '{dc}' not found!")
        return []
    df[dc] = pd.to_numeric(df[dc], errors='coerce')
    df = df.dropna(subset=[dc])

    # Log available columns
    avail = [c for c in ['parallax', 'vlos', 'vloserr', 'FeH', 'FeHerr',
                          'g_mag', 'bp_rp', 'ebv'] if c in df.columns]
    logger.info(f"  SGR catalog: {len(df)} stars, columns: {avail}")

    # V11: finer bins — 5 kpc width, 15-80 kpc
    be = np.arange(cfg.SGR_BIN_START_KPC,
                   cfg.SGR_BIN_END_KPC + cfg.SGR_BIN_WIDTH_KPC,
                   cfg.SGR_BIN_WIDTH_KPC)
    bl = [f'{be[j]:.0f}-{be[j+1]:.0f} kpc' for j in range(len(be) - 1)]
    df['dist_bin'] = pd.cut(df[dc], bins=be, labels=bl, right=False)
    df = df.dropna(subset=['dist_bin'])
    bc = df['dist_bin'].value_counts()
    vb = bc[bc >= cfg.SGR_MIN_STARS_PER_BIN].index.tolist()
    bs = sorted([(b, df[df['dist_bin'] == b][dc].mean()) for b in vb], key=lambda x: x[1])
    bins_list = [x[0] for x in bs]
    if not bins_list:
        logger.warning("  No SGR bins with enough stars")
        return []
    logger.info(f"  SGR bins ({cfg.SGR_BIN_WIDTH_KPC} kpc): {len(bins_list)} valid bins")
    for bl_name in bins_list:
        n_b = len(df[df['dist_bin'] == bl_name])
        logger.info(f"    {bl_name}: {n_b} stars")

    results = []
    for i, bl_name in enumerate(bins_list):
        logger.info(f"\n  [{i+1}/{len(bins_list)}] SGR {bl_name}  [{epoch_mode}]")
        bdf = df[df['dist_bin'] == bl_name].copy()

        ra  = pd.to_numeric(bdf[cols['ra']],  errors='coerce').values
        dec = pd.to_numeric(bdf[cols['dec']], errors='coerce').values
        vm  = np.isfinite(ra) & np.isfinite(dec)
        bdf = bdf[vm].reset_index(drop=True); ra = ra[vm]; dec = dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue

        ra_query, dec_query = _get_query_coords(bdf, cols, logger, epoch_mode)
        c_pmra  = pd.to_numeric(bdf[cols['pmra']],  errors='coerce').median()
        c_pmdec = pd.to_numeric(bdf[cols['pmdec']], errors='coerce').median()

        # V11: Reference distance = bin center +/- half-width
        bin_edges = bl_name.replace(' kpc', '').split('-')
        bin_lo = float(bin_edges[0])
        bin_hi = float(bin_edges[1])
        rd  = (bin_lo + bin_hi) / 2.0  # bin center
        rde = (bin_hi - bin_lo) / 2.0  # half-width
        dist_mod = 5.0 * np.log10(max(rd, 0.01) * 1000) - 5.0

        # V11: Reference RV from member catalog
        rv_mem_col = cols.get('rv', 'vlos')
        if rv_mem_col in bdf.columns:
            rv_mem = pd.to_numeric(bdf[rv_mem_col], errors='coerce').values
            rv_fin = rv_mem[np.isfinite(rv_mem)]
            ref_rv = float(np.median(rv_fin)) if len(rv_fin) >= 3 else None
            ref_rv_err = float(median_abs_deviation(rv_fin, nan_policy='omit')) if len(rv_fin) >= 3 else None
        else:
            ref_rv = ref_rv_err = None

        # V11: Reference [Fe/H] from member catalog
        feh_col = cols.get('feh', 'FeH')
        feh_sys = feh_sys_sigma = None
        if feh_col in bdf.columns:
            feh_mem = pd.to_numeric(bdf[feh_col], errors='coerce').values
            feh_fin = feh_mem[np.isfinite(feh_mem)]
            if len(feh_fin) >= 3:
                feh_sys = float(np.median(feh_fin))
                feh_sys_sigma = float(median_abs_deviation(feh_fin, nan_policy='omit'))
                logger.info(f"    [Fe/H] ref: {feh_sys:.2f} +/- {feh_sys_sigma:.2f}")

        algo = {'status': 'no_matches', 'algorithm': 'None',
                'center_pmra': c_pmra, 'center_pmdec': c_pmdec}
        mdf = None
        master_dist_arr = member_dist_arr = np.array([])
        master_rv_arr = member_rv_arr = np.array([])
        plx_dist_arr = np.array([])
        diag = {}

        midx, memidx, seps = master.query(ra_query, dec_query)
        logger.info(f"    Members: {len(ra)} | Matched: {len(midx)} "
                    f"({100*len(midx)/max(len(ra),1):.1f}%)")

        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm  = bdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx)
            mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1)
            mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            mdf = apply_quality_flags(mdf, logger)

            # V11: parallax distance
            plx_dist_arr = compute_parallax_distance(mdf, mm, cols)
            mdf['plx_dist'] = plx_dist_arr

            # V11: Build CMD template from SGR member photometry
            member_cmd = _build_cmd_kde_from_members(bdf, cols, dist_mod)
            field_cmd  = _build_field_cmd_kde(mdf)

            # V11: Extract member RV/dist for EM
            rv_obs_col = cols.get('rv', 'vlos')
            rv_err_col = cols.get('rv_err', 'vloserr')
            mem_rv = _safe_col(mm, rv_obs_col) if rv_obs_col in mm.columns else np.full(len(mm), np.nan)
            mem_rv_err = _safe_col(mm, rv_err_col) if rv_err_col in mm.columns else np.full(len(mm), np.nan)

            # Use master RV where member RV unavailable (fallback)
            master_rv_v = mdf['best_rv'].values
            master_rv_e = mdf['rv_err_adj'].values
            combined_rv = np.where(np.isfinite(mem_rv), mem_rv, master_rv_v)
            combined_rv_err = np.where(np.isfinite(mem_rv_err) & (mem_rv_err > 0),
                                        mem_rv_err, master_rv_e)

            # Member distances for EM
            mem_dist = _safe_col(mm, dc) if dc in mm.columns else np.full(len(mm), np.nan)
            mem_dist_err_col = cols.get('dist_err', 'disterr')
            mem_dist_err = _safe_col(mm, mem_dist_err_col) if mem_dist_err_col in mm.columns else np.full(len(mm), np.nan)

            # Use master dist where member dist unavailable
            master_dist_v = mdf['best_dist'].values
            master_dist_e = mdf.get('best_dist_err_master', pd.Series(dtype=float)).values.astype(float)
            combined_dist = np.where(np.isfinite(mem_dist), mem_dist, master_dist_v)
            combined_dist_err = np.where(np.isfinite(mem_dist_err) & (mem_dist_err > 0),
                                          mem_dist_err, master_dist_e)

            # V11: [Fe/H] from member catalog
            feh_obs_v = _safe_col(mm, feh_col) if feh_col in mm.columns else np.full(len(mm), np.nan)
            # Fallback to master feh
            master_feh = mdf.get('feh_master', pd.Series(dtype=float)).values.astype(float)
            combined_feh = np.where(np.isfinite(feh_obs_v), feh_obs_v, master_feh)

            # V11: SGR uses full Bayesian EM (same as GC/OC/DW)
            # No spatial Plummer — SGR is a stream, not a bound cluster
            P, algo = algorithm_bayesian_em(
                pmra=mdf['pmra'].values, pmdec=mdf['pmdec'].values,
                cp=c_pmra, cd=c_pmdec,
                pmra_err=mdf['pmra_err_adj'].values,
                pmdec_err=mdf['pmdec_err_adj'].values,
                pm_corr=mdf.get('pmRApmDEcor_master', pd.Series(dtype=float)).values,
                # No spatial prior for streams
                ra_deg=None, dec_deg=None,
                ra_center=None, dec_center=None,
                rhalf_deg=None,
                # RV
                rv_obs=combined_rv,
                rv_err_obs=combined_rv_err,
                rv_sys_prior=ref_rv,
                sigma_int_prior=cfg.SIGMA_INT_INIT,
                # Distance — critical for SGR
                dist_obs=combined_dist,
                dist_err_obs=combined_dist_err,
                ref_dist=rd, ref_dist_err=rde,
                # CMD
                gmag_obs=mdf.get('Gmag_master', pd.Series(dtype=float)).values,
                bprp_obs=mdf.get('BP-RP_master', pd.Series(dtype=float)).values,
                ag_obs=mdf.get('AG_master', pd.Series(dtype=float)).values,
                ebprp_obs=mdf.get('E(BP-RP)_master', pd.Series(dtype=float)).values,
                dist_modulus=dist_mod,
                member_cmd_kde=member_cmd, field_cmd_kde=field_cmd,
                # Spectroscopic
                logg_obs=mdf.get('logg_master', pd.Series(dtype=float)).values,
                feh_obs=combined_feh,
                feh_sys=feh_sys, feh_sys_sigma=feh_sys_sigma,
                obj_type='SGR',
                ms=cfg.MIN_STARS_FOR_ANALYSIS, logger=logger)

            # V11: If EM fails, fall back to stream DBSCAN
            if algo.get('status') != 'success':
                logger.warning(f"    EM failed ({algo.get('status')}), falling back to stream DBSCAN")
                P, algo = algorithm_stream_dbscan(
                    mdf['pmra'].values, mdf['pmdec'].values, c_pmra, c_pmdec,
                    cfg.DBSCAN_EPS_STREAM, cfg.DBSCAN_MIN_SAMPLES_STREAM,
                    cfg.MIN_STARS_FOR_ANALYSIS)

            mdf['P_mem'] = P
            master_dist_arr = mdf['best_dist'].values
            member_dist_arr = mem_dist if dc in mm.columns else np.array([])
            master_rv_arr = mdf['best_rv'].values
            member_rv_arr = mem_rv if rv_obs_col in mm.columns else np.array([])

            diag = compute_diagnostics(mdf, P, rd, ref_rv, 'SGR', algo, logger)
            n_hi = int(np.sum(np.where(np.isfinite(P), P, 0) > cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"    V11 EM: {algo.get('n_em_iterations',0)} iters | "
                        f"eta={algo.get('eta',0):.3f} | "
                        f"v_sys={algo.get('v_sys_recovered',0):.1f} | "
                        f"sigma_int={algo.get('sigma_int_km_s',0):.1f} | "
                        f"n(P>{cfg.P_MEM_PLOT_THRESHOLD})={n_hi}")

        results.append({
            'cluster_name': bl_name, 'obj_type': 'SGR',
            'member_df': bdf, 'matched_df': mdf,
            'algo_info': algo, 'mem_cols': cols,
            'n_members': len(bdf), 'n_matched': len(midx),
            'ref_dist': rd, 'ref_dist_err': rde,
            'ref_rv': ref_rv, 'ref_rv_err': ref_rv_err,
            'master_dist': master_dist_arr, 'member_dist': member_dist_arr,
            'master_rv': master_rv_arr, 'member_rv': member_rv_arr,
            'plx_dist': plx_dist_arr,
            'diagnostics': diag,
        })
        gcmod.collect()
    return results


# ============================================================================
# PROCESS DWARF GALAXIES
# ============================================================================

def process_dwg_members(master, logger, epoch_mode='2016', epoch_output_dir=None):
    if not cfg.DWG_MEMBERS_FILE or not os.path.exists(cfg.DWG_MEMBERS_FILE):
        return []
    logger.info(f"\n{'_'*50}")
    logger.info(f"Processing DWARF GALAXIES [{epoch_mode}] (V11)")
    logger.info(f"{'_'*50}")
    df   = pd.read_csv(cfg.DWG_MEMBERS_FILE)
    cols = cfg.DWG_MEM_COLS
    kc   = cols['key']
    for cname in [cols['ra'], cols['dec']]:
        if cname not in df.columns:
            logger.error(f"  Column '{cname}' not found!")
            return []
    gals    = df[kc].unique() if kc in df.columns else ['ALL']
    results = []

    for i, gn in enumerate(gals):
        logger.info(f"\n  [{i+1}/{len(gals)}] {gn}  [{epoch_mode}]")
        gal_df = df[df[kc] == gn].copy()
        if len(gal_df) < 1: continue
        gr         = gal_df.iloc[0]
        rd         = _safe_float(gr, cols.get('distance'))
        rde        = _safe_float(gr, cols.get('distance_err'))
        ref_rv     = _safe_float(gr, cols.get('rv_ref'))
        ref_rv_err = _safe_float(gr, cols.get('rv_ref_err'))
        rhalf_deg  = _safe_float(gr, cols.get('rhalf'))
        ellip      = _safe_float(gr, cols.get('ellipticity')) or 0.0
        pa         = _safe_float(gr, cols.get('position_angle')) or 0.0
        feh_sys    = _safe_float(gr, cols.get('metallicity'))
        feh_sigma  = _safe_float(gr, cols.get('metallicity_sigma'))
        sig_int_prior = _safe_float(gr, cols.get('rv_sigma')) or cfg.SIGMA_INT_INIT
        dist_mod   = _safe_float(gr, cols.get('distance_modulus'))
        edr3_pmra  = _safe_float(gr, cols.get('edr3_pmra'))
        edr3_pmdec = _safe_float(gr, cols.get('edr3_pmdec'))

        if rd is not None:
            logger.info(f"    dist={rd:.1f}+/-{rde or 0:.1f} kpc | "
                        f"RV={ref_rv or 0:.1f} km/s | [Fe/H]={feh_sys or 0:.2f}")

        ra  = pd.to_numeric(gal_df[cols['ra']],  errors='coerce').values
        dec = pd.to_numeric(gal_df[cols['dec']], errors='coerce').values
        vm  = np.isfinite(ra) & np.isfinite(dec)
        gal_df = gal_df[vm].reset_index(drop=True); ra = ra[vm]; dec = dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra_query, dec_query = _get_query_coords(gal_df, cols, logger, epoch_mode)

        if edr3_pmra is not None and np.isfinite(edr3_pmra):
            c_pmra = edr3_pmra
            c_pmdec = edr3_pmdec if edr3_pmdec is not None else 0.0
        else:
            c_pmra  = pd.to_numeric(gal_df[cols['pmra']],  errors='coerce').median() if cols['pmra'] in gal_df.columns else 0.0
            c_pmdec = pd.to_numeric(gal_df[cols['pmdec']], errors='coerce').median() if cols['pmdec'] in gal_df.columns else 0.0

        algo = {'status': 'no_matches', 'algorithm': 'BayesianEM-V11'}
        mdf = None
        master_dist_arr = member_dist_arr = np.array([])
        master_rv_arr = member_rv_arr = np.array([])
        plx_dist_arr = np.array([])
        n_plx_removed = 0; diag = {}

        midx, memidx, seps = master.query(ra_query, dec_query)
        logger.info(f"    Matched: {len(midx)}/{len(ra)}")

        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm  = gal_df.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx)
            mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1)
            mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            mdf = apply_quality_flags(mdf, logger)

            # Parallax foreground cut
            plx_v = mdf.get('plx_from_params_master', pd.Series(dtype=float)).values.astype(float)
            plx_e = mdf.get('plx_err_from_params_master', pd.Series(dtype=float)).values.astype(float)
            plx_lo = np.where(np.isfinite(plx_e), plx_v - cfg.PLX_FOREGROUND_SIGMA * plx_e, plx_v)
            fg_mask = (np.isfinite(plx_v) & np.isfinite(plx_e)
                       & (plx_lo > cfg.PLX_FOREGROUND_THRESHOLD))
            n_plx_removed = int(np.sum(fg_mask))
            if n_plx_removed > 0:
                logger.info(f"    Parallax cut: removed {n_plx_removed} foreground")
                mdf = mdf[~fg_mask].reset_index(drop=True)
                mm  = mm[~fg_mask].reset_index(drop=True)
            if len(mdf) < cfg.MIN_STARS_FOR_ANALYSIS:
                results.append({
                    'cluster_name': gn, 'obj_type': 'DW',
                    'member_df': gal_df, 'matched_df': None,
                    'algo_info': {'status': 'too_few_after_plx_cut', 'n_parallax_removed': n_plx_removed},
                    'mem_cols': cols, 'n_members': len(gal_df), 'n_matched': len(midx),
                    'ref_dist': rd, 'ref_dist_err': rde,
                    'ref_rv': ref_rv, 'ref_rv_err': ref_rv_err,
                    'master_dist': np.array([]), 'member_dist': np.array([]),
                    'master_rv': np.array([]), 'member_rv': np.array([]),
                    'plx_dist': np.array([]), 'diagnostics': {},
                })
                continue

            plx_dist_arr = compute_parallax_distance(mdf, mm, cols)
            mdf['plx_dist'] = plx_dist_arr

            ra_c = float(np.nanmedian(mdf['ra'].values))
            dec_c = float(np.nanmedian(mdf['dec'].values))
            if rhalf_deg and np.isfinite(rhalf_deg):
                estimate_control_field_contamination(master, ra_c, dec_c, rhalf_deg, logger=logger)
            field_cmd = _build_field_cmd_kde(mdf, None, rhalf_deg)

            P, algo = algorithm_bayesian_em(
                pmra=mdf['pmra'].values, pmdec=mdf['pmdec'].values,
                cp=c_pmra, cd=c_pmdec,
                pmra_err=mdf['pmra_err_adj'].values,
                pmdec_err=mdf['pmdec_err_adj'].values,
                pm_corr=mdf.get('pmRApmDEcor_master', pd.Series(dtype=float)).values,
                ra_deg=mdf['ra'].values, dec_deg=mdf['dec'].values,
                ra_center=ra_c, dec_center=dec_c,
                rhalf_deg=rhalf_deg, ellipticity=ellip, position_angle=pa,
                rv_obs=mdf['best_rv'].values,
                rv_err_obs=mdf['rv_err_adj'].values,
                rv_sys_prior=ref_rv, sigma_int_prior=sig_int_prior,
                dist_obs=mdf['best_dist'].values,
                dist_err_obs=mdf.get('best_dist_err_master', pd.Series(dtype=float)).values,
                ref_dist=rd, ref_dist_err=rde,
                gmag_obs=mdf.get('Gmag_master', pd.Series(dtype=float)).values,
                bprp_obs=mdf.get('BP-RP_master', pd.Series(dtype=float)).values,
                ag_obs=mdf.get('AG_master', pd.Series(dtype=float)).values,
                ebprp_obs=mdf.get('E(BP-RP)_master', pd.Series(dtype=float)).values,
                dist_modulus=dist_mod,
                member_cmd_kde=None, field_cmd_kde=field_cmd,
                logg_obs=mdf.get('logg_master', pd.Series(dtype=float)).values,
                feh_obs=mdf.get('feh_master', pd.Series(dtype=float)).values,
                feh_sys=feh_sys, feh_sys_sigma=feh_sigma,
                obj_type='DW', ms=cfg.MIN_STARS_FOR_ANALYSIS, logger=logger)

            algo['n_parallax_removed'] = n_plx_removed
            mdf['P_mem'] = P
            master_dist_arr = mdf['best_dist'].values
            master_rv_arr   = mdf['best_rv'].values
            rv_col = cols.get('rv', 'RV_km_s')
            member_rv_arr = (_safe_col(mm, rv_col) if rv_col in mm.columns else np.array([]))
            diag = compute_diagnostics(mdf, P, rd, ref_rv, 'DW', algo, logger)
            n_hi = int(np.sum(np.where(np.isfinite(P), P, 0) > cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"    V11 EM: {algo.get('n_em_iterations',0)} iters | "
                        f"eta={algo.get('eta',0):.3f} | "
                        f"v_sys={algo.get('v_sys_recovered',0):.1f} | "
                        f"n(P>{cfg.P_MEM_PLOT_THRESHOLD})={n_hi}")

        results.append({
            'cluster_name': gn, 'obj_type': 'DW',
            'member_df': gal_df, 'matched_df': mdf,
            'algo_info': algo, 'mem_cols': cols,
            'n_members': len(gal_df), 'n_matched': len(midx),
            'ref_dist': rd, 'ref_dist_err': rde,
            'ref_rv': ref_rv, 'ref_rv_err': ref_rv_err,
            'master_dist': master_dist_arr, 'member_dist': member_dist_arr,
            'master_rv': master_rv_arr, 'member_rv': member_rv_arr,
            'plx_dist': plx_dist_arr,
            'diagnostics': diag,
        })
        gcmod.collect()
    return results


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _clean(arr, plo=1, phi=99, min_n=3):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < min_n: return a
    lo, hi = np.percentile(a, [plo, phi])
    return a[(a >= lo) & (a <= hi)]


def _safe_kde(ax, data, bins, color, ls='--', alpha=0.7, zorder=5):
    d = data[np.isfinite(data)]
    if len(d) < 4: return
    try:
        kde = gaussian_kde(d)
        x = np.linspace(bins[0], bins[-1], 200)
        ax.plot(x, kde(x) * len(d) * (bins[1] - bins[0]),
                color=color, ls=ls, alpha=alpha, lw=2, zorder=zorder)
    except Exception: pass


def _med_mad_box(ax, data, color, loc='upper right', prefix=''):
    d = data[np.isfinite(data)]
    if len(d) < 2: return
    med = np.median(d)
    mad = median_abs_deviation(d, nan_policy='omit')
    txt = f"{prefix}Med = {med:.2f}\nMAD = {mad:.2f}\nN = {len(d)}"
    locs = {'upper right': (0.97, 0.97, 'right', 'top'),
            'upper left': (0.03, 0.97, 'left', 'top'),
            'lower right': (0.97, 0.03, 'right', 'bottom'),
            'lower left': (0.03, 0.03, 'left', 'bottom')}
    x, y, ha, va = locs.get(loc, (0.97, 0.97, 'right', 'top'))
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha=ha, va=va, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.9, lw=1.5))


def _hist_step_kde(ax, data, color, label, bins_range=None, n_bins=30, zorder=5):
    d = _clean(data)
    if len(d) < 3: return np.array([]), np.array([])
    if bins_range is None:
        lo, hi = np.percentile(d, [0.5, 99.5])
        rng = max(hi - lo, 1.0)
        bins_range = [lo - 0.1 * rng, hi + 0.1 * rng]
    counts, bins, _ = ax.hist(d, bins=n_bins, range=bins_range, alpha=0)
    ax.step(bins[:-1], counts, where='post', color=color, linewidth=2.5,
            label=label, zorder=zorder)
    _safe_kde(ax, d, bins, color, zorder=zorder)
    ax.axvline(np.median(d), color=color, ls=':', lw=2.5, alpha=0.9, zorder=zorder)
    return counts, bins


def _bins_range_with_ref(arrays, ref_val=None, positive_only=False):
    all_d = [_clean(a) for a in arrays if a is not None and len(a) > 0]
    if positive_only:
        all_d = [a[(a > 0) & (a < 300)] for a in all_d]
    all_d = [a for a in all_d if len(a) > 0]
    if not all_d and ref_val is None: return [0, 100]
    if not all_d:
        rng = max(abs(ref_val) * 0.3, 10)
        return [ref_val - rng, ref_val + rng]
    comb = np.concatenate(all_d)
    lo, hi = np.percentile(comb, [1, 99])
    rng = max(hi - lo, 1)
    if ref_val is not None and np.isfinite(ref_val):
        lo = min(lo, ref_val - 0.05 * rng)
        hi = max(hi, ref_val + 0.05 * rng)
        rng = max(hi - lo, 1)
    pad = 0.15 * rng
    br_lo = max(0, lo - pad) if positive_only else lo - pad
    return [br_lo, hi + pad]


def _add_mad_band(ax, data, color, alpha=0.15, zorder=2):
    d = _clean(data)
    if len(d) < 3: return
    med = np.median(d)
    mad = median_abs_deviation(d, nan_policy='omit')
    ax.axvspan(med - mad, med + mad, alpha=alpha, color=color, zorder=zorder)


def _add_median_mad_lines(ax, data, color, zorder=6):
    """V11: Add median line + MAD shading for any histogram dataset."""
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 3: return
    med = float(np.median(d))
    mad = float(median_abs_deviation(d, nan_policy='omit'))
    ax.axvline(med, color=color, ls=':', lw=2.5, alpha=0.9, zorder=zorder)
    ax.axvspan(med - mad, med + mad, alpha=0.15, color=color, zorder=zorder - 1)


def _filter_by_pmem(matched_df, col, threshold=None):
    if threshold is None: threshold = cfg.P_MEM_PLOT_THRESHOLD
    if col not in matched_df.columns: return np.array([]), np.array([])
    all_arr = matched_df[col].values.astype(float)
    if 'P_mem' not in matched_df.columns: return all_arr, all_arr
    pmem = matched_df['P_mem'].values.astype(float)
    mask = (pmem >= threshold) & np.isfinite(pmem)
    return all_arr[mask], all_arr


def _plot_ref_band(ax, ref_val, ref_err, label_prefix='Ref', zorder=20):
    """V11: Plot reference value + uncertainty band on HIGHEST zorder."""
    if ref_val is not None and np.isfinite(ref_val):
        ax.axvline(ref_val, color='black', lw=3, alpha=0.9,
                   label=f'{label_prefix}={ref_val:.1f}', zorder=zorder)
        if ref_err is not None and np.isfinite(ref_err):
            ax.axvspan(ref_val - ref_err, ref_val + ref_err,
                       alpha=0.2, color='gray', zorder=zorder - 1)


# ============================================================================
# V11: INDIVIDUAL PLOTS (4 panels: PM, Sky, Distance, RV)
# ============================================================================

def plot_individual_panels(matched_df, obj_name, obj_type, algo_info,
                           ref_dist=None, ref_dist_err=None,
                           ref_rv=None, ref_rv_err=None,
                           master_dist=None, member_dist=None,
                           master_rv=None, member_rv=None,
                           plx_dist=None,
                           prematch_ra=None, prematch_dec=None,
                           prematch_pmra=None, prematch_pmdec=None,
                           save_dir=None, epoch_mode='2016',
                           n_parallax_removed=0, diagnostics=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    set_paper_style()

    if save_dir is None:
        save_dir = os.path.join(cfg.OUTPUT_DIR, 'individual_plots')
    os.makedirs(save_dir, exist_ok=True)
    thr = cfg.P_MEM_PLOT_THRESHOLD
    epoch_tag = '[J2000->2016]' if epoch_mode == '2000' else '[J2016]'

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    af = axes.flatten()

    title = f'{obj_type}: {obj_name}  {epoch_tag}'
    if ref_dist is not None and np.isfinite(ref_dist):
        title += f'  ($d_{{ref}}$={ref_dist:.1f} kpc)'
    alg = algo_info.get('algorithm', '')
    n_terms = sum([1 for k in ['n_cmd_used', 'n_feh_used', 'n_logg_used',
                                'n_dist_used', 'n_rv_used']
                   if algo_info.get(k, 0) > 0])
    title += f'  [{alg}, {n_terms+2} terms]'
    if n_parallax_removed > 0:
        title += f'  (-{n_parallax_removed} fg)'
    fig.suptitle(title, fontsize=16, fontweight='bold', family='serif', y=0.98)

    hi_dist, all_dist = _filter_by_pmem(matched_df, 'best_dist', thr)
    hi_rv, all_rv = _filter_by_pmem(matched_df, 'best_rv', thr)

    # Parallax distance for high-P_mem stars
    hi_plx_dist = np.array([])
    if plx_dist is not None and len(plx_dist) > 0:
        plx_arr = np.asarray(plx_dist, dtype=float)
        if 'P_mem' in matched_df.columns:
            pmem = matched_df['P_mem'].values.astype(float)
            mask = (pmem >= thr) & np.isfinite(pmem) & np.isfinite(plx_arr) & (plx_arr > 0) & (plx_arr < 300)
            hi_plx_dist = plx_arr[mask]
        else:
            hi_plx_dist = plx_arr[np.isfinite(plx_arr) & (plx_arr > 0) & (plx_arr < 300)]

    # ===== Panel 0: PM =====================================================
    ax = af[0]
    # V11: Gray opaque background — all pre-match members
    if prematch_pmra is not None and prematch_pmdec is not None:
        vm = np.isfinite(prematch_pmra) & np.isfinite(prematch_pmdec)
        if np.sum(vm) > 0:
            ax.scatter(prematch_pmra[vm], prematch_pmdec[vm],
                       c=cfg.COL_GRAY_BG, s=cfg.GRAY_BG_SIZE,
                       alpha=cfg.GRAY_BG_ALPHA, zorder=1,
                       label=f'All members (n={np.sum(vm)})',
                       edgecolors='none', rasterized=True)
    if 'P_mem' in matched_df.columns:
        v = matched_df['P_mem'].notna()
        sc = ax.scatter(matched_df.loc[v, 'pmra'], matched_df.loc[v, 'pmdec'],
                        c=matched_df.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                        s=25, edgecolors='k', linewidths=0.3, vmin=0, vmax=1, zorder=5)
        plt.colorbar(sc, ax=ax).set_label('$P_{\\rm mem}$', fontsize=14, fontweight='bold')
    if algo_info.get('Sigma_cluster') is not None and obj_type not in ['SGR', 'STREAM']:
        mu = algo_info['mu_cluster']
        sig = np.array(algo_info['Sigma_cluster'])
        w, v2 = np.linalg.eigh(sig)
        ang = np.degrees(np.arctan2(v2[1, 0], v2[0, 0]))
        for ns in [1, 2]:
            ell = Ellipse(xy=mu, width=2*ns*np.sqrt(max(w[0], 0)),
                          height=2*ns*np.sqrt(max(w[1], 0)), angle=ang,
                          fill=False, edgecolor='lime', lw=2.5 if ns == 1 else 2,
                          ls='-' if ns == 1 else '--', zorder=10)
            ax.add_patch(ell)
        ax.scatter(*mu, marker='x', s=120, c='lime', linewidths=3, zorder=11)
    ax.set_xlabel('$\\mu_{\\alpha}\\cos\\delta$ (mas yr$^{-1}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel('$\\mu_{\\delta}$ (mas yr$^{-1}$)', fontsize=16, fontweight='bold')
    ax.set_title('Proper Motion', fontsize=18, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2); ax.set_aspect('equal', 'datalim')

    # ===== Panel 1: Sky =====================================================
    ax = af[1]
    # V11: Gray opaque background
    if prematch_ra is not None and prematch_dec is not None:
        vm = np.isfinite(prematch_ra) & np.isfinite(prematch_dec)
        if np.sum(vm) > 0:
            ax.scatter(prematch_ra[vm], prematch_dec[vm],
                       c=cfg.COL_GRAY_BG, s=cfg.GRAY_BG_SIZE,
                       alpha=cfg.GRAY_BG_ALPHA, zorder=1,
                       label=f'All members (n={np.sum(vm)})',
                       edgecolors='none', rasterized=True)
    if 'P_mem' in matched_df.columns:
        v = matched_df['P_mem'].notna()
        sc = ax.scatter(matched_df.loc[v, 'ra'], matched_df.loc[v, 'dec'],
                        c=matched_df.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                        s=25, edgecolors='k', linewidths=0.3, vmin=0, vmax=1, zorder=5)
        plt.colorbar(sc, ax=ax).set_label('$P_{\\rm mem}$', fontsize=14, fontweight='bold')
    ax.set_xlabel('RA (deg)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Dec (deg)', fontsize=16, fontweight='bold')
    ax.set_title('Sky Position', fontsize=18, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.invert_xaxis(); ax.grid(True, alpha=0.2)

    # ===== Panel 2: Distance (V11: red + navy blue + green) ================
    ax = af[2]
    hi_d  = _clean(hi_dist); hi_d = hi_d[(hi_d > 0) & (hi_d < 300)]
    all_d = _clean(all_dist); all_d = all_d[(all_d > 0) & (all_d < 300)]
    hi_plx_c = _clean(hi_plx_dist) if len(hi_plx_dist) > 0 else np.array([])
    hi_plx_c = hi_plx_c[(hi_plx_c > 0) & (hi_plx_c < 300)] if len(hi_plx_c) > 0 else np.array([])

    br = _bins_range_with_ref([all_d, hi_d, hi_plx_c], ref_dist, positive_only=True)

    # Red: all matched (lowest zorder)
    if len(all_d) >= 3:
        c_, b_, _ = ax.hist(all_d, bins=25, range=br, alpha=0)
        ax.step(b_[:-1], c_, where='post', color=cfg.COL_MASTER, lw=1.5,
                alpha=0.5, ls='--', label=f'All matched (n={len(all_d)})', zorder=3)
    # Navy blue: parallax distance (middle zorder)
    if len(hi_plx_c) >= 3:
        _hist_step_kde(ax, hi_plx_c, cfg.COL_MEMBER,
                       f'Plx dist P>{thr:.1f} (n={len(hi_plx_c)})', br, 25, zorder=6)
        _add_median_mad_lines(ax, hi_plx_c, cfg.COL_MEMBER, zorder=7)
        _med_mad_box(ax, hi_plx_c, cfg.COL_MEMBER, 'upper left', 'Plx ')
    # Green: high P_mem (HIGHEST zorder — on top)
    if len(hi_d) >= 3:
        _hist_step_kde(ax, hi_d, cfg.COL_HIGHMEM,
                       f'P>{thr:.1f} (n={len(hi_d)})', br, 25, zorder=8)
        _add_median_mad_lines(ax, hi_d, cfg.COL_HIGHMEM, zorder=9)
        _med_mad_box(ax, hi_d, cfg.COL_HIGHMEM, 'upper right', f'P>{thr:.1f} ')
    # Reference band on TOPMOST zorder
    _plot_ref_band(ax, ref_dist, ref_dist_err, 'Ref', zorder=20)
    # KS annotation
    if diagnostics and diagnostics.get('ks_dist_pval') is not None:
        pv = diagnostics['ks_dist_pval']
        ax.text(0.03, 0.03, f"KS p={pv:.3f}", transform=ax.transAxes, fontsize=9,
                color='green' if pv > 0.05 else 'red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=25)
    ax.set_xlabel('Distance (kpc)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.set_title('Distance Distribution', fontsize=18, fontweight='bold')
    ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.2, axis='y')

    # ===== Panel 3: RV ======================================================
    ax = af[3]
    hi_rv_c = _clean(hi_rv)
    all_rv_c = _clean(all_rv)
    show_rv = len(hi_rv_c) >= 3 or len(all_rv_c) >= 3
    if show_rv:
        br = _bins_range_with_ref([all_rv_c, hi_rv_c], ref_rv)
        if len(all_rv_c) >= 3:
            c_, b_, _ = ax.hist(all_rv_c, bins=25, range=br, alpha=0)
            ax.step(b_[:-1], c_, where='post', color=cfg.COL_MASTER, lw=1.5,
                    alpha=0.5, ls='--', label=f'All matched (n={len(all_rv_c)})', zorder=3)
        if len(hi_rv_c) >= 3:
            _hist_step_kde(ax, hi_rv_c, cfg.COL_HIGHMEM,
                           f'P>{thr:.1f} (n={len(hi_rv_c)})', br, 25, zorder=8)
            _add_median_mad_lines(ax, hi_rv_c, cfg.COL_HIGHMEM, zorder=9)
            _med_mad_box(ax, hi_rv_c, cfg.COL_HIGHMEM, 'upper right', f'P>{thr:.1f} ')
        _plot_ref_band(ax, ref_rv, ref_rv_err, 'Ref', zorder=20)
        v_sys_r = algo_info.get('v_sys_recovered', np.nan)
        if np.isfinite(v_sys_r):
            ax.axvline(v_sys_r, color='darkblue', lw=2.5, ls='-.',
                       alpha=0.9, label=f'$v_{{sys,rec}}$={v_sys_r:.1f}', zorder=15)
        if diagnostics and diagnostics.get('ks_rv_pval') is not None:
            pv = diagnostics['ks_rv_pval']
            ax.text(0.03, 0.03, f"KS p={pv:.3f}", transform=ax.transAxes, fontsize=9,
                    color='green' if pv > 0.05 else 'red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=25)
        ax.set_xlabel('RV (km s$^{-1}$)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Count', fontsize=16, fontweight='bold')
        ax.set_title('Radial Velocity', fontsize=18, fontweight='bold')
        ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.2, axis='y')
    else:
        ax.text(0.5, 0.5, 'No RV data', ha='center', va='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold', color='gray')
        ax.set_title('Radial Velocity', fontsize=18, fontweight='bold'); ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe = str(obj_name).replace(' ', '_').replace('/', '_').replace('-', '_')
    out = os.path.join(save_dir, f"{obj_type}_{safe}.{cfg.SAVE_FORMAT}")
    plt.savefig(out, dpi=cfg.PLOT_DPI); plt.close()
    return out


# ============================================================================
# V11: 4 SEPARATE SUMMARY SERIES (square panels, piecewise)
# ============================================================================

def _get_valid_results(all_results, min_match=5, min_green=1):
    """Filter results that have matches and P_mem column."""
    vr = []
    for r in all_results:
        if r['matched_df'] is None: continue
        if r['n_matched'] < min_match: continue
        if 'P_mem' not in r['matched_df'].columns: continue
        P = r['matched_df']['P_mem'].values
        n_hi = int(np.sum(np.where(np.isfinite(P), P, 0) > cfg.P_MEM_PLOT_THRESHOLD))
        r['_n_hi'] = n_hi
        vr.append(r)
    return vr


def _sort_results(vr):
    """Sort by type then n_hi descending."""
    type_order = ['GC', 'OC', 'DW', 'SGR']
    return sorted(vr, key=lambda x: (
        type_order.index(x['obj_type']) if x['obj_type'] in type_order else 99,
        -x['_n_hi']))


def generate_summary_series(all_results, logger, epoch_output_dir=None, epoch_mode='2016'):
    """V11: Generate 4 separate summary series + piecewise PM summary."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    set_paper_style()

    out_dir = epoch_output_dir or cfg.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    epoch_sub = f'  [J2000->2016]' if epoch_mode == '2000' else '  [J2016]'
    thr = cfg.P_MEM_PLOT_THRESHOLD
    tc  = {'GC': '#006400', 'OC': '#FF8C00', 'DW': '#00008B', 'SGR': '#8B0000'}

    vr = _get_valid_results(all_results, cfg.MIN_SUMMARY_MATCH)
    if not vr:
        logger.warning("No valid results for summary!"); return
    sr = _sort_results(vr)
    n = len(sr)
    ncols = cfg.SUMMARY_COLS_PER_PAGE
    nrows = cfg.SUMMARY_ROWS_PER_PAGE
    per_page = nrows * ncols

    logger.info(f"\n{'='*70}\nGENERATING V11 SUMMARY SERIES\n{'='*70}")
    logger.info(f"  {n} objects for summary")

    # ════════════════════════════════════════════════════════════════════════
    # SERIES 1: DISTANCE SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    n_pages = int(np.ceil(n / per_page))
    for page_idx in range(n_pages):
        i_start = page_idx * per_page
        i_end = min(i_start + per_page, n)
        page_sr = sr[i_start:i_end]
        n_this = len(page_sr)
        nr = int(np.ceil(n_this / ncols))
        fig, axes = plt.subplots(nr, ncols, figsize=(5 * ncols, 5 * nr), squeeze=False)
        af = axes.flatten()
        for j, r in enumerate(page_sr):
            ax = af[j]
            mdf = r['matched_df']; ot = r['obj_type']; cn = r['cluster_name']
            rd = r.get('ref_dist'); rde = r.get('ref_dist_err')
            col = tc.get(ot, 'k')

            hi_dist, all_dist = _filter_by_pmem(mdf, 'best_dist', thr)
            hi_d = _clean(hi_dist); hi_d = hi_d[(hi_d > 0) & (hi_d < 300)]
            all_d = _clean(all_dist); all_d = all_d[(all_d > 0) & (all_d < 300)]

            # Blue parallax
            plx_d = r.get('plx_dist', np.array([]))
            hi_plx = np.array([])
            if len(plx_d) > 0:
                plx_arr = np.asarray(plx_d, dtype=float)
                pmem = mdf['P_mem'].values.astype(float)
                mask = (pmem >= thr) & np.isfinite(pmem) & np.isfinite(plx_arr) & (plx_arr > 0) & (plx_arr < 300)
                hi_plx = plx_arr[mask]
            hi_plx_c = _clean(hi_plx) if len(hi_plx) > 0 else np.array([])

            br = _bins_range_with_ref([all_d, hi_d, hi_plx_c], rd, positive_only=True)
            if len(all_d) >= 3:
                c_, b_, _ = ax.hist(all_d, bins=20, range=br, alpha=0)
                ax.step(b_[:-1], c_, where='post', color=cfg.COL_MASTER, lw=1, alpha=0.4, ls='--', zorder=3)
            if len(hi_plx_c) >= 3:
                _hist_step_kde(ax, hi_plx_c, cfg.COL_MEMBER, f'Plx n={len(hi_plx_c)}', br, 20, zorder=6)
                _add_median_mad_lines(ax, hi_plx_c, cfg.COL_MEMBER, zorder=7)
            if len(hi_d) >= 3:
                _hist_step_kde(ax, hi_d, cfg.COL_HIGHMEM, f'n={len(hi_d)}', br, 20, zorder=8)
                _add_median_mad_lines(ax, hi_d, cfg.COL_HIGHMEM, zorder=9)
                _med_mad_box(ax, hi_d, cfg.COL_HIGHMEM, 'upper right')
            _plot_ref_band(ax, rd, rde, 'Ref', zorder=20)
            ax.set_xlabel('Dist (kpc)', fontsize=11, fontweight='bold')
            ax.set_ylabel('N', fontsize=11, fontweight='bold')
            ax.set_title(f'{ot}: {cn}', fontsize=11, fontweight='bold', color=col)
            ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.2, axis='y')
            ax.set_aspect('auto')
        for j in range(n_this, len(af)): af[j].set_visible(False)
        sfx = f'_p{page_idx+1}' if n_pages > 1 else ''
        fig.suptitle(f'V11 Distance Summary (P>{thr:.1f}){epoch_sub}{sfx}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(out_dir, f'SUMMARY_Dist{sfx}.{cfg.SAVE_FORMAT}'), dpi=cfg.PLOT_DPI)
        plt.close(); logger.info(f"  Saved: SUMMARY_Dist{sfx}")

    # ════════════════════════════════════════════════════════════════════════
    # SERIES 2: RV SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    for page_idx in range(n_pages):
        i_start = page_idx * per_page
        i_end = min(i_start + per_page, n)
        page_sr = sr[i_start:i_end]; n_this = len(page_sr)
        nr = int(np.ceil(n_this / ncols))
        fig, axes = plt.subplots(nr, ncols, figsize=(5 * ncols, 5 * nr), squeeze=False)
        af = axes.flatten()
        for j, r in enumerate(page_sr):
            ax = af[j]; mdf = r['matched_df']
            ot = r['obj_type']; cn = r['cluster_name']
            rrv = r.get('ref_rv'); rrve = r.get('ref_rv_err')
            col = tc.get(ot, 'k')
            hi_rv_arr, all_rv_arr = _filter_by_pmem(mdf, 'best_rv', thr)
            hi_rv_c = _clean(hi_rv_arr); all_rv_c = _clean(all_rv_arr)
            if len(hi_rv_c) >= 3 or len(all_rv_c) >= 3:
                br = _bins_range_with_ref([all_rv_c, hi_rv_c], rrv)
                if len(all_rv_c) >= 3:
                    c_, b_, _ = ax.hist(all_rv_c, bins=20, range=br, alpha=0)
                    ax.step(b_[:-1], c_, where='post', color=cfg.COL_MASTER, lw=1, alpha=0.4, ls='--', zorder=3)
                if len(hi_rv_c) >= 3:
                    _hist_step_kde(ax, hi_rv_c, cfg.COL_HIGHMEM, f'n={len(hi_rv_c)}', br, 20, zorder=8)
                    _add_median_mad_lines(ax, hi_rv_c, cfg.COL_HIGHMEM, zorder=9)
                    _med_mad_box(ax, hi_rv_c, cfg.COL_HIGHMEM, 'upper right')
                _plot_ref_band(ax, rrv, rrve, 'Ref', zorder=20)
                v_sys = r['algo_info'].get('v_sys_recovered', np.nan)
                if np.isfinite(v_sys):
                    ax.axvline(v_sys, color='darkblue', lw=2, ls='-.', alpha=0.8, zorder=15)
            else:
                ax.text(0.5, 0.5, 'No RV', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xlabel('RV (km/s)', fontsize=11, fontweight='bold')
            ax.set_ylabel('N', fontsize=11, fontweight='bold')
            ax.set_title(f'{ot}: {cn}', fontsize=11, fontweight='bold', color=col)
            ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.2, axis='y')
        for j in range(n_this, len(af)): af[j].set_visible(False)
        sfx = f'_p{page_idx+1}' if n_pages > 1 else ''
        fig.suptitle(f'V11 RV Summary (P>{thr:.1f}){epoch_sub}{sfx}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(out_dir, f'SUMMARY_RV{sfx}.{cfg.SAVE_FORMAT}'), dpi=cfg.PLOT_DPI)
        plt.close(); logger.info(f"  Saved: SUMMARY_RV{sfx}")

    # ════════════════════════════════════════════════════════════════════════
    # SERIES 3: PM SUMMARY (with gray background)
    # ════════════════════════════════════════════════════════════════════════
    for page_idx in range(n_pages):
        i_start = page_idx * per_page
        i_end = min(i_start + per_page, n)
        page_sr = sr[i_start:i_end]; n_this = len(page_sr)
        nr = int(np.ceil(n_this / ncols))
        fig, axes = plt.subplots(nr, ncols, figsize=(5 * ncols, 5 * nr), squeeze=False)
        af = axes.flatten()
        for j, r in enumerate(page_sr):
            ax = af[j]; mdf = r['matched_df']; algo = r['algo_info']
            ot = r['obj_type']; cn = r['cluster_name']
            col = tc.get(ot, 'k')
            cols_m = r['mem_cols']
            # Gray background: all members
            mem_df = r['member_df']
            if cols_m.get('pmra') in mem_df.columns and cols_m.get('pmdec') in mem_df.columns:
                pm_ra_all = pd.to_numeric(mem_df[cols_m['pmra']], errors='coerce').values
                pm_dec_all = pd.to_numeric(mem_df[cols_m['pmdec']], errors='coerce').values
                vm = np.isfinite(pm_ra_all) & np.isfinite(pm_dec_all)
                if np.sum(vm) > 0:
                    ax.scatter(pm_ra_all[vm], pm_dec_all[vm],
                               c=cfg.COL_GRAY_BG, s=cfg.GRAY_BG_SIZE,
                               alpha=cfg.GRAY_BG_ALPHA, zorder=1,
                               edgecolors='none', rasterized=True)
            if 'P_mem' in mdf.columns:
                v = mdf['P_mem'].notna()
                ax.scatter(mdf.loc[v, 'pmra'], mdf.loc[v, 'pmdec'],
                           c=mdf.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                           s=15, alpha=0.8, vmin=0, vmax=1, edgecolors='k', lw=0.2, zorder=5)
            if algo.get('Sigma_cluster') and ot not in ['SGR', 'STREAM']:
                mu = algo['mu_cluster']
                sig = np.array(algo['Sigma_cluster'])
                w, v2 = np.linalg.eigh(sig)
                ang = np.degrees(np.arctan2(v2[1, 0], v2[0, 0]))
                ell = Ellipse(xy=mu, width=4*np.sqrt(max(w[0], 0)),
                              height=4*np.sqrt(max(w[1], 0)), angle=ang,
                              fill=False, ec='lime', lw=2.5, zorder=10)
                ax.add_patch(ell)
            ax.set_title(f'{ot}: {cn}', fontsize=11, fontweight='bold', color=col)
            ax.set_xlabel('$\\mu_\\alpha\\cos\\delta$', fontsize=11, fontweight='bold')
            ax.set_ylabel('$\\mu_\\delta$', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.2); ax.set_aspect('equal', 'datalim')
        for j in range(n_this, len(af)): af[j].set_visible(False)
        sfx = f'_p{page_idx+1}' if n_pages > 1 else ''
        fig.suptitle(f'V11 PM Summary{epoch_sub}{sfx}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(out_dir, f'SUMMARY_PM{sfx}.{cfg.SAVE_FORMAT}'), dpi=cfg.PLOT_DPI)
        plt.close(); logger.info(f"  Saved: SUMMARY_PM{sfx}")

    # ════════════════════════════════════════════════════════════════════════
    # SERIES 4: RA/DEC SUMMARY (with gray background)
    # ════════════════════════════════════════════════════════════════════════
    for page_idx in range(n_pages):
        i_start = page_idx * per_page
        i_end = min(i_start + per_page, n)
        page_sr = sr[i_start:i_end]; n_this = len(page_sr)
        nr = int(np.ceil(n_this / ncols))
        fig, axes = plt.subplots(nr, ncols, figsize=(5 * ncols, 5 * nr), squeeze=False)
        af = axes.flatten()
        for j, r in enumerate(page_sr):
            ax = af[j]; mdf = r['matched_df']
            ot = r['obj_type']; cn = r['cluster_name']
            col = tc.get(ot, 'k')
            cols_m = r['mem_cols']
            mem_df = r['member_df']
            if cols_m.get('ra') in mem_df.columns and cols_m.get('dec') in mem_df.columns:
                ra_all = pd.to_numeric(mem_df[cols_m['ra']], errors='coerce').values
                dec_all = pd.to_numeric(mem_df[cols_m['dec']], errors='coerce').values
                vm = np.isfinite(ra_all) & np.isfinite(dec_all)
                if np.sum(vm) > 0:
                    ax.scatter(ra_all[vm], dec_all[vm],
                               c=cfg.COL_GRAY_BG, s=cfg.GRAY_BG_SIZE,
                               alpha=cfg.GRAY_BG_ALPHA, zorder=1,
                               edgecolors='none', rasterized=True)
            if 'P_mem' in mdf.columns:
                v = mdf['P_mem'].notna()
                ax.scatter(mdf.loc[v, 'ra'], mdf.loc[v, 'dec'],
                           c=mdf.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                           s=15, alpha=0.8, vmin=0, vmax=1, edgecolors='k', lw=0.2, zorder=5)
            ax.set_title(f'{ot}: {cn}', fontsize=11, fontweight='bold', color=col)
            ax.set_xlabel('RA (deg)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Dec (deg)', fontsize=11, fontweight='bold')
            ax.invert_xaxis(); ax.grid(True, alpha=0.2)
        for j in range(n_this, len(af)): af[j].set_visible(False)
        sfx = f'_p{page_idx+1}' if n_pages > 1 else ''
        fig.suptitle(f'V11 RA/Dec Summary{epoch_sub}{sfx}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(out_dir, f'SUMMARY_RADEC{sfx}.{cfg.SAVE_FORMAT}'), dpi=cfg.PLOT_DPI)
        plt.close(); logger.info(f"  Saved: SUMMARY_RADEC{sfx}")


# ============================================================================
# V11: GRAND SUMMARY — 4 columns x N rows (publication figure)
# ============================================================================

def generate_grand_summary(all_results, logger, epoch_output_dir=None, epoch_mode='2016'):
    """
    V11: GRAND_SUMMARY — the paper figure.
    4 columns: RA/Dec | PM | Distance | RV
    N rows: one per object with >= GRAND_MIN_GREEN green matches.
    Each panel is square. Full ApJ styling. Gray backgrounds for PM/Sky.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    set_paper_style()

    out_dir = epoch_output_dir or cfg.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    thr = cfg.P_MEM_PLOT_THRESHOLD
    tc  = {'GC': '#006400', 'OC': '#FF8C00', 'DW': '#00008B', 'SGR': '#8B0000'}
    epoch_sub = f'[J2000->2016]' if epoch_mode == '2000' else '[J2016]'

    # Filter: objects with >= GRAND_MIN_GREEN green matches
    gold = []
    for r in all_results:
        if r['matched_df'] is None: continue
        if 'P_mem' not in r['matched_df'].columns: continue
        P = r['matched_df']['P_mem'].values
        n_hi = int(np.sum(np.where(np.isfinite(P), P, 0) > thr))
        if n_hi >= cfg.GRAND_MIN_GREEN:
            r['_n_hi'] = n_hi
            gold.append(r)

    if not gold:
        logger.warning("No objects with >= GRAND_MIN_GREEN green matches for grand summary!")
        return

    gold = _sort_results(gold)
    N = len(gold)
    logger.info(f"\n{'='*70}")
    logger.info(f"GRAND SUMMARY: {N} objects with >= {cfg.GRAND_MIN_GREEN} green matches")
    logger.info(f"{'='*70}")

    # Adaptive layout: 4 columns, N rows
    panel_size = 4.2  # inches per panel (square)
    fig_w = 4 * panel_size + 2.0  # extra for labels/padding
    fig_h = N * panel_size + 2.0

    fig, axes = plt.subplots(N, 4, figsize=(fig_w, fig_h), squeeze=False)

    for row_idx, r in enumerate(gold):
        mdf    = r['matched_df']
        ot     = r['obj_type']
        cn     = r['cluster_name']
        algo   = r['algo_info']
        cols_m = r['mem_cols']
        mem_df = r['member_df']
        rd     = r.get('ref_dist')
        rde    = r.get('ref_dist_err')
        rrv    = r.get('ref_rv')
        rrve   = r.get('ref_rv_err')
        col    = tc.get(ot, 'k')
        diag   = r.get('diagnostics', {})

        # ── Column 0: RA/Dec ──────────────────────────────────────────────
        ax = axes[row_idx, 0]
        if cols_m.get('ra') in mem_df.columns and cols_m.get('dec') in mem_df.columns:
            ra_all = pd.to_numeric(mem_df[cols_m['ra']], errors='coerce').values
            dec_all = pd.to_numeric(mem_df[cols_m['dec']], errors='coerce').values
            vm = np.isfinite(ra_all) & np.isfinite(dec_all)
            if np.sum(vm) > 0:
                ax.scatter(ra_all[vm], dec_all[vm],
                           c=cfg.COL_GRAY_BG, s=cfg.GRAY_BG_SIZE,
                           alpha=cfg.GRAY_BG_ALPHA, zorder=1,
                           edgecolors='none', rasterized=True)
        if 'P_mem' in mdf.columns:
            v = mdf['P_mem'].notna()
            sc = ax.scatter(mdf.loc[v, 'ra'], mdf.loc[v, 'dec'],
                            c=mdf.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                            s=18, edgecolors='k', linewidths=0.2,
                            vmin=0, vmax=1, zorder=5)
        ax.set_xlabel('RA (deg)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dec (deg)', fontsize=12, fontweight='bold')
        ax.set_title(f'{ot}: {cn}', fontsize=12, fontweight='bold', color=col)
        ax.invert_xaxis(); ax.grid(True, alpha=0.2)

        # ── Column 1: PM ─────────────────────────────────────────────────
        ax = axes[row_idx, 1]
        if cols_m.get('pmra') in mem_df.columns and cols_m.get('pmdec') in mem_df.columns:
            pm_ra_all = pd.to_numeric(mem_df[cols_m['pmra']], errors='coerce').values
            pm_dec_all = pd.to_numeric(mem_df[cols_m['pmdec']], errors='coerce').values
            vm = np.isfinite(pm_ra_all) & np.isfinite(pm_dec_all)
            if np.sum(vm) > 0:
                ax.scatter(pm_ra_all[vm], pm_dec_all[vm],
                           c=cfg.COL_GRAY_BG, s=cfg.GRAY_BG_SIZE,
                           alpha=cfg.GRAY_BG_ALPHA, zorder=1,
                           edgecolors='none', rasterized=True)
        if 'P_mem' in mdf.columns:
            v = mdf['P_mem'].notna()
            ax.scatter(mdf.loc[v, 'pmra'], mdf.loc[v, 'pmdec'],
                       c=mdf.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                       s=18, edgecolors='k', linewidths=0.2,
                       vmin=0, vmax=1, zorder=5)
        if algo.get('Sigma_cluster') and ot not in ['SGR', 'STREAM']:
            mu = algo['mu_cluster']
            sig = np.array(algo['Sigma_cluster'])
            w, v2 = np.linalg.eigh(sig)
            ang = np.degrees(np.arctan2(v2[1, 0], v2[0, 0]))
            for ns in [1, 2]:
                ell = Ellipse(xy=mu, width=2*ns*np.sqrt(max(w[0], 0)),
                              height=2*ns*np.sqrt(max(w[1], 0)), angle=ang,
                              fill=False, ec='lime', lw=2.0, ls='-' if ns == 1 else '--',
                              zorder=10)
                ax.add_patch(ell)
            ax.scatter(*mu, marker='x', s=80, c='lime', linewidths=2, zorder=11)
        ax.set_xlabel('$\\mu_\\alpha\\cos\\delta$ (mas/yr)', fontsize=12, fontweight='bold')
        ax.set_ylabel('$\\mu_\\delta$ (mas/yr)', fontsize=12, fontweight='bold')
        ax.set_title('PM', fontsize=12, fontweight='bold', color=col)
        ax.grid(True, alpha=0.2); ax.set_aspect('equal', 'datalim')

        # ── Column 2: Distance (red + blue + green) ──────────────────────
        ax = axes[row_idx, 2]
        hi_dist, all_dist = _filter_by_pmem(mdf, 'best_dist', thr)
        hi_d = _clean(hi_dist); hi_d = hi_d[(hi_d > 0) & (hi_d < 300)]
        all_d = _clean(all_dist); all_d = all_d[(all_d > 0) & (all_d < 300)]
        plx_d = r.get('plx_dist', np.array([]))
        hi_plx = np.array([])
        if len(plx_d) > 0:
            plx_arr = np.asarray(plx_d, dtype=float)
            pmem_v = mdf['P_mem'].values.astype(float)
            mask = (pmem_v >= thr) & np.isfinite(pmem_v) & np.isfinite(plx_arr) & (plx_arr > 0) & (plx_arr < 300)
            hi_plx = plx_arr[mask]
        hi_plx_c = _clean(hi_plx) if len(hi_plx) > 0 else np.array([])

        br = _bins_range_with_ref([all_d, hi_d, hi_plx_c], rd, positive_only=True)
        if len(all_d) >= 3:
            c_, b_, _ = ax.hist(all_d, bins=20, range=br, alpha=0)
            ax.step(b_[:-1], c_, where='post', color=cfg.COL_MASTER, lw=1, alpha=0.4, ls='--', zorder=3)
        if len(hi_plx_c) >= 3:
            _hist_step_kde(ax, hi_plx_c, cfg.COL_MEMBER, f'Plx n={len(hi_plx_c)}', br, 20, zorder=6)
            _add_median_mad_lines(ax, hi_plx_c, cfg.COL_MEMBER, zorder=7)
        if len(hi_d) >= 3:
            _hist_step_kde(ax, hi_d, cfg.COL_HIGHMEM, f'n={len(hi_d)}', br, 20, zorder=8)
            _add_median_mad_lines(ax, hi_d, cfg.COL_HIGHMEM, zorder=9)
            _med_mad_box(ax, hi_d, cfg.COL_HIGHMEM, 'upper right')
        _plot_ref_band(ax, rd, rde, 'Ref', zorder=20)
        ax.set_xlabel('Dist (kpc)', fontsize=12, fontweight='bold')
        ax.set_ylabel('N', fontsize=12, fontweight='bold')
        ax.set_title('Distance', fontsize=12, fontweight='bold', color=col)
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.2, axis='y')

        # ── Column 3: RV ─────────────────────────────────────────────────
        ax = axes[row_idx, 3]
        hi_rv_arr, all_rv_arr = _filter_by_pmem(mdf, 'best_rv', thr)
        hi_rv_c = _clean(hi_rv_arr); all_rv_c = _clean(all_rv_arr)
        if len(hi_rv_c) >= 3 or len(all_rv_c) >= 3:
            br = _bins_range_with_ref([all_rv_c, hi_rv_c], rrv)
            if len(all_rv_c) >= 3:
                c_, b_, _ = ax.hist(all_rv_c, bins=20, range=br, alpha=0)
                ax.step(b_[:-1], c_, where='post', color=cfg.COL_MASTER, lw=1, alpha=0.4, ls='--', zorder=3)
            if len(hi_rv_c) >= 3:
                _hist_step_kde(ax, hi_rv_c, cfg.COL_HIGHMEM, f'n={len(hi_rv_c)}', br, 20, zorder=8)
                _add_median_mad_lines(ax, hi_rv_c, cfg.COL_HIGHMEM, zorder=9)
                _med_mad_box(ax, hi_rv_c, cfg.COL_HIGHMEM, 'upper right')
            _plot_ref_band(ax, rrv, rrve, 'Ref', zorder=20)
            v_sys = algo.get('v_sys_recovered', np.nan)
            if np.isfinite(v_sys):
                ax.axvline(v_sys, color='darkblue', lw=2, ls='-.', alpha=0.8, zorder=15)
        else:
            ax.text(0.5, 0.5, 'No RV', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_xlabel('RV (km/s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('N', fontsize=12, fontweight='bold')
        ax.set_title('RV', fontsize=12, fontweight='bold', color=col)
        ax.legend(fontsize=7, loc='best'); ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle(f'GRAND SUMMARY — V11 Membership (P>{thr:.1f}, n>={cfg.GRAND_MIN_GREEN})  {epoch_sub}',
                 fontsize=18, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(out_dir, f'GRAND_SUMMARY.{cfg.SAVE_FORMAT}')
    plt.savefig(out_path, dpi=cfg.PLOT_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: GRAND_SUMMARY ({N} rows x 4 cols)")
    return out_path


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_result_catalog(result, save_dir, logger):
    """Save matched DataFrame with P_mem to CSV."""
    if result['matched_df'] is None:
        return None
    mdf = result['matched_df']
    cn  = str(result['cluster_name']).replace(' ', '_').replace('/', '_')
    ot  = result['obj_type']
    fname = f"{ot}_{cn}_membership.csv"
    out_path = os.path.join(save_dir, fname)
    cols_to_save = [c for c in mdf.columns
                    if not c.endswith('_master') or c in [
                        'best_rv_master', 'best_dist_master',
                        'Gmag_master', 'BP-RP_master',
                        'feh_master', 'logg_master',
                        'RUWE_master', 'plx_from_params_master']]
    for c in ['P_mem', 'xmatch_sep_arcsec', 'plx_dist',
              'pmra_err_adj', 'pmdec_err_adj', 'rv_err_adj']:
        if c in mdf.columns and c not in cols_to_save:
            cols_to_save.append(c)
    cols_to_save = [c for c in cols_to_save if c in mdf.columns]
    mdf[cols_to_save].to_csv(out_path, index=False)
    logger.info(f"    Saved: {fname} ({len(mdf)} rows)")
    return out_path


def save_result_json(result, save_dir, logger):
    """Save algorithm info + diagnostics to JSON."""
    cn  = str(result['cluster_name']).replace(' ', '_').replace('/', '_')
    ot  = result['obj_type']
    fname = f"{ot}_{cn}_info.json"
    out_path = os.path.join(save_dir, fname)
    info = {
        'cluster_name': result['cluster_name'],
        'obj_type': ot,
        'n_members': result['n_members'],
        'n_matched': result['n_matched'],
        'ref_dist': result.get('ref_dist'),
        'ref_dist_err': result.get('ref_dist_err'),
        'ref_rv': result.get('ref_rv'),
        'ref_rv_err': result.get('ref_rv_err'),
        'algo_info': _json_safe(result.get('algo_info', {})),
        'diagnostics': _json_safe(result.get('diagnostics', {})),
    }
    with open(out_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    return out_path


def _json_safe(d):
    """Make dict JSON-serialisable (convert numpy types)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _json_safe(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                       for x in v]
        else:
            out[k] = v
    return out


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_analysis(args, logger):
    """Main analysis pipeline."""
    t_start = time.time()

    # ── Configure ──────────────────────────────────────────────────────────
    cfg.MASTER_CATALOG   = args.master_catalog
    cfg.GC_MEMBERS_FILE  = args.gc_members
    cfg.OC_MEMBERS_FILE  = args.oc_members
    cfg.SGR_MEMBERS_FILE = args.sgr_members
    cfg.DWG_MEMBERS_FILE = args.dwg_members
    cfg.GC_DIST_FILE     = args.gc_dist
    cfg.OUTPUT_DIR       = args.output_dir
    cfg.CHECKPOINT_DIR   = args.checkpoint_dir

    if args.xmatch_radius is not None:
        cfg.CROSSMATCH_RADIUS_ARCSEC = args.xmatch_radius
    if args.pmem_threshold is not None:
        cfg.P_MEM_PLOT_THRESHOLD = args.pmem_threshold

    epoch_modes = [args.epoch] if args.epoch != 'both' else ['2016', '2000']

    # ── Load master ────────────────────────────────────────────────────────
    master = MasterCatalog(logger)
    if not master.load(cfg.MASTER_CATALOG, cfg.CHECKPOINT_DIR):
        logger.error("FATAL: Master catalog failed to load.")
        sys.exit(1)

    # ── Load reference distances ───────────────────────────────────────────
    gc_dists = load_gc_reference_distances(cfg.GC_DIST_FILE, logger) if cfg.GC_DIST_FILE else {}

    # ── Per-epoch loop ─────────────────────────────────────────────────────
    for epoch_mode in epoch_modes:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# EPOCH MODE: {epoch_mode}")
        logger.info(f"{'#'*70}")

        epoch_sub = f'epoch{epoch_mode}'
        epoch_output_dir = os.path.join(cfg.OUTPUT_DIR, epoch_sub)
        os.makedirs(epoch_output_dir, exist_ok=True)
        indiv_dir = os.path.join(epoch_output_dir, 'individual')
        os.makedirs(indiv_dir, exist_ok=True)
        cats_dir = os.path.join(epoch_output_dir, 'catalogs')
        os.makedirs(cats_dir, exist_ok=True)

        all_results = []

        # ── GC ─────────────────────────────────────────────────────────
        gc_results = process_gc_members(
            master, gc_dists, logger, epoch_mode, epoch_output_dir)
        for r in gc_results:
            if r['matched_df'] is not None:
                prematch_ra   = pd.to_numeric(r['member_df'][r['mem_cols']['ra']],   errors='coerce').values
                prematch_dec  = pd.to_numeric(r['member_df'][r['mem_cols']['dec']],  errors='coerce').values
                prematch_pmra = pd.to_numeric(r['member_df'][r['mem_cols']['pmra']], errors='coerce').values
                prematch_pmdec = pd.to_numeric(r['member_df'][r['mem_cols']['pmdec']], errors='coerce').values
                plot_individual_panels(
                    r['matched_df'], r['cluster_name'], r['obj_type'],
                    r['algo_info'],
                    ref_dist=r.get('ref_dist'), ref_dist_err=r.get('ref_dist_err'),
                    ref_rv=r.get('ref_rv'), ref_rv_err=r.get('ref_rv_err'),
                    master_dist=r.get('master_dist'), member_dist=r.get('member_dist'),
                    master_rv=r.get('master_rv'), member_rv=r.get('member_rv'),
                    plx_dist=r.get('plx_dist'),
                    prematch_ra=prematch_ra, prematch_dec=prematch_dec,
                    prematch_pmra=prematch_pmra, prematch_pmdec=prematch_pmdec,
                    save_dir=indiv_dir, epoch_mode=epoch_mode,
                    diagnostics=r.get('diagnostics'))
                save_result_catalog(r, cats_dir, logger)
                save_result_json(r, cats_dir, logger)
        all_results.extend(gc_results)

        # ── OC ─────────────────────────────────────────────────────────
        oc_results = process_oc_members(
            master, logger, epoch_mode, epoch_output_dir)
        for r in oc_results:
            if r['matched_df'] is not None:
                prematch_ra   = pd.to_numeric(r['member_df'][r['mem_cols']['ra']],   errors='coerce').values
                prematch_dec  = pd.to_numeric(r['member_df'][r['mem_cols']['dec']],  errors='coerce').values
                prematch_pmra = pd.to_numeric(r['member_df'][r['mem_cols']['pmra']], errors='coerce').values
                prematch_pmdec = pd.to_numeric(r['member_df'][r['mem_cols']['pmdec']], errors='coerce').values
                plot_individual_panels(
                    r['matched_df'], r['cluster_name'], r['obj_type'],
                    r['algo_info'],
                    ref_dist=r.get('ref_dist'), ref_dist_err=r.get('ref_dist_err'),
                    ref_rv=r.get('ref_rv'), ref_rv_err=r.get('ref_rv_err'),
                    master_dist=r.get('master_dist'), member_dist=r.get('member_dist'),
                    master_rv=r.get('master_rv'), member_rv=r.get('member_rv'),
                    plx_dist=r.get('plx_dist'),
                    prematch_ra=prematch_ra, prematch_dec=prematch_dec,
                    prematch_pmra=prematch_pmra, prematch_pmdec=prematch_pmdec,
                    save_dir=indiv_dir, epoch_mode=epoch_mode,
                    diagnostics=r.get('diagnostics'))
                save_result_catalog(r, cats_dir, logger)
                save_result_json(r, cats_dir, logger)
        all_results.extend(oc_results)

        # ── SGR ────────────────────────────────────────────────────────
        sgr_results = process_sgr_members(
            master, logger, epoch_mode, epoch_output_dir)
        for r in sgr_results:
            if r['matched_df'] is not None:
                prematch_ra   = pd.to_numeric(r['member_df'][r['mem_cols']['ra']],   errors='coerce').values
                prematch_dec  = pd.to_numeric(r['member_df'][r['mem_cols']['dec']],  errors='coerce').values
                prematch_pmra = pd.to_numeric(r['member_df'][r['mem_cols']['pmra']], errors='coerce').values
                prematch_pmdec = pd.to_numeric(r['member_df'][r['mem_cols']['pmdec']], errors='coerce').values
                plot_individual_panels(
                    r['matched_df'], r['cluster_name'], r['obj_type'],
                    r['algo_info'],
                    ref_dist=r.get('ref_dist'), ref_dist_err=r.get('ref_dist_err'),
                    ref_rv=r.get('ref_rv'), ref_rv_err=r.get('ref_rv_err'),
                    master_dist=r.get('master_dist'), member_dist=r.get('member_dist'),
                    master_rv=r.get('master_rv'), member_rv=r.get('member_rv'),
                    plx_dist=r.get('plx_dist'),
                    prematch_ra=prematch_ra, prematch_dec=prematch_dec,
                    prematch_pmra=prematch_pmra, prematch_pmdec=prematch_pmdec,
                    save_dir=indiv_dir, epoch_mode=epoch_mode,
                    diagnostics=r.get('diagnostics'))
                save_result_catalog(r, cats_dir, logger)
                save_result_json(r, cats_dir, logger)
        all_results.extend(sgr_results)

        # ── DWG ────────────────────────────────────────────────────────
        dwg_results = process_dwg_members(
            master, logger, epoch_mode, epoch_output_dir)
        for r in dwg_results:
            if r['matched_df'] is not None:
                prematch_ra   = pd.to_numeric(r['member_df'][r['mem_cols']['ra']],   errors='coerce').values
                prematch_dec  = pd.to_numeric(r['member_df'][r['mem_cols']['dec']],  errors='coerce').values
                prematch_pmra = pd.to_numeric(r['member_df'][r['mem_cols']['pmra']], errors='coerce').values
                prematch_pmdec = pd.to_numeric(r['member_df'][r['mem_cols']['pmdec']], errors='coerce').values
                n_plx_rm = r['algo_info'].get('n_parallax_removed', 0)
                plot_individual_panels(
                    r['matched_df'], r['cluster_name'], r['obj_type'],
                    r['algo_info'],
                    ref_dist=r.get('ref_dist'), ref_dist_err=r.get('ref_dist_err'),
                    ref_rv=r.get('ref_rv'), ref_rv_err=r.get('ref_rv_err'),
                    master_dist=r.get('master_dist'), member_dist=r.get('member_dist'),
                    master_rv=r.get('master_rv'), member_rv=r.get('member_rv'),
                    plx_dist=r.get('plx_dist'),
                    prematch_ra=prematch_ra, prematch_dec=prematch_dec,
                    prematch_pmra=prematch_pmra, prematch_pmdec=prematch_pmdec,
                    save_dir=indiv_dir, epoch_mode=epoch_mode,
                    n_parallax_removed=n_plx_rm,
                    diagnostics=r.get('diagnostics'))
                save_result_catalog(r, cats_dir, logger)
                save_result_json(r, cats_dir, logger)
        all_results.extend(dwg_results)

        # ── Summaries ──────────────────────────────────────────────────
        if all_results:
            generate_summary_series(
                all_results, logger, epoch_output_dir, epoch_mode)
            generate_grand_summary(
                all_results, logger, epoch_output_dir, epoch_mode)

        # ── Summary statistics log ─────────────────────────────────────
        n_gc  = len([r for r in gc_results  if r['matched_df'] is not None])
        n_oc  = len([r for r in oc_results  if r['matched_df'] is not None])
        n_sgr = len([r for r in sgr_results if r['matched_df'] is not None])
        n_dw  = len([r for r in dwg_results if r['matched_df'] is not None])
        logger.info(f"\n{'='*70}")
        logger.info(f"EPOCH {epoch_mode} COMPLETE")
        logger.info(f"  GC: {n_gc} | OC: {n_oc} | SGR: {n_sgr} | DW: {n_dw}")
        logger.info(f"  Output: {epoch_output_dir}")
        logger.info(f"{'='*70}")

    t_total = time.time() - t_start
    logger.info(f"\nTotal runtime: {t_total:.1f}s ({t_total/60:.1f} min)")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Adaptive Membership V11 — Full Phase-Space + Publication Plots',
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--master-catalog', '-m', required=True,
                   help='Master catalog path (FITS or directory)')
    p.add_argument('--gc-members',  default=None,
                   help='GC member catalog CSV')
    p.add_argument('--oc-members',  default=None,
                   help='OC member catalog CSV')
    p.add_argument('--sgr-members', default=None,
                   help='SGR stream member catalog CSV')
    p.add_argument('--dwg-members', default=None,
                   help='Dwarf galaxy member catalog CSV')
    p.add_argument('--gc-dist',     default=None,
                   help='GC reference distances CSV')
    p.add_argument('--output-dir',  '-o', default='./outputs',
                   help='Output directory')
    p.add_argument('--checkpoint-dir', default='./checkpoints',
                   help='Checkpoint directory')
    p.add_argument('--epoch', choices=['2016', '2000', 'both'], default='2016',
                   help='Epoch mode')
    p.add_argument('--xmatch-radius', type=float, default=None,
                   help='Cross-match radius (arcsec)')
    p.add_argument('--pmem-threshold', type=float, default=None,
                   help='P_mem threshold for "high" membership')
    p.add_argument('--log-file', default='adaptive_membership_v11.log',
                   help='Log file')
    return p.parse_args()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    args   = parse_args()
    logger = setup_logging(args.log_file)
    logger.info("=" * 70)
    logger.info("ADAPTIVE MEMBERSHIP ANALYSIS V11")
    logger.info(f"  Started: {datetime.now().isoformat()}")
    logger.info(f"  Master:  {args.master_catalog}")
    logger.info(f"  GC:      {args.gc_members}")
    logger.info(f"  OC:      {args.oc_members}")
    logger.info(f"  SGR:     {args.sgr_members}")
    logger.info(f"  DWG:     {args.dwg_members}")
    logger.info(f"  Epoch:   {args.epoch}")
    logger.info(f"  Output:  {args.output_dir}")
    logger.info("=" * 70)
    run_analysis(args, logger)
    logger.info(f"\nFinished: {datetime.now().isoformat()}")