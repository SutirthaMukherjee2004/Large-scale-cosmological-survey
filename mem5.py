#!/usr/bin/env python3
"""
================================================================================
ADAPTIVE MEMBERSHIP ANALYSIS V7 — P_MEM-FILTERED DISTANCE & RV PANELS
================================================================================

V7 CHANGES over V6:
 1. Distance & RV histograms now plot ONLY high-probability members (P_mem > threshold)
    instead of all crossmatched stars. This fixes the systematic distance underestimate
    caused by foreground MW contamination dominating the unfiltered sample.
 2. Both individual and summary plots apply this filtering.
 3. "Master (all)" shown as faint background; "Members (P>thr)" as prominent histogram.
 4. Added P_MEM_PLOT_THRESHOLD config (default 0.5) — separate from P_MEM_HIGH (0.8)
    to allow a moderate cut that retains more stars while removing clear contaminants.
 5. Member catalog distances (1/plx) also filtered to only include stars that passed
    the P_mem cut, ensuring apples-to-apples comparison.

Author: Sutirtha
================================================================================
"""

import os, sys, json, time, logging, argparse, warnings, glob
import gc as gcmod
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal, gaussian_kde, median_abs_deviation
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file=None, level=logging.INFO):
    logger = logging.getLogger('AdaptiveMembership')
    logger.setLevel(level); logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MASTER_CATALOG = None
    GC_MEMBERS_FILE = None; OC_MEMBERS_FILE = None
    SGR_MEMBERS_FILE = None; DWG_MEMBERS_FILE = None
    GC_DIST_FILE = None
    OUTPUT_DIR = './outputs'; CHECKPOINT_DIR = './checkpoints'

    # ---- V6 column mapping for Entire_catalogue_chunk*.fits ----
    MASTER_COLS = {
        'ra': 'RA_final', 'dec': 'DEC_final',
        'pmra': 'pmra_final', 'pmdec': 'pmdec_final',
        'pmra_err': 'pmra_err_final', 'pmdec_err': 'pmdec_err_final',
        'dist': 'distance_final', 'dist_err': 'distance_err_final',
        'rv': 'Weighted_Avg_final', 'rv_err': 'Weighted_Avg_err_final',
        'parallax': 'parallax_final', 'parallax_err': 'parallax_err_final',
        'params_est': 'stellar_params_est',
        'params_err': 'stellar_params_err',
    }
    ALT_MASTER_COLS = {
        'rv':     ['Weighted_Avg_final', 'ZP_final', 'RV_final',
                   'Weighted_Avg', 'radial_velocity', 'RV'],
        'rv_err': ['Weighted_Avg_err_final', 'ZP_err_final', 'RV_err_final',
                   'radial_velocity_error', 'RV_err'],
    }

    GC_MEM_COLS = {
        'key': 'source', 'ra': 'ra', 'dec': 'dec',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmra_error', 'pmdec_err': 'pmdec_error',
        'pmra_pmdec_corr': 'pmra_pmdec_corr',
        'parallax': 'parallax',
        'membership_prob': 'membership_probability',
        'rv': 'RV_weighted_avg', 'rv_err': 'e_RV_weighted_avg',
    }
    GC_DIST_COLS = {
        'name': 'Name', 'lit_dist': 'Lit. dist. (kpc)',
        'lit_dist_err': 'Lit. dist. Err+',
        'mean_dist': 'Mean distance (kpc)',
        'mean_dist_err': 'Mean distance Err+',
    }
    OC_MEM_COLS = {
        'key': 'Cluster', 'ra': 'RAdeg', 'dec': 'DEdeg',
        'pmra': 'pmRA', 'pmdec': 'pmDE',
        'pmra_err': 'e_pmRA', 'pmdec_err': 'e_pmDE',
        'pmra_pmdec_corr': 'pmRApmDEcor',
        'parallax': 'Plx', 'membership_prob': 'Proba',
        'rv': 'RV', 'rv_err': 'e_RV',
    }
    SGR_MEM_COLS = {
        'key': None, 'ra': 'ra', 'dec': 'dec',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmraerr', 'pmdec_err': 'pmdecerr',
        'parallax': 'parallax',
        'dist': 'dist', 'dist_err': 'disterr',
        'rv': 'vlos', 'rv_err': 'vloserr',
    }
    DWG_MEM_COLS = {
        'key': 'name',
        'ra': 'RA_master', 'dec': 'Dec_master',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmra_error', 'pmdec_err': 'pmdec_error',
        'distance': 'distance', 'distance_err': 'distance_error',
        'rhalf': 'rhalf',
        'rv': 'RV_km_s', 'rv_err': 'e_RV_km_s',
        'rv_ref': 'vlos_systemic', 'rv_ref_err': 'vlos_systemic_error',
    }

    CROSSMATCH_RADIUS_ARCSEC = 1.0
    GMM_N_COMPONENTS = 2; GMM_MAX_ITER = 300; GMM_N_INIT = 10; GMM_RANDOM_STATE = 42
    DBSCAN_EPS_CLEANUP = 0.3; DBSCAN_MIN_SAMPLES_CLEANUP = 3
    DBSCAN_EPS_OC = 0.25; DBSCAN_MIN_SAMPLES_OC = 5
    DBSCAN_EPS_STREAM = 0.4; DBSCAN_MIN_SAMPLES_STREAM = 3
    P_MEM_HIGH = 0.8; P_MEM_LOW = 0.2; MIN_STARS_FOR_ANALYSIS = 10
    SGR_BIN_START_KPC = 15.0; SGR_BIN_WIDTH_KPC = 10.0; SGR_MIN_STARS_PER_BIN = 5
    MIN_SUMMARY_MATCH = 5

    # V7: Threshold for plotting distance/RV panels
    # Using 0.5 as default — keeps probable members, removes clear contaminants
    # Set to 0.8 for stricter filtering
    P_MEM_PLOT_THRESHOLD = 0.5

    CMAP_PMEM = 'RdYlGn'; PLOT_DPI = 150; SAVE_FORMAT = 'png'
    COL_MASTER = '#DC143C'   # crimson-red  (now: all matched, faint background)
    COL_MEMBER = '#000080'   # navy blue    (now: member catalog)
    COL_HIGHMEM = '#006400'  # dark green   (V7: high-P_mem filtered)

cfg = Config()

try:
    import hdbscan; HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

def normalize_name(name):
    import re
    if not isinstance(name, str): name = str(name)
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())

normalize_cluster_name = normalize_name

def _safe_float(row, col):
    if col is None: return None
    v = row.get(col)
    if v is None or pd.isna(v): return None
    try: return float(v)
    except: return None

# ============================================================================
# APJ STYLE
# ============================================================================

def set_paper_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 14,
        'axes.labelsize': 16, 'axes.titlesize': 18,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'axes.linewidth': 1.8, 'axes.edgecolor': 'black',
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'xtick.major.size': 7, 'ytick.major.size': 7,
        'xtick.minor.size': 4, 'ytick.minor.size': 4,
        'xtick.major.width': 1.4, 'ytick.major.width': 1.4,
        'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'legend.fontsize': 11, 'legend.framealpha': 0.85,
        'legend.edgecolor': 'black', 'legend.fancybox': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.bbox': 'tight',
        'figure.dpi': 150, 'savefig.dpi': 300, 'text.usetex': False,
    })


# ============================================================================
# MASTER CATALOG — V6: CHUNK LOADING (unchanged)
# ============================================================================

class MasterCatalog:
    def __init__(self, logger):
        self.logger = logger; self.df = None; self.tree = None
        self.coords_3d = None; self.max_chord = None; self.nrows = 0

    def _find_col(self, key, available_cols):
        primary = cfg.MASTER_COLS.get(key)
        if primary and primary in available_cols: return primary
        for alt in cfg.ALT_MASTER_COLS.get(key, []):
            if alt in available_cols: return alt
        return None

    def load(self, filepath, checkpoint_dir=None):
        from astropy.io import fits
        self.logger.info("=" * 70)
        self.logger.info("LOADING MASTER CATALOG (V7)")
        self.logger.info("=" * 70)

        tree_cp = os.path.join(checkpoint_dir, 'master_tree_v6.npz') if checkpoint_dir else None
        data_cp = os.path.join(checkpoint_dir, 'master_data_v6.parquet') if checkpoint_dir else None
        if tree_cp and os.path.exists(tree_cp) and data_cp and os.path.exists(data_cp):
            self.logger.info("Loading from V6 checkpoint...")
            try:
                self.df = pd.read_parquet(data_cp)
                self.coords_3d = np.load(tree_cp)['coords']
                self.tree = cKDTree(self.coords_3d)
                self.nrows = len(self.df); self._compute_max_chord()
                n_rv = self.df['best_rv'].notna().sum() if 'best_rv' in self.df.columns else 0
                self.logger.info(f"✓ Checkpoint: {self.nrows:,} rows ({n_rv:,} with RV)")
                return True
            except Exception as e:
                self.logger.warning(f"Checkpoint failed: {e}")

        files = self._resolve_files(filepath)
        if not files:
            self.logger.error(f"No FITS files found for: {filepath}")
            return False

        self.logger.info(f"Loading {len(files)} file(s)...")
        t0 = time.time()
        all_dfs = []

        for fi, fpath in enumerate(files):
            self.logger.info(f"  [{fi+1}/{len(files)}] {os.path.basename(fpath)}")
            try:
                with fits.open(fpath, memmap=True) as hdul:
                    data_hdu = None
                    for hdu in hdul:
                        if hasattr(hdu, 'columns') and hdu.columns is not None:
                            data_hdu = hdu; break
                    if data_hdu is None:
                        self.logger.warning(f"    No data table — skipping"); continue

                    col_names = [c.name for c in data_hdu.columns]
                    nchunk = data_hdu.data.shape[0]
                    chunk = {}

                    ra_col = cfg.MASTER_COLS['ra']
                    dec_col = cfg.MASTER_COLS['dec']
                    if ra_col in col_names:
                        chunk[ra_col] = np.array(data_hdu.data[ra_col], dtype=np.float64)
                    else:
                        self.logger.warning(f"    Missing {ra_col}"); continue
                    if dec_col in col_names:
                        chunk[dec_col] = np.array(data_hdu.data[dec_col], dtype=np.float64)
                    else:
                        self.logger.warning(f"    Missing {dec_col}"); continue

                    for key in ['pmra', 'pmdec', 'pmra_err', 'pmdec_err']:
                        col = cfg.MASTER_COLS.get(key)
                        if col and col in col_names:
                            chunk[col] = np.array(data_hdu.data[col], dtype=np.float64)

                    dist_col = cfg.MASTER_COLS['dist']
                    if dist_col in col_names:
                        chunk['best_dist'] = np.array(data_hdu.data[dist_col], dtype=np.float64)
                    else:
                        chunk['best_dist'] = np.full(nchunk, np.nan)

                    dist_err_col = cfg.MASTER_COLS.get('dist_err')
                    if dist_err_col and dist_err_col in col_names:
                        chunk['best_dist_err'] = np.array(data_hdu.data[dist_err_col], dtype=np.float64)

                    rv_col = self._find_col('rv', col_names)
                    rv_err_col = self._find_col('rv_err', col_names)
                    if rv_col:
                        chunk['best_rv'] = np.array(data_hdu.data[rv_col], dtype=np.float64)
                        if fi == 0:
                            self.logger.info(f"    RV from: {rv_col}")
                    else:
                        chunk['best_rv'] = np.full(nchunk, np.nan)
                    chunk['best_rv_err'] = (np.array(data_hdu.data[rv_err_col], dtype=np.float64)
                                            if rv_err_col else np.full(nchunk, np.nan))

                    plx_col = cfg.MASTER_COLS.get('parallax', 'parallax_final')
                    plx_err_col = cfg.MASTER_COLS.get('parallax_err', 'parallax_err_final')
                    if plx_col in col_names:
                        chunk['plx_from_params'] = np.array(data_hdu.data[plx_col], dtype=np.float64)
                    else:
                        pe_col = cfg.MASTER_COLS.get('params_est')
                        if pe_col and pe_col in col_names:
                            try:
                                chunk['plx_from_params'] = data_hdu.data[pe_col][:, 4].copy().astype(np.float64)
                            except:
                                chunk['plx_from_params'] = np.full(nchunk, np.nan)
                        else:
                            chunk['plx_from_params'] = np.full(nchunk, np.nan)

                    if plx_err_col in col_names:
                        chunk['plx_err_from_params'] = np.array(data_hdu.data[plx_err_col], dtype=np.float64)
                    else:
                        pe_err_col = cfg.MASTER_COLS.get('params_err')
                        if pe_err_col and pe_err_col in col_names:
                            try:
                                chunk['plx_err_from_params'] = data_hdu.data[pe_err_col][:, 4].copy().astype(np.float64)
                            except:
                                chunk['plx_err_from_params'] = np.full(nchunk, np.nan)
                        else:
                            chunk['plx_err_from_params'] = np.full(nchunk, np.nan)

                    cdf = pd.DataFrame(chunk)
                    cdf = cdf.dropna(subset=[ra_col, dec_col])
                    all_dfs.append(cdf)
                    self.logger.info(f"    → {len(cdf):,} rows")

            except Exception as e:
                self.logger.warning(f"    [ERROR] {e}")
                import traceback; traceback.print_exc()
                continue

        if not all_dfs:
            self.logger.error("No data loaded!"); return False

        self.df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs; gcmod.collect()
        self.nrows = len(self.df)
        n_rv = np.sum(np.isfinite(self.df['best_rv'].values))
        n_dist = np.sum(np.isfinite(self.df['best_dist'].values))
        self.logger.info(f"  Combined: {self.nrows:,} rows | "
                         f"{n_rv:,} with RV | {n_dist:,} with dist")

        self._build_kdtree()

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.df.to_parquet(data_cp)
            np.savez_compressed(tree_cp, coords=self.coords_3d)
            self.logger.info("  Checkpoint saved")

        self.logger.info(f"✓ Master loaded in {time.time()-t0:.1f}s")
        gcmod.collect(); return True

    def _resolve_files(self, filepath):
        if '*' in filepath or '?' in filepath:
            return sorted(glob.glob(filepath))
        if os.path.isdir(filepath):
            pattern = os.path.join(filepath, 'Entire_catalogue_chunk*.fits')
            found = sorted(glob.glob(pattern))
            if found: return found
            return sorted(glob.glob(os.path.join(filepath, '*.fits')))
        if os.path.exists(filepath):
            return [filepath]
        return []

    def _build_kdtree(self):
        ra = np.radians(self.df[cfg.MASTER_COLS['ra']].values)
        dec = np.radians(self.df[cfg.MASTER_COLS['dec']].values)
        self.coords_3d = np.column_stack([
            np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
        self.tree = cKDTree(self.coords_3d); self._compute_max_chord()

    def _compute_max_chord(self):
        self.max_chord = 2 * np.sin(np.radians(cfg.CROSSMATCH_RADIUS_ARCSEC / 3600.0) / 2)

    def query(self, ra, dec):
        rr, dr = np.radians(ra), np.radians(dec)
        c = np.column_stack([np.cos(dr)*np.cos(rr), np.cos(dr)*np.sin(rr), np.sin(dr)])
        d, i = self.tree.query(c, k=1, distance_upper_bound=self.max_chord)
        v = np.isfinite(d)
        return i[v], np.where(v)[0], np.degrees(2*np.arcsin(d[v]/2))*3600

    def get_matched_data(self, master_idx):
        return self.df.iloc[master_idx].reset_index(drop=True)


# ============================================================================
# ALGORITHMS  (unchanged from V5/V6)
# ============================================================================

def build_measurement_covariance(pmra_err, pmdec_err, corr=None):
    n = len(pmra_err); C = np.zeros((n, 2, 2))
    C[:, 0, 0] = pmra_err**2; C[:, 1, 1] = pmdec_err**2
    if corr is not None:
        cv = corr * pmra_err * pmdec_err; C[:, 0, 1] = cv; C[:, 1, 0] = cv
    return C

def algorithm_gmm_with_errors(pmra, pmdec, cp, cd, pe=None, pde=None, corr=None, ms=10):
    n = len(pmra); vm = ~(np.isnan(pmra)|np.isnan(pmdec)); nv = np.sum(vm)
    if nv < ms: return np.full(n, 0.5), {'status':'insufficient_data','algorithm':'GMM'}
    X = np.column_stack([pmra[vm], pmdec[vm]])
    try:
        gmm = GaussianMixture(n_components=cfg.GMM_N_COMPONENTS, max_iter=cfg.GMM_MAX_ITER,
                              n_init=cfg.GMM_N_INIT, random_state=cfg.GMM_RANDOM_STATE,
                              covariance_type='full')
        gmm.fit(X)
        ref = np.array([cp, cd])
        ci = np.argmin([np.linalg.norm(gmm.means_[i]-ref) for i in range(cfg.GMM_N_COMPONENTS)])
        fi = 1 - ci
        mc, Sc = gmm.means_[ci], gmm.covariances_[ci]
        mf, Sf = gmm.means_[fi], gmm.covariances_[fi]
        eta = gmm.weights_[ci]
        P = np.zeros(nv)
        if pe is not None and pde is not None:
            C = build_measurement_covariance(pe[vm], pde[vm],
                                             corr[vm] if corr is not None else None)
            for i in range(nv):
                try:
                    pc = eta * multivariate_normal.pdf(X[i], mc, Sc+C[i])
                    pf = (1-eta) * multivariate_normal.pdf(X[i], mf, Sf+C[i])
                    P[i] = pc/(pc+pf) if (pc+pf) > 0 else 0.5
                except: P[i] = 0.5
        else:
            P = gmm.predict_proba(X)[:, ci]
        Pm = np.full(n, np.nan); Pm[vm] = P
        return Pm, {'status':'success','algorithm':'GMM',
                    'mu_cluster':mc.tolist(),'Sigma_cluster':Sc.tolist(),
                    'mu_field':mf.tolist(),'Sigma_field':Sf.tolist(),
                    'eta':float(eta),
                    'pm_dispersion':float(np.sqrt(np.trace(Sc))),
                    'center_pmra':cp,'center_pmdec':cd}
    except Exception as e:
        return np.full(n, 0.5), {'status':f'error: {e}','algorithm':'GMM'}

def algorithm_dbscan(pmra, pmdec, cp, cd, eps=0.25, minsamp=5, use_h=False, ms=10):
    n = len(pmra); vm = ~(np.isnan(pmra)|np.isnan(pmdec)); nv = np.sum(vm)
    if nv < ms: return np.full(n, 0.5), {'status':'insufficient_data','algorithm':'DBSCAN'}
    X = np.column_stack([pmra[vm], pmdec[vm]])
    sc = StandardScaler(); Xs = sc.fit_transform(X); rs = sc.transform([[cp, cd]])[0]
    try:
        if use_h and HAS_HDBSCAN:
            cl = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
            labels = cl.fit_predict(Xs); an = 'HDBSCAN'
        else:
            cl = DBSCAN(eps=eps, min_samples=minsamp)
            labels = cl.fit_predict(Xs); an = 'DBSCAN'
        ul = set(labels) - {-1}
        if len(ul) == 0:
            d = np.linalg.norm(Xs-rs, axis=1); P = np.exp(-d**2/2)
            Pm = np.full(n, np.nan); Pm[vm] = P
            return Pm, {'status':'no_cluster_found','algorithm':an,
                        'center_pmra':cp,'center_pmdec':cd}
        bc = min(ul, key=lambda l: np.linalg.norm(Xs[labels==l].mean(axis=0)-rs))
        cm = labels == bc; P = np.zeros(nv)
        if use_h and HAS_HDBSCAN and hasattr(cl, 'probabilities_'):
            P[cm] = cl.probabilities_[cm]
        else:
            P[cm] = 1.0
        cc = Xs[cm].mean(axis=0); cs = Xs[cm].std(axis=0).mean()
        nc2 = ~cm
        if np.any(nc2):
            d = np.linalg.norm(Xs[nc2]-cc, axis=1)
            P[nc2] = np.exp(-d**2/(2*cs**2)) * 0.3
        Pm = np.full(n, np.nan); Pm[vm] = P; cpm = X[cm]
        mu = cpm.mean(axis=0) if len(cpm) > 0 else np.array([cp, cd])
        Sig = np.cov(cpm.T) if len(cpm) > 2 else np.eye(2)*0.1
        return Pm, {'status':'success','algorithm':an,
                    'n_cluster_members':int(np.sum(cm)),
                    'mu_cluster':mu.tolist(),'Sigma_cluster':Sig.tolist(),
                    'pm_dispersion':float(np.sqrt(np.trace(Sig))) if len(cpm)>2 else np.nan,
                    'eta':float(np.sum(cm)/nv),
                    'center_pmra':cp,'center_pmdec':cd}
    except Exception as e:
        return np.full(n, 0.5), {'status':f'error: {e}','algorithm':'DBSCAN'}

def algorithm_hybrid(pmra, pmdec, cp, cd, pe=None, pde=None, corr=None, ms=10):
    n = len(pmra); vm = ~(np.isnan(pmra)|np.isnan(pmdec)); nv = np.sum(vm)
    if nv < ms: return np.full(n, 0.5), {'status':'insufficient_data','algorithm':'Hybrid'}
    X = np.column_stack([pmra[vm], pmdec[vm]])
    sc = StandardScaler(); Xs = sc.fit_transform(X); rs = sc.transform([[cp, cd]])[0]
    try:
        db = DBSCAN(eps=cfg.DBSCAN_EPS_CLEANUP, min_samples=cfg.DBSCAN_MIN_SAMPLES_CLEANUP)
        labels = db.fit_predict(Xs); ul = set(labels) - {-1}
        if len(ul) == 0:
            return algorithm_gmm_with_errors(pmra, pmdec, cp, cd, pe, pde, corr, ms)
        bc = min(ul, key=lambda l: np.linalg.norm(Xs[labels==l].mean(axis=0)-rs))
        cmask = labels == bc
        if np.sum(cmask) < ms:
            return algorithm_gmm_with_errors(pmra, pmdec, cp, cd, pe, pde, corr, ms)
        Xc = X[cmask]
        gmm = GaussianMixture(n_components=2, max_iter=cfg.GMM_MAX_ITER,
                              n_init=cfg.GMM_N_INIT, random_state=cfg.GMM_RANDOM_STATE)
        gmm.fit(Xc)
        ref = np.array([cp, cd])
        ci = np.argmin([np.linalg.norm(gmm.means_[i]-ref) for i in range(2)])
        mc, Sc = gmm.means_[ci], gmm.covariances_[ci]
        mf, Sf = gmm.means_[1-ci], gmm.covariances_[1-ci]
        eta = gmm.weights_[ci]
        P = np.zeros(nv)
        for i in range(nv):
            try:
                pc = eta * multivariate_normal.pdf(X[i], mc, Sc)
                pf = (1-eta) * multivariate_normal.pdf(X[i], mf, Sf)
                P[i] = pc/(pc+pf) if (pc+pf) > 0 else 0.5
            except: P[i] = 0.5
        P[labels == -1] *= 0.5
        Pm = np.full(n, np.nan); Pm[vm] = P
        return Pm, {'status':'success','algorithm':'Hybrid',
                    'mu_cluster':mc.tolist(),'Sigma_cluster':Sc.tolist(),
                    'mu_field':mf.tolist(),'Sigma_field':Sf.tolist(),
                    'eta':float(eta),
                    'pm_dispersion':float(np.sqrt(np.trace(Sc))),
                    'center_pmra':cp,'center_pmdec':cd}
    except Exception as e:
        return np.full(n, 0.5), {'status':f'error: {e}','algorithm':'Hybrid'}

def algorithm_stream_dbscan(pmra, pmdec, cp, cd, eps=0.4, minsamp=3, ms=10):
    n = len(pmra); vm = ~(np.isnan(pmra)|np.isnan(pmdec)); nv = np.sum(vm)
    if nv < ms: return np.full(n, 0.5), {'status':'insufficient_data','algorithm':'Stream-DBSCAN'}
    X = np.column_stack([pmra[vm], pmdec[vm]])
    sc = StandardScaler(); Xs = sc.fit_transform(X); rs = sc.transform([[cp, cd]])[0]
    try:
        db = DBSCAN(eps=eps, min_samples=minsamp)
        labels = db.fit_predict(Xs); ul = set(labels) - {-1}
        if len(ul) == 0:
            d = np.linalg.norm(Xs-rs, axis=1); P = np.exp(-d**2/(2*eps**2))
            Pm = np.full(n, np.nan); Pm[vm] = P
            return Pm, {'status':'no_cluster_found','algorithm':'Stream-DBSCAN',
                        'center_pmra':cp,'center_pmdec':cd}
        bc = min(ul, key=lambda l: np.min(np.linalg.norm(Xs[labels==l]-rs, axis=1)))
        cm = labels == bc; cp2 = Xs[cm]; P = np.zeros(nv)
        if len(cp2) > 0:
            tr = cKDTree(cp2)
            for idx in np.where(cm)[0]:
                d, _ = tr.query(Xs[idx], k=min(3, len(cp2)))
                P[idx] = np.clip(1.0 - np.mean(d[1:] if len(d)>1 else [0])/eps, 0.5, 1.0)
        nc2 = ~cm
        if np.any(nc2):
            for idx in np.where(nc2)[0]:
                d = np.min(np.linalg.norm(cp2-Xs[idx], axis=1))
                P[idx] = 0.3 * np.exp(-d**2/(2*eps**2))
        Pm = np.full(n, np.nan); Pm[vm] = P; cpm = X[cm]
        mu = cpm.mean(axis=0) if len(cpm) > 0 else np.array([cp, cd])
        Sig = np.cov(cpm.T) if len(cpm) > 2 else np.eye(2)*0.1
        return Pm, {'status':'success','algorithm':'Stream-DBSCAN',
                    'n_stream_members':int(np.sum(cm)),
                    'mu_cluster':mu.tolist(),'Sigma_cluster':Sig.tolist(),
                    'pm_dispersion':float(np.sqrt(np.trace(Sig))) if len(cpm)>2 else np.nan,
                    'eta':float(np.sum(cm)/nv),
                    'center_pmra':cp,'center_pmdec':cd}
    except Exception as e:
        return np.full(n, 0.5), {'status':f'error: {e}','algorithm':'Stream-DBSCAN'}

def compute_adaptive_membership(pmra, pmdec, cp, cd, ot, pe=None, pde=None, corr=None, ms=10):
    o = ot.upper()
    if o == 'GC':
        return algorithm_gmm_with_errors(pmra, pmdec, cp, cd, pe, pde, corr, ms)
    elif o == 'OC':
        return algorithm_dbscan(pmra, pmdec, cp, cd, cfg.DBSCAN_EPS_OC,
                                cfg.DBSCAN_MIN_SAMPLES_OC, HAS_HDBSCAN, ms)
    elif o == 'DW':
        return algorithm_hybrid(pmra, pmdec, cp, cd, pe, pde, corr, ms)
    elif o in ['SGR', 'STREAM']:
        return algorithm_stream_dbscan(pmra, pmdec, cp, cd,
                                       cfg.DBSCAN_EPS_STREAM, cfg.DBSCAN_MIN_SAMPLES_STREAM, ms)
    else:
        return algorithm_gmm_with_errors(pmra, pmdec, cp, cd, pe, pde, corr, ms)


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _get_member_rv(member_df, cols):
    rv_col = cols.get('rv'); rv_err_col = cols.get('rv_err')
    rv = rv_err = None
    if rv_col and rv_col in member_df.columns:
        rv = pd.to_numeric(member_df[rv_col], errors='coerce').values
    else:
        for alt in ['RV_weighted_avg','RV_km_s','vlos','radial_velocity','RV','rv','Vrad','HRV']:
            if alt in member_df.columns:
                rv = pd.to_numeric(member_df[alt], errors='coerce').values; break
    if rv_err_col and rv_err_col in member_df.columns:
        rv_err = pd.to_numeric(member_df[rv_err_col], errors='coerce').values
    return rv, rv_err

def _clean(arr, plo=1, phi=99, min_n=3):
    a = np.asarray(arr, dtype=float); a = a[np.isfinite(a)]
    if len(a) < min_n: return a
    lo, hi = np.percentile(a, [plo, phi])
    return a[(a >= lo) & (a <= hi)]

def _safe_kde(ax, data, bins, color, ls='--', alpha=0.7):
    d = data[np.isfinite(data)]
    if len(d) < 4: return
    try:
        kde = gaussian_kde(d); x = np.linspace(bins[0], bins[-1], 200)
        ax.plot(x, kde(x)*len(d)*(bins[1]-bins[0]), color=color, ls=ls, alpha=alpha, lw=2)
    except: pass

def _med_mad_box(ax, data, color, loc='upper right', prefix=''):
    d = data[np.isfinite(data)]
    if len(d) < 2: return
    med = np.median(d); mad = median_abs_deviation(d, nan_policy='omit')
    txt = f"{prefix}Med = {med:.2f}\nMAD = {mad:.2f}\nN = {len(d)}"
    locs = {'upper right': (0.97, 0.97, 'right', 'top'),
            'upper left':  (0.03, 0.97, 'left',  'top'),
            'lower right': (0.97, 0.03, 'right', 'bottom'),
            'lower left':  (0.03, 0.03, 'left',  'bottom')}
    x, y, ha, va = locs.get(loc, (0.97, 0.97, 'right', 'top'))
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha=ha, va=va, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.9, lw=1.5))

def _hist_step_kde(ax, data, color, label, bins_range=None, n_bins=30):
    d = _clean(data)
    if len(d) < 3: return np.array([]), np.array([])
    if bins_range is None:
        lo, hi = np.percentile(d, [0.5, 99.5]); rng = max(hi-lo, 1.0)
        bins_range = [lo - 0.1*rng, hi + 0.1*rng]
    counts, bins, _ = ax.hist(d, bins=n_bins, range=bins_range, alpha=0)
    ax.step(bins[:-1], counts, where='post', color=color, linewidth=2.5, label=label)
    _safe_kde(ax, d, bins, color)
    ax.axvline(np.median(d), color=color, ls=':', lw=2.5, alpha=0.9)
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
    lo, hi = np.percentile(comb, [1, 99]); rng = max(hi - lo, 1)
    if ref_val is not None and np.isfinite(ref_val):
        lo = min(lo, ref_val - 0.05 * rng)
        hi = max(hi, ref_val + 0.05 * rng)
        rng = max(hi - lo, 1)
    pad = 0.15 * rng
    br_lo = max(0, lo - pad) if positive_only else lo - pad
    return [br_lo, hi + pad]

def _best_grid(n):
    if n <= 0: return 1, 1
    best = (n, 1); best_score = n * 10
    for nc in range(1, n + 1):
        nr = int(np.ceil(n / nc))
        if nc > nr: break
        waste = nr * nc - n
        score = waste * 10 + abs(nr - nc)
        if score < best_score:
            best = (nr, nc); best_score = score
    return best

def _compute_rv_reference(member_rv_arr):
    if member_rv_arr is None: return None, None
    rv = np.asarray(member_rv_arr, dtype=float); rv = rv[np.isfinite(rv)]
    if len(rv) < 3: return None, None
    return float(np.median(rv)), float(median_abs_deviation(rv, nan_policy='omit'))

def _add_mad_band(ax, data, color, alpha=0.15):
    d = _clean(data)
    if len(d) < 3: return
    med = np.median(d)
    mad = median_abs_deviation(d, nan_policy='omit')
    ax.axvspan(med - mad, med + mad, alpha=alpha, color=color, zorder=0)


# ============================================================================
# V7 HELPER: Extract P_mem-filtered arrays from matched_df
# ============================================================================

def _filter_by_pmem(matched_df, col, threshold=None):
    """Extract values from matched_df[col] only for stars with P_mem > threshold.

    Returns:
        filtered_arr: numpy array of values passing the P_mem cut
        all_arr: numpy array of all values (for faint background)
    """
    if threshold is None:
        threshold = cfg.P_MEM_PLOT_THRESHOLD

    if col not in matched_df.columns:
        return np.array([]), np.array([])

    all_arr = matched_df[col].values.astype(float)

    if 'P_mem' not in matched_df.columns:
        return all_arr, all_arr

    pmem = matched_df['P_mem'].values.astype(float)
    mask = (pmem >= threshold) & np.isfinite(pmem)
    filtered_arr = all_arr[mask]

    return filtered_arr, all_arr


# ============================================================================
# INDIVIDUAL PLOTS (V7: P_mem-filtered distance & RV)
# ============================================================================

def plot_individual_panels(matched_df, obj_name, obj_type, algo_info,
                           ref_dist=None, ref_dist_err=None,
                           ref_rv=None, ref_rv_err=None,
                           master_dist=None, member_dist=None,
                           master_rv=None, member_rv=None,
                           prematch_ra=None, prematch_dec=None,
                           prematch_pmra=None, prematch_pmdec=None,
                           save_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    set_paper_style()

    if save_dir is None:
        save_dir = os.path.join(cfg.OUTPUT_DIR, 'individual_plots')
    os.makedirs(save_dir, exist_ok=True)

    thr = cfg.P_MEM_PLOT_THRESHOLD

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    af = axes.flatten()

    title_str = f'{obj_type}: {obj_name}'
    if ref_dist is not None and np.isfinite(ref_dist):
        title_str += f'  ($d_{{\\rm ref}}$ = {ref_dist:.1f} kpc)'
    fig.suptitle(title_str, fontsize=22, fontweight='bold', family='serif', y=0.98)

    # ---------------------------------------------------------------
    # V7: Extract P_mem-filtered distance and RV from matched_df
    # ---------------------------------------------------------------
    hi_dist, all_dist = _filter_by_pmem(matched_df, 'best_dist', thr)
    hi_rv, all_rv     = _filter_by_pmem(matched_df, 'best_rv', thr)

    # Also filter member arrays by same P_mem mask (they're row-aligned)
    if member_dist is not None and len(member_dist) > 0 and 'P_mem' in matched_df.columns:
        pmem = matched_df['P_mem'].values.astype(float)
        mask = (pmem >= thr) & np.isfinite(pmem)
        if len(member_dist) == len(mask):
            hi_mem_dist = np.asarray(member_dist, dtype=float)[mask]
        else:
            hi_mem_dist = np.asarray(member_dist, dtype=float)
    else:
        hi_mem_dist = np.asarray(member_dist, dtype=float) if member_dist is not None else np.array([])

    if member_rv is not None and len(member_rv) > 0 and 'P_mem' in matched_df.columns:
        pmem = matched_df['P_mem'].values.astype(float)
        mask = (pmem >= thr) & np.isfinite(pmem)
        if len(member_rv) == len(mask):
            hi_mem_rv = np.asarray(member_rv, dtype=float)[mask]
        else:
            hi_mem_rv = np.asarray(member_rv, dtype=float)
    else:
        hi_mem_rv = np.asarray(member_rv, dtype=float) if member_rv is not None else np.array([])

    n_hi = int(np.sum((matched_df['P_mem'].values >= thr) & np.isfinite(matched_df['P_mem'].values))) if 'P_mem' in matched_df.columns else len(matched_df)

    # ===== Panel 0: PM (unchanged — shows P_mem coloring) =====
    ax = af[0]
    if prematch_pmra is not None and prematch_pmdec is not None:
        vm = np.isfinite(prematch_pmra) & np.isfinite(prematch_pmdec)
        if np.sum(vm) > 0:
            ax.scatter(prematch_pmra[vm], prematch_pmdec[vm],
                       c='lightgray', s=8, alpha=0.4, zorder=1, label='All members')
    if 'P_mem' in matched_df.columns:
        v = matched_df['P_mem'].notna()
        sc1 = ax.scatter(matched_df.loc[v, 'pmra'], matched_df.loc[v, 'pmdec'],
                         c=matched_df.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                         s=25, edgecolors='k', linewidths=0.3, vmin=0, vmax=1, zorder=5)
        cb = plt.colorbar(sc1, ax=ax)
        cb.set_label('$P_{\\rm mem}$', fontsize=14, fontweight='bold')
    if algo_info.get('Sigma_cluster') is not None and obj_type != 'SGR':
        mu = algo_info['mu_cluster']; sig = np.array(algo_info['Sigma_cluster'])
        w, v2 = np.linalg.eigh(sig)
        ang = np.degrees(np.arctan2(v2[1, 0], v2[0, 0]))
        for ns in [1, 2]:
            ell = Ellipse(xy=mu,
                          width=2*ns*np.sqrt(max(w[0], 0)),
                          height=2*ns*np.sqrt(max(w[1], 0)),
                          angle=ang, fill=False, edgecolor='lime',
                          lw=2.5 if ns == 1 else 2,
                          ls='-' if ns == 1 else '--')
            ax.add_patch(ell)
        ax.scatter(*mu, marker='x', s=120, c='lime', linewidths=3, zorder=10)
    ax.set_xlabel('$\\mu_{\\alpha}\\cos\\delta$ (mas yr$^{-1}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel('$\\mu_{\\delta}$ (mas yr$^{-1}$)', fontsize=16, fontweight='bold')
    ax.set_title('Proper Motion', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.2); ax.set_aspect('equal', 'datalim')

    # ===== Panel 1: RA-Dec (unchanged) =====
    ax = af[1]
    if prematch_ra is not None and prematch_dec is not None:
        vm = np.isfinite(prematch_ra) & np.isfinite(prematch_dec)
        if np.sum(vm) > 0:
            ax.scatter(prematch_ra[vm], prematch_dec[vm],
                       c='lightgray', s=8, alpha=0.4, zorder=1, label='All members')
    if 'P_mem' in matched_df.columns:
        v = matched_df['P_mem'].notna()
        sc2 = ax.scatter(matched_df.loc[v, 'ra'], matched_df.loc[v, 'dec'],
                         c=matched_df.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                         s=25, edgecolors='k', linewidths=0.3, vmin=0, vmax=1, zorder=5)
        cb = plt.colorbar(sc2, ax=ax)
        cb.set_label('$P_{\\rm mem}$', fontsize=14, fontweight='bold')
    ax.set_xlabel('RA (deg)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Dec (deg)', fontsize=16, fontweight='bold')
    ax.set_title('Sky Position', fontsize=18, fontweight='bold')
    ax.invert_xaxis(); ax.grid(True, alpha=0.2)

    # ===== Panel 2: Distance — V7: P_mem-filtered =====
    ax = af[2]

    # Clean arrays
    all_d_c = _clean(all_dist); all_d_c = all_d_c[(all_d_c > 0) & (all_d_c < 300)]
    hi_d_c  = _clean(hi_dist);  hi_d_c  = hi_d_c[(hi_d_c > 0) & (hi_d_c < 300)]
    mem_d_c = _clean(hi_mem_dist); mem_d_c = mem_d_c[(mem_d_c > 0) & (mem_d_c < 300)]

    br = _bins_range_with_ref([all_d_c, hi_d_c, mem_d_c], ref_dist, positive_only=True)

    # Faint background: ALL matched stars (shows contamination context)
    if len(all_d_c) >= 3:
        counts_bg, bins_bg, _ = ax.hist(all_d_c, bins=25, range=br, alpha=0)
        ax.step(bins_bg[:-1], counts_bg, where='post', color=cfg.COL_MASTER,
                linewidth=1.5, alpha=0.35, ls='--',
                label=f'All matched (n={len(all_d_c)})')

    # V7: PROMINENT: High-P_mem master distances
    if len(hi_d_c) >= 3:
        _hist_step_kde(ax, hi_d_c, cfg.COL_HIGHMEM,
                       f'P$_{{mem}}$>{thr:.1f} (n={len(hi_d_c)})', br, 25)
        _med_mad_box(ax, hi_d_c, cfg.COL_HIGHMEM, 'upper right',
                     f'P>{thr:.1f} ')

    # Member catalog distances (also P_mem-filtered)
    if len(mem_d_c) >= 3:
        _hist_step_kde(ax, mem_d_c, cfg.COL_MEMBER,
                       f'Member (n={len(mem_d_c)})', br, 25)
        _med_mad_box(ax, mem_d_c, cfg.COL_MEMBER, 'upper left', 'Member ')

    # Reference line or MAD band
    if ref_dist is not None and np.isfinite(ref_dist):
        ax.axvline(ref_dist, color='black', lw=3, alpha=0.8, ls='-',
                   label=f'Ref = {ref_dist:.1f} kpc')
        if ref_dist_err is not None and np.isfinite(ref_dist_err):
            ax.axvspan(ref_dist - ref_dist_err, ref_dist + ref_dist_err,
                       alpha=0.15, color='gray')
    else:
        if len(hi_d_c) >= 3:
            _add_mad_band(ax, hi_d_c, cfg.COL_HIGHMEM, alpha=0.12)

    ax.set_xlabel('Distance (kpc)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.set_title('Distance Distribution', fontsize=18, fontweight='bold')
    ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.2, axis='y')

    # ===== Panel 3: RV — V7: P_mem-filtered =====
    ax = af[3]

    all_rv_c = _clean(all_rv)
    hi_rv_c  = _clean(hi_rv)
    mem_rv_c = _clean(hi_mem_rv)

    show_rv = len(hi_rv_c) >= 3 or len(mem_rv_c) >= 3 or len(all_rv_c) >= 3

    if show_rv:
        br = _bins_range_with_ref([all_rv_c, hi_rv_c, mem_rv_c], ref_rv)

        # Faint background: all matched
        if len(all_rv_c) >= 3:
            counts_bg, bins_bg, _ = ax.hist(all_rv_c, bins=25, range=br, alpha=0)
            ax.step(bins_bg[:-1], counts_bg, where='post', color=cfg.COL_MASTER,
                    linewidth=1.5, alpha=0.35, ls='--',
                    label=f'All matched (n={len(all_rv_c)})')

        # V7: PROMINENT: High-P_mem RVs
        if len(hi_rv_c) >= 3:
            _hist_step_kde(ax, hi_rv_c, cfg.COL_HIGHMEM,
                           f'P$_{{mem}}$>{thr:.1f} (n={len(hi_rv_c)})', br, 25)
            _med_mad_box(ax, hi_rv_c, cfg.COL_HIGHMEM, 'upper right',
                         f'P>{thr:.1f} ')

        # Member catalog RVs (P_mem-filtered)
        if len(mem_rv_c) >= 3:
            _hist_step_kde(ax, mem_rv_c, cfg.COL_MEMBER,
                           f'Member (n={len(mem_rv_c)})', br, 25)
            _med_mad_box(ax, mem_rv_c, cfg.COL_MEMBER, 'upper left', 'Member ')

        # Reference line or MAD band
        if ref_rv is not None and np.isfinite(ref_rv):
            ax.axvline(ref_rv, color='black', lw=3, alpha=0.8, ls='-',
                       label=f'Ref = {ref_rv:.1f} km/s')
            if ref_rv_err is not None and np.isfinite(ref_rv_err):
                ax.axvspan(ref_rv - ref_rv_err, ref_rv + ref_rv_err,
                           alpha=0.15, color='gray')
        else:
            if len(hi_rv_c) >= 3:
                _add_mad_band(ax, hi_rv_c, cfg.COL_HIGHMEM, alpha=0.12)

        ax.set_xlabel('Radial Velocity (km s$^{-1}$)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Count', fontsize=16, fontweight='bold')
        ax.set_title('Radial Velocity', fontsize=18, fontweight='bold')
        ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.2, axis='y')
    else:
        ax.text(0.5, 0.5, 'No RV data', ha='center', va='center',
                transform=ax.transAxes, fontsize=16, fontweight='bold', color='gray')
        ax.set_title('Radial Velocity', fontsize=18, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe = str(obj_name).replace(' ', '_').replace('/', '_').replace('-', '_')
    out = os.path.join(save_dir, f"{obj_type}_{safe}.{cfg.SAVE_FORMAT}")
    plt.savefig(out, dpi=cfg.PLOT_DPI); plt.close()
    return out


# ============================================================================
# PROCESSING PIPELINES (V7: pass matched_df with P_mem, let plot func filter)
# ============================================================================

def load_gc_reference_distances(filepath, logger):
    if not filepath or not os.path.exists(filepath):
        logger.warning(f"GC_dist.csv not found"); return {}
    logger.info(f"Loading GC reference distances from {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    gc_dists = {}
    def empty(val):
        if pd.isna(val): return True
        s = str(val).strip()
        return s in ['', '-', '–', '—', '−', 'nan', 'NaN', 'N/A', 'n/a'] or len(s) == 0
    nl = nm = 0
    for _, row in df.iterrows():
        name = str(row[cfg.GC_DIST_COLS['name']]).strip()
        dist = err = np.nan; src = None
        try:
            v = row[cfg.GC_DIST_COLS['lit_dist']]
            if not empty(v):
                dist = float(v)
                ev = row[cfg.GC_DIST_COLS['lit_dist_err']]
                err = float(str(ev).replace('+', '').strip()) if not empty(ev) else np.nan
                src = 'lit'
        except: pass
        if not np.isfinite(dist):
            try:
                v = row[cfg.GC_DIST_COLS['mean_dist']]
                if not empty(v):
                    dist = float(v)
                    ev = row[cfg.GC_DIST_COLS['mean_dist_err']]
                    err = float(str(ev).replace('+', '').strip()) if not empty(ev) else np.nan
                    src = 'mean'
            except: pass
        if np.isfinite(dist):
            if src == 'lit': nl += 1
            else: nm += 1
            nn = normalize_name(name)
            gc_dists[nn] = (dist, err); gc_dists[name] = (dist, err)
            gc_dists[name.lower()] = (dist, err); gc_dists[name.upper()] = (dist, err)
    logger.info(f"  Loaded {nl} lit + {nm} mean = {nl+nm} total")
    return gc_dists


def _standard_match_columns(mdf):
    mdf['pmra'] = mdf.get(f"{cfg.MASTER_COLS['pmra']}_master", pd.Series(dtype=float))
    mdf['pmdec'] = mdf.get(f"{cfg.MASTER_COLS['pmdec']}_master", pd.Series(dtype=float))
    mdf['ra'] = mdf.get(f"{cfg.MASTER_COLS['ra']}_master", pd.Series(dtype=float))
    mdf['dec'] = mdf.get(f"{cfg.MASTER_COLS['dec']}_master", pd.Series(dtype=float))
    mdf['best_dist'] = mdf.get('best_dist_master', pd.Series(dtype=float))
    mdf['best_rv'] = mdf.get('best_rv_master', pd.Series(dtype=float))
    return mdf


def process_gc_members(master, gc_dists, logger):
    if not cfg.GC_MEMBERS_FILE or not os.path.exists(cfg.GC_MEMBERS_FILE):
        return []
    logger.info("\n" + "-"*50 + "\nProcessing GLOBULAR CLUSTERS...\n" + "-"*50)
    df = pd.read_csv(cfg.GC_MEMBERS_FILE); cols = cfg.GC_MEM_COLS; kc = cols['key']
    clusters = sorted(df[kc].unique()) if kc in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters")
    results = []
    for i, cn in enumerate(clusters):
        logger.info(f"\n[{i+1}/{len(clusters)}] {cn}")
        cdf = df[df[kc] == cn].copy()
        if len(cdf) < cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra = cdf[cols['ra']].values; dec = cdf[cols['dec']].values
        vm = ~(np.isnan(ra) | np.isnan(dec))
        cdf = cdf[vm].reset_index(drop=True); ra = ra[vm]; dec = dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue

        midx, memidx, seps = master.query(ra, dec)
        logger.info(f"  Members: {len(ra)}, Matched: {len(midx)} ({100*len(midx)/len(ra):.1f}%)")

        c_pmra = cdf[cols['pmra']].median(); c_pmdec = cdf[cols['pmdec']].median()
        rd, rde = None, None
        nn = normalize_name(cn)
        for nt in [nn, cn, cn.lower(), cn.upper()]:
            if nt in gc_dists: rd, rde = gc_dists[nt]; break

        algo = {'status':'no_matches', 'algorithm':'None',
                'center_pmra':c_pmra, 'center_pmdec':c_pmdec}
        mdf = None
        master_dist_arr = np.array([]); member_dist_arr = np.array([])
        master_rv_arr = np.array([]); member_rv_arr = np.array([])
        ref_rv = None; ref_rv_err = None

        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm = cdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx)
            mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1); mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            pe = mdf.get(f"{cfg.MASTER_COLS['pmra_err']}_master",
                         pd.Series(dtype=float)).values
            pde = mdf.get(f"{cfg.MASTER_COLS['pmdec_err']}_master",
                          pd.Series(dtype=float)).values
            P, algo = compute_adaptive_membership(
                mdf['pmra'].values, mdf['pmdec'].values,
                c_pmra, c_pmdec, 'GC', pe, pde, None, cfg.MIN_STARS_FOR_ANALYSIS)
            mdf['P_mem'] = P

            # V7: pass raw arrays; plot_individual_panels will filter by P_mem
            master_dist_arr = mdf['best_dist'].values
            plx_arr = (mm[cols['parallax']].values
                       if cols.get('parallax') and cols['parallax'] in mm.columns
                       else np.array([]))
            member_dist_arr = (np.where(plx_arr > 0, 1.0/plx_arr, np.nan)
                               if len(plx_arr) > 0 else np.array([]))
            master_rv_arr = mdf['best_rv'].values
            mrv, _ = _get_member_rv(mm, cols)
            member_rv_arr = mrv if mrv is not None else np.array([])
            ref_rv, ref_rv_err = _compute_rv_reference(member_rv_arr)

            n_hi = int(np.sum(P > cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"  Algo: {algo['algorithm']} | "
                        f"n(P>{cfg.P_MEM_PLOT_THRESHOLD}): {n_hi} | "
                        f"n(P>0.8): {int(np.sum(P>cfg.P_MEM_HIGH))}")

            pm_ra = cdf[cols['ra']].values; pm_dec = cdf[cols['dec']].values
            pm_pmra = cdf[cols['pmra']].values; pm_pmdec = cdf[cols['pmdec']].values
            plot_individual_panels(
                mdf, cn, 'GC', algo, rd, rde, ref_rv, ref_rv_err,
                master_dist_arr, member_dist_arr, master_rv_arr, member_rv_arr,
                pm_ra, pm_dec, pm_pmra, pm_pmdec)

        results.append({
            'cluster_name':cn, 'obj_type':'GC', 'member_df':cdf, 'matched_df':mdf,
            'algo_info':algo, 'mem_cols':cols,
            'n_members':len(cdf), 'n_matched':len(midx),
            'ref_dist':rd, 'ref_dist_err':rde,
            'ref_rv':ref_rv, 'ref_rv_err':ref_rv_err,
            'master_dist':master_dist_arr, 'member_dist':member_dist_arr,
            'master_rv':master_rv_arr, 'member_rv':member_rv_arr})
        gcmod.collect()
    return results


def process_oc_members(master, logger):
    if not cfg.OC_MEMBERS_FILE or not os.path.exists(cfg.OC_MEMBERS_FILE):
        return []
    logger.info("\n" + "-"*50 + "\nProcessing OPEN CLUSTERS...\n" + "-"*50)
    df = pd.read_csv(cfg.OC_MEMBERS_FILE); cols = cfg.OC_MEM_COLS; kc = cols['key']
    clusters = sorted(df[kc].unique()) if kc in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters")
    results = []
    for i, cn in enumerate(clusters):
        logger.info(f"\n[{i+1}/{len(clusters)}] {cn}")
        cdf = df[df[kc] == cn].copy()
        if len(cdf) < cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra = cdf[cols['ra']].values; dec = cdf[cols['dec']].values
        vm = ~(np.isnan(ra) | np.isnan(dec))
        cdf = cdf[vm].reset_index(drop=True); ra = ra[vm]; dec = dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue

        plx = cdf[cols['parallax']].values; vp = plx[plx > 0]
        rd = np.mean(1.0/vp) if len(vp) > 0 else None
        rde = np.std(1.0/vp) if len(vp) > 0 else None

        midx, memidx, seps = master.query(ra, dec)
        logger.info(f"  Members: {len(ra)}, Matched: {len(midx)} ({100*len(midx)/len(ra):.1f}%)")
        c_pmra = cdf[cols['pmra']].median(); c_pmdec = cdf[cols['pmdec']].median()

        algo = {'status':'no_matches', 'algorithm':'None',
                'center_pmra':c_pmra, 'center_pmdec':c_pmdec}
        mdf = None
        master_dist_arr = np.array([]); member_dist_arr = np.array([])
        master_rv_arr = np.array([]); member_rv_arr = np.array([])
        ref_rv = None; ref_rv_err = None

        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm = cdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx)
            mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1); mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            P, algo = compute_adaptive_membership(
                mdf['pmra'].values, mdf['pmdec'].values,
                c_pmra, c_pmdec, 'OC', None, None, None, cfg.MIN_STARS_FOR_ANALYSIS)
            mdf['P_mem'] = P

            master_dist_arr = mdf['best_dist'].values
            member_dist_arr = np.where(mm[cols['parallax']].values > 0,
                                       1.0/mm[cols['parallax']].values, np.nan)
            master_rv_arr = mdf['best_rv'].values
            mrv, _ = _get_member_rv(mm, cols)
            member_rv_arr = mrv if mrv is not None else np.array([])
            ref_rv, ref_rv_err = _compute_rv_reference(member_rv_arr)

            n_hi = int(np.sum(P > cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"  Algo: {algo['algorithm']} | "
                        f"n(P>{cfg.P_MEM_PLOT_THRESHOLD}): {n_hi} | "
                        f"n(P>0.8): {int(np.sum(P>cfg.P_MEM_HIGH))}")

            pm_ra = cdf[cols['ra']].values; pm_dec = cdf[cols['dec']].values
            pm_pmra = cdf[cols['pmra']].values; pm_pmdec = cdf[cols['pmdec']].values
            plot_individual_panels(
                mdf, cn, 'OC', algo, rd, rde, ref_rv, ref_rv_err,
                master_dist_arr, member_dist_arr, master_rv_arr, member_rv_arr,
                pm_ra, pm_dec, pm_pmra, pm_pmdec)

        results.append({
            'cluster_name':cn, 'obj_type':'OC', 'member_df':cdf, 'matched_df':mdf,
            'algo_info':algo, 'mem_cols':cols,
            'n_members':len(cdf), 'n_matched':len(midx),
            'ref_dist':rd, 'ref_dist_err':rde,
            'ref_rv':ref_rv, 'ref_rv_err':ref_rv_err,
            'master_dist':master_dist_arr, 'member_dist':member_dist_arr,
            'master_rv':master_rv_arr, 'member_rv':member_rv_arr})
        gcmod.collect()
    return results


def process_sgr_members(master, logger):
    if not cfg.SGR_MEMBERS_FILE or not os.path.exists(cfg.SGR_MEMBERS_FILE):
        return []
    logger.info("\n" + "-"*50 + "\nProcessing SGR STREAM...\n" + "-"*50)
    df = pd.read_csv(cfg.SGR_MEMBERS_FILE); cols = cfg.SGR_MEM_COLS
    dc = cols.get('dist', 'dist')
    if dc not in df.columns: return []
    df = df.dropna(subset=[dc])
    df[dc] = pd.to_numeric(df[dc], errors='coerce'); df = df.dropna(subset=[dc])

    be = np.arange(cfg.SGR_BIN_START_KPC,
                   df[dc].max() + cfg.SGR_BIN_WIDTH_KPC, cfg.SGR_BIN_WIDTH_KPC)
    bl = [f'{be[i]:.0f}-{be[i+1]:.0f} kpc' for i in range(len(be)-1)]
    df['dist_bin'] = pd.cut(df[dc], bins=be, labels=bl, right=False)
    df = df.dropna(subset=['dist_bin'])
    bc = df['dist_bin'].value_counts()
    vb = bc[bc >= cfg.SGR_MIN_STARS_PER_BIN].index.tolist()
    bs = sorted([(b, df[df['dist_bin']==b][dc].mean()) for b in vb], key=lambda x: x[1])
    bins_list = [x[0] for x in bs]
    if len(bins_list) == 0: return []

    results = []
    for i, bl_name in enumerate(bins_list):
        logger.info(f"\n[{i+1}/{len(bins_list)}] {bl_name}")
        bdf = df[df['dist_bin'] == bl_name].copy()
        ra = bdf[cols['ra']].values; dec = bdf[cols['dec']].values
        vm = ~(np.isnan(ra) | np.isnan(dec))
        bdf = bdf[vm].reset_index(drop=True); ra = ra[vm]; dec = dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue

        midx, memidx, seps = master.query(ra, dec)
        logger.info(f"  Members: {len(ra)}, Matched: {len(midx)} ({100*len(midx)/len(ra):.1f}%)")
        c_pmra = bdf[cols['pmra']].median(); c_pmdec = bdf[cols['pmdec']].median()

        algo = {'status':'no_matches', 'algorithm':'None',
                'center_pmra':c_pmra, 'center_pmdec':c_pmdec}
        mdf = None
        master_dist_arr = np.array([]); member_dist_arr = np.array([])
        master_rv_arr = np.array([]); member_rv_arr = np.array([])
        ref_rv = None; ref_rv_err = None

        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm = bdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx)
            mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1); mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            P, algo = compute_adaptive_membership(
                mdf['pmra'].values, mdf['pmdec'].values,
                c_pmra, c_pmdec, 'SGR', None, None, None, cfg.MIN_STARS_FOR_ANALYSIS)
            mdf['P_mem'] = P

            master_dist_arr = mdf['best_dist'].values
            member_dist_arr = mm[dc].values if dc in mm.columns else np.array([])
            master_rv_arr = mdf['best_rv'].values
            mrv, _ = _get_member_rv(mm, cols)
            member_rv_arr = mrv if mrv is not None else np.array([])
            ref_rv, ref_rv_err = _compute_rv_reference(member_rv_arr)

            pm_ra = bdf[cols['ra']].values; pm_dec = bdf[cols['dec']].values
            pm_pmra = bdf[cols['pmra']].values; pm_pmdec = bdf[cols['pmdec']].values
            plot_individual_panels(
                mdf, bl_name, 'SGR', algo, None, None, ref_rv, ref_rv_err,
                master_dist_arr, member_dist_arr, master_rv_arr, member_rv_arr,
                pm_ra, pm_dec, pm_pmra, pm_pmdec)

        results.append({
            'cluster_name':bl_name, 'obj_type':'SGR', 'member_df':bdf, 'matched_df':mdf,
            'algo_info':algo, 'mem_cols':cols,
            'n_members':len(bdf), 'n_matched':len(midx),
            'ref_dist':None, 'ref_dist_err':None,
            'ref_rv':ref_rv, 'ref_rv_err':ref_rv_err,
            'master_dist':master_dist_arr, 'member_dist':member_dist_arr,
            'master_rv':master_rv_arr, 'member_rv':member_rv_arr})
        gcmod.collect()
    return results


def process_dwg_members(master, logger):
    if not cfg.DWG_MEMBERS_FILE or not os.path.exists(cfg.DWG_MEMBERS_FILE):
        return []
    logger.info("\n" + "-"*50 + "\nProcessing DWARF GALAXIES (V7)...\n" + "-"*50)
    df = pd.read_csv(cfg.DWG_MEMBERS_FILE)
    cols = cfg.DWG_MEM_COLS; kc = cols['key']

    for cname in [cols['ra'], cols['dec']]:
        if cname not in df.columns:
            logger.error(f"  Column '{cname}' not found! Available: {list(df.columns[:15])}")
            return []

    gals = df[kc].unique() if kc in df.columns else ['ALL']
    logger.info(f"Found {len(gals)} galaxies")
    results = []

    for i, gn in enumerate(gals):
        logger.info(f"\n[{i+1}/{len(gals)}] {gn}")
        gal_df = df[df[kc] == gn].copy()
        if len(gal_df) < 1:
            logger.warning(f"  Empty DataFrame for '{gn}' - skipping"); continue

        gr = gal_df.iloc[0]
        rd = _safe_float(gr, cols.get('distance'))
        rde = _safe_float(gr, cols.get('distance_err'))
        ref_rv = _safe_float(gr, cols.get('rv_ref'))
        ref_rv_err = _safe_float(gr, cols.get('rv_ref_err'))
        if rd is not None:
            logger.info(f"  Ref dist: {rd:.2f} ± {rde if rde else 0:.2f} kpc")
        if ref_rv is not None:
            logger.info(f"  Ref RV: {ref_rv:.1f} ± {ref_rv_err if ref_rv_err else 0:.1f} km/s")

        ra = pd.to_numeric(gal_df[cols['ra']], errors='coerce').values
        dec = pd.to_numeric(gal_df[cols['dec']], errors='coerce').values
        vm = np.isfinite(ra) & np.isfinite(dec)
        gal_df = gal_df[vm].reset_index(drop=True); ra = ra[vm]; dec = dec[vm]

        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS:
            logger.info(f"  Only {len(ra)} valid positions - skipping")
            results.append({
                'cluster_name':gn, 'obj_type':'DW', 'member_df':gal_df,
                'matched_df':None, 'algo_info':{'status':'too_few_positions','algorithm':'None'},
                'mem_cols':cols, 'n_members':len(gal_df), 'n_matched':0,
                'ref_dist':rd, 'ref_dist_err':rde,
                'ref_rv':ref_rv, 'ref_rv_err':ref_rv_err,
                'master_dist':np.array([]), 'member_dist':np.array([]),
                'master_rv':np.array([]), 'member_rv':np.array([])})
            continue

        midx, memidx, seps = master.query(ra, dec)
        logger.info(f"  Members: {len(ra)}, Matched: {len(midx)} ({100*len(midx)/len(ra):.1f}%)")

        c_pmra = (pd.to_numeric(gal_df[cols['pmra']], errors='coerce').median()
                  if cols['pmra'] in gal_df.columns else 0.0)
        c_pmdec = (pd.to_numeric(gal_df[cols['pmdec']], errors='coerce').median()
                   if cols['pmdec'] in gal_df.columns else 0.0)

        algo = {'status':'no_matches', 'algorithm':'None',
                'center_pmra':c_pmra, 'center_pmdec':c_pmdec}
        mdf = None
        master_dist_arr = np.array([]); member_dist_arr = np.array([])
        master_rv_arr = np.array([]); member_rv_arr = np.array([])

        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm = gal_df.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx)
            mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1); mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)

            pe = mdf.get(f"{cfg.MASTER_COLS['pmra_err']}_master",
                         pd.Series(dtype=float)).values
            pde = mdf.get(f"{cfg.MASTER_COLS['pmdec_err']}_master",
                          pd.Series(dtype=float)).values
            P, algo = compute_adaptive_membership(
                mdf['pmra'].values, mdf['pmdec'].values,
                c_pmra, c_pmdec, 'DW', pe, pde, None, cfg.MIN_STARS_FOR_ANALYSIS)
            mdf['P_mem'] = P

            n_hi = int(np.sum(P > cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"  Algo: {algo['algorithm']} | "
                        f"n(P>{cfg.P_MEM_PLOT_THRESHOLD}): {n_hi} | "
                        f"n(P>0.8): {int(np.sum(P>cfg.P_MEM_HIGH))}")

            master_dist_arr = mdf['best_dist'].values
            plx_master = mdf.get('plx_from_params_master', pd.Series(dtype=float)).values
            plx_master = np.asarray(plx_master, dtype=float)
            member_dist_arr = np.where(plx_master > 0, 1.0 / plx_master, np.nan)

            master_rv_arr = mdf['best_rv'].values
            rv_col = cols.get('rv', 'RV_km_s')
            if rv_col in mm.columns:
                member_rv_arr = pd.to_numeric(mm[rv_col], errors='coerce').values
            else:
                member_rv_arr = np.array([])

        pm_ra = pd.to_numeric(gal_df[cols['ra']], errors='coerce').values
        pm_dec = pd.to_numeric(gal_df[cols['dec']], errors='coerce').values
        pm_pmra = (pd.to_numeric(gal_df[cols['pmra']], errors='coerce').values
                   if cols['pmra'] in gal_df.columns else np.array([]))
        pm_pmdec = (pd.to_numeric(gal_df[cols['pmdec']], errors='coerce').values
                    if cols['pmdec'] in gal_df.columns else np.array([]))

        if mdf is not None:
            plot_individual_panels(
                mdf, gn, 'DW', algo, rd, rde, ref_rv, ref_rv_err,
                master_dist_arr, member_dist_arr, master_rv_arr, member_rv_arr,
                pm_ra, pm_dec, pm_pmra, pm_pmdec)

        results.append({
            'cluster_name':gn, 'obj_type':'DW', 'member_df':gal_df, 'matched_df':mdf,
            'algo_info':algo, 'mem_cols':cols,
            'n_members':len(gal_df), 'n_matched':len(midx),
            'ref_dist':rd, 'ref_dist_err':rde,
            'ref_rv':ref_rv, 'ref_rv_err':ref_rv_err,
            'master_dist':master_dist_arr, 'member_dist':member_dist_arr,
            'master_rv':master_rv_arr, 'member_rv':member_rv_arr})
        gcmod.collect()
    return results


# ============================================================================
# SUMMARY PLOTS (V7: P_mem-filtered distance & RV)
# ============================================================================

def generate_summary_plots(all_results, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    set_paper_style()

    logger.info("\n" + "="*70 + "\nGENERATING SUMMARY PLOTS (V7)\n" + "="*70)
    vr = [r for r in all_results
          if r['matched_df'] is not None and r['n_matched'] >= cfg.MIN_SUMMARY_MATCH]
    if not vr: logger.warning("No valid results!"); return
    sr = sorted(vr, key=lambda x: x['n_matched'], reverse=True)
    n = len(sr); nr, nc = _best_grid(n)
    logger.info(f"Plotting {n} objects in {nr}×{nc} grid")
    tc = {'GC':'#006400', 'OC':'#FF8C00', 'DW':'#00008B', 'SGR':'#8B0000'}
    thr = cfg.P_MEM_PLOT_THRESHOLD

    def _get_prematch(r):
        cols = r['mem_cols']; mdf2 = r['member_df']
        try:
            pra = pd.to_numeric(mdf2[cols['ra']], errors='coerce').values
            pdec = pd.to_numeric(mdf2[cols['dec']], errors='coerce').values
            ppmra = pd.to_numeric(mdf2[cols['pmra']], errors='coerce').values
            ppmdec = pd.to_numeric(mdf2[cols['pmdec']], errors='coerce').values
        except:
            pra = pdec = ppmra = ppmdec = np.array([])
        return pra, pdec, ppmra, ppmdec

    # ===== 1. PM SUMMARY (unchanged) =====
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if n == 1: axes = np.array([axes])
    af = axes.flatten()
    for i, r in enumerate(sr):
        ax = af[i]; mdf = r['matched_df']; algo = r['algo_info']
        pra, pdec, ppmra, ppmdec = _get_prematch(r)
        if len(ppmra) > 0:
            vm = np.isfinite(ppmra) & np.isfinite(ppmdec)
            if np.sum(vm) > 0:
                ax.scatter(ppmra[vm], ppmdec[vm], c='lightgray', s=5, alpha=0.3, zorder=1)
        if 'P_mem' in mdf.columns:
            v = mdf['P_mem'].notna()
            ax.scatter(mdf.loc[v, 'pmra'], mdf.loc[v, 'pmdec'],
                       c=mdf.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                       s=15, alpha=0.8, vmin=0, vmax=1,
                       edgecolors='k', linewidth=0.2, zorder=5)
        if algo.get('Sigma_cluster') and r['obj_type'] != 'SGR':
            mu = algo['mu_cluster']; sig = np.array(algo['Sigma_cluster'])
            w, v2 = np.linalg.eigh(sig)
            ang = np.degrees(np.arctan2(v2[1, 0], v2[0, 0]))
            ell = Ellipse(xy=mu, width=4*np.sqrt(max(w[0], 0)),
                          height=4*np.sqrt(max(w[1], 0)),
                          angle=ang, fill=False, ec='lime', lw=2.5)
            ax.add_patch(ell)
        ax.set_title(f"{r['obj_type']}: {r['cluster_name']}", fontsize=14,
                     fontweight='bold', color=tc.get(r['obj_type'], 'k'))
        ax.set_xlabel('$\\mu_\\alpha\\cos\\delta$', fontsize=13, fontweight='bold')
        ax.set_ylabel('$\\mu_\\delta$', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2); ax.set_aspect('equal', 'datalim')
    for j in range(n, len(af)): af[j].set_visible(False)
    fig.suptitle('Proper Motion Membership Summary', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f'SUMMARY_PM.{cfg.SAVE_FORMAT}'),
                dpi=cfg.PLOT_DPI); plt.close()
    logger.info("  Saved: SUMMARY_PM")

    # ===== 2. RA-DEC SUMMARY (unchanged) =====
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if n == 1: axes = np.array([axes])
    af = axes.flatten()
    for i, r in enumerate(sr):
        ax = af[i]; mdf = r['matched_df']
        pra, pdec, _, _ = _get_prematch(r)
        if len(pra) > 0:
            vm = np.isfinite(pra) & np.isfinite(pdec)
            if np.sum(vm) > 0:
                ax.scatter(pra[vm], pdec[vm], c='lightgray', s=5, alpha=0.3, zorder=1)
        if 'P_mem' in mdf.columns:
            v = mdf['P_mem'].notna()
            ax.scatter(mdf.loc[v, 'ra'], mdf.loc[v, 'dec'],
                       c=mdf.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                       s=15, alpha=0.8, vmin=0, vmax=1,
                       edgecolors='k', linewidth=0.2, zorder=5)
        ax.set_title(f"{r['obj_type']}: {r['cluster_name']}", fontsize=14,
                     fontweight='bold', color=tc.get(r['obj_type'], 'k'))
        ax.set_xlabel('RA (deg)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Dec (deg)', fontsize=13, fontweight='bold')
        ax.invert_xaxis(); ax.grid(True, alpha=0.2)
    for j in range(n, len(af)): af[j].set_visible(False)
    fig.suptitle('RA-Dec Spatial Distribution Summary', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f'SUMMARY_RADEC.{cfg.SAVE_FORMAT}'),
                dpi=cfg.PLOT_DPI); plt.close()
    logger.info("  Saved: SUMMARY_RADEC")

    # ===== 3. DISTANCE SUMMARY — V7: P_mem-filtered =====
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if n == 1: axes = np.array([axes])
    af = axes.flatten()
    for i, r in enumerate(sr):
        ax = af[i]; mdf = r['matched_df']
        rd = r.get('ref_dist'); rde = r.get('ref_dist_err')

        # V7: Filter by P_mem
        hi_dist, all_dist = _filter_by_pmem(mdf, 'best_dist', thr)
        hi_d = _clean(hi_dist); hi_d = hi_d[(hi_d > 0) & (hi_d < 300)]
        all_d = _clean(all_dist); all_d = all_d[(all_d > 0) & (all_d < 300)]

        br = _bins_range_with_ref([all_d, hi_d], rd, positive_only=True)

        # Faint background
        if len(all_d) >= 3:
            c_bg, b_bg, _ = ax.hist(all_d, bins=20, range=br, alpha=0)
            ax.step(b_bg[:-1], c_bg, where='post', color=cfg.COL_MASTER,
                    linewidth=1.2, alpha=0.3, ls='--')

        # Prominent: high-P_mem
        if len(hi_d) >= 3:
            _hist_step_kde(ax, hi_d, cfg.COL_HIGHMEM,
                           f'P>{thr:.1f} (n={len(hi_d)})', br, 20)
            _med_mad_box(ax, hi_d, cfg.COL_HIGHMEM, 'upper right', f'P>{thr:.1f} ')

        if rd is not None and np.isfinite(rd):
            ax.axvline(rd, color='k', lw=2.5, alpha=0.8)
            if rde is not None and np.isfinite(rde):
                ax.axvspan(rd - rde, rd + rde, alpha=0.12, color='gray')
        else:
            if len(hi_d) >= 3:
                _add_mad_band(ax, hi_d, cfg.COL_HIGHMEM, alpha=0.10)

        ax.set_xlabel('Dist (kpc)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax.set_title(f"{r['obj_type']}: {r['cluster_name']}", fontsize=14,
                     fontweight='bold', color=tc.get(r['obj_type'], 'k'))
        ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.2, axis='y')
    for j in range(n, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Distance Distribution Summary (P$_{{mem}}$>{thr:.1f})',
                 fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f'SUMMARY_Distance.{cfg.SAVE_FORMAT}'),
                dpi=cfg.PLOT_DPI); plt.close()
    logger.info("  Saved: SUMMARY_Distance")

    # ===== 4. RV SUMMARY — V7: P_mem-filtered =====
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if n == 1: axes = np.array([axes])
    af = axes.flatten()
    for i, r in enumerate(sr):
        ax = af[i]; mdf = r['matched_df']
        rrv = r.get('ref_rv'); rrve = r.get('ref_rv_err')

        # V7: Filter by P_mem
        hi_rv, all_rv_arr = _filter_by_pmem(mdf, 'best_rv', thr)
        hi_rv_c = _clean(hi_rv)
        all_rv_c = _clean(all_rv_arr)

        if len(hi_rv_c) < 3 and len(all_rv_c) < 3:
            ax.text(0.5, 0.5, 'No RV data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, fontweight='bold', color='gray')
        else:
            br = _bins_range_with_ref([all_rv_c, hi_rv_c], rrv)

            # Faint background
            if len(all_rv_c) >= 3:
                c_bg, b_bg, _ = ax.hist(all_rv_c, bins=25, range=br, alpha=0)
                ax.step(b_bg[:-1], c_bg, where='post', color=cfg.COL_MASTER,
                        linewidth=1.2, alpha=0.3, ls='--')

            # Prominent: high-P_mem
            if len(hi_rv_c) >= 3:
                _hist_step_kde(ax, hi_rv_c, cfg.COL_HIGHMEM,
                               f'P>{thr:.1f} (n={len(hi_rv_c)})', br, 25)
                _med_mad_box(ax, hi_rv_c, cfg.COL_HIGHMEM, 'upper right', f'P>{thr:.1f} ')

            if rrv is not None and np.isfinite(rrv):
                ax.axvline(rrv, color='k', lw=2.5, alpha=0.8)
                if rrve is not None and np.isfinite(rrve):
                    ax.axvspan(rrv - rrve, rrv + rrve, alpha=0.12, color='gray')
            else:
                if len(hi_rv_c) >= 3:
                    _add_mad_band(ax, hi_rv_c, cfg.COL_HIGHMEM, alpha=0.10)

        ax.set_xlabel('RV (km s$^{-1}$)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax.set_title(f"{r['obj_type']}: {r['cluster_name']}", fontsize=14,
                     fontweight='bold', color=tc.get(r['obj_type'], 'k'))
        ax.legend(fontsize=8, loc='best'); ax.grid(True, alpha=0.2, axis='y')
    for j in range(n, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Radial Velocity Summary (P$_{{mem}}$>{thr:.1f})',
                 fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f'SUMMARY_RV.{cfg.SAVE_FORMAT}'),
                dpi=cfg.PLOT_DPI); plt.close()
    logger.info("  Saved: SUMMARY_RV")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(all_results, logger):
    logger.info("\n" + "-"*50 + "\nSaving results...\n" + "-"*50)
    summary_data = []; master_dfs = []
    for r in all_results:
        nh = 0
        if r['matched_df'] is not None and 'P_mem' in r['matched_df'].columns:
            nh = int(np.sum(r['matched_df']['P_mem'] > cfg.P_MEM_HIGH))
            t = r['matched_df'].copy()
            t.insert(0, 'Object_Type', r['obj_type'])
            t.insert(1, 'Cluster_Name', r['cluster_name'])
            master_dfs.append(t)
        summary_data.append({
            'Object': r['cluster_name'], 'Type': r['obj_type'],
            'N_members': r['n_members'], 'N_matched': r['n_matched'],
            'Match_pct': (f"{100*r['n_matched']/r['n_members']:.1f}"
                          if r['n_members'] > 0 else '0'),
            'N_high_prob': nh,
            'Ref_dist_kpc': r.get('ref_dist', np.nan),
            'Ref_RV_kms': r.get('ref_rv', np.nan),
            'Algorithm': r['algo_info'].get('algorithm', 'None'),
            'Status': r['algo_info'].get('status', 'N/A')})
    if master_dfs:
        full = pd.concat(master_dfs, ignore_index=True)
        fp = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_full_membership_results.csv')
        full.to_csv(fp, index=False)
        logger.info(f"  Saved: {fp} ({len(full):,} rows)")
    sdf = (pd.DataFrame(summary_data)
           .sort_values('N_matched', ascending=False)
           .reset_index(drop=True))
    sf = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_summary.csv')
    sdf.to_csv(sf, index=False); logger.info(f"  Saved: {sf}")
    af2 = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_algorithm_results.json')
    with open(af2, 'w') as f:
        json.dump({f"{r['obj_type']}_{r['cluster_name']}": r['algo_info']
                   for r in all_results}, f, indent=2, default=str)
    logger.info(f"  Saved: {af2}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Adaptive Membership V7 (P_mem-filtered plots)')
    p.add_argument('--master', required=True,
                   help='Master catalog: glob pattern, directory, or single FITS file')
    p.add_argument('--gc', default=None)
    p.add_argument('--oc', default=None)
    p.add_argument('--sgr', default=None)
    p.add_argument('--dwg', default=None)
    p.add_argument('--gc-dist', default=None)
    p.add_argument('--output', default='./outputs')
    p.add_argument('--checkpoint', default='./checkpoints')
    p.add_argument('--log', default=None)
    p.add_argument('--skip-plots', action='store_true')
    p.add_argument('--pmem-threshold', type=float, default=0.5,
                   help='P_mem threshold for distance/RV plots (default: 0.5)')
    return p.parse_args()

def main():
    args = parse_args()
    cfg.MASTER_CATALOG = args.master
    cfg.GC_MEMBERS_FILE = args.gc
    cfg.OC_MEMBERS_FILE = args.oc
    cfg.SGR_MEMBERS_FILE = args.sgr
    cfg.DWG_MEMBERS_FILE = args.dwg
    cfg.GC_DIST_FILE = args.gc_dist
    cfg.OUTPUT_DIR = args.output
    cfg.CHECKPOINT_DIR = args.checkpoint
    cfg.P_MEM_PLOT_THRESHOLD = args.pmem_threshold

    for d in [cfg.OUTPUT_DIR, cfg.CHECKPOINT_DIR,
              os.path.join(cfg.OUTPUT_DIR, 'individual_plots')]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logging(
        args.log or os.path.join(cfg.OUTPUT_DIR, 'adaptive_membership_v7.log'))
    logger.info("=" * 70)
    logger.info("ADAPTIVE MEMBERSHIP ANALYSIS V7")
    logger.info(f"  P_mem-filtered distance/RV | threshold = {cfg.P_MEM_PLOT_THRESHOLD}")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    t0 = time.time()

    master = MasterCatalog(logger)
    if not master.load(cfg.MASTER_CATALOG, cfg.CHECKPOINT_DIR):
        logger.error("Failed to load master catalog!"); sys.exit(1)

    gc_dists = load_gc_reference_distances(cfg.GC_DIST_FILE, logger)

    all_results = []
    all_results.extend(process_gc_members(master, gc_dists, logger))
    all_results.extend(process_oc_members(master, logger))
    all_results.extend(process_sgr_members(master, logger))
    all_results.extend(process_dwg_members(master, logger))

    logger.info(f"\n{'='*70}\nPROCESSING COMPLETE: {len(all_results)} objects\n{'='*70}")
    if not args.skip_plots:
        generate_summary_plots(all_results, logger)
    save_results(all_results, logger)
    logger.info(f"\n{'='*70}\nALL DONE! {(time.time()-t0)/60:.1f} minutes\n{'='*70}")

if __name__ == '__main__':
    main()