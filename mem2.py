#!/usr/bin/env python3
"""
================================================================================
ADAPTIVE MEMBERSHIP ANALYSIS V4 - PAPER QUALITY
================================================================================

V4 CHANGES:
 - Correct column mappings: GC(RV_weighted_avg), OC(RV), SGR(vlos), DWG(vlos_systemic)
 - ApJ/AAS dark-axes serif-font style throughout
 - Consistent colors: RED=master, BLUE=member in ALL plots
 - Dual y-axes in RV summary (normalize when N differs)
 - Median ± MAD text boxes on every histogram
 - ΔRV = (master − member) inset in individual RV panels
 - Summary plots use individual-style (step + KDE + median + stats)
 - Summary filtered to ≥ MIN_SUMMARY_MATCH matched stars
 - Bold, large fonts everywhere for publication

Author: Sutirtha
================================================================================
"""

import os, sys, json, time, logging, argparse, warnings
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
    logger.setLevel(level)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
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

    MASTER_COLS = {
        'ra': 'RA_all', 'dec': 'DEC_all',
        'pmra': 'pmRA_x', 'pmdec': 'pmDE',
        'pmra_err': 'e_pmRA', 'pmdec_err': 'e_pmDE',
        'dist': 'DIST', 'dist_err': 'DISTERR',
        'rv': 'RV_final', 'rv_err': 'RV_err_final',
        'params_est': 'stellar_params_est', 'params_err': 'stellar_params_err',
    }
    ALT_MASTER_COLS = {
        'rv': ['RV_final', 'Weighted_Avg', 'radial_velocity', 'RV'],
        'rv_err': ['RV_err_final', 'radial_velocity_error', 'RV_err'],
    }

    # ===== MEMBER CATALOG COLUMNS (V4: corrected) =====
    GC_MEM_COLS = {
        'key': 'source',
        'ra': 'ra', 'dec': 'dec',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmra_error', 'pmdec_err': 'pmdec_error',
        'pmra_pmdec_corr': 'pmra_pmdec_corr',
        'parallax': 'parallax',
        'membership_prob': 'membership_probability',
        'rv': 'RV_weighted_avg',
        'rv_err': 'e_RV_weighted_avg',
    }
    GC_DIST_COLS = {
        'name': 'Name', 'lit_dist': 'Lit. dist. (kpc)',
        'lit_dist_err': 'Lit. dist. Err+',
        'mean_dist': 'Mean distance (kpc)', 'mean_dist_err': 'Mean distance Err+',
    }
    OC_MEM_COLS = {
        'key': 'Cluster',
        'ra': 'RAdeg', 'dec': 'DEdeg',
        'pmra': 'pmRA', 'pmdec': 'pmDE',
        'pmra_err': 'e_pmRA', 'pmdec_err': 'e_pmDE',
        'pmra_pmdec_corr': 'pmRApmDEcor',
        'parallax': 'Plx',
        'membership_prob': 'Proba',
        'rv': 'RV', 'rv_err': 'e_RV',
    }
    SGR_MEM_COLS = {
        'key': None,
        'ra': 'ra', 'dec': 'dec',
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
    }

    CROSSMATCH_RADIUS_ARCSEC = 1.0; DWG_SEARCH_RADIUS_FACTOR = 3.0
    GMM_N_COMPONENTS = 2; GMM_MAX_ITER = 300; GMM_N_INIT = 10; GMM_RANDOM_STATE = 42
    DBSCAN_EPS_CLEANUP = 0.3; DBSCAN_MIN_SAMPLES_CLEANUP = 3
    DBSCAN_EPS_OC = 0.25; DBSCAN_MIN_SAMPLES_OC = 5
    DBSCAN_EPS_STREAM = 0.4; DBSCAN_MIN_SAMPLES_STREAM = 3
    P_MEM_HIGH = 0.8; P_MEM_LOW = 0.2; MIN_STARS_FOR_ANALYSIS = 10
    SGR_BIN_START_KPC = 15.0; SGR_BIN_WIDTH_KPC = 10.0; SGR_MIN_STARS_PER_BIN = 5
    MIN_SUMMARY_MATCH = 5  # minimum matched stars to appear in summary

    MAX_PANELS = 36; CMAP_PMEM = 'RdYlGn'; PLOT_DPI = 150; SAVE_FORMAT = 'png'

    # Colors
    COL_MASTER = '#DC143C'   # crimson-red for master
    COL_MEMBER = '#000080'   # navy blue for member

cfg = Config()

try:
    import hdbscan; HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

def normalize_name(name):
    import re
    if not isinstance(name, str): name = str(name)
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())

normalize_cluster_name = normalize_name  # alias

# ============================================================================
# APJ / AAS PLOT STYLE
# ============================================================================

def set_paper_style():
    """Set ApJ/AAS-style rcParams: serif fonts, dark axes, large bold labels."""
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
        'figure.dpi': 150, 'savefig.dpi': 300,
        'text.usetex': False,
    })

# ============================================================================
# MASTER CATALOG
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
        self.logger.info("LOADING MASTER CATALOG")
        self.logger.info("=" * 70)

        tree_cp = os.path.join(checkpoint_dir, 'master_tree.npz') if checkpoint_dir else None
        data_cp = os.path.join(checkpoint_dir, 'master_data_v4.parquet') if checkpoint_dir else None

        if tree_cp and os.path.exists(tree_cp) and data_cp and os.path.exists(data_cp):
            self.logger.info("Loading from checkpoint...")
            try:
                self.df = pd.read_parquet(data_cp)
                self.coords_3d = np.load(tree_cp)['coords']
                self.tree = cKDTree(self.coords_3d)
                self.nrows = len(self.df)
                self._compute_max_chord()
                n_rv = self.df['best_rv'].notna().sum() if 'best_rv' in self.df.columns else 0
                self.logger.info(f"✓ Loaded {self.nrows:,} rows from checkpoint ({n_rv:,} with RV)")
                return True
            except Exception as e:
                self.logger.warning(f"Checkpoint load failed: {e}")

        t0 = time.time()
        self.logger.info(f"Loading: {filepath}")
        try:
            with fits.open(filepath, memmap=True) as hdul:
                data_hdu = None
                for hdu in hdul:
                    if hasattr(hdu, 'columns') and hdu.columns is not None:
                        data_hdu = hdu; break
                if data_hdu is None:
                    self.logger.error("No data table found!"); return False

                self.nrows = data_hdu.data.shape[0]
                self.logger.info(f"  Total rows: {self.nrows:,}")
                col_names = [c.name for c in data_hdu.columns]
                data_dict = {}

                skip = {'params_est', 'params_err', 'rv', 'rv_err'}
                for key, col in cfg.MASTER_COLS.items():
                    if key in skip: continue
                    if col in col_names:
                        self.logger.info(f"  Loading: {col}")
                        data_dict[col] = data_hdu.data[col].copy()

                # RV
                self.logger.info("  Resolving RV columns...")
                rv_col = self._find_col('rv', col_names)
                rv_err_col = self._find_col('rv_err', col_names)
                if rv_col:
                    self.logger.info(f"  Loading RV from: {rv_col}")
                    data_dict['best_rv'] = np.array(data_hdu.data[rv_col], dtype=np.float64)
                    self.logger.info(f"  ✓ {np.sum(np.isfinite(data_dict['best_rv'])):,} with valid RV")
                else:
                    self.logger.warning("  No RV column found!"); data_dict['best_rv'] = np.full(self.nrows, np.nan)
                data_dict['best_rv_err'] = np.array(data_hdu.data[rv_err_col], dtype=np.float64) if rv_err_col else np.full(self.nrows, np.nan)

                # Distance selection
                self.logger.info("  Computing best distance...")
                dist_arr = data_dict.get(cfg.MASTER_COLS['dist'], np.full(self.nrows, np.nan))
                dist_err_arr = data_dict.get(cfg.MASTER_COLS['dist_err'], np.full(self.nrows, np.nan))
                plx_from_params = np.full(self.nrows, np.nan)
                plx_err_from_params = np.full(self.nrows, np.nan)
                if cfg.MASTER_COLS['params_est'] in col_names:
                    plx_from_params = data_hdu.data[cfg.MASTER_COLS['params_est']][:, 4].copy()
                if cfg.MASTER_COLS['params_err'] in col_names:
                    plx_err_from_params = data_hdu.data[cfg.MASTER_COLS['params_err']][:, 4].copy()

                with np.errstate(divide='ignore', invalid='ignore'):
                    dist_from_plx = 1.0 / plx_from_params; dist_from_plx[plx_from_params <= 0] = np.nan
                    rel_err_d = np.abs(dist_err_arr / dist_arr); rel_err_p = np.abs(plx_err_from_params / plx_from_params)

                best_dist = np.full(self.nrows, np.nan)
                hd = np.isfinite(dist_arr) & (dist_arr > 0); hp = np.isfinite(dist_from_plx) & (dist_from_plx > 0)
                both = hd & hp
                best_dist[(both & (rel_err_d <= rel_err_p)) | (hd & ~hp)] = dist_arr[(both & (rel_err_d <= rel_err_p)) | (hd & ~hp)]
                best_dist[(both & (rel_err_d > rel_err_p)) | (hp & ~hd)] = dist_from_plx[(both & (rel_err_d > rel_err_p)) | (hp & ~hd)]
                data_dict['best_dist'] = best_dist
                data_dict['plx_from_params'] = plx_from_params
                data_dict['plx_err_from_params'] = plx_err_from_params
                self.logger.info(f"  ✓ {np.sum(np.isfinite(best_dist)):,} with valid distance")

                self.df = pd.DataFrame(data_dict)

            ra_c, dec_c = cfg.MASTER_COLS['ra'], cfg.MASTER_COLS['dec']
            self.df = self.df.dropna(subset=[ra_c, dec_c]); self.nrows = len(self.df)
            self._build_kdtree()

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.df.to_parquet(data_cp)
                np.savez_compressed(tree_cp, coords=self.coords_3d)

            self.logger.info(f"✓ Master catalog loaded in {time.time()-t0:.1f}s")
            gcmod.collect(); return True
        except Exception as e:
            self.logger.error(f"Failed: {e}"); import traceback; traceback.print_exc(); return False

    def _build_kdtree(self):
        ra = np.radians(self.df[cfg.MASTER_COLS['ra']].values)
        dec = np.radians(self.df[cfg.MASTER_COLS['dec']].values)
        self.coords_3d = np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
        self.tree = cKDTree(self.coords_3d); self._compute_max_chord()

    def _compute_max_chord(self):
        self.max_chord = 2 * np.sin(np.radians(cfg.CROSSMATCH_RADIUS_ARCSEC / 3600.0) / 2)

    def query(self, ra, dec):
        ra_r, dec_r = np.radians(ra), np.radians(dec)
        coords = np.column_stack([np.cos(dec_r)*np.cos(ra_r), np.cos(dec_r)*np.sin(ra_r), np.sin(dec_r)])
        dists, idxs = self.tree.query(coords, k=1, distance_upper_bound=self.max_chord)
        valid = np.isfinite(dists)
        return idxs[valid], np.where(valid)[0], np.degrees(2*np.arcsin(dists[valid]/2))*3600

    def get_matched_data(self, master_idx):
        return self.df.iloc[master_idx].reset_index(drop=True)


# ============================================================================
# ALGORITHMS (unchanged)
# ============================================================================

def build_measurement_covariance(pmra_err, pmdec_err, corr=None):
    n = len(pmra_err); C = np.zeros((n, 2, 2))
    C[:, 0, 0] = pmra_err**2; C[:, 1, 1] = pmdec_err**2
    if corr is not None:
        cov = corr * pmra_err * pmdec_err; C[:, 0, 1] = cov; C[:, 1, 0] = cov
    return C

def algorithm_gmm_with_errors(pmra, pmdec, c_pmra, c_pmdec, pmra_err=None, pmdec_err=None, corr=None, min_stars=10):
    n = len(pmra); vm = ~(np.isnan(pmra)|np.isnan(pmdec)); nv = np.sum(vm)
    if nv < min_stars: return np.full(n, 0.5), {'status': 'insufficient_data', 'algorithm': 'GMM'}
    X = np.column_stack([pmra[vm], pmdec[vm]])
    try:
        gmm = GaussianMixture(n_components=cfg.GMM_N_COMPONENTS, max_iter=cfg.GMM_MAX_ITER,
                              n_init=cfg.GMM_N_INIT, random_state=cfg.GMM_RANDOM_STATE, covariance_type='full')
        gmm.fit(X)
        ref = np.array([c_pmra, c_pmdec])
        ci = np.argmin([np.linalg.norm(gmm.means_[i]-ref) for i in range(cfg.GMM_N_COMPONENTS)])
        fi = 1-ci
        mu_c, S_c = gmm.means_[ci], gmm.covariances_[ci]
        mu_f, S_f = gmm.means_[fi], gmm.covariances_[fi]
        eta = gmm.weights_[ci]
        P = np.zeros(nv)
        if pmra_err is not None and pmdec_err is not None:
            C = build_measurement_covariance(pmra_err[vm], pmdec_err[vm], corr[vm] if corr is not None else None)
            for i in range(nv):
                try:
                    pc = eta*multivariate_normal.pdf(X[i], mu_c, S_c+C[i])
                    pf = (1-eta)*multivariate_normal.pdf(X[i], mu_f, S_f+C[i])
                    P[i] = pc/(pc+pf) if (pc+pf)>0 else 0.5
                except: P[i]=0.5
        else: P = gmm.predict_proba(X)[:, ci]
        Pm = np.full(n, np.nan); Pm[vm] = P
        return Pm, {'status':'success','algorithm':'GMM','mu_cluster':mu_c.tolist(),'Sigma_cluster':S_c.tolist(),
                     'mu_field':mu_f.tolist(),'Sigma_field':S_f.tolist(),'eta':float(eta),
                     'pm_dispersion':float(np.sqrt(np.trace(S_c))),'center_pmra':c_pmra,'center_pmdec':c_pmdec}
    except Exception as e:
        return np.full(n, 0.5), {'status':f'error: {e}','algorithm':'GMM'}

def algorithm_dbscan(pmra, pmdec, c_pmra, c_pmdec, eps=0.25, min_samples=5, use_hdbscan=False, min_stars=10):
    n = len(pmra); vm = ~(np.isnan(pmra)|np.isnan(pmdec)); nv = np.sum(vm)
    if nv < min_stars: return np.full(n, 0.5), {'status':'insufficient_data','algorithm':'DBSCAN'}
    X = np.column_stack([pmra[vm], pmdec[vm]])
    sc = StandardScaler(); Xs = sc.fit_transform(X); rs = sc.transform([[c_pmra, c_pmdec]])[0]
    try:
        if use_hdbscan and HAS_HDBSCAN:
            cl = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3); labels = cl.fit_predict(Xs); an='HDBSCAN'
        else:
            cl = DBSCAN(eps=eps, min_samples=min_samples); labels = cl.fit_predict(Xs); an='DBSCAN'
        ul = set(labels)-{-1}
        if len(ul)==0:
            d=np.linalg.norm(Xs-rs,axis=1); P=np.exp(-d**2/2)
            Pm=np.full(n,np.nan);Pm[vm]=P; return Pm,{'status':'no_cluster_found','algorithm':an,'center_pmra':c_pmra,'center_pmdec':c_pmdec}
        bc = min(ul, key=lambda l: np.linalg.norm(Xs[labels==l].mean(axis=0)-rs))
        cm = labels==bc; P=np.zeros(nv)
        if use_hdbscan and HAS_HDBSCAN and hasattr(cl,'probabilities_'): P[cm]=cl.probabilities_[cm]
        else: P[cm]=1.0
        cc=Xs[cm].mean(axis=0); cs=Xs[cm].std(axis=0).mean()
        nc=~cm
        if np.any(nc): d=np.linalg.norm(Xs[nc]-cc,axis=1); P[nc]=np.exp(-d**2/(2*cs**2))*0.3
        Pm=np.full(n,np.nan); Pm[vm]=P
        cpm=X[cm]; mu=cpm.mean(axis=0) if len(cpm)>0 else np.array([c_pmra,c_pmdec])
        Sig=np.cov(cpm.T) if len(cpm)>2 else np.eye(2)*0.1
        return Pm,{'status':'success','algorithm':an,'n_cluster_members':int(np.sum(cm)),
                    'mu_cluster':mu.tolist(),'Sigma_cluster':Sig.tolist(),
                    'pm_dispersion':float(np.sqrt(np.trace(Sig))) if len(cpm)>2 else np.nan,
                    'eta':float(np.sum(cm)/nv),'center_pmra':c_pmra,'center_pmdec':c_pmdec}
    except Exception as e:
        return np.full(n,0.5),{'status':f'error: {e}','algorithm':'DBSCAN'}

def algorithm_hybrid(pmra, pmdec, c_pmra, c_pmdec, pmra_err=None, pmdec_err=None, corr=None, min_stars=10):
    n=len(pmra); vm=~(np.isnan(pmra)|np.isnan(pmdec)); nv=np.sum(vm)
    if nv<min_stars: return np.full(n,0.5),{'status':'insufficient_data','algorithm':'Hybrid'}
    X=np.column_stack([pmra[vm],pmdec[vm]]); sc=StandardScaler(); Xs=sc.fit_transform(X); rs=sc.transform([[c_pmra,c_pmdec]])[0]
    try:
        db=DBSCAN(eps=cfg.DBSCAN_EPS_CLEANUP,min_samples=cfg.DBSCAN_MIN_SAMPLES_CLEANUP); labels=db.fit_predict(Xs)
        ul=set(labels)-{-1}
        if len(ul)==0: return algorithm_gmm_with_errors(pmra,pmdec,c_pmra,c_pmdec,pmra_err,pmdec_err,corr,min_stars)
        bc=min(ul,key=lambda l:np.linalg.norm(Xs[labels==l].mean(axis=0)-rs)); cmask=labels==bc
        if np.sum(cmask)<min_stars: return algorithm_gmm_with_errors(pmra,pmdec,c_pmra,c_pmdec,pmra_err,pmdec_err,corr,min_stars)
        Xc=X[cmask]; gmm=GaussianMixture(n_components=2,max_iter=cfg.GMM_MAX_ITER,n_init=cfg.GMM_N_INIT,random_state=cfg.GMM_RANDOM_STATE); gmm.fit(Xc)
        ref=np.array([c_pmra,c_pmdec]); ci=np.argmin([np.linalg.norm(gmm.means_[i]-ref) for i in range(2)])
        mu_c,S_c=gmm.means_[ci],gmm.covariances_[ci]; mu_f,S_f=gmm.means_[1-ci],gmm.covariances_[1-ci]; eta=gmm.weights_[ci]
        P=np.zeros(nv)
        for i in range(nv):
            try:
                pc=eta*multivariate_normal.pdf(X[i],mu_c,S_c); pf=(1-eta)*multivariate_normal.pdf(X[i],mu_f,S_f)
                P[i]=pc/(pc+pf) if (pc+pf)>0 else 0.5
            except: P[i]=0.5
        P[labels==-1]*=0.5; Pm=np.full(n,np.nan); Pm[vm]=P
        return Pm,{'status':'success','algorithm':'Hybrid','mu_cluster':mu_c.tolist(),'Sigma_cluster':S_c.tolist(),
                    'mu_field':mu_f.tolist(),'Sigma_field':S_f.tolist(),'eta':float(eta),
                    'pm_dispersion':float(np.sqrt(np.trace(S_c))),'center_pmra':c_pmra,'center_pmdec':c_pmdec}
    except Exception as e:
        return np.full(n,0.5),{'status':f'error: {e}','algorithm':'Hybrid'}

def algorithm_stream_dbscan(pmra, pmdec, c_pmra, c_pmdec, eps=0.4, min_samples=3, min_stars=10):
    n=len(pmra); vm=~(np.isnan(pmra)|np.isnan(pmdec)); nv=np.sum(vm)
    if nv<min_stars: return np.full(n,0.5),{'status':'insufficient_data','algorithm':'Stream-DBSCAN'}
    X=np.column_stack([pmra[vm],pmdec[vm]]); sc=StandardScaler(); Xs=sc.fit_transform(X); rs=sc.transform([[c_pmra,c_pmdec]])[0]
    try:
        db=DBSCAN(eps=eps,min_samples=min_samples); labels=db.fit_predict(Xs); ul=set(labels)-{-1}
        if len(ul)==0:
            d=np.linalg.norm(Xs-rs,axis=1); P=np.exp(-d**2/(2*eps**2))
            Pm=np.full(n,np.nan); Pm[vm]=P; return Pm,{'status':'no_cluster_found','algorithm':'Stream-DBSCAN','center_pmra':c_pmra,'center_pmdec':c_pmdec}
        bc=min(ul,key=lambda l:np.min(np.linalg.norm(Xs[labels==l]-rs,axis=1))); cm=labels==bc; cp=Xs[cm]; P=np.zeros(nv)
        if len(cp)>0:
            tr=cKDTree(cp)
            for idx in np.where(cm)[0]:
                d,_=tr.query(Xs[idx],k=min(3,len(cp))); P[idx]=np.clip(1.0-np.mean(d[1:] if len(d)>1 else [0])/eps,0.5,1.0)
        nc=~cm
        if np.any(nc):
            for idx in np.where(nc)[0]:
                d=np.min(np.linalg.norm(cp-Xs[idx],axis=1)); P[idx]=0.3*np.exp(-d**2/(2*eps**2))
        Pm=np.full(n,np.nan); Pm[vm]=P
        cpm=X[cm]; mu=cpm.mean(axis=0) if len(cpm)>0 else np.array([c_pmra,c_pmdec])
        Sig=np.cov(cpm.T) if len(cpm)>2 else np.eye(2)*0.1
        return Pm,{'status':'success','algorithm':'Stream-DBSCAN','n_stream_members':int(np.sum(cm)),
                    'mu_cluster':mu.tolist(),'Sigma_cluster':Sig.tolist(),
                    'pm_dispersion':float(np.sqrt(np.trace(Sig))) if len(cpm)>2 else np.nan,
                    'eta':float(np.sum(cm)/nv),'center_pmra':c_pmra,'center_pmdec':c_pmdec}
    except Exception as e:
        return np.full(n,0.5),{'status':f'error: {e}','algorithm':'Stream-DBSCAN'}

def compute_adaptive_membership(pmra, pmdec, c_pmra, c_pmdec, obj_type, pmra_err=None, pmdec_err=None, corr=None, min_stars=10):
    ot = obj_type.upper()
    if ot=='GC': return algorithm_gmm_with_errors(pmra,pmdec,c_pmra,c_pmdec,pmra_err,pmdec_err,corr,min_stars)
    elif ot=='OC': return algorithm_dbscan(pmra,pmdec,c_pmra,c_pmdec,cfg.DBSCAN_EPS_OC,cfg.DBSCAN_MIN_SAMPLES_OC,HAS_HDBSCAN,min_stars)
    elif ot=='DW': return algorithm_hybrid(pmra,pmdec,c_pmra,c_pmdec,pmra_err,pmdec_err,corr,min_stars)
    elif ot in ['SGR','STREAM']: return algorithm_stream_dbscan(pmra,pmdec,c_pmra,c_pmdec,cfg.DBSCAN_EPS_STREAM,cfg.DBSCAN_MIN_SAMPLES_STREAM,min_stars)
    else: return algorithm_gmm_with_errors(pmra,pmdec,c_pmra,c_pmdec,pmra_err,pmdec_err,corr,min_stars)


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def _get_member_rv(member_df, cols):
    rv_col = cols.get('rv'); rv_err_col = cols.get('rv_err')
    rv = rv_err = None
    if rv_col and rv_col in member_df.columns:
        rv = pd.to_numeric(member_df[rv_col], errors='coerce').values
    else:
        for alt in ['RV_weighted_avg','vlos','vlos_systemic','radial_velocity','RV','rv','Vrad','HRV']:
            if alt in member_df.columns: rv = pd.to_numeric(member_df[alt], errors='coerce').values; break
    if rv_err_col and rv_err_col in member_df.columns:
        rv_err = pd.to_numeric(member_df[rv_err_col], errors='coerce').values
    return rv, rv_err

def _clean(arr, plo=1, phi=99, min_n=3):
    """Clean array: finite, clip percentile extremes."""
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
    """Add median ± MAD text box."""
    d = data[np.isfinite(data)]
    if len(d) < 2: return
    med = np.median(d); mad = median_abs_deviation(d, nan_policy='omit')
    txt = f"{prefix}Med = {med:.2f}\nMAD = {mad:.2f}\nN = {len(d)}"
    locs = {'upper right': (0.97, 0.97, 'right', 'top'),
            'upper left': (0.03, 0.97, 'left', 'top'),
            'lower right': (0.97, 0.03, 'right', 'bottom')}
    x, y, ha, va = locs.get(loc, (0.97, 0.97, 'right', 'top'))
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha=ha, va=va, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9, lw=1.5))

def _hist_step_kde_med(ax, data, color, label, bins_range=None, n_bins=30):
    """Standard paper histogram: step + KDE + median line + median±MAD box."""
    d = _clean(data)
    if len(d) < 3:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
        return np.array([]), np.array([])
    if bins_range is None:
        lo, hi = np.percentile(d, [0.5, 99.5]); rng = max(hi-lo, 1.0)
        bins_range = [lo - 0.1*rng, hi + 0.1*rng]
    counts, bins, _ = ax.hist(d, bins=n_bins, range=bins_range, alpha=0)
    ax.step(bins[:-1], counts, where='post', color=color, linewidth=2.5, label=label)
    _safe_kde(ax, d, bins, color)
    med = np.median(d)
    ax.axvline(med, color=color, ls=':', lw=2.5, alpha=0.9)
    return counts, bins


def _best_inset_loc(ax, data1, data2):
    """Pick the least-dense quadrant (matplotlib loc code) for inset placement.
    Checks upper-left, upper-right, lower-left, lower-right by counting data points."""
    all_d = np.concatenate([d for d in [data1, data2] if len(d) > 0])
    if len(all_d) < 2:
        return 'lower right'
    mid = np.median(all_d)
    # Get current y-axis: count-based, so "upper" = high count, "lower" = low count
    # For histogram: left half = data < median, right half = data >= median
    n_left = np.sum(all_d < mid)
    n_right = np.sum(all_d >= mid)
    # Place inset on the side with fewer data points, at the top (away from x-axis labels)
    # matplotlib loc codes: 1=upper right, 2=upper left, 3=lower left, 4=lower right
    if n_left <= n_right:
        return 'upper left'   # fewer points on left side
    else:
        return 'upper right'  # fewer points on right side


# ============================================================================
# INDIVIDUAL PLOTS (V4: paper quality)
# ============================================================================

def plot_individual_panels(matched_df, obj_name, obj_type, algo_info,
                           ref_dist=None, ref_dist_err=None,
                           member_plx=None, member_rv=None, member_rv_err=None,
                           save_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    set_paper_style()

    if save_dir is None: save_dir = os.path.join(cfg.OUTPUT_DIR, 'individual_plots')
    os.makedirs(save_dir, exist_ok=True)

    # Determine RV availability
    master_rv = _clean(matched_df['best_rv'].values) if 'best_rv' in matched_df.columns else np.array([])
    mem_rv_clean = _clean(member_rv) if member_rv is not None else np.array([])
    show_rv = len(master_rv) >= 3 or len(mem_rv_clean) >= 3

    if show_rv:
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    af = axes.flatten(); nc = 3 if show_rv else 2

    # Title
    title_str = f'{obj_type}: {obj_name}'
    if ref_dist is not None and np.isfinite(ref_dist):
        title_str += f'  ($d_{{\\rm ref}}$ = {ref_dist:.1f} kpc)'
    elif obj_type == 'SGR':
        title_str += f'  (distance bin)'
    fig.suptitle(title_str, fontsize=22, fontweight='bold', family='serif', y=0.98)

    # ===== Panel 1: PM =====
    ax = af[0]
    if 'P_mem' in matched_df.columns:
        v = matched_df['P_mem'].notna()
        sc = ax.scatter(matched_df.loc[v, 'pmra'], matched_df.loc[v, 'pmdec'],
                        c=matched_df.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                        s=25, edgecolors='k', linewidths=0.3, vmin=0, vmax=1)
        cb = plt.colorbar(sc, ax=ax); cb.set_label('$P_{\\rm mem}$', fontsize=14, fontweight='bold')
    if algo_info.get('Sigma_cluster') is not None and obj_type != 'SGR':
        mu = algo_info['mu_cluster']; sig = np.array(algo_info['Sigma_cluster'])
        w, v2 = np.linalg.eigh(sig); ang = np.degrees(np.arctan2(v2[1,0], v2[0,0]))
        for ns in [1, 2]:
            ell = Ellipse(xy=mu, width=2*ns*np.sqrt(max(w[0],0)), height=2*ns*np.sqrt(max(w[1],0)),
                         angle=ang, fill=False, edgecolor='lime', lw=2.5 if ns==1 else 2, ls='-' if ns==1 else '--')
            ax.add_patch(ell)
        ax.scatter(*mu, marker='x', s=120, c='lime', linewidths=3, zorder=10)
    if 'center_pmra' in algo_info:
        ax.scatter(algo_info['center_pmra'], algo_info['center_pmdec'], marker='*', s=200, c='blue', edgecolors='white', zorder=11)
    ax.set_xlabel('$\\mu_{\\alpha}\\cos\\delta$ (mas yr$^{-1}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel('$\\mu_{\\delta}$ (mas yr$^{-1}$)', fontsize=16, fontweight='bold')
    ax.set_title('Proper Motion', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.2); ax.set_aspect('equal', 'datalim')

    # ===== Panel 2: RA-Dec =====
    ax = af[1]
    if 'P_mem' in matched_df.columns:
        v = matched_df['P_mem'].notna()
        sc = ax.scatter(matched_df.loc[v, 'ra'], matched_df.loc[v, 'dec'],
                        c=matched_df.loc[v, 'P_mem'], cmap=cfg.CMAP_PMEM,
                        s=25, edgecolors='k', linewidths=0.3, vmin=0, vmax=1)
        cb = plt.colorbar(sc, ax=ax); cb.set_label('$P_{\\rm mem}$', fontsize=14, fontweight='bold')
    ax.set_xlabel('RA (deg)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Dec (deg)', fontsize=16, fontweight='bold')
    ax.set_title('Sky Position', fontsize=18, fontweight='bold')
    ax.invert_xaxis(); ax.grid(True, alpha=0.2)

    # ===== Panel 3: Distance (row1, col0) =====
    ax = af[nc]
    md = _clean(matched_df['best_dist'].values) if 'best_dist' in matched_df.columns else np.array([])
    md = md[(md > 0) & (md < 300)]
    dp = np.array([])
    if member_plx is not None:
        pv = np.asarray(member_plx); pv = pv[pv > 0]; dp = 1.0/pv; dp = dp[(dp > 0)&(dp < 300)]

    if obj_type in ['SGR', 'STREAM']:
        all_v = [x for x in [md, dp] if len(x) > 0]
        if all_v:
            comb = np.concatenate(all_v); lo, hi = np.percentile(comb, [1, 99]); rng = max(hi-lo, 1)
            br = [max(0, lo-0.1*rng), hi+0.1*rng]
        else: br = [0, 100]
        if len(md) > 0: _hist_step_kde_med(ax, md, cfg.COL_MASTER, f'Master (n={len(md)})', br)
        if len(dp) > 0: _hist_step_kde_med(ax, dp, cfg.COL_MEMBER, f'1/plx (n={len(dp)})', br)
        if len(md) > 0: _med_mad_box(ax, md, cfg.COL_MASTER, 'upper right', 'Master ')
        ax.set_xlabel('Distance (kpc)', fontsize=16, fontweight='bold')
        ax.set_title('Distance Distribution', fontsize=18, fontweight='bold')
    else:
        if ref_dist is not None and np.isfinite(ref_dist):
            dm = ref_dist - md if len(md) > 0 else np.array([])
            dpx = ref_dist - dp if len(dp) > 0 else np.array([])
            all_d = [x for x in [dm, dpx] if len(x) > 0]
            if all_d:
                comb = np.concatenate(all_d); lo, hi = np.percentile(comb, [2, 98]); rng = max(hi-lo, 1)
                br = [lo-0.15*rng, hi+0.15*rng]
            else: br = [-50, 50]
            if len(dm) > 0: _hist_step_kde_med(ax, dm, cfg.COL_MASTER, f'$\\Delta$(ref$-$master)', br); _med_mad_box(ax, dm, cfg.COL_MASTER, 'upper right', 'Master ')
            if len(dpx) > 0: _hist_step_kde_med(ax, dpx, cfg.COL_MEMBER, f'$\\Delta$(ref$-$1/plx)', br); _med_mad_box(ax, dpx, cfg.COL_MEMBER, 'upper left', 'Plx ')
            ax.axvline(0, color='k', lw=2.5, alpha=0.8)
            if ref_dist_err is not None and np.isfinite(ref_dist_err):
                ax.axvspan(-ref_dist_err, ref_dist_err, alpha=0.12, color='gray')
            ax.set_xlabel('$\\Delta$ Distance (kpc)', fontsize=16, fontweight='bold')
            ax.set_title(f'Distance Offset (ref = {ref_dist:.1f} kpc)', fontsize=18, fontweight='bold')
        else:
            if len(md) > 0: _hist_step_kde_med(ax, md, cfg.COL_MASTER, f'Master (n={len(md)})'); _med_mad_box(ax, md, cfg.COL_MASTER)
            ax.set_xlabel('Distance (kpc)', fontsize=16, fontweight='bold')
            ax.set_title('Distance Distribution', fontsize=18, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best'); ax.grid(True, alpha=0.2, axis='y')

    # ===== Panel 4: RV (if available) =====
    if show_rv:
        ax = af[nc + 1]
        # Compute common range
        all_rv = [x for x in [master_rv, mem_rv_clean] if len(x) > 0]
        if all_rv:
            comb = np.concatenate(all_rv); lo, hi = np.percentile(comb, [1, 99]); rng = max(hi-lo, 5)
            br = [lo-0.1*rng, hi+0.1*rng]
        else: br = [-200, 200]

        if len(master_rv) >= 3:
            _hist_step_kde_med(ax, master_rv, cfg.COL_MASTER, f'Master (n={len(master_rv)})', br)
            _med_mad_box(ax, master_rv, cfg.COL_MASTER, 'upper right', 'Master ')
        if len(mem_rv_clean) >= 3:
            _hist_step_kde_med(ax, mem_rv_clean, cfg.COL_MEMBER, f'Member (n={len(mem_rv_clean)})', br)
            _med_mad_box(ax, mem_rv_clean, cfg.COL_MEMBER, 'upper left', 'Member ')
        ax.set_xlabel('Radial Velocity (km s$^{-1}$)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Count', fontsize=16, fontweight='bold')
        ax.set_title('Radial Velocity', fontsize=18, fontweight='bold')
        ax.legend(fontsize=11, loc='best'); ax.grid(True, alpha=0.2, axis='y')

        # ΔRV inset — only when paired (same-length) master & member RV exist
        if len(master_rv) >= 3 and len(mem_rv_clean) >= 3:
            if 'best_rv' in matched_df.columns and member_rv is not None:
                mrv_matched = matched_df['best_rv'].values
                mrv_mem = np.asarray(member_rv, dtype=float)
                # Only compute element-wise ΔRV if arrays are paired (same length)
                if len(mrv_matched) == len(mrv_mem):
                    both_valid = np.isfinite(mrv_matched) & np.isfinite(mrv_mem)
                    if np.sum(both_valid) >= 3:
                        delta_rv = mrv_matched[both_valid] - mrv_mem[both_valid]
                        delta_rv = delta_rv[np.isfinite(delta_rv)]
                        if len(delta_rv) >= 3:
                            # Auto-position: find least-dense quadrant
                            inset_loc = _best_inset_loc(ax, master_rv, mem_rv_clean)
                            axins = inset_axes(ax, width="35%", height="35%", loc=inset_loc, borderpad=2)
                            axins.set_facecolor('white')  # opaque background
                            axins.patch.set_alpha(1.0)
                            axins.hist(delta_rv, bins=20, color='gray', alpha=0.7, edgecolor='black', lw=1)
                            axins.axvline(0, color='k', lw=1.5)
                            med_drv = np.median(delta_rv)
                            axins.axvline(med_drv, color='purple', ls=':', lw=2)
                            axins.set_xlabel('$\\Delta$RV', fontsize=10, fontweight='bold')
                            axins.set_title(f'Med={med_drv:.1f}', fontsize=10, fontweight='bold')
                            axins.tick_params(labelsize=8)
                            for spine in axins.spines.values():
                                spine.set_linewidth(1.5); spine.set_edgecolor('black')
                            axins.set_zorder(10)  # draw on top

        # Stats panel (top right)
        ax_s = af[2]
        _plot_stats_panel(ax_s, matched_df, algo_info, obj_name, obj_type, ref_dist)
        af[5].axis('off')  # blank
    else:
        ax_s = af[nc + 1]
        _plot_stats_panel(ax_s, matched_df, algo_info, obj_name, obj_type, ref_dist)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe = str(obj_name).replace(' ', '_').replace('/', '_').replace('-', '_')
    out = os.path.join(save_dir, f"{obj_type}_{safe}.{cfg.SAVE_FORMAT}")
    plt.savefig(out, dpi=cfg.PLOT_DPI); plt.close(); return out


def _plot_stats_panel(ax, mdf, algo, name, otype, ref_dist):
    ax.axis('off')
    lines = [f"Object:  {name}", f"Type:    {otype}", f"Algo:    {algo.get('algorithm','N/A')}",
             f"Status:  {algo.get('status','N/A')}", f"N match: {len(mdf)}"]
    if 'P_mem' in mdf.columns:
        pm = mdf['P_mem'].dropna()
        lines += [f"N(P>0.8): {int(np.sum(pm>cfg.P_MEM_HIGH))}", f"<P_mem>:  {pm.mean():.3f}"]
    if 'eta' in algo: lines.append(f"η:       {algo['eta']:.3f}")
    if ref_dist is not None and np.isfinite(ref_dist): lines.append(f"d_ref:   {ref_dist:.2f} kpc")
    if 'best_dist' in mdf.columns:
        d = mdf['best_dist'].dropna()
        if len(d) > 0: lines.append(f"d_med:   {d.median():.2f} kpc")
    if 'best_rv' in mdf.columns:
        r = mdf['best_rv'].dropna()
        if len(r) > 0: lines.append(f"RV_med:  {r.median():.1f} km/s")
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='black', alpha=0.9, lw=1.5))
    ax.set_title('Summary', fontsize=18, fontweight='bold')


# ============================================================================
# PROCESSING PIPELINES
# ============================================================================

def load_gc_reference_distances(filepath, logger):
    if not filepath or not os.path.exists(filepath):
        logger.warning(f"GC_dist.csv not found: {filepath}"); return {}
    logger.info(f"Loading GC reference distances from {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8-sig'); df.columns = df.columns.str.strip()
    gc_dists = {}
    def empty(val):
        if pd.isna(val): return True
        s = str(val).strip(); return s in ['','-','–','—','−','nan','NaN','N/A','n/a'] or len(s)==0
    n_lit = n_mean = 0
    for _, row in df.iterrows():
        name = str(row[cfg.GC_DIST_COLS['name']]).strip(); dist = err = np.nan; src = None
        try:
            v = row[cfg.GC_DIST_COLS['lit_dist']]
            if not empty(v): dist = float(v); ev = row[cfg.GC_DIST_COLS['lit_dist_err']]; err = float(str(ev).replace('+','').strip()) if not empty(ev) else np.nan; src='lit'
        except: pass
        if not np.isfinite(dist):
            try:
                v = row[cfg.GC_DIST_COLS['mean_dist']]
                if not empty(v): dist = float(v); ev = row[cfg.GC_DIST_COLS['mean_dist_err']]; err = float(str(ev).replace('+','').strip()) if not empty(ev) else np.nan; src='mean'
            except: pass
        if np.isfinite(dist):
            if src=='lit': n_lit+=1
            else: n_mean+=1
            nn = normalize_name(name); gc_dists[nn] = (dist, err); gc_dists[name] = (dist, err)
            gc_dists[name.lower()] = (dist, err); gc_dists[name.upper()] = (dist, err)
    logger.info(f"  Loaded {n_lit} lit + {n_mean} mean = {n_lit+n_mean} total")
    return gc_dists


def _standard_match_columns(matched_df):
    """Add standard column aliases from master suffixed columns."""
    matched_df['pmra'] = matched_df.get(f"{cfg.MASTER_COLS['pmra']}_master", pd.Series(dtype=float))
    matched_df['pmdec'] = matched_df.get(f"{cfg.MASTER_COLS['pmdec']}_master", pd.Series(dtype=float))
    matched_df['ra'] = matched_df.get(f"{cfg.MASTER_COLS['ra']}_master", pd.Series(dtype=float))
    matched_df['dec'] = matched_df.get(f"{cfg.MASTER_COLS['dec']}_master", pd.Series(dtype=float))
    matched_df['best_dist'] = matched_df.get('best_dist_master', pd.Series(dtype=float))
    matched_df['best_rv'] = matched_df.get('best_rv_master', pd.Series(dtype=float))
    return matched_df


def process_gc_members(master, gc_dists, logger):
    if not cfg.GC_MEMBERS_FILE or not os.path.exists(cfg.GC_MEMBERS_FILE):
        logger.info("GC members file not found - skipping"); return []
    logger.info("\n" + "-"*50 + "\nProcessing GLOBULAR CLUSTERS...\n" + "-"*50)
    df = pd.read_csv(cfg.GC_MEMBERS_FILE); cols = cfg.GC_MEM_COLS; kc = cols['key']
    clusters = sorted(df[kc].unique()) if kc in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters")
    results = []
    for i, cn in enumerate(clusters):
        logger.info(f"\n[{i+1}/{len(clusters)}] {cn}")
        cdf = df[df[kc]==cn].copy()
        if len(cdf) < cfg.MIN_STARS_FOR_ANALYSIS: logger.info(f"  Skipping - {len(cdf)} members"); continue
        ra = cdf[cols['ra']].values; dec = cdf[cols['dec']].values
        vm = ~(np.isnan(ra)|np.isnan(dec)); cdf = cdf[vm].reset_index(drop=True)
        ra = ra[vm]; dec = dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue

        midx, memidx, seps = master.query(ra, dec)
        logger.info(f"  Members: {len(ra)}, Matched: {len(midx)} ({100*len(midx)/len(ra):.1f}%)")

        c_pmra = cdf[cols['pmra']].median(); c_pmdec = cdf[cols['pmdec']].median()
        rd, rde = None, None
        nn = normalize_name(cn)
        for nt in [nn, cn, cn.lower(), cn.upper()]:
            if nt in gc_dists: rd, rde = gc_dists[nt]; logger.info(f"  Ref dist: {rd:.2f} kpc"); break
        if rd is None: logger.warning(f"  No ref dist for '{cn}'")

        algo = {'status':'no_matches','algorithm':'None','center_pmra':c_pmra,'center_pmdec':c_pmdec}
        mdf = None
        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm = cdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx); mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1); mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            pma = mdf['pmra'].values; pmd = mdf['pmdec'].values
            pe = mdf.get(f"{cfg.MASTER_COLS['pmra_err']}_master", pd.Series(dtype=float)).values
            pde = mdf.get(f"{cfg.MASTER_COLS['pmdec_err']}_master", pd.Series(dtype=float)).values
            P, algo = compute_adaptive_membership(pma, pmd, c_pmra, c_pmdec, 'GC', pe, pde, None, cfg.MIN_STARS_FOR_ANALYSIS)
            mdf['P_mem'] = P
            logger.info(f"  Algo: {algo['algorithm']} | n(P>0.8): {np.sum(P>cfg.P_MEM_HIGH)}")
            mplx = mm[cols['parallax']].values if cols.get('parallax') and cols['parallax'] in mm.columns else None
            mrv, mrve = _get_member_rv(mm, cols)
            plot_individual_panels(mdf, cn, 'GC', algo, rd, rde, mplx, mrv, mrve)

        results.append({'cluster_name':cn,'obj_type':'GC','member_df':cdf,'matched_df':mdf,
                        'algo_info':algo,'mem_cols':cols,'n_members':len(cdf),'n_matched':len(midx),
                        'ref_dist':rd,'ref_dist_err':rde})
        gcmod.collect()
    return results


def process_oc_members(master, logger):
    if not cfg.OC_MEMBERS_FILE or not os.path.exists(cfg.OC_MEMBERS_FILE):
        logger.info("OC members file not found - skipping"); return []
    logger.info("\n" + "-"*50 + "\nProcessing OPEN CLUSTERS...\n" + "-"*50)
    df = pd.read_csv(cfg.OC_MEMBERS_FILE); cols = cfg.OC_MEM_COLS; kc = cols['key']
    clusters = sorted(df[kc].unique()) if kc in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters")
    results = []
    for i, cn in enumerate(clusters):
        logger.info(f"\n[{i+1}/{len(clusters)}] {cn}")
        cdf = df[df[kc]==cn].copy()
        if len(cdf) < cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra = cdf[cols['ra']].values; dec = cdf[cols['dec']].values
        vm = ~(np.isnan(ra)|np.isnan(dec)); cdf = cdf[vm].reset_index(drop=True); ra=ra[vm]; dec=dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue

        plx = cdf[cols['parallax']].values; vp = plx[plx>0]
        rd = np.mean(1.0/vp) if len(vp)>0 else None
        rde = np.std(1.0/vp) if len(vp)>0 else None
        if rd: logger.info(f"  Ref dist (1/plx): {rd:.2f} ± {rde:.2f} kpc")

        midx, memidx, seps = master.query(ra, dec)
        logger.info(f"  Members: {len(ra)}, Matched: {len(midx)} ({100*len(midx)/len(ra):.1f}%)")
        c_pmra = cdf[cols['pmra']].median(); c_pmdec = cdf[cols['pmdec']].median()
        algo = {'status':'no_matches','algorithm':'None','center_pmra':c_pmra,'center_pmdec':c_pmdec}
        mdf = None
        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm = cdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx); mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1); mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            P, algo = compute_adaptive_membership(mdf['pmra'].values, mdf['pmdec'].values, c_pmra, c_pmdec, 'OC',
                                                  None, None, None, cfg.MIN_STARS_FOR_ANALYSIS)
            mdf['P_mem'] = P
            logger.info(f"  Algo: {algo['algorithm']} | n(P>0.8): {np.sum(P>cfg.P_MEM_HIGH)}")
            mplx = mm[cols['parallax']].values
            mrv, mrve = _get_member_rv(mm, cols)
            plot_individual_panels(mdf, cn, 'OC', algo, rd, rde, mplx, mrv, mrve)

        results.append({'cluster_name':cn,'obj_type':'OC','member_df':cdf,'matched_df':mdf,
                        'algo_info':algo,'mem_cols':cols,'n_members':len(cdf),'n_matched':len(midx),
                        'ref_dist':rd,'ref_dist_err':rde})
        gcmod.collect()
    return results


def process_sgr_members(master, logger):
    if not cfg.SGR_MEMBERS_FILE or not os.path.exists(cfg.SGR_MEMBERS_FILE):
        logger.info("SGR members file not found - skipping"); return []
    logger.info("\n" + "-"*50 + "\nProcessing SGR STREAM...\n" + "-"*50)
    df = pd.read_csv(cfg.SGR_MEMBERS_FILE); cols = cfg.SGR_MEM_COLS
    dc = cols.get('dist', 'dist')
    if dc not in df.columns: logger.warning(f"Dist col '{dc}' not found"); return []
    df = df.dropna(subset=[dc]); df[dc] = pd.to_numeric(df[dc], errors='coerce'); df = df.dropna(subset=[dc])
    logger.info(f"  Distance range: {df[dc].min():.1f} - {df[dc].max():.1f} kpc")

    be = np.arange(cfg.SGR_BIN_START_KPC, df[dc].max()+cfg.SGR_BIN_WIDTH_KPC, cfg.SGR_BIN_WIDTH_KPC)
    bl = [f'{be[i]:.0f}-{be[i+1]:.0f} kpc' for i in range(len(be)-1)]
    df['dist_bin'] = pd.cut(df[dc], bins=be, labels=bl, right=False); df = df.dropna(subset=['dist_bin'])
    bc = df['dist_bin'].value_counts(); vb = bc[bc>=cfg.SGR_MIN_STARS_PER_BIN].index.tolist()
    bs = sorted([(b, df[df['dist_bin']==b][dc].mean()) for b in vb], key=lambda x: x[1])
    bins = [x[0] for x in bs]
    logger.info(f"Found {len(bins)} valid bins")
    if len(bins)==0: return []

    results = []
    for i, bl_name in enumerate(bins):
        logger.info(f"\n[{i+1}/{len(bins)}] {bl_name}")
        bdf = df[df['dist_bin']==bl_name].copy()
        ra = bdf[cols['ra']].values; dec = bdf[cols['dec']].values
        vm = ~(np.isnan(ra)|np.isnan(dec)); bdf = bdf[vm].reset_index(drop=True); ra=ra[vm]; dec=dec[vm]
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS: continue

        midx, memidx, seps = master.query(ra, dec)
        logger.info(f"  Members: {len(ra)}, Matched: {len(midx)} ({100*len(midx)/len(ra):.1f}%)")
        c_pmra = bdf[cols['pmra']].median(); c_pmdec = bdf[cols['pmdec']].median()
        algo = {'status':'no_matches','algorithm':'None','center_pmra':c_pmra,'center_pmdec':c_pmdec}
        mdf = None
        if len(midx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            mm = bdf.iloc[memidx].reset_index(drop=True)
            mst = master.get_matched_data(midx); mst.columns = [f"{c}_master" for c in mst.columns]
            mdf = pd.concat([mm, mst], axis=1); mdf['xmatch_sep_arcsec'] = seps
            mdf = _standard_match_columns(mdf)
            P, algo = compute_adaptive_membership(mdf['pmra'].values, mdf['pmdec'].values, c_pmra, c_pmdec, 'SGR',
                                                  None, None, None, cfg.MIN_STARS_FOR_ANALYSIS)
            mdf['P_mem'] = P
            logger.info(f"  Algo: {algo['algorithm']} | n(P>0.8): {np.sum(P>cfg.P_MEM_HIGH)}")
            mplx = mm[cols['parallax']].values if cols.get('parallax') and cols['parallax'] in mm.columns else None
            mrv, mrve = _get_member_rv(mm, cols)
            plot_individual_panels(mdf, bl_name, 'SGR', algo, None, None, mplx, mrv, mrve)

        results.append({'cluster_name':bl_name,'obj_type':'SGR','member_df':bdf,'matched_df':mdf,
                        'algo_info':algo,'mem_cols':cols,'n_members':len(bdf),'n_matched':len(midx),'ref_dist':None})
        gcmod.collect()
    return results


def process_dwg_members(master, logger):
    if not cfg.DWG_MEMBERS_FILE or not os.path.exists(cfg.DWG_MEMBERS_FILE):
        logger.info("DWG members file not found - skipping"); return []
    logger.info("\n" + "-"*50 + "\nProcessing DWARF GALAXIES...\n" + "-"*50)
    df = pd.read_csv(cfg.DWG_MEMBERS_FILE); cols = cfg.DWG_MEM_COLS; kc = cols['key']
    gals = df[kc].unique() if kc in df.columns else ['ALL']
    logger.info(f"Found {len(gals)} galaxies")
    results = []
    for i, gn in enumerate(gals):
        logger.info(f"\n[{i+1}/{len(gals)}] {gn}")
        gr = df[df[kc]==gn].iloc[0]
        gra = gr[cols['ra']]; gdec = gr[cols['dec']]
        gpmra = gr[cols['pmra']] if cols['pmra'] in df.columns else 0
        gpmdec = gr[cols['pmdec']] if cols['pmdec'] in df.columns else 0
        rd = gr[cols['distance']] if cols['distance'] in df.columns else None
        rde = gr[cols['distance_err']] if cols['distance_err'] in df.columns else None
        if rd: logger.info(f"  Ref dist: {rd:.2f} kpc")

        rhc = cols.get('rhalf')
        srd = (cfg.DWG_SEARCH_RADIUS_FACTOR * gr[rhc] / 60.0) if (rhc and rhc in df.columns) else 0.5
        sc = 2*np.sin(np.radians(srd)/2)
        ra_r = np.radians(gra); dec_r = np.radians(gdec)
        gc = np.array([np.cos(dec_r)*np.cos(ra_r), np.cos(dec_r)*np.sin(ra_r), np.sin(dec_r)])
        indices = master.tree.query_ball_point(gc, sc)
        logger.info(f"  Search: {srd*60:.1f}', Found: {len(indices)}")
        if len(indices) < cfg.MIN_STARS_FOR_ANALYSIS: continue

        mst = master.df.iloc[indices].reset_index(drop=True)
        mdf = mst.copy(); mdf.columns = [f"{c}_master" for c in mdf.columns]
        mdf = _standard_match_columns(mdf)
        pe = mdf.get(f"{cfg.MASTER_COLS['pmra_err']}_master", pd.Series(dtype=float)).values
        pde = mdf.get(f"{cfg.MASTER_COLS['pmdec_err']}_master", pd.Series(dtype=float)).values
        P, algo = compute_adaptive_membership(mdf['pmra'].values, mdf['pmdec'].values, gpmra, gpmdec, 'DW',
                                              pe, pde, None, cfg.MIN_STARS_FOR_ANALYSIS)
        mdf['P_mem'] = P
        logger.info(f"  Algo: {algo['algorithm']} | n(P>0.8): {np.sum(P>cfg.P_MEM_HIGH)}")

        mplx = mdf['plx_from_params_master'].values if 'plx_from_params_master' in mdf.columns else None
        mrv, mrve = _get_member_rv(df[df[kc]==gn], cols)
        plot_individual_panels(mdf, gn, 'DW', algo, rd, rde, mplx, mrv, mrve)

        results.append({'cluster_name':gn,'obj_type':'DW','member_df':df[df[kc]==gn],
                        'matched_df':mdf,'algo_info':algo,'mem_cols':cols,
                        'n_members':len(indices),'n_matched':len(indices),'ref_dist':rd,'ref_dist_err':rde})
        gcmod.collect()
    return results


# ============================================================================
# SUMMARY PLOTS (V4: individual-style per panel)
# ============================================================================

def _grid(n):
    nc = int(np.ceil(np.sqrt(n))); nr = int(np.ceil(n/nc)); return nr, nc

def generate_summary_plots(all_results, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    set_paper_style()

    logger.info("\n" + "="*70 + "\nGENERATING SUMMARY PLOTS (V4)\n" + "="*70)
    vr = [r for r in all_results if r['matched_df'] is not None and r['n_matched'] >= cfg.MIN_SUMMARY_MATCH]
    if not vr: logger.warning("No valid results!"); return
    sr = sorted(vr, key=lambda x: x['n_matched'], reverse=True)[:cfg.MAX_PANELS]
    n = len(sr); nr, nc = _grid(n)
    logger.info(f"Plotting {n} objects (≥{cfg.MIN_SUMMARY_MATCH} matches) in summary panels")

    tc = {'GC': '#006400', 'OC': '#FF8C00', 'DW': '#00008B', 'SGR': '#8B0000'}

    # ===== 1. PM SUMMARY =====
    from matplotlib.patches import Ellipse
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if n==1: axes = np.array([axes])
    af = axes.flatten()
    for i, r in enumerate(sr):
        ax = af[i]; mdf = r['matched_df']; algo = r['algo_info']
        if 'P_mem' in mdf.columns:
            v = mdf['P_mem'].notna()
            ax.scatter(mdf.loc[v,'pmra'], mdf.loc[v,'pmdec'], c=mdf.loc[v,'P_mem'], cmap=cfg.CMAP_PMEM,
                       s=18, alpha=0.8, vmin=0, vmax=1, edgecolors='k', linewidth=0.2)
        if algo.get('Sigma_cluster') and r['obj_type']!='SGR':
            mu=algo['mu_cluster']; sig=np.array(algo['Sigma_cluster']); w,v2=np.linalg.eigh(sig); ang=np.degrees(np.arctan2(v2[1,0],v2[0,0]))
            ell=Ellipse(xy=mu,width=4*np.sqrt(max(w[0],0)),height=4*np.sqrt(max(w[1],0)),angle=ang,fill=False,ec='lime',lw=2.5)
            ax.add_patch(ell)
        ax.set_title(f"{r['obj_type']}: {r['cluster_name']}", fontsize=14, fontweight='bold', color=tc.get(r['obj_type'],'k'))
        ax.set_xlabel('$\\mu_\\alpha\\cos\\delta$', fontsize=13, fontweight='bold')
        ax.set_ylabel('$\\mu_\\delta$', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2); ax.set_aspect('equal','datalim')
    for j in range(n, len(af)): af[j].set_visible(False)
    fig.suptitle('Proper Motion Membership Summary', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(cfg.OUTPUT_DIR, f'SUMMARY_PM.{cfg.SAVE_FORMAT}'), dpi=cfg.PLOT_DPI); plt.close()
    logger.info("  Saved: SUMMARY_PM")

    # ===== 2. RA-DEC SUMMARY =====
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if n==1: axes = np.array([axes])
    af = axes.flatten()
    for i, r in enumerate(sr):
        ax = af[i]; mdf = r['matched_df']
        if 'P_mem' in mdf.columns:
            v = mdf['P_mem'].notna()
            ax.scatter(mdf.loc[v,'ra'], mdf.loc[v,'dec'], c=mdf.loc[v,'P_mem'], cmap=cfg.CMAP_PMEM,
                       s=18, alpha=0.8, vmin=0, vmax=1, edgecolors='k', linewidth=0.2)
        ax.set_title(f"{r['obj_type']}: {r['cluster_name']}", fontsize=14, fontweight='bold', color=tc.get(r['obj_type'],'k'))
        ax.set_xlabel('RA (deg)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Dec (deg)', fontsize=13, fontweight='bold')
        ax.invert_xaxis(); ax.grid(True, alpha=0.2)
    for j in range(n, len(af)): af[j].set_visible(False)
    fig.suptitle('RA-Dec Spatial Distribution Summary', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(cfg.OUTPUT_DIR, f'SUMMARY_RADEC.{cfg.SAVE_FORMAT}'), dpi=cfg.PLOT_DPI); plt.close()
    logger.info("  Saved: SUMMARY_RADEC")

    # ===== 3. DISTANCE SUMMARY (individual style) =====
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if n==1: axes = np.array([axes])
    af = axes.flatten()
    for i, r in enumerate(sr):
        ax = af[i]; mdf = r['matched_df']
        md = _clean(mdf['best_dist'].values) if 'best_dist' in mdf.columns else np.array([])
        md = md[(md>0)&(md<300)]; rd = r.get('ref_dist'); ot = r['obj_type']
        if ot in ['SGR','STREAM']:
            if len(md)>0: _hist_step_kde_med(ax, md, cfg.COL_MASTER, 'Master', n_bins=20); _med_mad_box(ax, md, cfg.COL_MASTER)
            ax.set_xlabel('Dist (kpc)', fontsize=13, fontweight='bold')
        else:
            if rd is not None and np.isfinite(rd) and len(md)>0:
                dm = rd - md
                _hist_step_kde_med(ax, dm, cfg.COL_MASTER, '$\\Delta$d', n_bins=20)
                ax.axvline(0, color='k', lw=2, alpha=0.7); _med_mad_box(ax, dm, cfg.COL_MASTER)
                ax.set_xlabel('$\\Delta$d (kpc)', fontsize=13, fontweight='bold')
            elif len(md)>0:
                _hist_step_kde_med(ax, md, cfg.COL_MASTER, 'Master', n_bins=20); _med_mad_box(ax, md, cfg.COL_MASTER)
                ax.set_xlabel('Dist (kpc)', fontsize=13, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=13, fontweight='bold')
        ax.set_title(f"{ot}: {r['cluster_name']}", fontsize=14, fontweight='bold', color=tc.get(ot,'k'))
        ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.2, axis='y')
    for j in range(n, len(af)): af[j].set_visible(False)
    fig.suptitle('Distance Comparison Summary', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(cfg.OUTPUT_DIR, f'SUMMARY_Distance.{cfg.SAVE_FORMAT}'), dpi=cfg.PLOT_DPI); plt.close()
    logger.info("  Saved: SUMMARY_Distance")

    # ===== 4. RV SUMMARY (dual y-axes, individual style) =====
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    if n==1: axes = np.array([axes])
    af = axes.flatten()
    for i, r in enumerate(sr):
        ax = af[i]; mdf = r['matched_df']
        mrv = _clean(mdf['best_rv'].values) if 'best_rv' in mdf.columns else np.array([])
        # Get member RV from the stored member_df and cols
        mem_rv_raw, _ = _get_member_rv(r.get('member_df', pd.DataFrame()), r.get('mem_cols', {}))
        mem_rv = _clean(mem_rv_raw) if mem_rv_raw is not None else np.array([])

        all_rv = [x for x in [mrv, mem_rv] if len(x)>=3]
        if not all_rv:
            ax.text(0.5, 0.5, 'No RV data', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold', color='gray')
        else:
            comb = np.concatenate(all_rv); lo, hi = np.percentile(comb, [1, 99]); rng = max(hi-lo, 5)
            br = [lo-0.1*rng, hi+0.1*rng]

            if len(mrv) >= 3:
                _hist_step_kde_med(ax, mrv, cfg.COL_MASTER, f'Master (n={len(mrv)})', br, 25)
                _med_mad_box(ax, mrv, cfg.COL_MASTER, 'upper right', 'M ')
            ax.set_ylabel('Count (Master)', fontsize=13, fontweight='bold', color=cfg.COL_MASTER)

            if len(mem_rv) >= 3:
                # Dual y-axis for member
                ax2 = ax.twinx()
                counts2, bins2, _ = ax2.hist(mem_rv, bins=25, range=br, alpha=0)
                ax2.step(bins2[:-1], counts2, where='post', color=cfg.COL_MEMBER, linewidth=2.5,
                         label=f'Member (n={len(mem_rv)})')
                _safe_kde(ax2, mem_rv, bins2, cfg.COL_MEMBER)
                ax2.axvline(np.median(mem_rv), color=cfg.COL_MEMBER, ls=':', lw=2.5)
                ax2.set_ylabel('Count (Member)', fontsize=13, fontweight='bold', color=cfg.COL_MEMBER)
                ax2.tick_params(axis='y', labelcolor=cfg.COL_MEMBER)
                ax2.legend(fontsize=9, loc='upper left')
                _med_mad_box(ax2, mem_rv, cfg.COL_MEMBER, 'lower right', 'Mem ')

        ax.set_xlabel('RV (km s$^{-1}$)', fontsize=13, fontweight='bold')
        ax.set_title(f"{r['obj_type']}: {r['cluster_name']}", fontsize=14, fontweight='bold', color=tc.get(r['obj_type'],'k'))
        ax.legend(fontsize=9, loc='best'); ax.grid(True, alpha=0.2, axis='y')

    for j in range(n, len(af)): af[j].set_visible(False)
    fig.suptitle('Radial Velocity Distribution Summary', fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(cfg.OUTPUT_DIR, f'SUMMARY_RV.{cfg.SAVE_FORMAT}'), dpi=cfg.PLOT_DPI); plt.close()
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
            t = r['matched_df'].copy(); t.insert(0,'Object_Type',r['obj_type']); t.insert(1,'Cluster_Name',r['cluster_name']); master_dfs.append(t)
        summary_data.append({'Object':r['cluster_name'],'Type':r['obj_type'],'N_members':r['n_members'],
                             'N_matched':r['n_matched'],'Match_pct':f"{100*r['n_matched']/r['n_members']:.1f}" if r['n_members']>0 else '0',
                             'N_high_prob':nh,'Ref_dist_kpc':r.get('ref_dist',np.nan),
                             'Algorithm':r['algo_info'].get('algorithm','None'),'Status':r['algo_info'].get('status','N/A')})
    if master_dfs:
        full = pd.concat(master_dfs, ignore_index=True)
        fp = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_full_membership_results.csv')
        full.to_csv(fp, index=False); logger.info(f"  Saved: {fp} ({len(full):,} rows)")
    sdf = pd.DataFrame(summary_data).sort_values('N_matched', ascending=False).reset_index(drop=True)
    sf = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_summary.csv'); sdf.to_csv(sf, index=False); logger.info(f"  Saved: {sf}")
    af = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_algorithm_results.json')
    with open(af, 'w') as f: json.dump({f"{r['obj_type']}_{r['cluster_name']}": r['algo_info'] for r in all_results}, f, indent=2, default=str)
    logger.info(f"  Saved: {af}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Adaptive Membership Analysis V4')
    p.add_argument('--master', required=True); p.add_argument('--gc', default=None)
    p.add_argument('--oc', default=None); p.add_argument('--sgr', default=None)
    p.add_argument('--dwg', default=None); p.add_argument('--gc-dist', default=None)
    p.add_argument('--output', default='./outputs'); p.add_argument('--checkpoint', default='./checkpoints')
    p.add_argument('--log', default=None); p.add_argument('--skip-plots', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    cfg.MASTER_CATALOG=args.master; cfg.GC_MEMBERS_FILE=args.gc; cfg.OC_MEMBERS_FILE=args.oc
    cfg.SGR_MEMBERS_FILE=args.sgr; cfg.DWG_MEMBERS_FILE=args.dwg; cfg.GC_DIST_FILE=args.gc_dist
    cfg.OUTPUT_DIR=args.output; cfg.CHECKPOINT_DIR=args.checkpoint

    for d in [cfg.OUTPUT_DIR, cfg.CHECKPOINT_DIR, os.path.join(cfg.OUTPUT_DIR, 'individual_plots')]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logging(args.log or os.path.join(cfg.OUTPUT_DIR, 'adaptive_membership_v4.log'))
    logger.info("="*70 + f"\nADAPTIVE MEMBERSHIP ANALYSIS V4\n" + "="*70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    t0 = time.time()

    master = MasterCatalog(logger)
    if not master.load(cfg.MASTER_CATALOG, cfg.CHECKPOINT_DIR):
        logger.error("Failed!"); sys.exit(1)

    gc_dists = load_gc_reference_distances(cfg.GC_DIST_FILE, logger)
    all_results = []
    all_results.extend(process_gc_members(master, gc_dists, logger))
    all_results.extend(process_oc_members(master, logger))
    all_results.extend(process_sgr_members(master, logger))
    all_results.extend(process_dwg_members(master, logger))

    logger.info(f"\n{'='*70}\nPROCESSING COMPLETE: {len(all_results)} objects\n{'='*70}")
    if not args.skip_plots: generate_summary_plots(all_results, logger)
    save_results(all_results, logger)
    logger.info(f"\n{'='*70}\nALL DONE! {(time.time()-t0)/60:.1f} minutes\n{'='*70}")

if __name__ == '__main__':
    main()