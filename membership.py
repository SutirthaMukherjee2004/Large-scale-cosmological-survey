
#!/usr/bin/env python3
"""
================================================================================
ADAPTIVE MEMBERSHIP ANALYSIS - CLUSTER OPTIMIZED (PRODUCTION)
================================================================================

Production-ready script for HPC cluster execution.
Cross-matches member catalogs against 52M master catalog, runs adaptive 
clustering algorithms (GMM/DBSCAN), and generates visualizations.

UPDATES:
- FIXED: FileNotFoundError in plotting (Force directory creation)
- Distance fallback logic (1/parallax)
- Single consolidated output CSV
- Individual PNG plots in dedicated folder
- SGR Aspect Ratio handling

Author: Sutirtha
Optimized for: PBS/SLURM cluster with 64+ cores, 700GB RAM
"""

import os
import sys
import gc
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with file and console handlers."""
    logger = logging.getLogger('AdaptiveMembership')
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration container with validation."""
    
    # File paths (to be set by user)
    MASTER_CATALOG: str = None
    GC_MEMBERS_FILE: str = None
    OC_MEMBERS_FILE: str = None
    SGR_MEMBERS_FILE: str = None
    DWG_MEMBERS_FILE: str = None
    OUTPUT_DIR: str = './outputs'
    CHECKPOINT_DIR: str = './checkpoints'
    
    # Master catalog columns
    MASTER_COLS = {
        'ra': 'RA_all',
        'dec': 'DEC_all',
        'pmra': 'pmRA_x',
        'pmdec': 'pmDE',
        'pmra_err': 'e_pmRA',
        'pmdec_err': 'e_pmDE',
        'parallax': 'Plx',
        'parallax_err': 'e_Plx',
        'rv': 'Weighted_Avg',
        'dist': 'e_weighted_avg',
        'params': 'stellar_params_est'
    }
    
    # GC member catalog columns
    GC_MEM_COLS = {
        'key': 'source',
        'ra': 'ra', 'dec': 'dec',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmra_error', 'pmdec_err': 'pmdec_error',
        'pmra_pmdec_corr': 'pmra_pmdec_corr',
        'parallax': 'parallax',
        'membership_prob': 'membership_probability',
    }
    
    # OC member catalog columns
    OC_MEM_COLS = {
        'key': 'Cluster',
        'ra': 'RAdeg', 'dec': 'DEdeg',
        'pmra': 'pmRA', 'pmdec': 'pmDE',
        'pmra_err': 'e_pmRA', 'pmdec_err': 'e_pmDE',
        'pmra_pmdec_corr': 'pmRApmDEcor',
        'parallax': 'Plx',
        'membership_prob': 'Proba',
    }
    
    # SGR stream member columns
    SGR_MEM_COLS = {
        'key': None,  # Use distance bins
        'ra': 'ra', 'dec': 'dec',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmraerr', 'pmdec_err': 'pmdecerr',
        'pmra_pmdec_corr': None,
        'parallax': 'parallax',
        'dist': 'dist',
    }
    
    # DWG member columns (galaxy-level data)
    DWG_MEM_COLS = {
        'key': 'source',
        'ra': 'ra_x', 'dec': 'dec_x',
        'pmra': 'pmra', 'pmdec': 'pmdec',
        'pmra_err': 'pmra_error', 'pmdec_err': 'pmdec_error',
        'distance': 'distance',
        'rhalf': 'rhalf',  # Half-light radius for search
    }
    
    
    # Cross-match parameters
    CROSSMATCH_RADIUS_ARCSEC: float = 1.0
    DWG_SEARCH_RADIUS_FACTOR: float = 3.0  # Search within N × r_half
    
    # Algorithm parameters
    GMM_N_COMPONENTS: int = 2
    GMM_MAX_ITER: int = 300
    GMM_N_INIT: int = 10
    GMM_RANDOM_STATE: int = 42
    
    DBSCAN_EPS_CLEANUP: float = 0.3
    DBSCAN_MIN_SAMPLES_CLEANUP: int = 3
    DBSCAN_EPS_OC: float = 0.25
    DBSCAN_MIN_SAMPLES_OC: int = 5
    DBSCAN_EPS_STREAM: float = 0.4
    DBSCAN_MIN_SAMPLES_STREAM: int = 3
    
    HDBSCAN_MIN_CLUSTER_SIZE: int = 5
    HDBSCAN_MIN_SAMPLES: int = 3
    
    # Membership thresholds
    P_MEM_HIGH: float = 0.8
    P_MEM_LOW: float = 0.2
    MIN_STARS_FOR_ANALYSIS: int = 10
    
    # SGR binning
    SGR_BIN_START_KPC: float = 15.0
    SGR_BIN_WIDTH_KPC: float = 10.0
    
    # Plotting
    MAX_PANELS: int = 36
    CMAP_PMEM: str = 'RdYlGn'
    PLOT_DPI: int = 150
    SAVE_FORMAT: str = 'png'  # Changed to PNG for stability


# Global config instance
cfg = Config()

# Check for HDBSCAN
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# ============================================================================
# PHASE 0: PRE-FLIGHT VALIDATION
# ============================================================================

def validate_fits_file(filepath: str, required_cols: List[str], logger: logging.Logger) -> Tuple[bool, int, List[str]]:
    """
    Validate FITS file exists and has required columns.
    Returns: (success, nrows, available_columns)
    """
    from astropy.io import fits
    
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False, 0, []
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            # Find the data HDU (usually index 1 for BINTABLE)
            data_hdu = None
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'columns') and hdu.columns is not None:
                    data_hdu = hdu
                    break
            
            if data_hdu is None:
                logger.error(f"No data table found in {filepath}")
                return False, 0, []
            
            available_cols = [c.name for c in data_hdu.columns]
            nrows = data_hdu.data.shape[0]
            
            # Note: We relax strict check for 'params' column to avoid init crash if user file varies
            # but main logic relies on it.
            missing = [c for c in required_cols if c not in available_cols and c != cfg.MASTER_COLS['params']]
            if missing:
                logger.error(f"Missing columns in {filepath}: {missing}")
                logger.info(f"Available columns: {available_cols[:20]}...")
                return False, nrows, available_cols
            
            logger.info(f"✓ {filepath}: {nrows:,} rows, all required columns present")
            return True, nrows, available_cols
            
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return False, 0, []


def validate_csv_file(filepath: str, required_cols: List[str], logger: logging.Logger) -> Tuple[bool, int, List[str]]:
    """
    Validate CSV file exists and has required columns.
    Returns: (success, nrows, available_columns)
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found (will skip): {filepath}")
        return False, 0, []
    
    try:
        # Just read header and count rows
        df_sample = pd.read_csv(filepath, nrows=5)
        available_cols = list(df_sample.columns)
        
        # Count total rows efficiently
        with open(filepath, 'r') as f:
            nrows = sum(1 for _ in f) - 1  # Subtract header
        
        missing = [c for c in required_cols if c not in available_cols]
        if missing:
            logger.warning(f"Missing columns in {filepath}: {missing}")
            logger.info(f"Available columns: {available_cols}")
            # Don't fail, just warn
        
        logger.info(f"✓ {filepath}: {nrows:,} rows")
        return True, nrows, available_cols
        
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return False, 0, []


def preflight_check(logger: logging.Logger) -> bool:
    """
    Run all pre-flight validation checks.
    Returns True if all critical checks pass.
    """
    logger.info("=" * 70)
    logger.info("PHASE 0: PRE-FLIGHT VALIDATION")
    logger.info("=" * 70)
    
    all_ok = True
    
    # Check master catalog (CRITICAL)
    logger.info("\n[1/5] Checking Master Catalog...")
    master_required = list(cfg.MASTER_COLS.values())
    master_ok, master_rows, _ = validate_fits_file(
        cfg.MASTER_CATALOG, master_required, logger
    )
    if not master_ok:
        logger.error("CRITICAL: Master catalog validation failed!")
        all_ok = False
    else:
        # Estimate memory
        mem_estimate_gb = master_rows * 8 * len(master_required) / 1e9
        logger.info(f"  Estimated memory for master data: {mem_estimate_gb:.1f} GB")
    
    # Check member catalogs
    logger.info("\n[2/5] Checking GC Members...")
    if cfg.GC_MEMBERS_FILE:
        gc_required = [v for v in cfg.GC_MEM_COLS.values() if v is not None]
        validate_csv_file(cfg.GC_MEMBERS_FILE, gc_required, logger)
    
    logger.info("\n[3/5] Checking OC Members...")
    if cfg.OC_MEMBERS_FILE:
        oc_required = [v for v in cfg.OC_MEM_COLS.values() if v is not None]
        validate_csv_file(cfg.OC_MEMBERS_FILE, oc_required, logger)
    
    logger.info("\n[4/5] Checking SGR Members...")
    if cfg.SGR_MEMBERS_FILE:
        sgr_required = [v for v in cfg.SGR_MEM_COLS.values() if v is not None]
        validate_csv_file(cfg.SGR_MEMBERS_FILE, sgr_required, logger)
    
    logger.info("\n[5/5] Checking DWG Members...")
    if cfg.DWG_MEMBERS_FILE:
        dwg_required = [v for v in cfg.DWG_MEM_COLS.values() if v is not None]
        validate_csv_file(cfg.DWG_MEMBERS_FILE, dwg_required, logger)
    
    # Check output directories
    logger.info("\n[Checking Output Directories]")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'individual_plots'), exist_ok=True)
    logger.info(f"  Output dir: {cfg.OUTPUT_DIR}")
    logger.info(f"  Checkpoint dir: {cfg.CHECKPOINT_DIR}")
    
    # Check available memory
    logger.info("\n[System Resources]")
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"  Total RAM: {mem.total / 1e9:.1f} GB")
        logger.info(f"  Available RAM: {mem.available / 1e9:.1f} GB")
    except ImportError:
        logger.warning("  psutil not available - cannot check memory")
    
    # Check dependencies
    logger.info("\n[Dependencies]")
    logger.info(f"  HDBSCAN: {'✓ Available' if HAS_HDBSCAN else '✗ Not available (will use DBSCAN)'}")
    
    try:
        from astropy.io import fits
        logger.info("  Astropy: ✓ Available")
    except ImportError:
        logger.error("  Astropy: ✗ NOT AVAILABLE (CRITICAL)")
        all_ok = False
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for cluster
        logger.info("  Matplotlib: ✓ Available (using Agg backend)")
    except ImportError:
        logger.error("  Matplotlib: ✗ NOT AVAILABLE")
        all_ok = False
    
    logger.info("\n" + "=" * 70)
    if all_ok:
        logger.info("PRE-FLIGHT CHECK: ✓ PASSED")
    else:
        logger.error("PRE-FLIGHT CHECK: ✗ FAILED")
    logger.info("=" * 70)
    
    return all_ok


# ============================================================================
# PHASE 1: LOAD MASTER CATALOG & BUILD KDTREE
# ============================================================================

class MasterCatalog:
    """Memory-efficient handler for the 52M master catalog."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.df: pd.DataFrame = None
        self.tree: cKDTree = None
        self.coords_3d: np.ndarray = None
        self.max_chord: float = None
        self.nrows: int = 0
    
    def load(self, filepath: str, checkpoint_dir: str = None) -> bool:
        """Load master catalog and build KDTree."""
        from astropy.io import fits
        
        self.logger.info("=" * 70)
        self.logger.info("PHASE 1: LOADING MASTER CATALOG")
        self.logger.info("=" * 70)
        
        # Check for checkpoint
        tree_checkpoint = os.path.join(checkpoint_dir, 'master_tree.npz') if checkpoint_dir else None
        data_checkpoint = os.path.join(checkpoint_dir, 'master_data.parquet') if checkpoint_dir else None
        
        if tree_checkpoint and os.path.exists(tree_checkpoint) and os.path.exists(data_checkpoint):
            self.logger.info("Found checkpoint - loading from cache...")
            try:
                self.df = pd.read_parquet(data_checkpoint)
                loaded = np.load(tree_checkpoint)
                self.coords_3d = loaded['coords']
                self.tree = cKDTree(self.coords_3d)
                self.nrows = len(self.df)
                self._compute_max_chord()
                self.logger.info(f"✓ Loaded from checkpoint: {self.nrows:,} rows")
                return True
            except Exception as e:
                self.logger.warning(f"Checkpoint load failed: {e}, loading from FITS...")
        
        # Load from FITS
        start_time = time.time()
        self.logger.info(f"Loading: {filepath}")
        
        try:
            # Only load needed columns
            needed_cols = list(cfg.MASTER_COLS.values())
            
            with fits.open(filepath, memmap=True) as hdul:
                # Find data HDU
                data_hdu = None
                for hdu in hdul:
                    if hasattr(hdu, 'columns') and hdu.columns is not None:
                        data_hdu = hdu
                        break
                
                if data_hdu is None:
                    self.logger.error("No data table found!")
                    return False
                
                self.nrows = data_hdu.data.shape[0]
                self.logger.info(f"  Total rows: {self.nrows:,}")
                
                # Load only needed columns
                data_dict = {}
                for key, col in cfg.MASTER_COLS.items():
                    if key == 'params': continue # Skip special handling here
                    if col in data_hdu.columns.names:
                        self.logger.info(f"  Loading column: {col}")
                        data_dict[col] = data_hdu.data[col].copy()
                
                # --- NEW IMPLEMENTATION: Distance Fallback Logic ---
                # "iff there is NaN value in DIST column , then use stellar_params_est[4] as parallax"
                if cfg.MASTER_COLS['params'] in data_hdu.columns.names:
                    self.logger.info("  Processing Parallax Fallback from stellar_params_est...")
                    params_array = data_hdu.data[cfg.MASTER_COLS['params']]
                    # Extract index 4
                    parallax = params_array[:, 4]
                    
                    # Calculate fallback distance: 1000 / parallax
                    # Use errstate to handle divide by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        dist_fallback = 1000.0 / parallax
                        dist_fallback[parallax <= 0] = np.nan # Clean up non-physical values
                    
                    # Fill NaN in 'dist'
                    dist_col_name = cfg.MASTER_COLS['dist']
                    if dist_col_name in data_dict:
                        current_dist = data_dict[dist_col_name]
                        mask = np.isnan(current_dist)
                        # Fill NaNs where fallback is valid
                        current_dist[mask] = dist_fallback[mask]
                        data_dict[dist_col_name] = current_dist
                        self.logger.info(f"  ✓ Filled {np.sum(mask):,} NaN distances using parallax")
                    else:
                        data_dict[dist_col_name] = dist_fallback
                        self.logger.info("  ✓ Created dist column from parallax")
                # ---------------------------------------------------

                self.df = pd.DataFrame(data_dict)
                self.logger.info(f"  DataFrame created: {self.df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
            
            # Drop NaN coordinates
            ra_col = cfg.MASTER_COLS['ra']
            dec_col = cfg.MASTER_COLS['dec']
            
            n_before = len(self.df)
            self.df = self.df.dropna(subset=[ra_col, dec_col])
            n_dropped = n_before - len(self.df)
            if n_dropped > 0:
                self.logger.info(f"  Dropped {n_dropped:,} rows with NaN coordinates")
            
            self.nrows = len(self.df)
            
            # Build KDTree
            self.logger.info("Building KDTree...")
            self._build_kdtree()
            
            # Save checkpoint
            if checkpoint_dir:
                self.logger.info("Saving checkpoint...")
                self.df.to_parquet(data_checkpoint)
                np.savez_compressed(tree_checkpoint, coords=self.coords_3d)
                self.logger.info(f"  Checkpoint saved to {checkpoint_dir}")
            
            elapsed = time.time() - start_time
            self.logger.info(f"✓ Master catalog loaded in {elapsed:.1f}s")
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load master catalog: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _build_kdtree(self):
        """Build 3D KDTree for spherical matching."""
        ra = self.df[cfg.MASTER_COLS['ra']].values
        dec = self.df[cfg.MASTER_COLS['dec']].values
        
        # Convert to 3D Cartesian
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        self.coords_3d = np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ])
        
        self.tree = cKDTree(self.coords_3d)
        self._compute_max_chord()
        
        self.logger.info(f"  KDTree built with {len(self.coords_3d):,} points")
    
    def _compute_max_chord(self):
        """Pre-compute max chord length for cross-matching."""
        max_sep_rad = np.radians(cfg.CROSSMATCH_RADIUS_ARCSEC / 3600.0)
        self.max_chord = 2 * np.sin(max_sep_rad / 2)
    
    def query(self, ra: np.ndarray, dec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cross-match coordinates against master catalog.
        
        Returns:
            master_idx: Indices in master catalog
            member_idx: Indices in input arrays that matched
            sep_arcsec: Separation in arcseconds
        """
        # Convert to 3D
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        coords = np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ])
        
        # Query tree
        distances, indices = self.tree.query(coords, k=1, distance_upper_bound=self.max_chord)
        
        # Filter valid matches
        valid = np.isfinite(distances)
        member_idx = np.where(valid)[0]
        master_idx = indices[valid]
        
        # Convert to arcsec
        sep_arcsec = np.degrees(2 * np.arcsin(distances[valid] / 2)) * 3600
        
        return master_idx, member_idx, sep_arcsec
    
    def get_matched_data(self, master_idx: np.ndarray) -> pd.DataFrame:
        """Get master catalog data for matched indices."""
        return self.df.iloc[master_idx].reset_index(drop=True)


# ============================================================================
# ALGORITHMS
# ============================================================================

def build_measurement_covariance(pmra_err: np.ndarray, pmdec_err: np.ndarray, 
                                 corr: np.ndarray = None) -> np.ndarray:
    """Build measurement covariance matrices C_i for each star."""
    n = len(pmra_err)
    C = np.zeros((n, 2, 2))
    C[:, 0, 0] = pmra_err**2
    C[:, 1, 1] = pmdec_err**2
    if corr is not None:
        cov_xy = corr * pmra_err * pmdec_err
        C[:, 0, 1] = cov_xy
        C[:, 1, 0] = cov_xy
    return C


def algorithm_gmm_with_errors(pmra: np.ndarray, pmdec: np.ndarray, 
                               center_pmra: float, center_pmdec: float,
                               pmra_err: np.ndarray = None, pmdec_err: np.ndarray = None,
                               pmra_pmdec_corr: np.ndarray = None,
                               min_stars: int = 10) -> Tuple[np.ndarray, Dict]:
    """GMM with Error Deconvolution - for GLOBULAR CLUSTERS."""
    n_stars = len(pmra)
    valid_mask = ~(np.isnan(pmra) | np.isnan(pmdec))
    n_valid = np.sum(valid_mask)
    
    if n_valid < min_stars:
        return np.full(n_stars, 0.5), {'status': 'insufficient_data', 'algorithm': 'GMM'}
    
    X = np.column_stack([pmra[valid_mask], pmdec[valid_mask]])
    
    try:
        gmm = GaussianMixture(
            n_components=cfg.GMM_N_COMPONENTS,
            max_iter=cfg.GMM_MAX_ITER,
            n_init=cfg.GMM_N_INIT,
            random_state=cfg.GMM_RANDOM_STATE,
            covariance_type='full'
        )
        gmm.fit(X)
        
        # Identify cluster component
        ref_point = np.array([center_pmra, center_pmdec])
        distances = [np.linalg.norm(gmm.means_[i] - ref_point) for i in range(cfg.GMM_N_COMPONENTS)]
        cluster_idx = np.argmin(distances)
        field_idx = 1 - cluster_idx
        
        mu_cluster = gmm.means_[cluster_idx]
        Sigma_cluster = gmm.covariances_[cluster_idx]
        mu_field = gmm.means_[field_idx]
        Sigma_field = gmm.covariances_[field_idx]
        eta = gmm.weights_[cluster_idx]
        
        # Compute P_mem with error deconvolution
        P_mem_valid = np.zeros(n_valid)
        
        if pmra_err is not None and pmdec_err is not None:
            pmra_err_valid = pmra_err[valid_mask]
            pmdec_err_valid = pmdec_err[valid_mask]
            corr_valid = pmra_pmdec_corr[valid_mask] if pmra_pmdec_corr is not None else None
            C = build_measurement_covariance(pmra_err_valid, pmdec_err_valid, corr_valid)
            
            for i in range(n_valid):
                try:
                    Sigma_cl_eff = Sigma_cluster + C[i]
                    Sigma_f_eff = Sigma_field + C[i]
                    p_cl = eta * multivariate_normal.pdf(X[i], mu_cluster, Sigma_cl_eff)
                    p_f = (1 - eta) * multivariate_normal.pdf(X[i], mu_field, Sigma_f_eff)
                    P_mem_valid[i] = p_cl / (p_cl + p_f) if (p_cl + p_f) > 0 else 0.5
                except:
                    P_mem_valid[i] = 0.5
        else:
            probs = gmm.predict_proba(X)
            P_mem_valid = probs[:, cluster_idx]
        
        P_mem = np.full(n_stars, np.nan)
        P_mem[valid_mask] = P_mem_valid
        
        info = {
            'status': 'success',
            'algorithm': 'GMM',
            'mu_cluster': mu_cluster.tolist(),
            'Sigma_cluster': Sigma_cluster.tolist(),
            'mu_field': mu_field.tolist(),
            'Sigma_field': Sigma_field.tolist(),
            'eta': float(eta),
            'pm_dispersion': float(np.sqrt(np.trace(Sigma_cluster))),
            'center_pmra': center_pmra,
            'center_pmdec': center_pmdec,
        }
        return P_mem, info
        
    except Exception as e:
        return np.full(n_stars, 0.5), {'status': f'error: {e}', 'algorithm': 'GMM'}


def algorithm_dbscan(pmra: np.ndarray, pmdec: np.ndarray,
                     center_pmra: float, center_pmdec: float,
                     eps: float = 0.25, min_samples: int = 5,
                     use_hdbscan: bool = False, min_stars: int = 10) -> Tuple[np.ndarray, Dict]:
    """DBSCAN/HDBSCAN - for OPEN CLUSTERS."""
    n_stars = len(pmra)
    valid_mask = ~(np.isnan(pmra) | np.isnan(pmdec))
    n_valid = np.sum(valid_mask)
    
    if n_valid < min_stars:
        return np.full(n_stars, 0.5), {'status': 'insufficient_data', 'algorithm': 'DBSCAN'}
    
    X = np.column_stack([pmra[valid_mask], pmdec[valid_mask]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ref_scaled = scaler.transform([[center_pmra, center_pmdec]])[0]
    
    try:
        if use_hdbscan and HAS_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=cfg.HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=cfg.HDBSCAN_MIN_SAMPLES
            )
            labels = clusterer.fit_predict(X_scaled)
            algo_name = 'HDBSCAN'
        else:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(X_scaled)
            algo_name = 'DBSCAN'
        
        unique_labels = set(labels) - {-1}
        
        if len(unique_labels) == 0:
            distances = np.linalg.norm(X_scaled - ref_scaled, axis=1)
            P_mem_valid = np.exp(-distances**2 / 2)
            P_mem = np.full(n_stars, np.nan)
            P_mem[valid_mask] = P_mem_valid
            return P_mem, {
                'status': 'no_cluster_found',
                'algorithm': algo_name,
                'center_pmra': center_pmra,
                'center_pmdec': center_pmdec,
            }
        
        # Find cluster closest to reference
        best_cluster = min(unique_labels,
                           key=lambda l: np.linalg.norm(X_scaled[labels == l].mean(axis=0) - ref_scaled))
        cluster_mask = labels == best_cluster
        
        P_mem_valid = np.zeros(n_valid)
        
        if use_hdbscan and HAS_HDBSCAN and hasattr(clusterer, 'probabilities_'):
            P_mem_valid[cluster_mask] = clusterer.probabilities_[cluster_mask]
        else:
            P_mem_valid[cluster_mask] = 1.0
        
        # Distance-based for non-members
        cluster_center = X_scaled[cluster_mask].mean(axis=0)
        cluster_std = X_scaled[cluster_mask].std(axis=0).mean()
        non_cluster = ~cluster_mask
        if np.any(non_cluster):
            dists = np.linalg.norm(X_scaled[non_cluster] - cluster_center, axis=1)
            P_mem_valid[non_cluster] = np.exp(-dists**2 / (2 * cluster_std**2)) * 0.3
        
        P_mem = np.full(n_stars, np.nan)
        P_mem[valid_mask] = P_mem_valid
        
        cluster_pm = X[cluster_mask]
        mu_cluster = cluster_pm.mean(axis=0) if len(cluster_pm) > 0 else np.array([center_pmra, center_pmdec])
        Sigma_cluster = np.cov(cluster_pm.T) if len(cluster_pm) > 2 else np.eye(2) * 0.1
        
        info = {
            'status': 'success',
            'algorithm': algo_name,
            'n_cluster_members': int(np.sum(cluster_mask)),
            'mu_cluster': mu_cluster.tolist(),
            'Sigma_cluster': Sigma_cluster.tolist(),
            'pm_dispersion': float(np.sqrt(np.trace(Sigma_cluster))) if len(cluster_pm) > 2 else np.nan,
            'eta': float(np.sum(cluster_mask) / n_valid),
            'center_pmra': center_pmra,
            'center_pmdec': center_pmdec,
        }
        return P_mem, info
        
    except Exception as e:
        return np.full(n_stars, 0.5), {'status': f'error: {e}', 'algorithm': 'DBSCAN'}


def algorithm_hybrid_dbscan_gmm(pmra: np.ndarray, pmdec: np.ndarray,
                                center_pmra: float, center_pmdec: float,
                                pmra_err: np.ndarray = None, pmdec_err: np.ndarray = None,
                                pmra_pmdec_corr: np.ndarray = None,
                                min_stars: int = 10) -> Tuple[np.ndarray, Dict]:
    """Hybrid DBSCAN→GMM - for DWARF GALAXIES."""
    n_stars = len(pmra)
    valid_mask = ~(np.isnan(pmra) | np.isnan(pmdec))
    n_valid = np.sum(valid_mask)
    
    if n_valid < min_stars:
        return np.full(n_stars, 0.5), {'status': 'insufficient_data', 'algorithm': 'Hybrid'}
    
    X = np.column_stack([pmra[valid_mask], pmdec[valid_mask]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ref_scaled = scaler.transform([[center_pmra, center_pmdec]])[0]
    
    try:
        # Stage 1: DBSCAN cleanup
        dbscan = DBSCAN(eps=cfg.DBSCAN_EPS_CLEANUP, min_samples=cfg.DBSCAN_MIN_SAMPLES_CLEANUP)
        labels = dbscan.fit_predict(X_scaled)
        unique_labels = set(labels) - {-1}
        
        if len(unique_labels) == 0:
            return algorithm_gmm_with_errors(pmra, pmdec, center_pmra, center_pmdec,
                                             pmra_err, pmdec_err, pmra_pmdec_corr, min_stars)
        
        best_cluster = min(unique_labels,
                           key=lambda l: np.linalg.norm(X_scaled[labels == l].mean(axis=0) - ref_scaled))
        cluster_mask_dbscan = labels == best_cluster
        
        if np.sum(cluster_mask_dbscan) < min_stars:
            return algorithm_gmm_with_errors(pmra, pmdec, center_pmra, center_pmdec,
                                             pmra_err, pmdec_err, pmra_pmdec_corr, min_stars)
        
        # Stage 2: GMM on cleaned data
        X_clean = X[cluster_mask_dbscan]
        gmm = GaussianMixture(n_components=2, max_iter=cfg.GMM_MAX_ITER,
                              n_init=cfg.GMM_N_INIT, random_state=cfg.GMM_RANDOM_STATE)
        gmm.fit(X_clean)
        
        ref_point = np.array([center_pmra, center_pmdec])
        cluster_idx = np.argmin([np.linalg.norm(gmm.means_[i] - ref_point) for i in range(2)])
        
        mu_cluster = gmm.means_[cluster_idx]
        Sigma_cluster = gmm.covariances_[cluster_idx]
        mu_field = gmm.means_[1 - cluster_idx]
        Sigma_field = gmm.covariances_[1 - cluster_idx]
        eta = gmm.weights_[cluster_idx]
        
        P_mem_valid = np.zeros(n_valid)
        for i in range(n_valid):
            try:
                p_cl = eta * multivariate_normal.pdf(X[i], mu_cluster, Sigma_cluster)
                p_f = (1 - eta) * multivariate_normal.pdf(X[i], mu_field, Sigma_field)
                P_mem_valid[i] = p_cl / (p_cl + p_f) if (p_cl + p_f) > 0 else 0.5
            except:
                P_mem_valid[i] = 0.5
        
        P_mem_valid[labels == -1] *= 0.5  # Penalize noise
        
        P_mem = np.full(n_stars, np.nan)
        P_mem[valid_mask] = P_mem_valid
        
        info = {
            'status': 'success',
            'algorithm': 'Hybrid (DBSCAN→GMM)',
            'mu_cluster': mu_cluster.tolist(),
            'Sigma_cluster': Sigma_cluster.tolist(),
            'mu_field': mu_field.tolist(),
            'Sigma_field': Sigma_field.tolist(),
            'eta': float(eta),
            'pm_dispersion': float(np.sqrt(np.trace(Sigma_cluster))),
            'center_pmra': center_pmra,
            'center_pmdec': center_pmdec,
        }
        return P_mem, info
        
    except Exception as e:
        return np.full(n_stars, 0.5), {'status': f'error: {e}', 'algorithm': 'Hybrid'}


def algorithm_stream_dbscan(pmra: np.ndarray, pmdec: np.ndarray,
                            center_pmra: float, center_pmdec: float,
                            eps: float = 0.4, min_samples: int = 3,
                            min_stars: int = 10) -> Tuple[np.ndarray, Dict]:
    """Stream-DBSCAN - for SGR STREAMS."""
    n_stars = len(pmra)
    valid_mask = ~(np.isnan(pmra) | np.isnan(pmdec))
    n_valid = np.sum(valid_mask)
    
    if n_valid < min_stars:
        return np.full(n_stars, 0.5), {'status': 'insufficient_data', 'algorithm': 'Stream-DBSCAN'}
    
    X = np.column_stack([pmra[valid_mask], pmdec[valid_mask]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ref_scaled = scaler.transform([[center_pmra, center_pmdec]])[0]
    
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        unique_labels = set(labels) - {-1}
        
        if len(unique_labels) == 0:
            distances = np.linalg.norm(X_scaled - ref_scaled, axis=1)
            P_mem_valid = np.exp(-distances**2 / (2 * eps**2))
            P_mem = np.full(n_stars, np.nan)
            P_mem[valid_mask] = P_mem_valid
            return P_mem, {
                'status': 'no_cluster_found',
                'algorithm': 'Stream-DBSCAN',
                'center_pmra': center_pmra,
                'center_pmdec': center_pmdec,
            }
        
        # Find stream closest to reference (min dist to any point)
        best_cluster = min(unique_labels,
                           key=lambda l: np.min(np.linalg.norm(X_scaled[labels == l] - ref_scaled, axis=1)))
        cluster_mask = labels == best_cluster
        cluster_points = X_scaled[cluster_mask]
        
        P_mem_valid = np.zeros(n_valid)
        
        if len(cluster_points) > 0:
            tree = cKDTree(cluster_points)
            for idx in np.where(cluster_mask)[0]:
                dists, _ = tree.query(X_scaled[idx], k=min(3, len(cluster_points)))
                avg_dist = np.mean(dists[1:]) if len(dists) > 1 else 0
                P_mem_valid[idx] = np.clip(1.0 - avg_dist / eps, 0.5, 1.0)
        
        non_cluster = ~cluster_mask
        if np.any(non_cluster):
            for idx in np.where(non_cluster)[0]:
                dist = np.min(np.linalg.norm(cluster_points - X_scaled[idx], axis=1))
                P_mem_valid[idx] = 0.3 * np.exp(-dist**2 / (2 * eps**2))
        
        P_mem = np.full(n_stars, np.nan)
        P_mem[valid_mask] = P_mem_valid
        
        cluster_pm = X[cluster_mask]
        mu_cluster = cluster_pm.mean(axis=0) if len(cluster_pm) > 0 else np.array([center_pmra, center_pmdec])
        Sigma_cluster = np.cov(cluster_pm.T) if len(cluster_pm) > 2 else np.eye(2) * 0.1
        
        info = {
            'status': 'success',
            'algorithm': 'Stream-DBSCAN',
            'n_stream_members': int(np.sum(cluster_mask)),
            'mu_cluster': mu_cluster.tolist(),
            'Sigma_cluster': Sigma_cluster.tolist(),
            'pm_dispersion': float(np.sqrt(np.trace(Sigma_cluster))) if len(cluster_pm) > 2 else np.nan,
            'eta': float(np.sum(cluster_mask) / n_valid),
            'center_pmra': center_pmra,
            'center_pmdec': center_pmdec,
        }
        return P_mem, info
        
    except Exception as e:
        return np.full(n_stars, 0.5), {'status': f'error: {e}', 'algorithm': 'Stream-DBSCAN'}


def compute_adaptive_membership(pmra: np.ndarray, pmdec: np.ndarray,
                                center_pmra: float, center_pmdec: float,
                                object_type: str,
                                pmra_err: np.ndarray = None,
                                pmdec_err: np.ndarray = None,
                                pmra_pmdec_corr: np.ndarray = None,
                                min_stars: int = 10) -> Tuple[np.ndarray, Dict]:
    """Master dispatcher - selects algorithm based on object type."""
    obj_type = object_type.upper()
    
    if obj_type == 'GC':
        return algorithm_gmm_with_errors(pmra, pmdec, center_pmra, center_pmdec,
                                         pmra_err, pmdec_err, pmra_pmdec_corr, min_stars)
    elif obj_type == 'OC':
        return algorithm_dbscan(pmra, pmdec, center_pmra, center_pmdec,
                                cfg.DBSCAN_EPS_OC, cfg.DBSCAN_MIN_SAMPLES_OC,
                                use_hdbscan=HAS_HDBSCAN, min_stars=min_stars)
    elif obj_type == 'DW':
        return algorithm_hybrid_dbscan_gmm(pmra, pmdec, center_pmra, center_pmdec,
                                           pmra_err, pmdec_err, pmra_pmdec_corr, min_stars)
    elif obj_type in ['SGR', 'STREAM']:
        return algorithm_stream_dbscan(pmra, pmdec, center_pmra, center_pmdec,
                                       cfg.DBSCAN_EPS_STREAM, cfg.DBSCAN_MIN_SAMPLES_STREAM,
                                       min_stars)
    else:
        return algorithm_gmm_with_errors(pmra, pmdec, center_pmra, center_pmdec,
                                         pmra_err, pmdec_err, pmra_pmdec_corr, min_stars)


# ============================================================================
# PHASE 2 & 3: PROCESS MEMBER CATALOGS
# ============================================================================

def process_gc_members(master: MasterCatalog, logger: logging.Logger) -> List[Dict]:
    """Process Globular Cluster members."""
    if not cfg.GC_MEMBERS_FILE or not os.path.exists(cfg.GC_MEMBERS_FILE):
        logger.info("GC members file not found - skipping")
        return []
    
    logger.info("\n" + "-" * 50)
    logger.info("Processing GLOBULAR CLUSTERS...")
    logger.info("-" * 50)
    
    df = pd.read_csv(cfg.GC_MEMBERS_FILE)
    cols = cfg.GC_MEM_COLS
    key_col = cols['key']
    
    clusters = sorted(df[key_col].unique()) if key_col in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters")
    
    results = []
    for i, cluster_name in enumerate(clusters):
        logger.info(f"\n[{i+1}/{len(clusters)}] {cluster_name}")
        
        cluster_df = df[df[key_col] == cluster_name].copy() if key_col in df.columns else df.copy()
        
        if len(cluster_df) < cfg.MIN_STARS_FOR_ANALYSIS:
            logger.info(f"  Skipping - only {len(cluster_df)} members")
            continue
        
        # Drop NaN coords
        ra = cluster_df[cols['ra']].values
        dec = cluster_df[cols['dec']].values
        valid_coords = ~(np.isnan(ra) | np.isnan(dec))
        cluster_df = cluster_df[valid_coords].reset_index(drop=True)
        ra = ra[valid_coords]
        dec = dec[valid_coords]
        
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS:
            continue
        
        # Cross-match
        master_idx, mem_idx, seps = master.query(ra, dec)
        match_pct = 100 * len(master_idx) / len(ra)
        logger.info(f"  Members: {len(ra)}, Matched: {len(master_idx)} ({match_pct:.1f}%)")
        
        # Get reference PM
        center_pmra = cluster_df[cols['pmra']].median()
        center_pmdec = cluster_df[cols['pmdec']].median()
        
        # Run algorithm if enough matches
        algo_info = {'status': 'no_matches', 'algorithm': 'None',
                     'center_pmra': center_pmra, 'center_pmdec': center_pmdec}
        matched_df = None
        
        if len(master_idx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            matched_members = cluster_df.iloc[mem_idx].reset_index(drop=True)
            matched_master = master.get_matched_data(master_idx)
            matched_master.columns = [f"{c}_master" for c in matched_master.columns] # Rename master cols
            matched_df = pd.concat([matched_members, matched_master], axis=1)
            matched_df['xmatch_sep_arcsec'] = seps
            
            # --- Renaming back for consistency with algorithms ---
            # We need standard names for plotting and algorithms
            matched_df['pmra'] = matched_df[f"{cfg.MASTER_COLS['pmra']}_master"]
            matched_df['pmdec'] = matched_df[f"{cfg.MASTER_COLS['pmdec']}_master"]
            matched_df['ra'] = matched_df[f"{cfg.MASTER_COLS['ra']}_master"]
            matched_df['dec'] = matched_df[f"{cfg.MASTER_COLS['dec']}_master"]
            if f"{cfg.MASTER_COLS['dist']}_master" in matched_df.columns:
                matched_df['computed_dist'] = matched_df[f"{cfg.MASTER_COLS['dist']}_master"]
            
            pmra_arr = matched_df['pmra'].values
            pmdec_arr = matched_df['pmdec'].values
            pmra_err = matched_df[f"{cfg.MASTER_COLS['pmra_err']}_master"].values
            pmdec_err = matched_df[f"{cfg.MASTER_COLS['pmdec_err']}_master"].values
            
            P_mem, algo_info = compute_adaptive_membership(
                pmra_arr, pmdec_arr, center_pmra, center_pmdec, 'GC',
                pmra_err, pmdec_err, None, cfg.MIN_STARS_FOR_ANALYSIS
            )
            matched_df['P_mem'] = P_mem
            
            n_high = np.sum(P_mem > cfg.P_MEM_HIGH)
            logger.info(f"  Algorithm: {algo_info['algorithm']} | n(P>0.8): {n_high}")
            
            # --- NEW: Generate Individual Plot ---
            plot_individual_match(matched_df, cluster_name, 'GC', algo_info, 
                                  ref_dist=None) # GC csv usually has no dist col
        
        results.append({
            'cluster_name': cluster_name,
            'obj_type': 'GC',
            'member_df': cluster_df,
            'matched_df': matched_df,
            'algo_info': algo_info,
            'mem_cols': cols,
            'n_members': len(cluster_df),
            'n_matched': len(master_idx),
        })
        
        gc.collect()
    
    return results


def process_oc_members(master: MasterCatalog, logger: logging.Logger) -> List[Dict]:
    """Process Open Cluster members."""
    if not cfg.OC_MEMBERS_FILE or not os.path.exists(cfg.OC_MEMBERS_FILE):
        logger.info("OC members file not found - skipping")
        return []
    
    logger.info("\n" + "-" * 50)
    logger.info("Processing OPEN CLUSTERS...")
    logger.info("-" * 50)
    
    df = pd.read_csv(cfg.OC_MEMBERS_FILE)
    cols = cfg.OC_MEM_COLS
    key_col = cols['key']
    
    clusters = sorted(df[key_col].unique()) if key_col in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters")
    
    results = []
    for i, cluster_name in enumerate(clusters):
        logger.info(f"\n[{i+1}/{len(clusters)}] {cluster_name}")
        
        cluster_df = df[df[key_col] == cluster_name].copy() if key_col in df.columns else df.copy()
        
        if len(cluster_df) < cfg.MIN_STARS_FOR_ANALYSIS:
            logger.info(f"  Skipping - only {len(cluster_df)} members")
            continue
        
        ra = cluster_df[cols['ra']].values
        dec = cluster_df[cols['dec']].values
        valid_coords = ~(np.isnan(ra) | np.isnan(dec))
        cluster_df = cluster_df[valid_coords].reset_index(drop=True)
        ra = ra[valid_coords]
        dec = dec[valid_coords]
        
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS:
            continue
        
        master_idx, mem_idx, seps = master.query(ra, dec)
        match_pct = 100 * len(master_idx) / len(ra)
        logger.info(f"  Members: {len(ra)}, Matched: {len(master_idx)} ({match_pct:.1f}%)")
        
        center_pmra = cluster_df[cols['pmra']].median()
        center_pmdec = cluster_df[cols['pmdec']].median()
        
        algo_info = {'status': 'no_matches', 'algorithm': 'None',
                     'center_pmra': center_pmra, 'center_pmdec': center_pmdec}
        matched_df = None
        
        if len(master_idx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            matched_members = cluster_df.iloc[mem_idx].reset_index(drop=True)
            matched_master = master.get_matched_data(master_idx)
            matched_master.columns = [f"{c}_master" for c in matched_master.columns]
            matched_df = pd.concat([matched_members, matched_master], axis=1)
            matched_df['xmatch_sep_arcsec'] = seps
            
            matched_df['pmra'] = matched_df[f"{cfg.MASTER_COLS['pmra']}_master"]
            matched_df['pmdec'] = matched_df[f"{cfg.MASTER_COLS['pmdec']}_master"]
            matched_df['ra'] = matched_df[f"{cfg.MASTER_COLS['ra']}_master"]
            matched_df['dec'] = matched_df[f"{cfg.MASTER_COLS['dec']}_master"]
            if f"{cfg.MASTER_COLS['dist']}_master" in matched_df.columns:
                matched_df['computed_dist'] = matched_df[f"{cfg.MASTER_COLS['dist']}_master"]
            
            pmra_arr = matched_df['pmra'].values
            pmdec_arr = matched_df['pmdec'].values
            
            P_mem, algo_info = compute_adaptive_membership(
                pmra_arr, pmdec_arr, center_pmra, center_pmdec, 'OC',
                None, None, None, cfg.MIN_STARS_FOR_ANALYSIS
            )
            matched_df['P_mem'] = P_mem
            
            n_high = np.sum(P_mem > cfg.P_MEM_HIGH)
            logger.info(f"  Algorithm: {algo_info['algorithm']} | n(P>0.8): {n_high}")
            
            plot_individual_match(matched_df, cluster_name, 'OC', algo_info)
        
        results.append({
            'cluster_name': cluster_name,
            'obj_type': 'OC',
            'member_df': cluster_df,
            'matched_df': matched_df,
            'algo_info': algo_info,
            'mem_cols': cols,
            'n_members': len(cluster_df),
            'n_matched': len(master_idx),
        })
        
        gc.collect()
    
    return results


def process_sgr_members(master: MasterCatalog, logger: logging.Logger) -> List[Dict]:
    """Process SGR Stream members (by distance bins)."""
    if not cfg.SGR_MEMBERS_FILE or not os.path.exists(cfg.SGR_MEMBERS_FILE):
        logger.info("SGR members file not found - skipping")
        return []
    
    logger.info("\n" + "-" * 50)
    logger.info("Processing SGR STREAM...")
    logger.info("-" * 50)
    
    df = pd.read_csv(cfg.SGR_MEMBERS_FILE)
    cols = cfg.SGR_MEM_COLS
    
    # Bin by distance
    dist_col = cols.get('dist', 'dist')
    if dist_col not in df.columns:
        logger.warning(f"Distance column '{dist_col}' not found - treating as single object")
        bins = ['ALL']
        df['dist_bin'] = 'ALL'
    else:
        df = df.dropna(subset=[dist_col])
        max_dist = df[dist_col].max()
        bin_edges = np.arange(cfg.SGR_BIN_START_KPC, max_dist + cfg.SGR_BIN_WIDTH_KPC, cfg.SGR_BIN_WIDTH_KPC)
        bin_labels = [f'{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f} kpc' for i in range(len(bin_edges)-1)]
        df['dist_bin'] = pd.cut(df[dist_col], bins=bin_edges, labels=bin_labels, right=False)
        df = df.dropna(subset=['dist_bin'])
        bins = [l for l in bin_labels if (df['dist_bin'] == l).sum() >= cfg.MIN_STARS_FOR_ANALYSIS]
    
    logger.info(f"Found {len(bins)} distance bins")
    
    results = []
    for i, bin_label in enumerate(bins):
        logger.info(f"\n[{i+1}/{len(bins)}] {bin_label}")
        
        bin_df = df[df['dist_bin'] == bin_label].copy()
        
        ra = bin_df[cols['ra']].values
        dec = bin_df[cols['dec']].values
        valid_coords = ~(np.isnan(ra) | np.isnan(dec))
        bin_df = bin_df[valid_coords].reset_index(drop=True)
        ra = ra[valid_coords]
        dec = dec[valid_coords]
        
        if len(ra) < cfg.MIN_STARS_FOR_ANALYSIS:
            continue
        
        master_idx, mem_idx, seps = master.query(ra, dec)
        match_pct = 100 * len(master_idx) / len(ra)
        logger.info(f"  Members: {len(ra)}, Matched: {len(master_idx)} ({match_pct:.1f}%)")
        
        center_pmra = bin_df[cols['pmra']].median()
        center_pmdec = bin_df[cols['pmdec']].median()
        
        algo_info = {'status': 'no_matches', 'algorithm': 'None',
                     'center_pmra': center_pmra, 'center_pmdec': center_pmdec}
        matched_df = None
        
        if len(master_idx) >= cfg.MIN_STARS_FOR_ANALYSIS:
            matched_members = bin_df.iloc[mem_idx].reset_index(drop=True)
            matched_master = master.get_matched_data(master_idx)
            matched_master.columns = [f"{c}_master" for c in matched_master.columns]
            matched_df = pd.concat([matched_members, matched_master], axis=1)
            matched_df['xmatch_sep_arcsec'] = seps
            
            matched_df['pmra'] = matched_df[f"{cfg.MASTER_COLS['pmra']}_master"]
            matched_df['pmdec'] = matched_df[f"{cfg.MASTER_COLS['pmdec']}_master"]
            matched_df['ra'] = matched_df[f"{cfg.MASTER_COLS['ra']}_master"]
            matched_df['dec'] = matched_df[f"{cfg.MASTER_COLS['dec']}_master"]
            if f"{cfg.MASTER_COLS['dist']}_master" in matched_df.columns:
                matched_df['computed_dist'] = matched_df[f"{cfg.MASTER_COLS['dist']}_master"]
            
            pmra_arr = matched_df['pmra'].values
            pmdec_arr = matched_df['pmdec'].values
            
            P_mem, algo_info = compute_adaptive_membership(
                pmra_arr, pmdec_arr, center_pmra, center_pmdec, 'SGR',
                None, None, None, cfg.MIN_STARS_FOR_ANALYSIS
            )
            matched_df['P_mem'] = P_mem
            
            n_high = np.sum(P_mem > cfg.P_MEM_HIGH)
            logger.info(f"  Algorithm: {algo_info['algorithm']} | n(P>0.8): {n_high}")
            
            # Use bin mid-point distance for reference line if needed
            plot_individual_match(matched_df, bin_label, 'SGR', algo_info)
        
        results.append({
            'cluster_name': bin_label,
            'obj_type': 'SGR',
            'member_df': bin_df,
            'matched_df': matched_df,
            'algo_info': algo_info,
            'mem_cols': cols,
            'n_members': len(bin_df),
            'n_matched': len(master_idx),
        })
        
        gc.collect()
    
    return results


def process_dwg_members(master: MasterCatalog, logger: logging.Logger) -> List[Dict]:
    """
    Process Dwarf Galaxy members.
    NOTE: DWG_members.csv contains galaxy-level properties, NOT individual member stars.
    """
    if not cfg.DWG_MEMBERS_FILE or not os.path.exists(cfg.DWG_MEMBERS_FILE):
        logger.info("DWG members file not found - skipping")
        return []
    
    logger.info("\n" + "-" * 50)
    logger.info("Processing DWARF GALAXIES...")
    logger.info("-" * 50)
    logger.info("NOTE: Searching around galaxy centers in master catalog")
    
    df = pd.read_csv(cfg.DWG_MEMBERS_FILE)
    cols = cfg.DWG_MEM_COLS
    key_col = cols['key']
    
    galaxies = df[key_col].unique() if key_col in df.columns else ['ALL']
    logger.info(f"Found {len(galaxies)} galaxies")
    
    results = []
    for i, gal_name in enumerate(galaxies):
        logger.info(f"\n[{i+1}/{len(galaxies)}] {gal_name}")
        
        gal_row = df[df[key_col] == gal_name].iloc[0] if key_col in df.columns else df.iloc[0]
        
        # Get galaxy center and search radius
        gal_ra = gal_row[cols['ra']]
        gal_dec = gal_row[cols['dec']]
        gal_pmra = gal_row[cols['pmra']] if cols['pmra'] in df.columns else 0
        gal_pmdec = gal_row[cols['pmdec']] if cols['pmdec'] in df.columns else 0
        
        # Search radius in degrees (from r_half in arcmin or kpc)
        rhalf_col = cols.get('rhalf')
        if rhalf_col and rhalf_col in df.columns:
            rhalf = gal_row[rhalf_col]  # in arcmin
            search_radius_deg = cfg.DWG_SEARCH_RADIUS_FACTOR * rhalf / 60.0
        else:
            search_radius_deg = 0.5  # Default 0.5 degree radius
        
        # Convert search radius to 3D chord
        search_rad = np.radians(search_radius_deg)
        search_chord = 2 * np.sin(search_rad / 2)
        
        # Query master catalog within search radius
        ra_rad = np.radians(gal_ra)
        dec_rad = np.radians(gal_dec)
        gal_coord = np.array([[
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ]])
        
        # Find all sources within search radius
        indices = master.tree.query_ball_point(gal_coord[0], search_chord)
        
        logger.info(f"  Search radius: {search_radius_deg*60:.1f} arcmin")
        logger.info(f"  Found {len(indices)} sources in search region")
        
        if len(indices) < cfg.MIN_STARS_FOR_ANALYSIS:
            continue
        
        # Get matched data
        matched_master = master.df.iloc[indices].reset_index(drop=True)
        
        # Create a "member_df" with galaxy info repeated for plotting
        member_df = pd.DataFrame({
            cols['ra']: [gal_ra],
            cols['dec']: [gal_dec],
            cols['pmra']: [gal_pmra],
            cols['pmdec']: [gal_pmdec],
            key_col: [gal_name],
        })
        
        matched_df = matched_master.copy()
        matched_df.columns = [f"{c}_master" for c in matched_df.columns]
        
        matched_df['pmra'] = matched_df[f"{cfg.MASTER_COLS['pmra']}_master"]
        matched_df['pmdec'] = matched_df[f"{cfg.MASTER_COLS['pmdec']}_master"]
        matched_df['ra'] = matched_df[f"{cfg.MASTER_COLS['ra']}_master"]
        matched_df['dec'] = matched_df[f"{cfg.MASTER_COLS['dec']}_master"]
        if f"{cfg.MASTER_COLS['dist']}_master" in matched_df.columns:
            matched_df['computed_dist'] = matched_df[f"{cfg.MASTER_COLS['dist']}_master"]
        
        # Run algorithm
        pmra_arr = matched_df['pmra'].values
        pmdec_arr = matched_df['pmdec'].values
        pmra_err = matched_df[f"{cfg.MASTER_COLS['pmra_err']}_master"].values
        pmdec_err = matched_df[f"{cfg.MASTER_COLS['pmdec_err']}_master"].values
        
        P_mem, algo_info = compute_adaptive_membership(
            pmra_arr, pmdec_arr, gal_pmra, gal_pmdec, 'DW',
            pmra_err, pmdec_err, None, cfg.MIN_STARS_FOR_ANALYSIS
        )
        matched_df['P_mem'] = P_mem
        
        n_high = np.sum(P_mem > cfg.P_MEM_HIGH)
        logger.info(f"  Algorithm: {algo_info['algorithm']} | n(P>0.8): {n_high}")
        
        ref_dist = gal_row['distance'] if 'distance' in gal_row else None
        plot_individual_match(matched_df, gal_name, 'DW', algo_info, ref_dist)
        
        results.append({
            'cluster_name': gal_name,
            'obj_type': 'DW',
            'member_df': member_df,
            'matched_df': matched_df,
            'algo_info': algo_info,
            'mem_cols': cols,
            'n_members': len(indices),  # Sources in search region
            'n_matched': len(indices),
            'search_radius_deg': search_radius_deg,
        })
        
        gc.collect()
    
    return results


# ============================================================================
# PHASE 4: PLOTTING
# ============================================================================

def plot_individual_match(matched_df, obj_name, obj_type, algo_info, ref_dist=None):
    """
    Generate individual 3-panel plot: PM, RA-Dec, and Distance Histogram.
    Saves to 'individual_plots' directory.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    save_dir = os.path.join(cfg.OUTPUT_DIR, 'individual_plots')
    # FIX: FORCE DIRECTORY CREATION to avoid FileNotFoundError
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. PM Plot
    sc1 = axes[0].scatter(matched_df['pmra'], matched_df['pmdec'], c=matched_df['P_mem'], 
                          cmap=cfg.CMAP_PMEM, s=15, edgecolors='k', linewidths=0.2, vmin=0, vmax=1)
    
    if algo_info.get('Sigma_cluster') is not None and obj_type != 'SGR':
        mu = algo_info['mu_cluster']
        sigma = np.array(algo_info['Sigma_cluster'])
        w, v = np.linalg.eigh(sigma)
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0])) # Keep original logic
        width, height = 2 * 2 * np.sqrt(w) # 2-sigma
        ell = Ellipse(xy=mu, width=width, height=height, angle=angle, fill=False, edgecolor='green', lw=2)
        axes[0].add_patch(ell)
    
    axes[0].set_xlabel('pmRA (mas/yr)')
    axes[0].set_ylabel('pmDE (mas/yr)')
    axes[0].set_title(f'PM Space: {obj_name}')
    axes[0].grid(True, alpha=0.3)
    
    # 2. RA-Dec Plot (SGR Aware)
    axes[1].scatter(matched_df['ra'], matched_df['dec'], c=matched_df['P_mem'], 
                    cmap=cfg.CMAP_PMEM, s=15, edgecolors='k', linewidths=0.2, vmin=0, vmax=1)
    axes[1].set_xlabel('RA (deg)')
    axes[1].set_ylabel('Dec (deg)')
    axes[1].set_title('Spatial Distribution')
    axes[1].invert_xaxis()
    
    # Auto-decidable aspect ratio for SGR
    if obj_type in ['SGR', 'STREAM']:
        axes[1].set_aspect('auto')
    else:
        axes[1].set_aspect('equal', 'datalim')
    
    # 3. Distance Histogram (New)
    if 'computed_dist' in matched_df.columns:
        dists = matched_df['computed_dist'].dropna()
        # Filter extremes for plot clarity
        dists = dists[(dists > 0) & (dists < 200)] 
        
        if len(dists) > 0:
            axes[2].hist(dists, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Master (calc)')
            if ref_dist is not None and not np.isnan(ref_dist):
                axes[2].axvline(ref_dist, color='red', linestyle='--', lw=2, label=f'Ref: {ref_dist:.1f} kpc')
            axes[2].set_xlabel('Distance (kpc)')
            axes[2].set_ylabel('Count')
            axes[2].set_title('Distance Comparison')
            axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "No Distance Data", ha='center')
    
    plt.colorbar(sc1, ax=axes[2], label='P_mem')
    plt.tight_layout()
    
    safe_name = str(obj_name).replace(' ', '_').replace('/', '_')
    out_file = os.path.join(save_dir, f"{obj_type}_{safe_name}.{cfg.SAVE_FORMAT}")
    plt.savefig(out_file, dpi=cfg.PLOT_DPI)
    plt.close()


def generate_plots(all_results: List[Dict], logger: logging.Logger):
    """Generate summary panel plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: GENERATING PLOTS")
    logger.info("=" * 70)
    
    if not all_results:
        logger.warning("No results to plot!")
        return
    
    # Sort by number of matches
    sorted_results = sorted(all_results, key=lambda x: x['n_matched'], reverse=True)
    
    # Ensure SGRs are included in summary panels
    sgr_res = [r for r in sorted_results if r['obj_type'] in ['SGR', 'STREAM']]
    other_res = [r for r in sorted_results if r['obj_type'] not in ['SGR', 'STREAM']]
    
    # Mix top results: e.g., Top 30 clusters + Top 6 SGR/Streams
    plot_results = other_res[:max(0, cfg.MAX_PANELS - len(sgr_res[:6]))] + sgr_res[:6]
    n_plots = len(plot_results)
    
    logger.info(f"Plotting {n_plots} objects (Summary Panels)...")
    
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # PM Membership plot
    logger.info("\nGenerating PM membership plot...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows), dpi=cfg.PLOT_DPI)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    sc = None
    for i, result in enumerate(plot_results):
        sc = plot_pm_panel(axes[i], result, cfg)
    
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes[:n_plots], shrink=0.6, pad=0.02)
        cbar.set_label('P_mem', fontsize=10)
    
    plt.suptitle('Proper Motion Membership Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    pm_file = os.path.join(cfg.OUTPUT_DIR, f'ADAPTIVE_PM_membership.{cfg.SAVE_FORMAT}')
    plt.savefig(pm_file, dpi=cfg.PLOT_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {pm_file}")
    
    # RA-Dec plot
    logger.info("\nGenerating RA-Dec plot...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows), dpi=cfg.PLOT_DPI)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    sc = None
    for i, result in enumerate(plot_results):
        sc = plot_radec_panel(axes[i], result, cfg)
    
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes[:n_plots], shrink=0.6, pad=0.02)
        cbar.set_label('P_mem', fontsize=10)
    
    plt.suptitle('RA-Dec Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    radec_file = os.path.join(cfg.OUTPUT_DIR, f'ADAPTIVE_RA_DEC.{cfg.SAVE_FORMAT}')
    plt.savefig(radec_file, dpi=cfg.PLOT_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {radec_file}")


def plot_pm_panel(ax, result: Dict, config: Config):
    """Plot PM diagram with background members + P_mem coloring."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    member_df = result['member_df']
    matched_df = result['matched_df']
    cols = result['mem_cols']
    obj_type = result['obj_type']
    cluster_name = result['cluster_name']
    algo_info = result['algo_info']
    
    pmra_col = cols['pmra']
    pmdec_col = cols['pmdec']
    
    # 1. ALL member stars as gray background (for GC/OC/SGR)
    if obj_type != 'DW' and pmra_col in member_df.columns:
        valid = member_df[pmra_col].notna() & member_df[pmdec_col].notna()
        ax.scatter(
            member_df.loc[valid, pmra_col],
            member_df.loc[valid, pmdec_col],
            s=8, alpha=0.3, color='gray', edgecolors='none',
            label=f'All members (n={valid.sum()})'
        )
    
    # 2. Cross-matched stars with P_mem coloring
    sc = None
    if matched_df is not None and len(matched_df) > 0 and 'P_mem' in matched_df.columns:
        valid = matched_df['P_mem'].notna()
        plot_df = matched_df[valid]
        if len(plot_df) > 0:
            sc = ax.scatter(
                plot_df['pmra'], plot_df['pmdec'],
                c=plot_df['P_mem'], cmap=config.CMAP_PMEM,
                s=25, alpha=0.9, vmin=0, vmax=1,
                edgecolors='k', linewidth=0.3,
                label=f'X-matched (n={len(plot_df)})'
            )
    
    # 3. GMM ellipses
    if algo_info.get('Sigma_cluster') is not None and obj_type != 'SGR':
        mu_cluster = np.array(algo_info['mu_cluster'])
        Sigma_cluster = np.array(algo_info['Sigma_cluster'])
        
        for n_std in [1, 2]:
            eigenvalues, eigenvectors = np.linalg.eigh(Sigma_cluster)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 0))
            ellipse = Ellipse(xy=mu_cluster, width=width, height=height, angle=angle,
                            fill=False, edgecolor='green',
                            linewidth=2 if n_std == 1 else 1.5,
                            linestyle='-' if n_std == 1 else '--', alpha=0.8)
            ax.add_patch(ellipse)
        ax.scatter(*mu_cluster, marker='x', s=100, c='green', linewidths=2, zorder=10)
    
    # 4. Reference PM
    if 'center_pmra' in algo_info:
        ax.scatter(
            algo_info['center_pmra'], algo_info['center_pmdec'],
            marker='*', s=150, c='blue', edgecolors='white', linewidth=1, zorder=11
        )
    
    ax.set_xlabel('pmra (mas/yr)', fontsize=7)
    ax.set_ylabel('pmdec (mas/yr)', fontsize=7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6)
    
    colors = {'GC': 'darkgreen', 'OC': 'darkorange', 'DW': 'darkblue', 'SGR': 'darkred'}
    ax.set_title(f'{obj_type}: {cluster_name}', fontsize=8, fontweight='bold',
                 color=colors.get(obj_type, 'black'))
    
    return sc


def plot_radec_panel(ax, result: Dict, config: Config):
    """Plot RA-Dec with background members."""
    import matplotlib.pyplot as plt
    
    member_df = result['member_df']
    matched_df = result['matched_df']
    cols = result['mem_cols']
    obj_type = result['obj_type']
    cluster_name = result['cluster_name']
    
    ra_col = cols['ra']
    dec_col = cols['dec']
    
    if obj_type != 'DW' and ra_col in member_df.columns:
        valid = member_df[ra_col].notna() & member_df[dec_col].notna()
        ax.scatter(member_df.loc[valid, ra_col], member_df.loc[valid, dec_col],
                   s=8, alpha=0.3, color='gray', edgecolors='none')
    
    sc = None
    if matched_df is not None and len(matched_df) > 0 and 'P_mem' in matched_df.columns:
        valid = matched_df['P_mem'].notna()
        plot_df = matched_df[valid]
        if len(plot_df) > 0:
            sc = ax.scatter(plot_df['ra'], plot_df['dec'],
                            c=plot_df['P_mem'], cmap=config.CMAP_PMEM,
                            s=25, alpha=0.9, vmin=0, vmax=1,
                            edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel('RA (deg)', fontsize=7)
    ax.set_ylabel('Dec (deg)', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6)
    ax.invert_xaxis()
    
    # Auto-aspect for SGR in panel plots too
    if obj_type in ['SGR', 'STREAM']:
        ax.set_aspect('auto')
    
    colors = {'GC': 'darkgreen', 'OC': 'darkorange', 'DW': 'darkblue', 'SGR': 'darkred'}
    ax.set_title(f'{obj_type}: {cluster_name}', fontsize=8, fontweight='bold',
                 color=colors.get(obj_type, 'black'))
    
    return sc


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(all_results: List[Dict], logger: logging.Logger):
    """Save summary table and ONE consolidated matched catalog."""
    logger.info("\n" + "-" * 50)
    logger.info("Saving results...")
    logger.info("-" * 50)
    
    # 1. Summary table
    summary_data = []
    
    # 2. Prepare consolidated CSV file
    full_csv_path = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_full_membership_results.csv')
    
    # Initialize list for bulk concat (keeping logic simple as per instruction)
    master_dfs = []
    
    for r in all_results:
        n_high = 0
        if r['matched_df'] is not None and 'P_mem' in r['matched_df'].columns:
            n_high = int(np.sum(r['matched_df']['P_mem'] > cfg.P_MEM_HIGH))
            
            # Prepare DF for master CSV
            temp_df = r['matched_df'].copy()
            # Add identification columns
            temp_df.insert(0, 'Object_Type', r['obj_type'])
            temp_df.insert(1, 'Cluster_Name', r['cluster_name'])
            master_dfs.append(temp_df)
        
        summary_data.append({
            'Object': r['cluster_name'],
            'Type': r['obj_type'],
            'N_members': r['n_members'],
            'N_matched': r['n_matched'],
            'Match_pct': f"{100*r['n_matched']/r['n_members']:.1f}" if r['n_members'] > 0 else '0',
            'N_high_prob': n_high,
            'Algorithm': r['algo_info'].get('algorithm', 'None'),
            'Status': r['algo_info'].get('status', 'N/A'),
        })
    
    # Save Consolidated CSV
    if master_dfs:
        logger.info(f"Consolidating {len(master_dfs)} datasets into one CSV...")
        full_df = pd.concat(master_dfs, ignore_index=True)
        full_df.to_csv(full_csv_path, index=False)
        logger.info(f"  Saved Master CSV: {full_csv_path} ({len(full_df):,} rows)")
    else:
        logger.warning("  No matched data found to save.")
    
    # Save Summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('N_matched', ascending=False).reset_index(drop=True)
    summary_file = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"  Summary: {summary_file}")
    
    # Save algorithm results as JSON
    algo_results = {}
    for r in all_results:
        key = f"{r['obj_type']}_{r['cluster_name']}"
        algo_results[key] = r['algo_info']
    
    algo_file = os.path.join(cfg.OUTPUT_DIR, 'ADAPTIVE_algorithm_results.json')
    with open(algo_file, 'w') as f:
        json.dump(algo_results, f, indent=2, default=str)
    logger.info(f"  Algorithm results: {algo_file}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Adaptive Membership Analysis - Cluster Optimized',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--master', required=True, help='Path to master FITS catalog (52M rows)')
    parser.add_argument('--gc', default=None, help='Path to GC members CSV')
    parser.add_argument('--oc', default=None, help='Path to OC members CSV')
    parser.add_argument('--sgr', default=None, help='Path to SGR members CSV')
    parser.add_argument('--dwg', default=None, help='Path to DWG members CSV')
    parser.add_argument('--output', default='./outputs', help='Output directory')
    parser.add_argument('--checkpoint', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log', default=None, help='Log file path')
    parser.add_argument('--skip-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--preflight-only', action='store_true', help='Run pre-flight checks only')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set config from args
    cfg.MASTER_CATALOG = args.master
    cfg.GC_MEMBERS_FILE = args.gc
    cfg.OC_MEMBERS_FILE = args.oc
    cfg.SGR_MEMBERS_FILE = args.sgr
    cfg.DWG_MEMBERS_FILE = args.dwg
    cfg.OUTPUT_DIR = args.output
    cfg.CHECKPOINT_DIR = args.checkpoint
    
    # Setup logging
    log_file = args.log or os.path.join(cfg.OUTPUT_DIR, 'adaptive_membership.log')
    logger = setup_logging(log_file)
    
    logger.info("=" * 70)
    logger.info("ADAPTIVE MEMBERSHIP ANALYSIS - CLUSTER OPTIMIZED (PRO)")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Master catalog: {cfg.MASTER_CATALOG}")
    logger.info(f"Output directory: {cfg.OUTPUT_DIR}")
    
    start_time = time.time()
    
    # Phase 0: Pre-flight validation
    if not preflight_check(logger):
        logger.error("Pre-flight validation FAILED - exiting")
        sys.exit(1)
    
    if args.preflight_only:
        logger.info("Pre-flight only mode - exiting")
        sys.exit(0)
    
    # Phase 1: Load master catalog
    master = MasterCatalog(logger)
    if not master.load(cfg.MASTER_CATALOG, cfg.CHECKPOINT_DIR):
        logger.error("Failed to load master catalog - exiting")
        sys.exit(1)
    
    # Phase 2 & 3: Process member catalogs
    all_results = []
    
    gc_results = process_gc_members(master, logger)
    all_results.extend(gc_results)
    
    oc_results = process_oc_members(master, logger)
    all_results.extend(oc_results)
    
    sgr_results = process_sgr_members(master, logger)
    all_results.extend(sgr_results)
    
    dwg_results = process_dwg_members(master, logger)
    all_results.extend(dwg_results)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"PROCESSING COMPLETE: {len(all_results)} objects")
    logger.info("=" * 70)
    
    # Phase 4: Generate plots
    if not args.skip_plots:
        generate_plots(all_results, logger)
    
    # Save results
    save_results(all_results, logger)
    
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"ALL DONE! Total time: {elapsed/60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()