#!/usr/bin/env python3
"""
================================================================================
RIGOROUS COMPLETENESS & PURITY ANALYSIS - VERSION 6
================================================================================

Implements statistically rigorous methods for error modeling:

1. LOG-NORMAL MIXTURE MODEL
   - Proper support on (0, ∞) for relative errors
   - Heavy-tail modeling through log-space Gaussian mixture
   - Error deconvolution in log-space

2. EXTREME DECONVOLUTION (Bovy et al. 2011)
   - Per-star measurement uncertainty handling
   - Separates intrinsic population scatter from measurement noise
   - Gold standard for heteroscedastic astronomical data

3. GAMMA MIXTURE MODEL
   - Alternative for exponential-like tails
   - Shape flexibility for different error distributions

4. REGULARIZED GAUSSIAN MIXTURE (Improved)
   - Priors preventing component collapse
   - Robust initialization using MAD
   - Minimum tail fraction enforced

Key improvement: Prevents η → 1 collapse that makes purity meaningless.

Author: Sutirtha
Version: 6.0 (Rigorous Statistical Methods)
================================================================================
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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp, gammaln, digamma
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree
from scipy.integrate import quad
from astropy.io import fits
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class MixtureMethod(Enum):
    """Available mixture model methods."""
    GAUSSIAN = "gaussian"           # Standard (often fails)
    LOGNORMAL = "lognormal"         # Log-normal mixture
    GAMMA = "gamma"                 # Gamma mixture
    EXTREME_DECONV = "xd"           # Extreme Deconvolution
    REGULARIZED_GAUSSIAN = "reg_gaussian"  # Regularized Gaussian with priors


@dataclass
class AnalysisConfig:
    """Configuration for rigorous completeness & purity analysis."""
    
    master_catalog_dir: str = ""
    gc_members_file: str = ""
    oc_members_file: str = ""
    sgr_members_file: str = ""
    gc_dist_file: str = ""
    output_dir: str = "./rigorous_comp_purity_results_v6"
    
    # Mixture model method selection
    mixture_method: MixtureMethod = MixtureMethod.LOGNORMAL
    
    # Column mappings
    master_cols: Dict[str, str] = field(default_factory=lambda: {
        'ra': 'RA_final', 'dec': 'DEC_final',
        'distance': 'distance_final', 'distance_err': 'distance_err_final',
        'rv': 'RV_final', 'rv_err': 'RV_err_final',
        'parallax': 'parallax_final', 'parallax_err': 'parallax_err_final',
        'ruwe': 'RUWE', 'gmag': 'Gmag',
    })
    
    alt_master_cols: Dict[str, List[str]] = field(default_factory=lambda: {
        'ra': ['RA_final', 'RA_all', 'ra'],
        'dec': ['DEC_final', 'DEC_all', 'dec'],
        'distance': ['distance_final', 'DIST', 'Dist_x'],
        'distance_err': ['distance_err_final', 'DISTERR'],
        'rv': ['RV_final', 'radial_velocity', 'RV'],
        'rv_err': ['RV_err_final', 'radial_velocity_error'],
        'parallax': ['parallax_final', 'parallax'],
        'parallax_err': ['parallax_err_final', 'parallax_error'],
        'ruwe': ['RUWE', 'ruwe'],
        'gmag': ['Gmag', 'phot_g_mean_mag'],
    })
    
    # Member catalog columns
    gc_mem_cols: Dict[str, str] = field(default_factory=lambda: {
        'key': 'source', 'ra': 'ra', 'dec': 'dec',
        'membership_prob': 'membership_probability',
    })
    oc_mem_cols: Dict[str, str] = field(default_factory=lambda: {
        'key': 'Cluster', 'ra': 'RAdeg', 'dec': 'DEdeg',
        'membership_prob': 'Proba',
    })
    sgr_mem_cols: Dict[str, str] = field(default_factory=lambda: {
        'ra': 'ra', 'dec': 'dec', 'dist': 'dist',
    })
    
    # Cross-match parameters
    xmatch_radius_arcsec: float = 1.0
    
    # GMM parameters
    gmm_n_components: int = 2  # Core + Tail
    gmm_max_iter: int = 500
    gmm_tol: float = 1e-6
    gmm_n_init: int = 10
    
    # REGULARIZATION PARAMETERS (CRITICAL FOR PREVENTING COLLAPSE)
    # Prior on eta: Beta(alpha_eta, beta_eta)
    eta_prior_alpha: float = 5.0   # Encourages eta away from 1
    eta_prior_beta: float = 2.0    # Allows high but not extreme eta
    min_eta: float = 0.5           # Hard minimum core fraction
    max_eta: float = 0.95          # Hard maximum (ensures some tail)
    
    # Minimum separation between components (in units of core sigma)
    min_component_separation: float = 2.0
    
    # Extreme Deconvolution parameters
    xd_n_iter: int = 200
    xd_tol: float = 1e-5
    
    # Binning for diagnostic plots
    n_radial_bins: int = 100
    rgal_min: float = 0.0
    rgal_max: float = 300.0
    
    # Quality cut thresholds
    rel_dist_err_cuts: List[float] = field(default_factory=lambda: [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0
    ])
    rel_rv_err_cuts: List[float] = field(default_factory=lambda: [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0
    ])
    ruwe_cuts: List[float] = field(default_factory=lambda: [
        1.0, 1.2, 1.4, 1.6, 2.0, 3.0, 5.0
    ])
    
    # Legacy bins for summary
    rgal_bins: List[float] = field(default_factory=lambda: [0, 20, 50, 100, 200, 500])
    
    # Membership thresholds
    p_mem_high: float = 0.8
    min_stars_for_analysis: int = 10
    
    # Plotting
    plot_dpi: int = 150
    
    # Processing limits
    max_chunks: Optional[int] = None
    max_rows_per_chunk: Optional[int] = None


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_file: str = None) -> logging.Logger:
    logger = logging.getLogger('RigorousAnalysisV6')
    logger.setLevel(logging.INFO)
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


# =============================================================================
# ABSTRACT BASE CLASS FOR MIXTURE MODELS
# =============================================================================

class BaseMixtureModel(ABC):
    """Abstract base class for mixture models."""
    
    def __init__(self, logger: logging.Logger, n_components: int = 2,
                 max_iter: int = 200, tol: float = 1e-4, n_init: int = 5):
        self.logger = logger
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.converged = False
        self.params = {}
    
    @abstractmethod
    def fit(self, data: np.ndarray, measurement_errors: np.ndarray = None,
            data_name: str = "data") -> 'BaseMixtureModel':
        """Fit the mixture model to data."""
        pass
    
    @abstractmethod
    def predict_purity(self, data: np.ndarray,
                       measurement_errors: np.ndarray = None) -> np.ndarray:
        """Compute per-star purity probability P_pure,i."""
        pass
    
    @abstractmethod
    def get_component_pdf(self, x: np.ndarray, component: int) -> np.ndarray:
        """Get PDF of a specific component for plotting."""
        pass


# =============================================================================
# 1. LOG-NORMAL MIXTURE MODEL
# =============================================================================

class LogNormalMixtureModel(BaseMixtureModel):
    """
    Log-Normal Mixture Model for positive quantities.
    
    For relative errors ε = σ_d/d > 0, we model in log-space:
        y = ln(ε)
        p(y) = η × N(y | μ_c, σ_c²) + (1-η) × N(y | μ_t, σ_t²)
    
    With error deconvolution:
        p(y_i | C_i) = η × N(y_i | μ_c, σ_c² + δ_i²) + (1-η) × N(y_i | μ_t, σ_t² + δ_i²)
    
    where δ_i is the measurement uncertainty in log-space.
    
    Key advantages:
    - Proper support on (0, ∞)
    - Heavy right tails naturally modeled
    - Prevents collapse by enforcing η ∈ [η_min, η_max]
    """
    
    def __init__(self, logger: logging.Logger, n_components: int = 2,
                 max_iter: int = 500, tol: float = 1e-6, n_init: int = 10,
                 eta_min: float = 0.5, eta_max: float = 0.95,
                 eta_prior_alpha: float = 5.0, eta_prior_beta: float = 2.0):
        super().__init__(logger, n_components, max_iter, tol, n_init)
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_prior_alpha = eta_prior_alpha
        self.eta_prior_beta = eta_prior_beta
        
        # Parameters (in log-space)
        self.eta = None          # Core fraction
        self.mu_core = None      # Log-space mean of core
        self.sigma_core = None   # Log-space std of core
        self.mu_tail = None      # Log-space mean of tail
        self.sigma_tail = None   # Log-space std of tail
    
    def _log_prior_eta(self, eta: float) -> float:
        """Log prior on eta: Beta(alpha, beta)."""
        if eta <= 0 or eta >= 1:
            return -np.inf
        return (self.eta_prior_alpha - 1) * np.log(eta) + \
               (self.eta_prior_beta - 1) * np.log(1 - eta)
    
    def _gaussian_log_likelihood(self, y: np.ndarray, mu: float, sigma: float,
                                  delta: np.ndarray = None) -> np.ndarray:
        """Gaussian log-likelihood with optional error deconvolution."""
        if delta is not None:
            total_var = sigma**2 + delta**2
        else:
            total_var = np.full_like(y, sigma**2)
        
        total_var = np.maximum(total_var, 1e-10)
        log_likelihood = -0.5 * np.log(2 * np.pi * total_var) - \
                         0.5 * (y - mu)**2 / total_var
        return log_likelihood
    
    def fit(self, data: np.ndarray, measurement_errors: np.ndarray = None,
            data_name: str = "data") -> 'LogNormalMixtureModel':
        """Fit log-normal mixture using MAP-EM algorithm."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"LOG-NORMAL MIXTURE MODEL: {data_name}")
        self.logger.info(f"{'='*60}")
        
        # Filter valid positive data
        valid = np.isfinite(data) & (data > 0)
        if measurement_errors is not None:
            measurement_errors = np.asarray(measurement_errors, dtype=np.float64)
            valid &= np.isfinite(measurement_errors) & (measurement_errors > 0)
        
        x = data[valid]
        n_samples = len(x)
        self.logger.info(f"  Valid samples: {n_samples:,}")
        
        if n_samples < 100:
            self.logger.warning("  Insufficient data for fitting")
            return self
        
        # Transform to log-space
        y = np.log(x)
        
        # Compute measurement errors in log-space (delta-method)
        # For ε with uncertainty σ_ε: δ = σ_ε / ε
        if measurement_errors is not None:
            err_valid = measurement_errors[valid[:len(measurement_errors)] if len(measurement_errors) > np.sum(valid) else valid]
            # Ensure we have matching lengths
            if len(err_valid) == len(x):
                delta = err_valid / x  # Fractional error ≈ error in log-space
                delta = np.clip(delta, 0.01, 2.0)  # Reasonable bounds
            else:
                delta = None
        else:
            delta = None
        
        # Robust statistics for initialization
        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))
        std_y = 1.4826 * mad_y  # Robust std estimate
        
        # Percentile-based initialization
        p10, p50, p90 = np.percentile(y, [10, 50, 90])
        
        self.logger.info(f"  Log-space statistics:")
        self.logger.info(f"    Median: {median_y:.4f} (original: {np.exp(median_y):.4f})")
        self.logger.info(f"    Robust std: {std_y:.4f}")
        self.logger.info(f"    P10/P50/P90: {p10:.3f}/{p50:.3f}/{p90:.3f}")
        
        best_posterior = -np.inf
        best_params = None
        
        for init_idx in range(self.n_init):
            # Initialize with different strategies
            if init_idx == 0:
                # Strategy 1: Core at median, tail at 90th percentile
                eta = 0.75
                mu_core = p50
                sigma_core = (p50 - p10) / 1.28  # ~1σ
                mu_tail = p90
                sigma_tail = 2.0 * sigma_core
            elif init_idx == 1:
                # Strategy 2: Split at 75th percentile
                p75 = np.percentile(y, 75)
                eta = 0.75
                mu_core = np.median(y[y < p75])
                sigma_core = np.std(y[y < p75])
                mu_tail = np.median(y[y >= p75])
                sigma_tail = np.std(y[y >= p75])
            else:
                # Random initialization with constraints
                eta = np.random.uniform(self.eta_min + 0.1, self.eta_max - 0.1)
                mu_core = median_y + 0.2 * std_y * np.random.randn()
                sigma_core = std_y * np.random.uniform(0.3, 0.8)
                mu_tail = mu_core + std_y * np.random.uniform(1.0, 3.0)
                sigma_tail = sigma_core * np.random.uniform(1.5, 4.0)
            
            # Ensure constraints
            sigma_core = max(sigma_core, 0.1)
            sigma_tail = max(sigma_tail, sigma_core * 1.5)
            
            # EM iterations with MAP (maximum a posteriori)
            for iteration in range(self.max_iter):
                # E-step: compute responsibilities
                log_l_core = self._gaussian_log_likelihood(y, mu_core, sigma_core, delta)
                log_l_tail = self._gaussian_log_likelihood(y, mu_tail, sigma_tail, delta)
                
                log_weighted_core = np.log(eta + 1e-10) + log_l_core
                log_weighted_tail = np.log(1 - eta + 1e-10) + log_l_tail
                log_total = np.logaddexp(log_weighted_core, log_weighted_tail)
                
                gamma_core = np.exp(log_weighted_core - log_total)
                gamma_tail = 1 - gamma_core
                
                # M-step with regularization
                N_core = np.sum(gamma_core) + 1e-10
                N_tail = np.sum(gamma_tail) + 1e-10
                
                # Update eta with Beta prior (MAP)
                eta_mle = N_core / n_samples
                eta_new = (N_core + self.eta_prior_alpha - 1) / \
                          (n_samples + self.eta_prior_alpha + self.eta_prior_beta - 2)
                eta_new = np.clip(eta_new, self.eta_min, self.eta_max)
                
                # Update means
                mu_core_new = np.sum(gamma_core * y) / N_core
                mu_tail_new = np.sum(gamma_tail * y) / N_tail
                
                # Update variances with error deconvolution
                if delta is not None:
                    var_core_obs = np.sum(gamma_core * (y - mu_core_new)**2) / N_core
                    var_core_err = np.sum(gamma_core * delta**2) / N_core
                    sigma_core_new = np.sqrt(max(var_core_obs - var_core_err, 0.01))
                    
                    var_tail_obs = np.sum(gamma_tail * (y - mu_tail_new)**2) / N_tail
                    var_tail_err = np.sum(gamma_tail * delta**2) / N_tail
                    sigma_tail_new = np.sqrt(max(var_tail_obs - var_tail_err, 0.05))
                else:
                    sigma_core_new = np.sqrt(np.sum(gamma_core * (y - mu_core_new)**2) / N_core)
                    sigma_tail_new = np.sqrt(np.sum(gamma_tail * (y - mu_tail_new)**2) / N_tail)
                
                # Ensure core has smaller mean (better quality)
                if mu_tail_new < mu_core_new:
                    mu_core_new, mu_tail_new = mu_tail_new, mu_core_new
                    sigma_core_new, sigma_tail_new = sigma_tail_new, sigma_core_new
                    eta_new = 1 - eta_new
                    eta_new = np.clip(eta_new, self.eta_min, self.eta_max)
                
                # Ensure minimum separation
                min_sep = 0.5 * sigma_core_new
                if mu_tail_new - mu_core_new < min_sep:
                    mu_tail_new = mu_core_new + min_sep
                
                # Ensure tail is wider
                sigma_tail_new = max(sigma_tail_new, sigma_core_new * 1.2)
                
                # Check convergence
                delta_params = (abs(eta_new - eta) + abs(mu_core_new - mu_core) +
                               abs(sigma_core_new - sigma_core) + abs(mu_tail_new - mu_tail))
                
                eta = eta_new
                mu_core, sigma_core = mu_core_new, sigma_core_new
                mu_tail, sigma_tail = mu_tail_new, sigma_tail_new
                
                if delta_params < self.tol:
                    break
            
            # Compute posterior (likelihood + prior)
            log_l_core = self._gaussian_log_likelihood(y, mu_core, sigma_core, delta)
            log_l_tail = self._gaussian_log_likelihood(y, mu_tail, sigma_tail, delta)
            log_weighted_core = np.log(eta + 1e-10) + log_l_core
            log_weighted_tail = np.log(1 - eta + 1e-10) + log_l_tail
            log_likelihood = np.sum(np.logaddexp(log_weighted_core, log_weighted_tail))
            
            # Add log prior
            log_posterior = log_likelihood + self._log_prior_eta(eta)
            
            if log_posterior > best_posterior:
                best_posterior = log_posterior
                best_params = (eta, mu_core, sigma_core, mu_tail, sigma_tail)
        
        # Store best parameters
        self.eta, self.mu_core, self.sigma_core, self.mu_tail, self.sigma_tail = best_params
        self.converged = True
        
        # Store in params dict for compatibility
        self.params = {
            'eta': self.eta,
            'mu_core': self.mu_core,
            'sigma_core': self.sigma_core,
            'mu_tail': self.mu_tail,
            'sigma_tail': self.sigma_tail,
            'log_posterior': best_posterior,
        }
        
        # Convert to original space for reporting
        median_core_orig = np.exp(self.mu_core)
        median_tail_orig = np.exp(self.mu_tail)
        
        self.logger.info(f"\n  LOG-NORMAL GMM CONVERGED:")
        self.logger.info(f"    η (core fraction): {self.eta:.4f}")
        self.logger.info(f"    Core: μ_log={self.mu_core:.4f}, σ_log={self.sigma_core:.4f}")
        self.logger.info(f"           → median={median_core_orig:.4f}")
        self.logger.info(f"    Tail: μ_log={self.mu_tail:.4f}, σ_log={self.sigma_tail:.4f}")
        self.logger.info(f"           → median={median_tail_orig:.4f}")
        self.logger.info(f"    Tail/Core median ratio: {median_tail_orig/median_core_orig:.2f}x")
        
        return self
    
    def predict_purity(self, data: np.ndarray,
                       measurement_errors: np.ndarray = None) -> np.ndarray:
        """Compute per-star purity probability in original space."""
        if not self.converged:
            return np.full(len(data), 0.5)
        
        # Handle non-positive data
        purity = np.full(len(data), 0.5)
        valid = np.isfinite(data) & (data > 0)
        
        if np.sum(valid) == 0:
            return purity
        
        x = data[valid]
        y = np.log(x)
        
        # Compute measurement error in log-space
        if measurement_errors is not None and len(measurement_errors) == len(data):
            err = measurement_errors[valid]
            delta = np.where((err > 0) & np.isfinite(err), err / x, 0.0)
            delta = np.clip(delta, 0, 2.0)
        else:
            delta = None
        
        log_l_core = self._gaussian_log_likelihood(y, self.mu_core, self.sigma_core, delta)
        log_l_tail = self._gaussian_log_likelihood(y, self.mu_tail, self.sigma_tail, delta)
        
        log_weighted_core = np.log(self.eta + 1e-10) + log_l_core
        log_weighted_tail = np.log(1 - self.eta + 1e-10) + log_l_tail
        
        log_total = np.logaddexp(log_weighted_core, log_weighted_tail)
        purity_valid = np.exp(log_weighted_core - log_total)
        purity_valid = np.where(np.isfinite(purity_valid), purity_valid, 0.5)
        
        purity[valid] = purity_valid
        return purity
    
    def get_component_pdf(self, x: np.ndarray, component: int) -> np.ndarray:
        """Get PDF of a component in original space."""
        if not self.converged:
            return np.zeros_like(x)
        
        valid = x > 0
        pdf = np.zeros_like(x)
        
        if component == 0:  # Core
            # Log-normal PDF: (1/(x*σ*√(2π))) * exp(-(ln(x)-μ)²/(2σ²))
            pdf[valid] = stats.lognorm.pdf(x[valid], s=self.sigma_core, 
                                           scale=np.exp(self.mu_core))
            pdf *= self.eta
        else:  # Tail
            pdf[valid] = stats.lognorm.pdf(x[valid], s=self.sigma_tail,
                                           scale=np.exp(self.mu_tail))
            pdf *= (1 - self.eta)
        
        return pdf


# =============================================================================
# 2. EXTREME DECONVOLUTION (Bovy et al. 2011)
# =============================================================================

class ExtremeDeconvolutionModel(BaseMixtureModel):
    """
    Extreme Deconvolution for heteroscedastic data.
    
    Models the distribution as:
        p(x_i | θ) = Σ_k π_k × N(x_i | μ_k, Σ_k + C_i)
    
    where C_i is the per-star measurement covariance.
    
    This properly separates:
    - Intrinsic population scatter (Σ_k)
    - Measurement noise (C_i)
    
    Reference: Bovy, Hogg, Roweis (2011) "Extreme Deconvolution"
    """
    
    def __init__(self, logger: logging.Logger, n_components: int = 2,
                 max_iter: int = 200, tol: float = 1e-5, n_init: int = 10,
                 eta_min: float = 0.5, eta_max: float = 0.95):
        super().__init__(logger, n_components, max_iter, tol, n_init)
        self.eta_min = eta_min
        self.eta_max = eta_max
        
        # Parameters (1D case)
        self.weights = None    # Component weights
        self.means = None      # Component means
        self.variances = None  # Component intrinsic variances
    
    def fit(self, data: np.ndarray, measurement_errors: np.ndarray = None,
            data_name: str = "data") -> 'ExtremeDeconvolutionModel':
        """Fit Extreme Deconvolution model."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EXTREME DECONVOLUTION: {data_name}")
        self.logger.info(f"{'='*60}")
        
        # Filter valid data
        valid = np.isfinite(data)
        if measurement_errors is not None:
            measurement_errors = np.asarray(measurement_errors, dtype=np.float64)
            valid &= np.isfinite(measurement_errors) & (measurement_errors > 0)
        
        x = data[valid].astype(np.float64)
        n_samples = len(x)
        self.logger.info(f"  Valid samples: {n_samples:,}")
        
        if n_samples < 100:
            self.logger.warning("  Insufficient data for XD fitting")
            return self
        
        # Get measurement variances
        if measurement_errors is not None:
            C = measurement_errors[valid].astype(np.float64)**2  # Variance
        else:
            # Assume 10% measurement error if not provided
            C = (0.1 * np.abs(x))**2
        
        # Robust statistics
        median_x = np.median(x)
        mad_x = np.median(np.abs(x - median_x))
        std_x = 1.4826 * mad_x
        
        self.logger.info(f"  Median: {median_x:.4f}, Robust std: {std_x:.4f}")
        self.logger.info(f"  Mean measurement variance: {np.mean(C):.6f}")
        
        best_ll = -np.inf
        best_params = None
        
        for init_idx in range(self.n_init):
            # Initialize
            if init_idx == 0:
                weights = np.array([0.75, 0.25])
                means = np.array([median_x, median_x + 2*std_x])
                variances = np.array([std_x**2 * 0.25, std_x**2 * 4.0])
            else:
                weights = np.array([np.random.uniform(0.6, 0.85), 0.0])
                weights[1] = 1 - weights[0]
                means = np.array([
                    median_x + 0.2*std_x*np.random.randn(),
                    median_x + std_x*np.random.uniform(1, 3)
                ])
                variances = np.array([
                    std_x**2 * np.random.uniform(0.1, 0.5),
                    std_x**2 * np.random.uniform(1.0, 5.0)
                ])
            
            # EM iterations
            for iteration in range(self.max_iter):
                # E-step: compute responsibilities
                log_probs = np.zeros((n_samples, self.n_components))
                
                for k in range(self.n_components):
                    total_var = variances[k] + C
                    log_probs[:, k] = (np.log(weights[k] + 1e-10) - 
                                       0.5*np.log(2*np.pi*total_var) -
                                       0.5*(x - means[k])**2 / total_var)
                
                # Normalize
                log_norm = logsumexp(log_probs, axis=1)
                log_resp = log_probs - log_norm[:, np.newaxis]
                resp = np.exp(log_resp)
                
                # M-step
                N_k = np.sum(resp, axis=0) + 1e-10
                
                # Update weights with regularization
                weights_new = N_k / n_samples
                weights_new = np.clip(weights_new, 1 - self.eta_max, self.eta_max)
                weights_new /= weights_new.sum()
                
                # Update means
                means_new = np.sum(resp * x[:, np.newaxis], axis=0) / N_k
                
                # Update intrinsic variances (deconvolved)
                variances_new = np.zeros(self.n_components)
                for k in range(self.n_components):
                    # Observed variance for this component
                    obs_var = np.sum(resp[:, k] * (x - means_new[k])**2) / N_k[k]
                    # Mean measurement variance for this component
                    mean_C = np.sum(resp[:, k] * C) / N_k[k]
                    # Intrinsic variance = observed - measurement
                    variances_new[k] = max(obs_var - mean_C, 0.01 * std_x**2)
                
                # Ensure component 0 is the "core" (smaller variance)
                if variances_new[1] < variances_new[0]:
                    means_new = means_new[::-1]
                    variances_new = variances_new[::-1]
                    weights_new = weights_new[::-1]
                
                # Check convergence
                delta = (np.sum(np.abs(weights_new - weights)) +
                        np.sum(np.abs(means_new - means)) +
                        np.sum(np.abs(np.sqrt(variances_new) - np.sqrt(variances))))
                
                weights, means, variances = weights_new, means_new, variances_new
                
                if delta < self.tol:
                    break
            
            # Compute log-likelihood
            log_probs = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                total_var = variances[k] + C
                log_probs[:, k] = (np.log(weights[k] + 1e-10) - 
                                   0.5*np.log(2*np.pi*total_var) -
                                   0.5*(x - means[k])**2 / total_var)
            ll = np.sum(logsumexp(log_probs, axis=1))
            
            if ll > best_ll:
                best_ll = ll
                best_params = (weights.copy(), means.copy(), variances.copy())
        
        self.weights, self.means, self.variances = best_params
        self.converged = True
        
        # For compatibility
        self.params = {
            'eta': self.weights[0],
            'mu_core': self.means[0],
            'sigma_core': np.sqrt(self.variances[0]),
            'mu_tail': self.means[1],
            'sigma_tail': np.sqrt(self.variances[1]),
            'log_likelihood': best_ll,
        }
        
        self.logger.info(f"\n  EXTREME DECONVOLUTION CONVERGED:")
        self.logger.info(f"    Weights: {self.weights}")
        self.logger.info(f"    Means: {self.means}")
        self.logger.info(f"    Intrinsic σ: {np.sqrt(self.variances)}")
        self.logger.info(f"    (After deconvolving measurement errors)")
        
        return self
    
    def predict_purity(self, data: np.ndarray,
                       measurement_errors: np.ndarray = None) -> np.ndarray:
        """Compute purity probability."""
        if not self.converged:
            return np.full(len(data), 0.5)
        
        purity = np.full(len(data), 0.5)
        valid = np.isfinite(data)
        
        if measurement_errors is not None and len(measurement_errors) == len(data):
            valid &= np.isfinite(measurement_errors)
        
        x = data[valid]
        
        if measurement_errors is not None and len(measurement_errors) == len(data):
            C = measurement_errors[valid]**2
        else:
            C = (0.1 * np.abs(x))**2
        
        log_probs = np.zeros((len(x), self.n_components))
        for k in range(self.n_components):
            total_var = self.variances[k] + C
            log_probs[:, k] = (np.log(self.weights[k] + 1e-10) -
                              0.5*np.log(2*np.pi*total_var) -
                              0.5*(x - self.means[k])**2 / total_var)
        
        log_norm = logsumexp(log_probs, axis=1)
        purity_valid = np.exp(log_probs[:, 0] - log_norm)
        purity_valid = np.where(np.isfinite(purity_valid), purity_valid, 0.5)
        
        purity[valid] = purity_valid
        return purity
    
    def get_component_pdf(self, x: np.ndarray, component: int) -> np.ndarray:
        """Get PDF of a component (without measurement error)."""
        if not self.converged:
            return np.zeros_like(x)
        
        return self.weights[component] * stats.norm.pdf(
            x, self.means[component], np.sqrt(self.variances[component])
        )


# =============================================================================
# 3. GAMMA MIXTURE MODEL
# =============================================================================

class GammaMixtureModel(BaseMixtureModel):
    """
    Gamma Mixture Model for positive quantities with exponential tails.
    
    p(x) = η × Gamma(x | α_c, β_c) + (1-η) × Gamma(x | α_t, β_t)
    
    Gamma is parameterized as:
        Gamma(x | α, β) = β^α / Γ(α) × x^(α-1) × exp(-βx)
    
    Mean = α/β, Variance = α/β²
    """
    
    def __init__(self, logger: logging.Logger, n_components: int = 2,
                 max_iter: int = 500, tol: float = 1e-6, n_init: int = 10,
                 eta_min: float = 0.5, eta_max: float = 0.95):
        super().__init__(logger, n_components, max_iter, tol, n_init)
        self.eta_min = eta_min
        self.eta_max = eta_max
        
        # Parameters for each component: (alpha, beta)
        self.eta = None
        self.alpha_core = None
        self.beta_core = None
        self.alpha_tail = None
        self.beta_tail = None
    
    def _gamma_log_likelihood(self, x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Compute Gamma log-likelihood."""
        return stats.gamma.logpdf(x, a=alpha, scale=1/beta)
    
    def _fit_gamma_mle(self, x: np.ndarray, weights: np.ndarray = None) -> Tuple[float, float]:
        """Fit single Gamma distribution using MLE with optional weights."""
        if weights is None:
            weights = np.ones(len(x))
        
        weights = weights / np.sum(weights)
        
        # Method of moments initialization
        mean_x = np.sum(weights * x)
        var_x = np.sum(weights * (x - mean_x)**2)
        
        alpha_init = mean_x**2 / max(var_x, 1e-6)
        beta_init = mean_x / max(var_x, 1e-6)
        
        # MLE refinement using Newton-Raphson for alpha
        # log(α) - ψ(α) = log(mean) - mean(log)
        mean_log_x = np.sum(weights * np.log(x + 1e-10))
        s = np.log(mean_x) - mean_log_x
        
        # Approximate solution
        if s > 0:
            alpha = (3 - s + np.sqrt((s-3)**2 + 24*s)) / (12*s)
        else:
            alpha = alpha_init
        
        alpha = np.clip(alpha, 0.1, 100)
        beta = alpha / mean_x
        
        return float(alpha), float(beta)
    
    def fit(self, data: np.ndarray, measurement_errors: np.ndarray = None,
            data_name: str = "data") -> 'GammaMixtureModel':
        """Fit Gamma mixture using EM algorithm."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"GAMMA MIXTURE MODEL: {data_name}")
        self.logger.info(f"{'='*60}")
        
        # Filter valid positive data
        valid = np.isfinite(data) & (data > 0)
        x = data[valid]
        n_samples = len(x)
        
        self.logger.info(f"  Valid samples: {n_samples:,}")
        
        if n_samples < 100:
            self.logger.warning("  Insufficient data for fitting")
            return self
        
        # Statistics
        median_x = np.median(x)
        mean_x = np.mean(x)
        std_x = np.std(x)
        
        self.logger.info(f"  Mean: {mean_x:.4f}, Median: {median_x:.4f}, Std: {std_x:.4f}")
        
        best_ll = -np.inf
        best_params = None
        
        for init_idx in range(self.n_init):
            # Initialize
            if init_idx == 0:
                eta = 0.75
                # Core: low mean, low variance
                alpha_core, beta_core = 4.0, 4.0 / median_x
                # Tail: high mean, high variance
                alpha_tail, beta_tail = 2.0, 1.0 / mean_x
            else:
                eta = np.random.uniform(self.eta_min + 0.1, self.eta_max - 0.1)
                alpha_core = np.random.uniform(2, 10)
                beta_core = alpha_core / (median_x * np.random.uniform(0.5, 1.5))
                alpha_tail = np.random.uniform(1, 5)
                beta_tail = alpha_tail / (mean_x * np.random.uniform(1.5, 3.0))
            
            for iteration in range(self.max_iter):
                # E-step
                log_l_core = self._gamma_log_likelihood(x, alpha_core, beta_core)
                log_l_tail = self._gamma_log_likelihood(x, alpha_tail, beta_tail)
                
                log_weighted_core = np.log(eta + 1e-10) + log_l_core
                log_weighted_tail = np.log(1 - eta + 1e-10) + log_l_tail
                log_total = np.logaddexp(log_weighted_core, log_weighted_tail)
                
                gamma_core = np.exp(log_weighted_core - log_total)
                gamma_tail = 1 - gamma_core
                
                # M-step
                N_core = np.sum(gamma_core) + 1e-10
                N_tail = np.sum(gamma_tail) + 1e-10
                
                eta_new = np.clip(N_core / n_samples, self.eta_min, self.eta_max)
                
                # Fit each component
                alpha_core_new, beta_core_new = self._fit_gamma_mle(x, gamma_core)
                alpha_tail_new, beta_tail_new = self._fit_gamma_mle(x, gamma_tail)
                
                # Ensure core has smaller mean
                mean_core = alpha_core_new / beta_core_new
                mean_tail = alpha_tail_new / beta_tail_new
                
                if mean_tail < mean_core:
                    alpha_core_new, alpha_tail_new = alpha_tail_new, alpha_core_new
                    beta_core_new, beta_tail_new = beta_tail_new, beta_core_new
                    eta_new = 1 - eta_new
                    eta_new = np.clip(eta_new, self.eta_min, self.eta_max)
                
                # Check convergence
                delta = (abs(eta_new - eta) + 
                        abs(alpha_core_new - alpha_core) + abs(beta_core_new - beta_core))
                
                eta = eta_new
                alpha_core, beta_core = alpha_core_new, beta_core_new
                alpha_tail, beta_tail = alpha_tail_new, beta_tail_new
                
                if delta < self.tol:
                    break
            
            # Compute log-likelihood
            log_l_core = self._gamma_log_likelihood(x, alpha_core, beta_core)
            log_l_tail = self._gamma_log_likelihood(x, alpha_tail, beta_tail)
            log_weighted_core = np.log(eta + 1e-10) + log_l_core
            log_weighted_tail = np.log(1 - eta + 1e-10) + log_l_tail
            ll = np.sum(np.logaddexp(log_weighted_core, log_weighted_tail))
            
            if ll > best_ll:
                best_ll = ll
                best_params = (eta, alpha_core, beta_core, alpha_tail, beta_tail)
        
        self.eta, self.alpha_core, self.beta_core, self.alpha_tail, self.beta_tail = best_params
        self.converged = True
        
        self.params = {
            'eta': self.eta,
            'alpha_core': self.alpha_core,
            'beta_core': self.beta_core,
            'alpha_tail': self.alpha_tail,
            'beta_tail': self.beta_tail,
            'mean_core': self.alpha_core / self.beta_core,
            'mean_tail': self.alpha_tail / self.beta_tail,
        }
        
        self.logger.info(f"\n  GAMMA MIXTURE CONVERGED:")
        self.logger.info(f"    η (core fraction): {self.eta:.4f}")
        self.logger.info(f"    Core: α={self.alpha_core:.3f}, β={self.beta_core:.3f}")
        self.logger.info(f"           → mean={self.alpha_core/self.beta_core:.4f}")
        self.logger.info(f"    Tail: α={self.alpha_tail:.3f}, β={self.beta_tail:.3f}")
        self.logger.info(f"           → mean={self.alpha_tail/self.beta_tail:.4f}")
        
        return self
    
    def predict_purity(self, data: np.ndarray,
                       measurement_errors: np.ndarray = None) -> np.ndarray:
        """Compute purity probability."""
        if not self.converged:
            return np.full(len(data), 0.5)
        
        purity = np.full(len(data), 0.5)
        valid = np.isfinite(data) & (data > 0)
        x = data[valid]
        
        log_l_core = self._gamma_log_likelihood(x, self.alpha_core, self.beta_core)
        log_l_tail = self._gamma_log_likelihood(x, self.alpha_tail, self.beta_tail)
        
        log_weighted_core = np.log(self.eta + 1e-10) + log_l_core
        log_weighted_tail = np.log(1 - self.eta + 1e-10) + log_l_tail
        log_total = np.logaddexp(log_weighted_core, log_weighted_tail)
        
        purity_valid = np.exp(log_weighted_core - log_total)
        purity_valid = np.where(np.isfinite(purity_valid), purity_valid, 0.5)
        
        purity[valid] = purity_valid
        return purity
    
    def get_component_pdf(self, x: np.ndarray, component: int) -> np.ndarray:
        """Get PDF of a component."""
        if not self.converged:
            return np.zeros_like(x)
        
        valid = x > 0
        pdf = np.zeros_like(x)
        
        if component == 0:
            pdf[valid] = self.eta * stats.gamma.pdf(x[valid], a=self.alpha_core, 
                                                     scale=1/self.beta_core)
        else:
            pdf[valid] = (1-self.eta) * stats.gamma.pdf(x[valid], a=self.alpha_tail,
                                                         scale=1/self.beta_tail)
        return pdf


# =============================================================================
# 4. REGULARIZED GAUSSIAN MIXTURE (Improved original)
# =============================================================================

class RegularizedGaussianMixture(BaseMixtureModel):
    """
    Regularized Gaussian Mixture with priors to prevent collapse.
    
    Improvements over standard GMM:
    1. Beta prior on η to prevent η → 1
    2. Robust initialization using MAD
    3. Minimum component separation constraint
    4. Hard bounds on η ∈ [η_min, η_max]
    """
    
    def __init__(self, logger: logging.Logger, n_components: int = 2,
                 max_iter: int = 500, tol: float = 1e-6, n_init: int = 10,
                 eta_min: float = 0.5, eta_max: float = 0.95,
                 eta_prior_alpha: float = 5.0, eta_prior_beta: float = 2.0,
                 min_separation: float = 1.0):
        super().__init__(logger, n_components, max_iter, tol, n_init)
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_prior_alpha = eta_prior_alpha
        self.eta_prior_beta = eta_prior_beta
        self.min_separation = min_separation
        
        self.eta = None
        self.mu_core = None
        self.sigma_core = None
        self.mu_tail = None
        self.sigma_tail = None
    
    def _gaussian_log_likelihood(self, x: np.ndarray, mu: float, sigma: float,
                                  errors: np.ndarray = None) -> np.ndarray:
        """Gaussian log-likelihood with error deconvolution."""
        if errors is not None:
            total_var = sigma**2 + errors**2
        else:
            total_var = np.full_like(x, sigma**2)
        
        total_var = np.maximum(total_var, 1e-10)
        return -0.5 * np.log(2 * np.pi * total_var) - 0.5 * (x - mu)**2 / total_var
    
    def fit(self, data: np.ndarray, measurement_errors: np.ndarray = None,
            data_name: str = "data") -> 'RegularizedGaussianMixture':
        """Fit regularized Gaussian mixture."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"REGULARIZED GAUSSIAN MIXTURE: {data_name}")
        self.logger.info(f"{'='*60}")
        
        valid = np.isfinite(data)
        if measurement_errors is not None:
            measurement_errors = np.asarray(measurement_errors, dtype=np.float64)
            valid_err = np.isfinite(measurement_errors) & (measurement_errors > 0)
            valid &= valid_err[:len(valid)] if len(valid_err) > len(valid) else valid_err
        
        x = data[valid]
        errors = measurement_errors[valid] if measurement_errors is not None else None
        n_samples = len(x)
        
        self.logger.info(f"  Valid samples: {n_samples:,}")
        
        if n_samples < 100:
            self.logger.warning("  Insufficient data")
            return self
        
        # Robust statistics
        median_x = np.median(x)
        mad_x = np.median(np.abs(x - median_x))
        std_x = 1.4826 * mad_x
        
        self.logger.info(f"  Median: {median_x:.4f}, Robust std: {std_x:.4f}")
        
        best_posterior = -np.inf
        best_params = None
        
        for init_idx in range(self.n_init):
            # Robust initialization
            if init_idx == 0:
                eta = 0.75
                mu_core = median_x
                sigma_core = std_x * 0.5
                mu_tail = median_x + 2 * std_x
                sigma_tail = std_x * 2.0
            else:
                eta = np.random.uniform(self.eta_min + 0.1, self.eta_max - 0.1)
                mu_core = median_x + 0.2 * std_x * np.random.randn()
                sigma_core = std_x * np.random.uniform(0.3, 0.7)
                mu_tail = mu_core + std_x * np.random.uniform(1.5, 3.0)
                sigma_tail = std_x * np.random.uniform(1.5, 4.0)
            
            for iteration in range(self.max_iter):
                # E-step
                log_l_core = self._gaussian_log_likelihood(x, mu_core, sigma_core, errors)
                log_l_tail = self._gaussian_log_likelihood(x, mu_tail, sigma_tail, errors)
                
                log_weighted_core = np.log(eta + 1e-10) + log_l_core
                log_weighted_tail = np.log(1 - eta + 1e-10) + log_l_tail
                log_total = np.logaddexp(log_weighted_core, log_weighted_tail)
                
                gamma_core = np.exp(log_weighted_core - log_total)
                gamma_tail = 1 - gamma_core
                
                # M-step with priors
                N_core = np.sum(gamma_core) + 1e-10
                N_tail = np.sum(gamma_tail) + 1e-10
                
                # MAP estimate for eta
                eta_new = (N_core + self.eta_prior_alpha - 1) / \
                          (n_samples + self.eta_prior_alpha + self.eta_prior_beta - 2)
                eta_new = np.clip(eta_new, self.eta_min, self.eta_max)
                
                mu_core_new = np.sum(gamma_core * x) / N_core
                mu_tail_new = np.sum(gamma_tail * x) / N_tail
                
                if errors is not None:
                    var_obs_core = np.sum(gamma_core * (x - mu_core_new)**2) / N_core
                    var_err_core = np.sum(gamma_core * errors**2) / N_core
                    sigma_core_new = np.sqrt(max(var_obs_core - var_err_core, 0.01 * std_x**2))
                    
                    var_obs_tail = np.sum(gamma_tail * (x - mu_tail_new)**2) / N_tail
                    var_err_tail = np.sum(gamma_tail * errors**2) / N_tail
                    sigma_tail_new = np.sqrt(max(var_obs_tail - var_err_tail, 0.05 * std_x**2))
                else:
                    sigma_core_new = np.sqrt(np.sum(gamma_core * (x - mu_core_new)**2) / N_core)
                    sigma_tail_new = np.sqrt(np.sum(gamma_tail * (x - mu_tail_new)**2) / N_tail)
                
                # Ensure proper ordering
                if sigma_tail_new < sigma_core_new:
                    mu_core_new, mu_tail_new = mu_tail_new, mu_core_new
                    sigma_core_new, sigma_tail_new = sigma_tail_new, sigma_core_new
                    eta_new = 1 - eta_new
                    eta_new = np.clip(eta_new, self.eta_min, self.eta_max)
                
                # Minimum separation
                min_sep = self.min_separation * sigma_core_new
                if abs(mu_tail_new - mu_core_new) < min_sep:
                    if mu_tail_new > mu_core_new:
                        mu_tail_new = mu_core_new + min_sep
                    else:
                        mu_tail_new = mu_core_new - min_sep
                
                # Ensure tail is wider
                sigma_tail_new = max(sigma_tail_new, sigma_core_new * 1.5)
                
                delta = (abs(eta_new - eta) + abs(mu_core_new - mu_core) + 
                        abs(sigma_core_new - sigma_core))
                
                eta, mu_core, sigma_core = eta_new, mu_core_new, sigma_core_new
                mu_tail, sigma_tail = mu_tail_new, sigma_tail_new
                
                if delta < self.tol:
                    break
            
            # Compute posterior
            log_l_core = self._gaussian_log_likelihood(x, mu_core, sigma_core, errors)
            log_l_tail = self._gaussian_log_likelihood(x, mu_tail, sigma_tail, errors)
            ll = np.sum(np.logaddexp(np.log(eta) + log_l_core, np.log(1-eta) + log_l_tail))
            
            # Add prior
            log_prior = (self.eta_prior_alpha - 1) * np.log(eta) + \
                       (self.eta_prior_beta - 1) * np.log(1 - eta)
            posterior = ll + log_prior
            
            if posterior > best_posterior:
                best_posterior = posterior
                best_params = (eta, mu_core, sigma_core, mu_tail, sigma_tail)
        
        self.eta, self.mu_core, self.sigma_core, self.mu_tail, self.sigma_tail = best_params
        self.converged = True
        
        self.params = {
            'eta': self.eta,
            'mu_core': self.mu_core,
            'sigma_core': self.sigma_core,
            'mu_tail': self.mu_tail,
            'sigma_tail': self.sigma_tail,
        }
        
        self.logger.info(f"\n  REGULARIZED GAUSSIAN CONVERGED:")
        self.logger.info(f"    η (core fraction): {self.eta:.4f}")
        self.logger.info(f"    Core: μ={self.mu_core:.4f}, σ={self.sigma_core:.4f}")
        self.logger.info(f"    Tail: μ={self.mu_tail:.4f}, σ={self.sigma_tail:.4f}")
        
        return self
    
    def predict_purity(self, data: np.ndarray,
                       measurement_errors: np.ndarray = None) -> np.ndarray:
        """Compute purity probability."""
        if not self.converged:
            return np.full(len(data), 0.5)
        
        errors = None
        if measurement_errors is not None and len(measurement_errors) == len(data):
            errors = measurement_errors
        
        log_l_core = self._gaussian_log_likelihood(data, self.mu_core, self.sigma_core, errors)
        log_l_tail = self._gaussian_log_likelihood(data, self.mu_tail, self.sigma_tail, errors)
        
        log_weighted_core = np.log(self.eta + 1e-10) + log_l_core
        log_weighted_tail = np.log(1 - self.eta + 1e-10) + log_l_tail
        log_total = np.logaddexp(log_weighted_core, log_weighted_tail)
        
        purity = np.exp(log_weighted_core - log_total)
        purity = np.where(np.isfinite(purity), purity, 0.5)
        
        return purity
    
    def get_component_pdf(self, x: np.ndarray, component: int) -> np.ndarray:
        """Get PDF of a component."""
        if not self.converged:
            return np.zeros_like(x)
        
        if component == 0:
            return self.eta * stats.norm.pdf(x, self.mu_core, self.sigma_core)
        else:
            return (1 - self.eta) * stats.norm.pdf(x, self.mu_tail, self.sigma_tail)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_mixture_model(method: MixtureMethod, logger: logging.Logger,
                         config: AnalysisConfig) -> BaseMixtureModel:
    """Factory function to create mixture models."""
    if method == MixtureMethod.LOGNORMAL:
        return LogNormalMixtureModel(
            logger, n_components=config.gmm_n_components,
            max_iter=config.gmm_max_iter, tol=config.gmm_tol,
            n_init=config.gmm_n_init,
            eta_min=config.min_eta, eta_max=config.max_eta,
            eta_prior_alpha=config.eta_prior_alpha,
            eta_prior_beta=config.eta_prior_beta
        )
    elif method == MixtureMethod.EXTREME_DECONV:
        return ExtremeDeconvolutionModel(
            logger, n_components=config.gmm_n_components,
            max_iter=config.xd_n_iter, tol=config.xd_tol,
            n_init=config.gmm_n_init,
            eta_min=config.min_eta, eta_max=config.max_eta
        )
    elif method == MixtureMethod.GAMMA:
        return GammaMixtureModel(
            logger, n_components=config.gmm_n_components,
            max_iter=config.gmm_max_iter, tol=config.gmm_tol,
            n_init=config.gmm_n_init,
            eta_min=config.min_eta, eta_max=config.max_eta
        )
    elif method == MixtureMethod.REGULARIZED_GAUSSIAN:
        return RegularizedGaussianMixture(
            logger, n_components=config.gmm_n_components,
            max_iter=config.gmm_max_iter, tol=config.gmm_tol,
            n_init=config.gmm_n_init,
            eta_min=config.min_eta, eta_max=config.max_eta,
            eta_prior_alpha=config.eta_prior_alpha,
            eta_prior_beta=config.eta_prior_beta,
            min_separation=config.min_component_separation
        )
    else:
        raise ValueError(f"Unknown mixture method: {method}")


# =============================================================================
# CATALOG LOADER (Same as v5)
# =============================================================================

class CatalogLoader:
    """Load and prepare the master catalog."""
    
    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.df = None
        self.tree = None
        self.col_mapping = {}
    
    def _find_column(self, key: str, available: List[str]) -> Optional[str]:
        primary = self.config.master_cols.get(key)
        if primary and primary in available:
            return primary
        for alt in self.config.alt_master_cols.get(key, []):
            if alt in available:
                return alt
        return None
    
    def load(self) -> bool:
        self.logger.info("=" * 70)
        self.logger.info("LOADING MASTER CATALOG")
        self.logger.info("=" * 70)
        
        p = Path(self.config.master_catalog_dir)
        if p.is_file() and str(p).endswith('.fits'):
            files = [str(p)]
        elif p.is_dir():
            files = []
            for pattern in ['Entire_catalogue_chunk*.fits', '*_chunk*.fits', '*.fits']:
                files = sorted(p.glob(pattern))
                if files:
                    files = [str(f) for f in files]
                    break
        else:
            self.logger.error(f"Path not found: {p}")
            return False
        
        if not files:
            self.logger.error("No FITS files found!")
            return False
        
        if self.config.max_chunks:
            files = files[:self.config.max_chunks]
        
        self.logger.info(f"Found {len(files)} files")
        
        dfs = []
        for i, fp in enumerate(files):
            self.logger.info(f"[{i+1}/{len(files)}] {Path(fp).name}")
            try:
                with fits.open(fp, memmap=True) as hdu:
                    data = hdu[1].data
                    cols = [c.name for c in hdu[1].columns]
                    
                    if i == 0:
                        for key in self.config.master_cols:
                            found = self._find_column(key, cols)
                            if found:
                                self.col_mapping[key] = found
                                self.logger.info(f"  {key} -> {found}")
                    
                    n_rows = len(data)
                    if self.config.max_rows_per_chunk:
                        n_rows = min(n_rows, self.config.max_rows_per_chunk)
                    
                    chunk = {}
                    for key, col in self.col_mapping.items():
                        try:
                            chunk[key] = np.array(data[col][:n_rows], dtype=np.float64)
                        except:
                            chunk[key] = np.full(n_rows, np.nan)
                    
                    dfs.append(pd.DataFrame(chunk))
                    self.logger.info(f"  Loaded {n_rows:,} rows")
            except Exception as e:
                self.logger.error(f"  Error: {e}")
            gc.collect()
        
        if not dfs:
            return False
        
        self.df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        
        self._calc_derived()
        self.df = self.df.dropna(subset=['ra', 'dec'])
        self._build_tree()
        
        self.logger.info(f"Loaded {len(self.df):,} stars")
        return True
    
    def _calc_derived(self):
        """Calculate derived columns."""
        if 'distance' in self.df.columns and 'distance_err' in self.df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.df['rel_dist_err'] = np.abs(self.df['distance_err'] / self.df['distance'])
        
        if 'rv' in self.df.columns and 'rv_err' in self.df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.df['rel_rv_err'] = np.abs(self.df['rv_err'] / np.abs(self.df['rv']))
        
        if 'parallax' in self.df.columns and 'parallax_err' in self.df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.df['parallax_over_error'] = self.df['parallax'] / self.df['parallax_err']
        
        if 'distance' not in self.df.columns or self.df['distance'].isna().all():
            if 'parallax' in self.df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.df['distance'] = 1.0 / self.df['parallax']
                    self.df.loc[self.df['parallax'] <= 0, 'distance'] = np.nan
        
        if 'ra' in self.df.columns and 'dec' in self.df.columns:
            try:
                coords = SkyCoord(ra=self.df['ra'].values * u.deg,
                                 dec=self.df['dec'].values * u.deg,
                                 frame='icrs')
                galactic = coords.galactic
                self.df['l'] = galactic.l.deg
                self.df['b'] = galactic.b.deg
            except:
                pass
        
        if 'distance' in self.df.columns and 'l' in self.df.columns and 'b' in self.df.columns:
            try:
                d = self.df['distance'].values
                l_rad = np.radians(self.df['l'].values)
                b_rad = np.radians(self.df['b'].values)
                
                R_sun = 8.2
                x = d * np.cos(b_rad) * np.cos(l_rad) - R_sun
                y = d * np.cos(b_rad) * np.sin(l_rad)
                z = d * np.sin(b_rad)
                
                self.df['R_gal'] = np.sqrt(x**2 + y**2 + z**2)
            except:
                pass
    
    def _build_tree(self):
        """Build KDTree for cross-matching."""
        ra_rad = np.radians(self.df['ra'].values)
        dec_rad = np.radians(self.df['dec'].values)
        
        coords_3d = np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ])
        
        self.tree = cKDTree(coords_3d)
        self.max_chord = 2 * np.sin(np.radians(self.config.xmatch_radius_arcsec / 3600) / 2)
    
    def query(self, ra: np.ndarray, dec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cross-match coordinates against catalog."""
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        coords = np.column_stack([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad)
        ])
        
        distances, indices = self.tree.query(coords, k=1, distance_upper_bound=self.max_chord)
        valid = np.isfinite(distances)
        
        return indices[valid], np.where(valid)[0]


# =============================================================================
# MEMBER CATALOG LOADER
# =============================================================================

class MemberCatalogLoader:
    """Load known member catalogs for completeness validation."""
    
    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.gc_members = None
        self.oc_members = None
        self.sgr_members = None
    
    def load_all(self):
        self.logger.info("\n" + "=" * 70)
        self.logger.info("LOADING MEMBER CATALOGS")
        self.logger.info("=" * 70)
        
        if self.config.gc_members_file and os.path.exists(self.config.gc_members_file):
            try:
                self.gc_members = pd.read_csv(self.config.gc_members_file)
                self.logger.info(f"  GC members: {len(self.gc_members):,}")
            except Exception as e:
                self.logger.error(f"  GC load error: {e}")
        
        if self.config.oc_members_file and os.path.exists(self.config.oc_members_file):
            try:
                self.oc_members = pd.read_csv(self.config.oc_members_file)
                self.logger.info(f"  OC members: {len(self.oc_members):,}")
            except Exception as e:
                self.logger.error(f"  OC load error: {e}")
        
        if self.config.sgr_members_file and os.path.exists(self.config.sgr_members_file):
            try:
                self.sgr_members = pd.read_csv(self.config.sgr_members_file)
                self.logger.info(f"  SGR members: {len(self.sgr_members):,}")
            except Exception as e:
                self.logger.error(f"  SGR load error: {e}")


# =============================================================================
# RIGOROUS PURITY ANALYZER (Multi-method)
# =============================================================================

class RigorousPurityAnalyzer:
    """
    Rigorous purity analysis using multiple statistical methods.
    
    Runs all available mixture models and compares results.
    """
    
    def __init__(self, config: AnalysisConfig, logger: logging.Logger,
                 catalog: CatalogLoader):
        self.config = config
        self.logger = logger
        self.catalog = catalog
        
        # Models for each error type
        self.models = {}
        self.results = {}
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run purity analysis with all configured methods."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("RIGOROUS PURITY ANALYSIS")
        self.logger.info(f"Primary method: {self.config.mixture_method.value}")
        self.logger.info("=" * 70)
        
        # Run primary method
        self._analyze_ruwe()
        self._analyze_distance_error()
        self._analyze_rv_error()
        self._compute_combined_purity()
        self._analyze_by_radius()
        
        # Run comparison with other methods
        self._run_method_comparison()
        
        return self.results
    
    def _analyze_ruwe(self):
        """Analyze RUWE distribution."""
        self.logger.info("\n" + "-" * 50)
        self.logger.info("RUWE PURITY ANALYSIS")
        self.logger.info("-" * 50)
        
        if 'ruwe' not in self.catalog.df.columns:
            self.logger.warning("  RUWE column not found")
            return
        
        ruwe = self.catalog.df['ruwe'].values
        valid = np.isfinite(ruwe) & (ruwe > 0) & (ruwe < 100)
        
        self.logger.info(f"  Valid RUWE values: {np.sum(valid):,}")
        
        # Use regularized Gaussian for RUWE (it's naturally Gaussian-ish)
        model = RegularizedGaussianMixture(
            self.logger, n_components=2,
            max_iter=self.config.gmm_max_iter, tol=self.config.gmm_tol,
            n_init=self.config.gmm_n_init,
            eta_min=self.config.min_eta, eta_max=self.config.max_eta,
            eta_prior_alpha=self.config.eta_prior_alpha,
            eta_prior_beta=self.config.eta_prior_beta
        )
        
        model.fit(ruwe[valid], data_name="RUWE")
        self.models['ruwe'] = model
        
        purity_ruwe = model.predict_purity(ruwe)
        self.catalog.df['purity_ruwe'] = purity_ruwe
        
        self.results['ruwe'] = {
            'method': 'regularized_gaussian',
            **model.params,
            'mean_purity': float(np.nanmean(purity_ruwe)),
            'median_purity': float(np.nanmedian(purity_ruwe)),
            'n_high_purity': int(np.sum(purity_ruwe > 0.8)),
        }
        
        self.logger.info(f"  Mean RUWE purity: {self.results['ruwe']['mean_purity']:.4f}")
    
    def _analyze_distance_error(self):
        """Analyze relative distance error distribution."""
        self.logger.info("\n" + "-" * 50)
        self.logger.info("DISTANCE ERROR PURITY ANALYSIS")
        self.logger.info("-" * 50)
        
        if 'rel_dist_err' not in self.catalog.df.columns:
            self.logger.warning("  Relative distance error not available")
            return
        
        rel_err = self.catalog.df['rel_dist_err'].values
        valid = np.isfinite(rel_err) & (rel_err > 0) & (rel_err < 10)
        
        self.logger.info(f"  Valid relative distance errors: {np.sum(valid):,}")
        
        # Get measurement errors for deconvolution
        dist_err = None
        if 'distance_err' in self.catalog.df.columns and 'distance' in self.catalog.df.columns:
            dist = self.catalog.df['distance'].values
            d_err = self.catalog.df['distance_err'].values
            
            with np.errstate(divide='ignore', invalid='ignore'):
                # Error on relative error: δ(σ_d/d) ≈ σ_d/d * sqrt((δσ_d/σ_d)² + (δd/d)²)
                # Approximation: use fractional error as proxy
                dist_err = np.abs(d_err / dist)
                dist_err = np.where(np.isfinite(dist_err) & (dist_err > 0), dist_err, np.nan)
        
        # Create model based on configured method
        model = create_mixture_model(self.config.mixture_method, self.logger, self.config)
        model.fit(rel_err[valid], measurement_errors=dist_err[valid] if dist_err is not None else None,
                  data_name="Relative Distance Error")
        
        self.models['distance_error'] = model
        
        purity_dist = model.predict_purity(rel_err, dist_err)
        self.catalog.df['purity_dist'] = purity_dist
        
        self.results['distance_error'] = {
            'method': self.config.mixture_method.value,
            **model.params,
            'mean_purity': float(np.nanmean(purity_dist)),
            'median_purity': float(np.nanmedian(purity_dist)),
            'n_high_purity': int(np.sum(purity_dist > 0.8)),
        }
        
        self.logger.info(f"  Mean distance purity: {self.results['distance_error']['mean_purity']:.4f}")
    
    def _analyze_rv_error(self):
        """Analyze relative RV error distribution."""
        self.logger.info("\n" + "-" * 50)
        self.logger.info("RV ERROR PURITY ANALYSIS")
        self.logger.info("-" * 50)
        
        if 'rel_rv_err' not in self.catalog.df.columns:
            self.logger.warning("  Relative RV error not available")
            return
        
        rel_err = self.catalog.df['rel_rv_err'].values
        valid = np.isfinite(rel_err) & (rel_err > 0) & (rel_err < 10)
        
        self.logger.info(f"  Valid relative RV errors: {np.sum(valid):,}")
        
        # Get measurement errors
        rv_err = None
        if 'rv_err' in self.catalog.df.columns and 'rv' in self.catalog.df.columns:
            rv = np.abs(self.catalog.df['rv'].values)
            rv_e = self.catalog.df['rv_err'].values
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rv_err = np.abs(rv_e / rv)
                rv_err = np.where(np.isfinite(rv_err) & (rv_err > 0), rv_err, np.nan)
        
        model = create_mixture_model(self.config.mixture_method, self.logger, self.config)
        model.fit(rel_err[valid], measurement_errors=rv_err[valid] if rv_err is not None else None,
                  data_name="Relative RV Error")
        
        self.models['rv_error'] = model
        
        purity_rv = model.predict_purity(rel_err, rv_err)
        self.catalog.df['purity_rv'] = purity_rv
        
        self.results['rv_error'] = {
            'method': self.config.mixture_method.value,
            **model.params,
            'mean_purity': float(np.nanmean(purity_rv)),
            'median_purity': float(np.nanmedian(purity_rv)),
            'n_high_purity': int(np.sum(purity_rv > 0.8)),
        }
        
        self.logger.info(f"  Mean RV purity: {self.results['rv_error']['mean_purity']:.4f}")
    
    def _compute_combined_purity(self):
        """Compute combined purity as product of individual purities."""
        self.logger.info("\n" + "-" * 50)
        self.logger.info("COMBINED PURITY")
        self.logger.info("-" * 50)
        
        combined = np.ones(len(self.catalog.df))
        
        for col in ['purity_ruwe', 'purity_dist', 'purity_rv']:
            if col in self.catalog.df.columns:
                combined *= self.catalog.df[col].fillna(1.0).values
        
        self.catalog.df['purity_combined'] = combined
        
        self.results['combined'] = {
            'mean_purity': float(np.nanmean(combined)),
            'median_purity': float(np.nanmedian(combined)),
            'n_high_purity': int(np.sum(combined > 0.8)),
            'n_very_high_purity': int(np.sum(combined > 0.95)),
        }
        
        self.logger.info(f"  Mean combined purity: {self.results['combined']['mean_purity']:.4f}")
        self.logger.info(f"  Median combined purity: {self.results['combined']['median_purity']:.4f}")
        self.logger.info(f"  Stars with P > 0.8: {self.results['combined']['n_high_purity']:,}")
        self.logger.info(f"  Stars with P > 0.95: {self.results['combined']['n_very_high_purity']:,}")
    
    def _analyze_by_radius(self):
        """Analyze purity by galactocentric radius."""
        self.logger.info("\n" + "-" * 50)
        self.logger.info("PURITY BY GALACTOCENTRIC RADIUS")
        self.logger.info("-" * 50)
        
        if 'R_gal' not in self.catalog.df.columns:
            self.logger.warning("  R_gal not available")
            return
        
        bins = self.config.rgal_bins
        self.results['by_radius'] = {}
        
        for i in range(len(bins) - 1):
            r_min, r_max = bins[i], bins[i+1]
            mask = (self.catalog.df['R_gal'] >= r_min) & (self.catalog.df['R_gal'] < r_max)
            n_stars = np.sum(mask)
            
            if n_stars < 10:
                continue
            
            bin_label = f"{r_min}-{r_max} kpc"
            purity_combined = self.catalog.df.loc[mask, 'purity_combined'].values
            
            self.results['by_radius'][bin_label] = {
                'n_stars': int(n_stars),
                'mean_purity': float(np.nanmean(purity_combined)),
                'median_purity': float(np.nanmedian(purity_combined)),
                'frac_high_purity': float(np.mean(purity_combined > 0.8)),
            }
            
            self.logger.info(f"  {bin_label}: N={n_stars:,}, "
                           f"P_mean={self.results['by_radius'][bin_label]['mean_purity']:.4f}")
    
    def _run_method_comparison(self):
        """Compare different mixture model methods."""
        self.logger.info("\n" + "-" * 50)
        self.logger.info("METHOD COMPARISON")
        self.logger.info("-" * 50)
        
        if 'rel_dist_err' not in self.catalog.df.columns:
            return
        
        rel_err = self.catalog.df['rel_dist_err'].values
        valid = np.isfinite(rel_err) & (rel_err > 0) & (rel_err < 10)
        
        methods = [
            MixtureMethod.LOGNORMAL,
            MixtureMethod.GAMMA,
            MixtureMethod.REGULARIZED_GAUSSIAN,
            MixtureMethod.EXTREME_DECONV,
        ]
        
        self.results['method_comparison'] = {}
        
        for method in methods:
            if method == self.config.mixture_method:
                # Already computed
                self.results['method_comparison'][method.value] = {
                    'eta': self.results['distance_error'].get('eta', 'N/A'),
                    'mean_purity': self.results['distance_error']['mean_purity'],
                }
                continue
            
            try:
                model = create_mixture_model(method, self.logger, self.config)
                model.fit(rel_err[valid], data_name=f"Distance Error ({method.value})")
                
                if model.converged:
                    purity = model.predict_purity(rel_err)
                    self.results['method_comparison'][method.value] = {
                        'eta': model.params.get('eta', 'N/A'),
                        'mean_purity': float(np.nanmean(purity)),
                    }
            except Exception as e:
                self.logger.warning(f"  {method.value} failed: {e}")
        
        self.logger.info("\n  Method comparison (distance error):")
        for method, result in self.results['method_comparison'].items():
            eta_value = result['eta']
            eta_str = f"{eta_value:.4f}" if isinstance(eta_value, (int, float)) else str(eta_value)
            self.logger.info(f"    {method}: η={eta_str}, P_mean={result['mean_purity']:.4f}")


# =============================================================================
# COMPLETENESS ANALYZER
# =============================================================================

class CompletenessAnalyzer:
    """Compute completeness using cross-match with known members."""
    
    def __init__(self, config: AnalysisConfig, logger: logging.Logger,
                 catalog: CatalogLoader, members: MemberCatalogLoader):
        self.config = config
        self.logger = logger
        self.catalog = catalog
        self.members = members
        self.results = {}
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run completeness analysis."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("COMPLETENESS ANALYSIS")
        self.logger.info("=" * 70)
        
        self._analyze_gc()
        self._analyze_oc()
        self._analyze_sgr()
        self._compute_summary()
        
        return self.results
    
    def _analyze_gc(self):
        """Analyze GC completeness."""
        if self.members.gc_members is None:
            return
        
        cols = self.config.gc_mem_cols
        df = self.members.gc_members
        
        if cols['membership_prob'] in df.columns:
            df = df[df[cols['membership_prob']] > self.config.p_mem_high]
        
        if len(df) < self.config.min_stars_for_analysis:
            return
        
        ra = df[cols['ra']].values
        dec = df[cols['dec']].values
        valid = ~(np.isnan(ra) | np.isnan(dec))
        ra, dec = ra[valid], dec[valid]
        
        matched_idx, query_idx = self.catalog.query(ra, dec)
        
        n_known = len(ra)
        n_matched = len(matched_idx)
        completeness = n_matched / n_known if n_known > 0 else 0
        
        self.results['gc'] = {
            'n_known': int(n_known),
            'n_matched': int(n_matched),
            'completeness': float(completeness),
            'completeness_err': float(np.sqrt(completeness * (1 - completeness) / n_known) if n_known > 0 else 0),
        }
        
        self.logger.info(f"  GC: {n_matched}/{n_known} = {completeness:.4f}")
    
    def _analyze_oc(self):
        """Analyze OC completeness."""
        if self.members.oc_members is None:
            return
        
        cols = self.config.oc_mem_cols
        df = self.members.oc_members
        
        if cols['membership_prob'] in df.columns:
            df = df[df[cols['membership_prob']] > self.config.p_mem_high]
        
        if len(df) < self.config.min_stars_for_analysis:
            return
        
        ra = df[cols['ra']].values
        dec = df[cols['dec']].values
        valid = ~(np.isnan(ra) | np.isnan(dec))
        ra, dec = ra[valid], dec[valid]
        
        matched_idx, query_idx = self.catalog.query(ra, dec)
        
        n_known = len(ra)
        n_matched = len(matched_idx)
        completeness = n_matched / n_known if n_known > 0 else 0
        
        self.results['oc'] = {
            'n_known': int(n_known),
            'n_matched': int(n_matched),
            'completeness': float(completeness),
            'completeness_err': float(np.sqrt(completeness * (1 - completeness) / n_known) if n_known > 0 else 0),
        }
        
        self.logger.info(f"  OC: {n_matched}/{n_known} = {completeness:.4f}")
    
    def _analyze_sgr(self):
        """Analyze SGR completeness."""
        if self.members.sgr_members is None:
            return
        
        cols = self.config.sgr_mem_cols
        df = self.members.sgr_members
        
        ra = df[cols['ra']].values
        dec = df[cols['dec']].values
        valid = ~(np.isnan(ra) | np.isnan(dec))
        ra, dec = ra[valid], dec[valid]
        
        if len(ra) < self.config.min_stars_for_analysis:
            return
        
        matched_idx, query_idx = self.catalog.query(ra, dec)
        
        n_known = len(ra)
        n_matched = len(matched_idx)
        completeness = n_matched / n_known if n_known > 0 else 0
        
        self.results['sgr'] = {
            'n_known': int(n_known),
            'n_matched': int(n_matched),
            'completeness': float(completeness),
            'completeness_err': float(np.sqrt(completeness * (1 - completeness) / n_known) if n_known > 0 else 0),
        }
        
        self.logger.info(f"  SGR: {n_matched}/{n_known} = {completeness:.4f}")
    
    def _compute_summary(self):
        """Compute overall completeness summary."""
        total_known = 0
        total_matched = 0
        
        for source in ['gc', 'oc', 'sgr']:
            if source in self.results:
                total_known += self.results[source]['n_known']
                total_matched += self.results[source]['n_matched']
        
        if total_known > 0:
            self.results['overall'] = {
                'n_known': int(total_known),
                'n_matched': int(total_matched),
                'completeness': float(total_matched / total_known),
            }
            self.logger.info(f"\n  Overall: {total_matched}/{total_known} = "
                           f"{self.results['overall']['completeness']:.4f}")


# =============================================================================
# DIAGNOSTIC ANALYZER
# =============================================================================

class DiagnosticAnalyzer:
    """Generate diagnostic plots for purity & completeness vs distance."""
    
    def __init__(self, config: AnalysisConfig, logger: logging.Logger,
                 catalog: CatalogLoader, members: MemberCatalogLoader):
        self.config = config
        self.logger = logger
        self.catalog = catalog
        self.members = members
        self.results = {}
    
    def compute_purity_completeness_vs_distance(self) -> Dict[str, Any]:
        """Compute purity and completeness vs distance for quality cuts."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("DIAGNOSTIC: PURITY & COMPLETENESS vs DISTANCE")
        self.logger.info(f"Using {self.config.n_radial_bins} radial bins")
        self.logger.info("=" * 70)
        
        r_gal = self.catalog.df['R_gal'].values
        valid_r = np.isfinite(r_gal) & (r_gal > 0) & (r_gal < self.config.rgal_max)
        
        r_edges = np.logspace(np.log10(0.1), np.log10(self.config.rgal_max),
                              self.config.n_radial_bins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        
        self.results['r_edges'] = r_edges.tolist()
        self.results['r_centers'] = r_centers.tolist()
        
        self._compute_for_rel_dist_err_cuts(r_edges, r_centers)
        self._compute_for_rel_rv_err_cuts(r_edges, r_centers)
        self._compute_for_ruwe_cuts(r_edges, r_centers)
        
        return self.results
    
    def _compute_for_rel_dist_err_cuts(self, r_edges: np.ndarray, r_centers: np.ndarray):
        """Compute for relative distance error cuts."""
        self.logger.info("\n--- Relative Distance Error Cuts ---")
        
        if 'rel_dist_err' not in self.catalog.df.columns:
            return
        
        r_gal = self.catalog.df['R_gal'].values
        rel_dist_err = self.catalog.df['rel_dist_err'].values
        purity_combined = self.catalog.df.get('purity_combined', 
                                               pd.Series(np.ones(len(self.catalog.df)))).values
        
        self.results['rel_dist_err_cuts'] = {}
        
        for cut in self.config.rel_dist_err_cuts:
            self.logger.info(f"  Processing cut: σ_d/d < {cut}")
            
            purity_profile = []
            completeness_profile = []
            n_stars_profile = []
            
            for i in range(len(r_centers)):
                r_min, r_max = r_edges[i], r_edges[i+1]
                
                mask_bin = (r_gal >= r_min) & (r_gal < r_max)
                n_total_bin = np.sum(mask_bin & np.isfinite(rel_dist_err))
                
                mask_quality = mask_bin & (rel_dist_err < cut)
                n_selected = np.sum(mask_quality)
                
                if n_selected > 0 and n_total_bin > 0:
                    purity = np.nanmean(purity_combined[mask_quality])
                    completeness = n_selected / n_total_bin
                else:
                    purity = np.nan
                    completeness = np.nan
                
                purity_profile.append(purity)
                completeness_profile.append(completeness)
                n_stars_profile.append(int(n_selected))
            
            self.results['rel_dist_err_cuts'][f'{cut}'] = {
                'purity': purity_profile,
                'completeness': completeness_profile,
                'n_stars': n_stars_profile,
            }
    
    def _compute_for_rel_rv_err_cuts(self, r_edges: np.ndarray, r_centers: np.ndarray):
        """Compute for relative RV error cuts."""
        self.logger.info("\n--- Relative RV Error Cuts ---")
        
        if 'rel_rv_err' not in self.catalog.df.columns:
            return
        
        r_gal = self.catalog.df['R_gal'].values
        rel_rv_err = self.catalog.df['rel_rv_err'].values
        purity_combined = self.catalog.df.get('purity_combined',
                                               pd.Series(np.ones(len(self.catalog.df)))).values
        
        self.results['rel_rv_err_cuts'] = {}
        
        for cut in self.config.rel_rv_err_cuts:
            self.logger.info(f"  Processing cut: σ_v/v < {cut}")
            
            purity_profile = []
            completeness_profile = []
            n_stars_profile = []
            
            for i in range(len(r_centers)):
                r_min, r_max = r_edges[i], r_edges[i+1]
                
                mask_bin = (r_gal >= r_min) & (r_gal < r_max)
                n_total_bin = np.sum(mask_bin & np.isfinite(rel_rv_err))
                
                mask_quality = mask_bin & (rel_rv_err < cut)
                n_selected = np.sum(mask_quality)
                
                if n_selected > 0 and n_total_bin > 0:
                    purity = np.nanmean(purity_combined[mask_quality])
                    completeness = n_selected / n_total_bin
                else:
                    purity = np.nan
                    completeness = np.nan
                
                purity_profile.append(purity)
                completeness_profile.append(completeness)
                n_stars_profile.append(int(n_selected))
            
            self.results['rel_rv_err_cuts'][f'{cut}'] = {
                'purity': purity_profile,
                'completeness': completeness_profile,
                'n_stars': n_stars_profile,
            }
    
    def _compute_for_ruwe_cuts(self, r_edges: np.ndarray, r_centers: np.ndarray):
        """Compute for RUWE cuts."""
        self.logger.info("\n--- RUWE Cuts ---")
        
        if 'ruwe' not in self.catalog.df.columns:
            return
        
        r_gal = self.catalog.df['R_gal'].values
        ruwe = self.catalog.df['ruwe'].values
        purity_combined = self.catalog.df.get('purity_combined',
                                               pd.Series(np.ones(len(self.catalog.df)))).values
        
        self.results['ruwe_cuts'] = {}
        
        for cut in self.config.ruwe_cuts:
            self.logger.info(f"  Processing cut: RUWE < {cut}")
            
            purity_profile = []
            completeness_profile = []
            n_stars_profile = []
            
            for i in range(len(r_centers)):
                r_min, r_max = r_edges[i], r_edges[i+1]
                
                mask_bin = (r_gal >= r_min) & (r_gal < r_max)
                n_total_bin = np.sum(mask_bin & np.isfinite(ruwe))
                
                mask_quality = mask_bin & (ruwe < cut)
                n_selected = np.sum(mask_quality)
                
                if n_selected > 0 and n_total_bin > 0:
                    purity = np.nanmean(purity_combined[mask_quality])
                    completeness = n_selected / n_total_bin
                else:
                    purity = np.nan
                    completeness = np.nan
                
                purity_profile.append(purity)
                completeness_profile.append(completeness)
                n_stars_profile.append(int(n_selected))
            
            self.results['ruwe_cuts'][f'{cut}'] = {
                'purity': purity_profile,
                'completeness': completeness_profile,
                'n_stars': n_stars_profile,
            }


# =============================================================================
# PRECISION-RECALL ANALYZER
# =============================================================================

class PrecisionRecallAnalyzer:
    """Generate Precision-Recall curves with fine galactocentric radius binning."""
    
    def __init__(self, config: AnalysisConfig, logger: logging.Logger,
                 catalog: CatalogLoader, purity_results: Dict, completeness_results: Dict):
        self.config = config
        self.logger = logger
        self.catalog = catalog
        self.purity_results = purity_results
        self.completeness_results = completeness_results
        self.results = {}
    
    def generate_curves(self) -> Dict[str, Any]:
        """Generate precision-recall curves."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PRECISION-RECALL ANALYSIS")
        self.logger.info("=" * 70)
        
        if 'purity_combined' not in self.catalog.df.columns:
            return self.results
        
        if 'R_gal' not in self.catalog.df.columns:
            return self.results
        
        n_pr_bins = min(20, self.config.n_radial_bins)
        r_edges = np.logspace(np.log10(0.1), np.log10(self.config.rgal_max), n_pr_bins + 1)
        
        thresholds = np.linspace(0, 1, 101)
        
        for i in range(len(r_edges) - 1):
            r_min, r_max = r_edges[i], r_edges[i+1]
            mask = (self.catalog.df['R_gal'] >= r_min) & (self.catalog.df['R_gal'] < r_max)
            
            purity = self.catalog.df.loc[mask, 'purity_combined'].values
            n_total = len(purity)
            
            if n_total < 100:
                continue
            
            bin_label = f"{r_min:.1f}-{r_max:.1f} kpc"
            
            precisions = []
            recalls = []
            
            for thresh in thresholds:
                selected = purity >= thresh
                n_selected = np.sum(selected)
                
                if n_selected == 0:
                    precisions.append(1.0)
                    recalls.append(0.0)
                else:
                    precision = np.sum(purity[selected]) / n_selected
                    precisions.append(precision)
                    
                    total_purity_mass = np.sum(purity)
                    captured_purity_mass = np.sum(purity[selected])
                    recall = captured_purity_mass / total_purity_mass if total_purity_mass > 0 else 0
                    recalls.append(recall)
            
            auc_pr = np.abs(np.trapz(precisions, recalls))
            
            self.results[bin_label] = {
                'thresholds': thresholds.tolist(),
                'precision': precisions,
                'recall': recalls,
                'auc': float(auc_pr),
                'n_stars': int(n_total),
                'r_min': float(r_min),
                'r_max': float(r_max),
            }
            
            self.logger.info(f"  {bin_label}: N={n_total:,}, AUC={auc_pr:.4f}")
        
        return self.results


# =============================================================================
# ENHANCED PLOTTER
# =============================================================================

class RigorousPlotter:
    """Generate all plots with enhanced diagnostics."""
    
    def __init__(self, config: AnalysisConfig, logger: logging.Logger, output_dir: str):
        self.config = config
        self.logger = logger
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.rcParams.update({
            'font.size': 11,
            'figure.dpi': 100,
            'savefig.dpi': config.plot_dpi,
            'figure.facecolor': 'white',
        })
    
    def plot_all(self, catalog: CatalogLoader, purity_analyzer: RigorousPurityAnalyzer,
                 purity_results: Dict, completeness_results: Dict, 
                 pr_results: Dict, diagnostic_results: Dict):
        """Generate all plots."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("GENERATING PLOTS")
        self.logger.info("=" * 70)
        
        self._plot_mixture_fits(catalog, purity_analyzer)
        self._plot_purity_distributions(catalog)
        self._plot_purity_by_radius(purity_results)
        self._plot_precision_recall(pr_results)
        self._plot_completeness_summary(completeness_results)
        
        # Diagnostic plots
        self._plot_purity_completeness_vs_distance_rel_dist_err(diagnostic_results)
        self._plot_purity_completeness_vs_distance_rel_rv_err(diagnostic_results)
        self._plot_purity_completeness_vs_distance_ruwe(diagnostic_results)
        
        # Method comparison
        self._plot_method_comparison(catalog, purity_analyzer)
        
        # Combined summary
        self._plot_combined_summary(purity_results, completeness_results)
        
        self.logger.info(f"\nPlots saved to: {self.output_dir}")
    
    def _plot_mixture_fits(self, catalog: CatalogLoader, analyzer: RigorousPurityAnalyzer):
        """Plot mixture model fits with proper distributions."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # RUWE
        ax = axes[0]
        if 'ruwe' in catalog.df.columns and 'ruwe' in analyzer.models:
            ruwe = catalog.df['ruwe'].dropna()
            ruwe = ruwe[(ruwe > 0) & (ruwe < 5)]
            
            ax.hist(ruwe, bins=100, density=True, alpha=0.7, color='steelblue', label='Data')
            
            model = analyzer.models['ruwe']
            if model.converged:
                x = np.linspace(0.01, 5, 1000)
                core_pdf = model.get_component_pdf(x, 0)
                tail_pdf = model.get_component_pdf(x, 1)
                
                ax.plot(x, core_pdf, 'g-', lw=2, 
                       label=f'Core (η={model.params.get("eta", 0):.2f})')
                ax.plot(x, tail_pdf, 'r-', lw=2, label='Tail')
                ax.plot(x, core_pdf + tail_pdf, 'k--', lw=2, label='Total')
            
            ax.axvline(1.4, color='orange', linestyle=':', lw=2, label='RUWE=1.4')
            ax.set_xlabel('RUWE')
            ax.set_ylabel('Density')
            ax.set_title(f'RUWE Distribution\n(Regularized Gaussian)')
            ax.legend()
            ax.set_xlim(0, 5)
        
        # Distance Error
        ax = axes[1]
        if 'rel_dist_err' in catalog.df.columns and 'distance_error' in analyzer.models:
            err = catalog.df['rel_dist_err'].dropna()
            err = err[(err > 0) & (err < 2)]
            
            ax.hist(err, bins=100, density=True, alpha=0.7, color='steelblue', label='Data')
            
            model = analyzer.models['distance_error']
            if model.converged:
                x = np.linspace(0.001, 2, 1000)
                core_pdf = model.get_component_pdf(x, 0)
                tail_pdf = model.get_component_pdf(x, 1)
                
                ax.plot(x, core_pdf, 'g-', lw=2, 
                       label=f'Core (η={model.params.get("eta", 0):.2f})')
                ax.plot(x, tail_pdf, 'r-', lw=2, label='Tail')
                ax.plot(x, core_pdf + tail_pdf, 'k--', lw=2, label='Total')
            
            ax.axvline(0.2, color='orange', linestyle=':', lw=2, label='20% error')
            ax.set_xlabel('Relative Distance Error (σ_d/d)')
            ax.set_ylabel('Density')
            ax.set_title(f'Distance Error Distribution\n({self.config.mixture_method.value})')
            ax.legend()
            ax.set_xlim(0, 2)
        
        # RV Error
        ax = axes[2]
        if 'rel_rv_err' in catalog.df.columns and 'rv_error' in analyzer.models:
            err = catalog.df['rel_rv_err'].dropna()
            err = err[(err > 0) & (err < 2)]
            
            ax.hist(err, bins=100, density=True, alpha=0.7, color='steelblue', label='Data')
            
            model = analyzer.models['rv_error']
            if model.converged:
                x = np.linspace(0.001, 2, 1000)
                core_pdf = model.get_component_pdf(x, 0)
                tail_pdf = model.get_component_pdf(x, 1)
                
                ax.plot(x, core_pdf, 'g-', lw=2,
                       label=f'Core (η={model.params.get("eta", 0):.2f})')
                ax.plot(x, tail_pdf, 'r-', lw=2, label='Tail')
                ax.plot(x, core_pdf + tail_pdf, 'k--', lw=2, label='Total')
            
            ax.set_xlabel('Relative RV Error (σ_v/v)')
            ax.set_ylabel('Density')
            ax.set_title(f'RV Error Distribution\n({self.config.mixture_method.value})')
            ax.legend()
            ax.set_xlim(0, 2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mixture_fits.png'))
        plt.close()
    
    def _plot_method_comparison(self, catalog: CatalogLoader, analyzer: RigorousPurityAnalyzer):
        """Plot comparison of different mixture methods."""
        if 'method_comparison' not in analyzer.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of eta values
        ax = axes[0]
        methods = list(analyzer.results['method_comparison'].keys())
        etas = [analyzer.results['method_comparison'][m].get('eta', 0) for m in methods]
        purities = [analyzer.results['method_comparison'][m].get('mean_purity', 0) for m in methods]
        
        # Handle non-numeric etas - convert 'N/A' to 0 for plotting
        etas_numeric = [float(e) if isinstance(e, (int, float)) else 0.0 for e in etas]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        
        bars = ax.bar(methods, etas_numeric, color=colors, edgecolor='black')
        ax.set_ylabel('Core Fraction (η)')
        ax.set_title('Component Mixing Fraction by Method')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='η=1 (collapse)')
        
        for bar, eta in zip(bars, etas_numeric):
            eta_text = f'{eta:.4f}' if eta != 0 else 'N/A'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                eta_text, ha='center')
        
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        
        # Bar chart of mean purity
        ax = axes[1]
        bars = ax.bar(methods, purities, color=colors, edgecolor='black')
        ax.set_ylabel('Mean Purity')
        ax.set_title('Mean Purity by Method')
        ax.set_ylim(0, 1)
        
        for bar, p in zip(bars, purities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{p:.3f}', ha='center')
        
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'method_comparison.png'))
        plt.close()
    
    def _plot_purity_distributions(self, catalog: CatalogLoader):
        """Plot purity probability distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        purity_cols = [('purity_ruwe', 'RUWE Purity'),
                       ('purity_dist', 'Distance Purity'),
                       ('purity_rv', 'RV Purity'),
                       ('purity_combined', 'Combined Purity')]
        
        for ax, (col, title) in zip(axes.flat, purity_cols):
            if col in catalog.df.columns:
                purity = catalog.df[col].dropna()
                
                # Check for meaningful distribution
                if len(purity) > 0:
                    ax.hist(purity, bins=50, range=(0, 1), alpha=0.7, 
                           color='steelblue', edgecolor='black')
                    ax.axvline(0.8, color='red', linestyle='--', lw=2, label='P=0.8')
                    ax.axvline(np.mean(purity), color='orange', linestyle=':', lw=2,
                              label=f'Mean={np.mean(purity):.3f}')
                    
                    # Check if distribution is informative
                    if np.std(purity) < 0.05:
                        ax.text(0.5, 0.9, 'WARNING: Low variance\n(model may have issues)',
                               transform=ax.transAxes, ha='center', color='red',
                               fontsize=10, fontweight='bold')
                    
                    ax.set_xlabel('Purity Probability P')
                    ax.set_ylabel('Count')
                    ax.set_title(title)
                    ax.legend()
        
        fig.text(0.5, 0.01,
                'P = Posterior probability of belonging to "good" (core) population\n'
                'Higher P = more reliable measurement | Low variance may indicate model issues',
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(os.path.join(self.output_dir, 'purity_distributions.png'))
        plt.close()
    
    def _plot_purity_by_radius(self, results: Dict):
        """Plot purity vs galactocentric radius."""
        if 'by_radius' not in results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        labels = list(results['by_radius'].keys())
        mean_purities = [results['by_radius'][l]['mean_purity'] for l in labels]
        high_purity_fracs = [results['by_radius'][l]['frac_high_purity'] for l in labels]
        
        x = range(len(labels))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], mean_purities, width, 
               label='Mean Purity', color='steelblue')
        ax.bar([i + width/2 for i in x], high_purity_fracs, width,
               label='Frac(P>0.8)', color='coral')
        
        ax.set_xlabel('Galactocentric Radius')
        ax.set_ylabel('Purity')
        ax.set_title('Purity by Galactocentric Radius')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'purity_by_radius.png'))
        plt.close()
    
    def _plot_precision_recall(self, results: Dict):
        """Plot precision-recall curves."""
        if not results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(results)))
        
        for (label, data), color in zip(results.items(), colors):
            ax.plot(data['recall'], data['precision'], '-', color=color, lw=2,
                   label=f'{label} (AUC={data["auc"]:.3f})')
        
        ax.set_xlabel('Recall (Completeness)', fontsize=12)
        ax.set_ylabel('Precision (Purity)', fontsize=12)
        ax.set_title('Precision-Recall Curves by Galactocentric Radius', fontsize=14)
        ax.legend(loc='lower left', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curves.png'))
        plt.close()
    
    def _plot_completeness_summary(self, results: Dict):
        """Plot completeness by source type."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sources = []
        completeness = []
        errors = []
        colors = []
        color_map = {'gc': '#2ecc71', 'oc': '#3498db', 'sgr': '#e74c3c'}
        
        for source in ['gc', 'oc', 'sgr']:
            if source in results:
                sources.append(source.upper())
                completeness.append(results[source]['completeness'])
                errors.append(results[source]['completeness_err'])
                colors.append(color_map[source])
        
        if sources:
            bars = ax.bar(sources, completeness, yerr=errors, capsize=5,
                         color=colors, edgecolor='black', alpha=0.8)
            ax.set_ylabel('Completeness')
            ax.set_title('Completeness by Source Type')
            ax.set_ylim(0, 1.05)
            ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='90% target')
            
            for bar, c in zip(bars, completeness):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{c:.3f}', ha='center')
            
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'completeness_summary.png'))
        plt.close()
    
    def _plot_purity_completeness_vs_distance_rel_dist_err(self, diagnostic_results: Dict):
        """Plot Purity & Completeness vs Distance for distance error cuts."""
        if 'rel_dist_err_cuts' not in diagnostic_results:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        r_centers = np.array(diagnostic_results['r_centers'])
        cuts = sorted([float(c) for c in diagnostic_results['rel_dist_err_cuts'].keys()])
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(cuts)))
        
        # Purity
        ax = axes[0]
        for cut, color in zip(cuts, colors):
            data = diagnostic_results['rel_dist_err_cuts'][f'{cut}']
            purity = np.array(data['purity'])
            valid = np.isfinite(purity)
            if np.sum(valid) > 5:
                ax.plot(r_centers[valid], purity[valid] * 100, '-', color=color,
                       lw=2, label=f'σ_d/d < {cut}')
        
        ax.set_ylabel('Purity [%]', fontsize=12)
        ax.set_title('Purity vs Galactocentric Distance\n(for different relative distance error cuts)',
                    fontsize=14)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Completeness
        ax = axes[1]
        for cut, color in zip(cuts, colors):
            data = diagnostic_results['rel_dist_err_cuts'][f'{cut}']
            completeness = np.array(data['completeness'])
            valid = np.isfinite(completeness)
            if np.sum(valid) > 5:
                ax.plot(r_centers[valid], completeness[valid] * 100, '--', color=color,
                       lw=2, label=f'σ_d/d < {cut}')
        
        ax.set_xlabel('Galactocentric Distance R [kpc]', fontsize=12)
        ax.set_ylabel('Completeness [%]', fontsize=12)
        ax.set_title('Completeness vs Galactocentric Distance', fontsize=14)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'purity_completeness_vs_distance_rel_dist_err.png'))
        plt.close()
    
    def _plot_purity_completeness_vs_distance_rel_rv_err(self, diagnostic_results: Dict):
        """Plot Purity & Completeness vs Distance for RV error cuts."""
        if 'rel_rv_err_cuts' not in diagnostic_results:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        r_centers = np.array(diagnostic_results['r_centers'])
        cuts = sorted([float(c) for c in diagnostic_results['rel_rv_err_cuts'].keys()])
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(cuts)))
        
        ax = axes[0]
        for cut, color in zip(cuts, colors):
            data = diagnostic_results['rel_rv_err_cuts'][f'{cut}']
            purity = np.array(data['purity'])
            valid = np.isfinite(purity)
            if np.sum(valid) > 5:
                ax.plot(r_centers[valid], purity[valid] * 100, '-', color=color,
                       lw=2, label=f'σ_v/v < {cut}')
        
        ax.set_ylabel('Purity [%]', fontsize=12)
        ax.set_title('Purity vs Galactocentric Distance\n(for different relative RV error cuts)',
                    fontsize=14)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        ax = axes[1]
        for cut, color in zip(cuts, colors):
            data = diagnostic_results['rel_rv_err_cuts'][f'{cut}']
            completeness = np.array(data['completeness'])
            valid = np.isfinite(completeness)
            if np.sum(valid) > 5:
                ax.plot(r_centers[valid], completeness[valid] * 100, '--', color=color,
                       lw=2, label=f'σ_v/v < {cut}')
        
        ax.set_xlabel('Galactocentric Distance R [kpc]', fontsize=12)
        ax.set_ylabel('Completeness [%]', fontsize=12)
        ax.set_title('Completeness vs Galactocentric Distance', fontsize=14)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'purity_completeness_vs_distance_rel_rv_err.png'))
        plt.close()
    
    def _plot_purity_completeness_vs_distance_ruwe(self, diagnostic_results: Dict):
        """Plot Purity & Completeness vs Distance for RUWE cuts."""
        if 'ruwe_cuts' not in diagnostic_results:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        r_centers = np.array(diagnostic_results['r_centers'])
        cuts = sorted([float(c) for c in diagnostic_results['ruwe_cuts'].keys()])
        colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(cuts)))
        
        ax = axes[0]
        for cut, color in zip(cuts, colors):
            data = diagnostic_results['ruwe_cuts'][f'{cut}']
            purity = np.array(data['purity'])
            valid = np.isfinite(purity)
            if np.sum(valid) > 5:
                ax.plot(r_centers[valid], purity[valid] * 100, '-', color=color,
                       lw=2, label=f'RUWE < {cut}')
        
        ax.set_ylabel('Purity [%]', fontsize=12)
        ax.set_title('Purity vs Galactocentric Distance\n(for different RUWE cuts)', fontsize=14)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        ax = axes[1]
        for cut, color in zip(cuts, colors):
            data = diagnostic_results['ruwe_cuts'][f'{cut}']
            completeness = np.array(data['completeness'])
            valid = np.isfinite(completeness)
            if np.sum(valid) > 5:
                ax.plot(r_centers[valid], completeness[valid] * 100, '--', color=color,
                       lw=2, label=f'RUWE < {cut}')
        
        ax.set_xlabel('Galactocentric Distance R [kpc]', fontsize=12)
        ax.set_ylabel('Completeness [%]', fontsize=12)
        ax.set_title('Completeness vs Galactocentric Distance', fontsize=14)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'purity_completeness_vs_distance_ruwe.png'))
        plt.close()
    
    def _plot_combined_summary(self, purity_results: Dict, completeness_results: Dict):
        """Plot combined summary."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('RIGOROUS COMPLETENESS & PURITY SUMMARY (v6)', fontsize=16, fontweight='bold')
        
        # Key metrics
        ax = axes[0, 0]
        
        comp = completeness_results.get('overall', {}).get('completeness', 0)
        purity = purity_results.get('combined', {}).get('mean_purity', 0)
        f1 = 2 * comp * purity / (comp + purity) if (comp + purity) > 0 else 0
        
        metrics = ['Completeness', 'Purity', 'F1 Score']
        values = [comp, purity, f1]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_title('Key Metrics')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center')
        
        # Model info
        ax = axes[0, 1]
        ax.axis('off')
        
        model_info = f"""
STATISTICAL METHODS USED
{'='*40}

Primary Method: {self.config.mixture_method.value.upper()}

RUWE: Regularized Gaussian Mixture
  - Best for approximately Gaussian data
  - Regularization prevents collapse

Distance/RV: {self.config.mixture_method.value}
  - Log-Normal: Proper support on (0,∞)
  - Gamma: For exponential-like tails
  - XD: Full error deconvolution

REGULARIZATION:
  - Beta prior on η: α={self.config.eta_prior_alpha}, β={self.config.eta_prior_beta}
  - η bounds: [{self.config.min_eta}, {self.config.max_eta}]
  - Prevents η → 1 collapse
"""
        
        ax.text(0.05, 0.95, model_info, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Purity breakdown
        ax = axes[1, 0]
        
        purity_types = ['RUWE', 'Distance', 'RV', 'Combined']
        purity_values = []
        eta_values = []
        
        for key in ['ruwe', 'distance_error', 'rv_error', 'combined']:
            if key in purity_results:
                purity_values.append(purity_results[key].get('mean_purity', 0))
                eta_values.append(purity_results[key].get('eta', 'N/A'))
            else:
                purity_values.append(0)
                eta_values.append('N/A')
        
        bars = ax.bar(purity_types, purity_values, 
                      color=['#9b59b6', '#f39c12', '#1abc9c', '#34495e'],
                      edgecolor='black', alpha=0.8)
        ax.set_ylabel('Mean Purity')
        ax.set_title('Purity by Error Type')
        ax.set_ylim(0, 1)
        
        for bar, p, eta in zip(bars, purity_values, eta_values):
            eta_str = f"η={eta:.2f}" if isinstance(eta, float) else ""
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{p:.3f}\n{eta_str}', ha='center', fontsize=9)
        
        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
ANALYSIS SUMMARY
{'='*40}

COMPLETENESS:
  GC:  {completeness_results.get('gc', {}).get('completeness', 0):.4f}
  OC:  {completeness_results.get('oc', {}).get('completeness', 0):.4f}
  SGR: {completeness_results.get('sgr', {}).get('completeness', 0):.4f}
  Overall: {comp:.4f}

PURITY (Rigorous Methods):
  RUWE:     {purity_results.get('ruwe', {}).get('mean_purity', 0):.4f}
  Distance: {purity_results.get('distance_error', {}).get('mean_purity', 0):.4f}
  RV:       {purity_results.get('rv_error', {}).get('mean_purity', 0):.4f}
  Combined: {purity:.4f}

COMBINED METRICS:
  F1 Score: {f1:.4f}
  Stars with P > 0.8: {purity_results.get('combined', {}).get('n_high_purity', 0):,}
  Stars with P > 0.95: {purity_results.get('combined', {}).get('n_very_high_purity', 0):,}

KEY IMPROVEMENT:
  η values now properly bounded
  Prevents meaningless P ≈ 1 for all stars
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'combined_summary.png'))
        plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_analysis(config: AnalysisConfig) -> Dict[str, Any]:
    """Run the complete rigorous analysis."""
    
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, 'analysis.log')
    logger = setup_logging(log_file)
    
    logger.info("=" * 70)
    logger.info("RIGOROUS COMPLETENESS & PURITY ANALYSIS v6")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Primary mixture method: {config.mixture_method.value}")
    logger.info(f"η bounds: [{config.min_eta}, {config.max_eta}]")
    logger.info(f"η prior: Beta({config.eta_prior_alpha}, {config.eta_prior_beta})")
    
    start_time = time.time()
    results = {}
    
    try:
        # Load catalog
        catalog = CatalogLoader(config, logger)
        if not catalog.load():
            logger.error("Failed to load catalog!")
            return results
        
        # Load member catalogs
        members = MemberCatalogLoader(config, logger)
        members.load_all()
        
        # Rigorous purity analysis
        purity_analyzer = RigorousPurityAnalyzer(config, logger, catalog)
        purity_results = purity_analyzer.run_analysis()
        results['purity'] = purity_results
        
        # Completeness analysis
        completeness_analyzer = CompletenessAnalyzer(config, logger, catalog, members)
        completeness_results = completeness_analyzer.run_analysis()
        results['completeness'] = completeness_results
        
        # Diagnostic analysis
        diagnostic_analyzer = DiagnosticAnalyzer(config, logger, catalog, members)
        diagnostic_results = diagnostic_analyzer.compute_purity_completeness_vs_distance()
        results['diagnostic'] = diagnostic_results
        
        # Precision-Recall curves
        pr_analyzer = PrecisionRecallAnalyzer(config, logger, catalog,
                                              purity_results, completeness_results)
        pr_results = pr_analyzer.generate_curves()
        results['precision_recall'] = pr_results
        
        # Generate plots
        plotter = RigorousPlotter(config, logger, config.output_dir)
        plotter.plot_all(catalog, purity_analyzer, purity_results, completeness_results,
                        pr_results, diagnostic_results)
        
        # Save results
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            elif isinstance(obj, MixtureMethod):
                return obj.value
            return obj
        
        results_file = os.path.join(config.output_dir, 'rigorous_analysis_results_v6.json')
        with open(results_file, 'w') as f:
            json.dump(convert_for_json(results), f, indent=2, default=str)
        
        # Final summary
        elapsed = time.time() - start_time
        comp = completeness_results.get('overall', {}).get('completeness', 0)
        purity = purity_results.get('combined', {}).get('mean_purity', 0)
        f1 = 2 * comp * purity / (comp + purity) if (comp + purity) > 0 else 0
        
        logger.info("\n" + "=" * 70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Time: {elapsed/60:.1f} min")
        logger.info(f"Method: {config.mixture_method.value}")
        logger.info(f"Completeness: {comp:.4f}")
        logger.info(f"Purity: {purity:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Results: {results_file}")
        
        # Highlight key improvements
        logger.info("\n" + "-" * 50)
        logger.info("KEY IMPROVEMENTS IN v6:")
        logger.info("-" * 50)
        logger.info(f"  - η (core fraction) properly bounded to [{config.min_eta}, {config.max_eta}]")
        logger.info(f"  - Log-normal mixture for positive quantities")
        logger.info(f"  - Extreme Deconvolution for error handling")
        logger.info(f"  - Gamma mixture alternative")
        logger.info(f"  - Regularized Gaussian with priors")
        logger.info(f"  - Method comparison included")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Rigorous Completeness & Purity Analysis v6')
    
    p.add_argument('--master', '-m', required=True, help='Master catalog path')
    p.add_argument('--gc', default=None, help='GC members CSV')
    p.add_argument('--oc', default=None, help='OC members CSV')
    p.add_argument('--sgr', default=None, help='SGR members CSV')
    p.add_argument('--output', '-o', default='./rigorous_comp_purity_results_v6')
    
    # Method selection
    p.add_argument('--method', choices=['lognormal', 'gamma', 'xd', 'reg_gaussian'],
                   default='lognormal', help='Mixture model method (default: lognormal)')
    
    # Regularization parameters
    p.add_argument('--eta-min', type=float, default=0.5,
                   help='Minimum core fraction (default: 0.5)')
    p.add_argument('--eta-max', type=float, default=0.95,
                   help='Maximum core fraction (default: 0.95)')
    p.add_argument('--eta-prior-alpha', type=float, default=5.0,
                   help='Beta prior alpha for eta (default: 5.0)')
    p.add_argument('--eta-prior-beta', type=float, default=2.0,
                   help='Beta prior beta for eta (default: 2.0)')
    
    # Binning
    p.add_argument('--n-radial-bins', type=int, default=100)
    p.add_argument('--rgal-max', type=float, default=300.0)
    
    # Processing limits
    p.add_argument('--max-chunks', type=int, default=None)
    p.add_argument('--max-rows-per-chunk', type=int, default=None)
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Map method string to enum
    method_map = {
        'lognormal': MixtureMethod.LOGNORMAL,
        'gamma': MixtureMethod.GAMMA,
        'xd': MixtureMethod.EXTREME_DECONV,
        'reg_gaussian': MixtureMethod.REGULARIZED_GAUSSIAN,
    }
    
    config = AnalysisConfig(
        master_catalog_dir=args.master,
        gc_members_file=args.gc,
        oc_members_file=args.oc,
        sgr_members_file=args.sgr,
        output_dir=args.output,
        mixture_method=method_map[args.method],
        min_eta=args.eta_min,
        max_eta=args.eta_max,
        eta_prior_alpha=args.eta_prior_alpha,
        eta_prior_beta=args.eta_prior_beta,
        n_radial_bins=args.n_radial_bins,
        rgal_max=args.rgal_max,
        max_chunks=args.max_chunks,
        max_rows_per_chunk=args.max_rows_per_chunk,
    )
    
    results = run_analysis(config)
    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())