#!/usr/bin/env python3
"""
================================================================================
STELLAR CATALOG DEDUPLICATION - MEMORY-OPTIMIZED VERSION
================================================================================
Optimized for: 52M rows × 214 columns on 64-core cluster with LIMITED MEMORY

Key Memory Optimizations:
    1. CHUNKED WRITING: Write each chunk immediately, don't accumulate results
    2. CHECKPOINTING: Save progress after each phase, resume if killed
    3. STREAMING AGGREGATION: Process groups in batches, free memory after each
    4. MINIMAL DICT OVERHEAD: Pre-allocate arrays where possible

Memory Usage Target: <400 GB (vs 700+ GB before)
================================================================================
"""

import os
import sys
import gc
import time
import json
import pickle
import logging
import warnings
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import cKDTree

from astropy.io import fits
from astropy.table import Table, Column

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False
    print("WARNING: healpy not available.")

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration for the deduplication pipeline."""
    
    input_fits: str = ""
    output_fits: str = ""
    temp_dir: str = "./temp_dedup"
    checkpoint_dir: str = "./checkpoints"
    
    ra_col: str = "RA_all"
    dec_col: str = "DEC_all"
    tolerance_arcsec: float = 1.0
    parallax_sigma_threshold: float = 3.0
    outlier_sigma_threshold: float = 3.0
    mad_sigma_threshold: float = 3.0
    
    parallax_columns: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("parallax_1", "parallax_error_1"),
        ("parallax_2", "parallax_error_2"),
        ("Plx", "e_Plx"),
        ("parallax", "parallax_error"),
    ])
    
    kinematic_groups: List[Tuple[List[str], List[str], str, Dict[str, float]]] = field(default_factory=list)
    string_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    array_columns: List[str] = field(default_factory=list)
    
    n_workers: int = 64
    chunk_size: int = 100_000
    healpix_nside: int = 32
    n_output_chunks: int = 8  # Number of output files
    
    def __post_init__(self):
        if not self.kinematic_groups:
            self.kinematic_groups = [
                (["parallax_1", "parallax_2", "Plx", "parallax"],
                 ["parallax_error_1", "parallax_error_2", "e_Plx", "parallax_error"],
                 "parallax", {}),
                (["pmra_1", "pmra_2", "pmra", "pmRA_x"],
                 ["pmra_error_1", "pmra_error_2", "pmra_error", "e_pmRA"],
                 "pmra", {}),
                (["pmdec_1", "pmdec_2", "pmdec", "pmDE"],
                 ["pmdec_error_1", "pmdec_error_2", "pmdec_error", "e_pmDE"],
                 "pmdec", {}),
                (["radial_velocity_1", "radial_velocity_2", "RV", "VRAD", "dr2_radial_velocity"],
                 ["radial_velocity_error_1", "radial_velocity_error_2", "e_RV", "VRAD_ERR", "dr2_radial_velocity_error"],
                 "RV", {}),
                (["Weighted_Avg"], ["e_weighted_avg"], "Weighted_Avg", {}),
                (["ZP"], ["e_ZP"], "ZP", {}),
                (["DIST", "Dist_x"], ["DISTERR", ""], "distance", {"Dist_x": 0.001}),
            ]
        
        if not self.string_columns:
            self.string_columns = ["Survey", "Code", "VarFlag", "Lib", "XYZ", "DR3Name"]
        
        if not self.id_columns:
            self.id_columns = ["gdr3_source_id", "source_id", "source_id_1", "Source", "SolID",
                              "panstarrs1", "sdssdr13", "skymapper2", "urat1"]
        
        if not self.array_columns:
            self.array_columns = ["stellar_params_est", "stellar_params_err"]
        
        self.skip_passthrough = set()
        self.skip_passthrough.add(self.ra_col)
        self.skip_passthrough.add(self.dec_col)
        self.skip_passthrough.update(self.string_columns)
        self.skip_passthrough.update(self.id_columns)
        self.skip_passthrough.update(self.array_columns)
        
        for val_cols, err_cols, _, _ in self.kinematic_groups:
            self.skip_passthrough.update(val_cols)
            for ec in err_cols:
                if ec:
                    self.skip_passthrough.add(ec)


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_file: str = "deduplication.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# =============================================================================
# MEMORY MONITORING
# =============================================================================

def get_memory_usage_gb():
    """Get current memory usage in GB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    except:
        return -1


def log_memory(logger, msg=""):
    """Log current memory usage."""
    mem = get_memory_usage_gb()
    if mem > 0:
        logger.info(f"  [MEMORY] {msg}: {mem:.1f} GB")


# =============================================================================
# UNION-FIND
# =============================================================================

class UnionFind:
    __slots__ = ['parent', 'rank', 'n']
    
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int32)
        self.n = n
    
    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def get_groups(self) -> Dict[int, List[int]]:
        groups = defaultdict(list)
        for i in range(self.n):
            groups[self.find(i)].append(i)
        return dict(groups)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_healpix_pixels(ra, dec, nside=32):
    if HEALPY_AVAILABLE:
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        return hp.ang2pix(nside, theta, phi, nest=True)
    else:
        ra_bin = np.floor(ra / 1.0).astype(np.int32) % 360
        dec_bin = np.floor((dec + 90) / 1.0).astype(np.int32) % 180
        return ra_bin * 180 + dec_bin


def find_pairs_in_region(indices, ra, dec, tolerance_arcsec):
    if len(indices) < 2:
        return []
    
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    tree = cKDTree(np.column_stack([x, y, z]))
    tolerance_cart = 2 * np.sin(tolerance_arcsec / 3600.0 * np.pi / 180.0 / 2)
    
    pairs_local = tree.query_pairs(r=tolerance_cart, output_type='ndarray')
    return [(indices[i], indices[j]) for i, j in pairs_local]


def parallax_consistent(plx1, plx2, sigma=3.0):
    if len(plx1) == 0 or len(plx2) == 0:
        return True
    for v1, e1 in plx1:
        for v2, e2 in plx2:
            combined_err = np.sqrt(e1**2 + e2**2)
            if combined_err > 0 and abs(v1 - v2) <= sigma * combined_err:
                return True
            max_val = max(abs(v1), abs(v2))
            if max_val > 0 and abs(v1 - v2) / max_val < 0.2:
                return True
    return False


def weighted_avg_with_outliers(values, errors, sigma=3.0):
    mask = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    vals, errs = values[mask], errors[mask]
    
    if len(vals) == 0:
        val_only = values[np.isfinite(values)]
        if len(val_only) > 0:
            return np.nanmean(val_only), np.nan, []
        return np.nan, np.nan, []
    
    if len(vals) == 1:
        return vals[0], errs[0], []
    
    weights = 1.0 / errs**2
    wmean = np.sum(vals * weights) / np.sum(weights)
    
    is_outlier = np.abs(vals - wmean) > sigma * errs
    
    if np.all(is_outlier):
        return np.mean(vals), np.std(vals), []
    
    inlier_v, inlier_e = vals[~is_outlier], errs[~is_outlier]
    outlier_v = vals[is_outlier].tolist()
    
    w = 1.0 / inlier_e**2
    return np.sum(inlier_v * w) / np.sum(w), 1.0 / np.sqrt(np.sum(w)), outlier_v


def mad_avg_with_outliers(values, sigma=3.0):
    vals = values[np.isfinite(values)]
    
    if len(vals) == 0:
        return np.nan, []
    if len(vals) == 1:
        return vals[0], []
    
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    
    if mad == 0:
        return np.mean(vals), []
    
    threshold = sigma * 1.4826 * mad
    is_outlier = np.abs(vals - median) > threshold
    
    if np.all(is_outlier):
        return np.mean(vals), []
    
    return np.mean(vals[~is_outlier]), vals[is_outlier].tolist()


def process_region_worker(args):
    (region_id, indices, ra, dec, plx_data, plx_cols, col_names, tol, sigma) = args
    
    pairs = find_pairs_in_region(indices, ra, dec, tol)
    if len(pairs) == 0:
        return region_id, []
    
    valid_pairs = []
    for i, j in pairs:
        local_i = np.where(indices == i)[0][0]
        local_j = np.where(indices == j)[0][0]
        
        plx_i, plx_j = [], []
        
        for val_col, err_col in plx_cols:
            if val_col not in col_names:
                continue
            val_idx = col_names.index(val_col)
            err_idx = col_names.index(err_col) if err_col in col_names else -1
            
            vi = plx_data[local_i, val_idx] if val_idx < plx_data.shape[1] else np.nan
            vj = plx_data[local_j, val_idx] if val_idx < plx_data.shape[1] else np.nan
            
            if not np.isnan(vi):
                ei = plx_data[local_i, err_idx] if err_idx >= 0 else abs(vi) * 0.1
                if np.isnan(ei) or ei <= 0:
                    ei = abs(vi) * 0.1
                plx_i.append((vi, ei))
            
            if not np.isnan(vj):
                ej = plx_data[local_j, err_idx] if err_idx >= 0 else abs(vj) * 0.1
                if np.isnan(ej) or ej <= 0:
                    ej = abs(vj) * 0.1
                plx_j.append((vj, ej))
        
        if parallax_consistent(plx_i, plx_j, sigma):
            valid_pairs.append((i, j))
    
    return region_id, valid_pairs


# =============================================================================
# MAIN PIPELINE - MEMORY OPTIMIZED
# =============================================================================

class StellarDeduplicator:
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()
        os.makedirs(config.temp_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        self.n_rows = 0
        self.col_names = []
        self.hdu = None
        self.passthrough_numeric_cols = []
    
    def run(self):
        start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("STELLAR DEDUPLICATION - MEMORY OPTIMIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"Input: {self.config.input_fits}")
        self.logger.info(f"Output: {self.config.output_fits}")
        log_memory(self.logger, "Initial")
        
        # Check for checkpoint
        checkpoint_file = Path(self.config.checkpoint_dir) / "groups.pkl"
        
        if checkpoint_file.exists():
            self.logger.info("\n*** RESUMING FROM CHECKPOINT ***")
            groups = self._load_checkpoint()
        else:
            self._load_metadata()
            pairs = self._find_all_pairs()
            groups = self._build_groups(pairs)
            self._save_checkpoint(groups)
            
            # Free pair memory
            del pairs
            gc.collect()
        
        log_memory(self.logger, "After grouping")
        
        # Aggregation with chunked writing
        self._aggregate_and_write_chunked(groups)
        
        elapsed = time.time() - start_time
        self.logger.info(f"\nCOMPLETE! Time: {elapsed/3600:.2f} hours")
    
    def _save_checkpoint(self, groups):
        """Save groups to checkpoint file."""
        checkpoint_file = Path(self.config.checkpoint_dir) / "groups.pkl"
        self.logger.info(f"Saving checkpoint to {checkpoint_file}...")
        
        # Save as list of tuples (more compact)
        groups_list = [(root, members) for root, members in groups.items()]
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'groups': groups_list,
                'n_rows': self.n_rows,
                'col_names': self.col_names,
                'input_fits': self.config.input_fits,
            }, f)
        
        self.logger.info(f"Checkpoint saved: {len(groups_list):,} groups")
    
    def _load_checkpoint(self):
        """Load groups from checkpoint file."""
        checkpoint_file = Path(self.config.checkpoint_dir) / "groups.pkl"
        
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        
        self.n_rows = data['n_rows']
        self.col_names = data['col_names']
        
        # Convert back to dict
        groups = {root: members for root, members in data['groups']}
        
        self.logger.info(f"Loaded checkpoint: {len(groups):,} groups, {self.n_rows:,} rows")
        return groups
    
    def _load_metadata(self):
        self.logger.info("\n[PHASE 1] Loading metadata...")
        
        self.hdu = fits.open(self.config.input_fits, memmap=True)
        data = self.hdu[1].data
        
        self.n_rows = len(data)
        self.col_names = [col.name for col in self.hdu[1].columns]
        
        self.logger.info(f"Rows: {self.n_rows:,}, Columns: {len(self.col_names)}")
        
        self.ra = np.array(data[self.config.ra_col], dtype=np.float64)
        self.dec = np.array(data[self.config.dec_col], dtype=np.float64)
        
        self.pixels = get_healpix_pixels(self.ra, self.dec, self.config.healpix_nside)
        self.logger.info(f"HEALPix regions: {len(np.unique(self.pixels)):,}")
        
        log_memory(self.logger, "After loading coords")
    
    def _find_all_pairs(self):
        self.logger.info("\n[PHASE 2] Finding pairs...")
        
        pixel_to_idx = defaultdict(list)
        for i, pix in enumerate(self.pixels):
            pixel_to_idx[pix].append(i)
        
        unique_pixels = list(pixel_to_idx.keys())
        self.logger.info(f"Processing {len(unique_pixels):,} regions...")
        
        plx_col_names = []
        for vc, ec in self.config.parallax_columns:
            if vc in self.col_names:
                plx_col_names.append(vc)
            if ec and ec in self.col_names:
                plx_col_names.append(ec)
        
        all_pairs = []
        batch_size = 500
        
        for batch_start in range(0, len(unique_pixels), batch_size):
            batch_pixels = unique_pixels[batch_start:batch_start + batch_size]
            
            worker_args = []
            for pix in batch_pixels:
                indices = np.array(pixel_to_idx[pix], dtype=np.int64)
                if len(indices) < 2:
                    continue
                
                ra_reg = self.ra[indices]
                dec_reg = self.dec[indices]
                
                if plx_col_names:
                    plx_data = np.column_stack([
                        self.hdu[1].data[c][indices] if c in self.col_names
                        else np.full(len(indices), np.nan)
                        for c in plx_col_names
                    ])
                else:
                    plx_data = np.empty((len(indices), 0))
                
                worker_args.append((
                    pix, indices, ra_reg, dec_reg, plx_data,
                    self.config.parallax_columns, plx_col_names,
                    self.config.tolerance_arcsec, self.config.parallax_sigma_threshold
                ))
            
            if worker_args:
                with ProcessPoolExecutor(max_workers=self.config.n_workers) as ex:
                    futures = [ex.submit(process_region_worker, a) for a in worker_args]
                    for f in as_completed(futures):
                        _, pairs = f.result()
                        all_pairs.extend(pairs)
            
            n_done = min(batch_start + batch_size, len(unique_pixels))
            if n_done % 2000 == 0 or n_done == len(unique_pixels):
                self.logger.info(f"  {n_done:,}/{len(unique_pixels):,} regions, {len(all_pairs):,} pairs")
        
        # Free memory
        del self.ra, self.dec, self.pixels
        gc.collect()
        
        self.logger.info(f"Total pairs: {len(all_pairs):,}")
        log_memory(self.logger, "After finding pairs")
        return all_pairs
    
    def _build_groups(self, pairs):
        self.logger.info("\n[PHASE 3] Building groups...")
        
        uf = UnionFind(self.n_rows)
        for i, j in pairs:
            uf.union(i, j)
        
        groups = uf.get_groups()
        
        sizes = [len(g) for g in groups.values()]
        n_unique = len(groups)
        n_dup = self.n_rows - n_unique
        
        self.logger.info(f"Unique stars: {n_unique:,}")
        self.logger.info(f"Duplicates merged: {n_dup:,} ({100*n_dup/self.n_rows:.2f}%)")
        self.logger.info(f"Largest group: {max(sizes)}")
        
        # Free UF memory
        del uf
        gc.collect()
        
        return groups
    
    def _aggregate_and_write_chunked(self, groups):
        """
        MEMORY-OPTIMIZED: Process and write in chunks without accumulating all results.
        """
        self.logger.info("\n[PHASE 4] Aggregating (memory-optimized chunked writing)...")
        log_memory(self.logger, "Start aggregation")
        
        # Open FITS file
        self.hdu = fits.open(self.config.input_fits, memmap=True)
        
        # Separate groups
        all_groups = [(root, members) for root, members in groups.items()]
        total_groups = len(all_groups)
        
        # Free groups dict memory
        del groups
        gc.collect()
        
        self.logger.info(f"Total unique stars: {total_groups:,}")
        
        # Identify pass-through numeric columns
        self._identify_passthrough_cols()
        
        # Calculate chunk boundaries
        n_chunks = self.config.n_output_chunks
        chunk_size = (total_groups + n_chunks - 1) // n_chunks
        
        self.logger.info(f"Writing {n_chunks} chunks, ~{chunk_size:,} stars each")
        
        # Get output path info
        base_path = Path(self.config.output_fits)
        base_name = base_path.stem
        base_dir = base_path.parent
        base_ext = base_path.suffix or '.fits'
        
        # Check which chunks already exist (for resume)
        completed_chunks = set()
        for i in range(1, n_chunks + 1):
            chunk_file = base_dir / f"{base_name}_chunk{i:02d}{base_ext}"
            if chunk_file.exists():
                completed_chunks.add(i)
                self.logger.info(f"  Chunk {i} already exists, skipping...")
        
        # Process each chunk
        for chunk_idx in range(n_chunks):
            chunk_num = chunk_idx + 1
            
            if chunk_num in completed_chunks:
                continue
            
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, total_groups)
            
            if chunk_start >= total_groups:
                break
            
            self.logger.info(f"\n--- Processing Chunk {chunk_num}/{n_chunks} ---")
            self.logger.info(f"  Groups {chunk_start:,} to {chunk_end:,}")
            log_memory(self.logger, f"Chunk {chunk_num} start")
            
            # Process this chunk's groups
            chunk_results = self._process_chunk(all_groups[chunk_start:chunk_end])
            
            log_memory(self.logger, f"Chunk {chunk_num} processed")
            
            # Write this chunk
            chunk_file = base_dir / f"{base_name}_chunk{chunk_num:02d}{base_ext}"
            self._write_chunk(chunk_results, str(chunk_file), chunk_num)
            
            # FREE MEMORY IMMEDIATELY
            del chunk_results
            gc.collect()
            
            log_memory(self.logger, f"Chunk {chunk_num} done")
        
        self.hdu.close()
        self.logger.info(f"\nAll chunks written to {base_dir}/{base_name}_chunk*{base_ext}")
    
    def _identify_passthrough_cols(self):
        """Identify numeric pass-through columns."""
        self.passthrough_numeric_cols = []
        for col in self.col_names:
            if col in self.config.skip_passthrough:
                continue
            try:
                sample = self.hdu[1].data[col][0]
                if hasattr(sample, '__len__') and not isinstance(sample, str):
                    continue
                float(sample)
                self.passthrough_numeric_cols.append(col)
            except:
                pass
        self.logger.info(f"Pass-through numeric columns: {len(self.passthrough_numeric_cols)}")
    
    def _process_chunk(self, chunk_groups):
        """Process a chunk of groups into results."""
        results = []
        
        n_groups = len(chunk_groups)
        for i, (root, members) in enumerate(chunk_groups):
            if i % 200000 == 0 and i > 0:
                self.logger.info(f"    Processed {i:,}/{n_groups:,}")
                gc.collect()  # Periodic GC
            
            if len(members) == 1:
                results.append(self._process_single(members[0]))
            else:
                results.append(self._process_multi(members))
        
        self.logger.info(f"    Processed {n_groups:,}/{n_groups:,}")
        return results
    
    def _get_float(self, col, idx):
        try:
            return float(self.hdu[1].data[col][idx])
        except:
            return np.nan
    
    def _format_id(self, val):
        if val is None:
            return ''
        try:
            f = float(val)
            return '' if np.isnan(f) else str(int(f))
        except:
            s = str(val).strip()
            return s if s.lower() != 'nan' else ''
    
    def _process_single(self, idx):
        """Process single-measurement star (minimal memory)."""
        result = {
            'n_measurements': 1,
            'original_indices': str(idx),
            'RA_final': self._get_float(self.config.ra_col, idx),
            'DEC_final': self._get_float(self.config.dec_col, idx),
        }
        
        # Kinematic groups
        for val_cols, err_cols, name, conv in self.config.kinematic_groups:
            val, err = np.nan, np.nan
            for vc, ec in zip(val_cols, err_cols):
                if vc in self.col_names:
                    v = self._get_float(vc, idx)
                    if not np.isnan(v):
                        if vc in conv:
                            v *= conv[vc]
                        val = v
                        if ec and ec in self.col_names:
                            e = self._get_float(ec, idx)
                            if vc in conv and not np.isnan(e):
                                e *= conv[vc]
                            err = e
                        break
            result[f'{name}_final'] = val
            result[f'{name}_err_final'] = err
            result[f'{name}_outliers'] = ''
        
        # String columns
        for col in self.config.string_columns:
            if col in self.col_names:
                v = self.hdu[1].data[col][idx]
                result[f'{col}_combined'] = str(v).strip() if v and str(v).lower() != 'nan' else ''
            else:
                result[f'{col}_combined'] = ''
        
        # IDs
        ids = []
        for col in self.config.id_columns:
            if col in self.col_names:
                s = self._format_id(self.hdu[1].data[col][idx])
                if s:
                    ids.append(f"{col}:{s}")
        result['all_survey_ids'] = '; '.join(ids)
        
        # Array columns
        for col in self.config.array_columns:
            if col in self.col_names:
                try:
                    result[col] = np.array(self.hdu[1].data[col][idx])
                except:
                    pass
        
        # Pass-through numeric
        for col in self.passthrough_numeric_cols:
            result[col] = self._get_float(col, idx)
            result[f'{col}_outliers'] = ''
        
        return result
    
    def _process_multi(self, members):
        """Process multi-measurement star."""
        result = {
            'n_measurements': len(members),
            'original_indices': ','.join(map(str, members)),
        }
        
        # Position
        ra_vals = [self._get_float(self.config.ra_col, i) for i in members]
        dec_vals = [self._get_float(self.config.dec_col, i) for i in members]
        result['RA_final'] = np.nanmean(ra_vals)
        result['DEC_final'] = np.nanmean(dec_vals)
        
        # Kinematic groups
        for val_cols, err_cols, name, conv in self.config.kinematic_groups:
            all_vals, all_errs = [], []
            
            for vc, ec in zip(val_cols, err_cols):
                if vc not in self.col_names:
                    continue
                for i in members:
                    v = self._get_float(vc, i)
                    if np.isnan(v):
                        continue
                    if vc in conv:
                        v *= conv[vc]
                    
                    e = np.nan
                    if ec and ec in self.col_names:
                        e = self._get_float(ec, i)
                        if vc in conv and not np.isnan(e):
                            e *= conv[vc]
                    
                    all_vals.append(v)
                    all_errs.append(e)
            
            if all_vals:
                avg, err, outliers = weighted_avg_with_outliers(
                    np.array(all_vals), np.array(all_errs), self.config.outlier_sigma_threshold
                )
                result[f'{name}_final'] = avg
                result[f'{name}_err_final'] = err
                result[f'{name}_outliers'] = ','.join(f'{x:.6g}' for x in outliers) if outliers else ''
            else:
                result[f'{name}_final'] = np.nan
                result[f'{name}_err_final'] = np.nan
                result[f'{name}_outliers'] = ''
        
        # String columns
        for col in self.config.string_columns:
            if col in self.col_names:
                uniq = set()
                for i in members:
                    v = self.hdu[1].data[col][i]
                    if v and str(v).strip().lower() != 'nan':
                        uniq.add(str(v).strip())
                result[f'{col}_combined'] = ','.join(sorted(uniq))
            else:
                result[f'{col}_combined'] = ''
        
        # IDs
        ids = []
        for col in self.config.id_columns:
            if col in self.col_names:
                uniq = set()
                for i in members:
                    s = self._format_id(self.hdu[1].data[col][i])
                    if s:
                        uniq.add(s)
                for u in sorted(uniq):
                    ids.append(f"{col}:{u}")
        result['all_survey_ids'] = '; '.join(ids)
        
        # Array columns
        for col in self.config.array_columns:
            if col in self.col_names:
                for i in members:
                    try:
                        arr = self.hdu[1].data[col][i]
                        if arr is not None and not np.all(np.isnan(arr)):
                            result[col] = np.array(arr)
                            break
                    except:
                        pass
        
        # Pass-through numeric
        for col in self.passthrough_numeric_cols:
            vals = np.array([self._get_float(col, i) for i in members])
            avg, outliers = mad_avg_with_outliers(vals, self.config.mad_sigma_threshold)
            result[col] = avg
            result[f'{col}_outliers'] = ','.join(f'{x:.6g}' for x in outliers) if outliers else ''
        
        return result
    
    def _write_chunk(self, results, filename, chunk_num):
        """Write a chunk to FITS file."""
        self.logger.info(f"  Writing chunk {chunk_num}: {len(results):,} rows -> {filename}")
        
        # Collect column names
        all_cols = set()
        for r in results[:1000]:
            all_cols.update(r.keys())
        
        # Build column arrays
        col_arrays = {c: [] for c in all_cols}
        
        for r in results:
            for c in all_cols:
                if c.endswith('_combined') or c.endswith('_outliers') or c == 'all_survey_ids' or c == 'original_indices':
                    col_arrays[c].append(r.get(c, ''))
                else:
                    col_arrays[c].append(r.get(c, np.nan))
        
        # Create table
        output = Table()
        
        for col, data in col_arrays.items():
            if col.endswith('_combined') or col.endswith('_outliers') or col == 'all_survey_ids' or col == 'original_indices':
                output[col] = Column(data, dtype=str)
            else:
                sample = None
                for v in data[:100]:
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        sample = v
                        break
                
                if sample is not None and hasattr(sample, '__len__') and not isinstance(sample, str):
                    try:
                        output[col] = Column(np.array(data))
                    except:
                        pass
                else:
                    try:
                        output[col] = Column(np.array(data, dtype=np.float64))
                    except:
                        try:
                            output[col] = Column(data, dtype=str)
                        except:
                            pass
        
        output.write(filename, format='fits', overwrite=True)
        self.logger.info(f"  Chunk {chunk_num} written: {len(output.colnames)} columns")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-optimized stellar deduplication')
    parser.add_argument('input_fits', help='Input FITS file')
    parser.add_argument('output_fits', help='Output FITS file (base name for chunks)')
    parser.add_argument('--workers', '-w', type=int, default=64)
    parser.add_argument('--tolerance', '-t', type=float, default=1.0)
    parser.add_argument('--sigma', '-s', type=float, default=3.0)
    parser.add_argument('--chunk-size', '-c', type=int, default=100000)
    parser.add_argument('--temp-dir', type=str, default='./temp_dedup')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--ra-col', type=str, default='RA_all')
    parser.add_argument('--dec-col', type=str, default='DEC_all')
    parser.add_argument('--n-chunks', type=int, default=8, help='Number of output files')
    
    args = parser.parse_args()
    
    config = Config(
        input_fits=args.input_fits,
        output_fits=args.output_fits,
        temp_dir=args.temp_dir,
        checkpoint_dir=args.checkpoint_dir,
        ra_col=args.ra_col,
        dec_col=args.dec_col,
        tolerance_arcsec=args.tolerance,
        outlier_sigma_threshold=args.sigma,
        mad_sigma_threshold=args.sigma,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        n_output_chunks=args.n_chunks,
    )
    
    StellarDeduplicator(config).run()


if __name__ == '__main__':
    main()