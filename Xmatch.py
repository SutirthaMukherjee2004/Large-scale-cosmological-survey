#!/usr/bin/env python3
"""
=============================================================================
FITS Cross-Match Script: Memory-Efficient Spatial Cross-Matching
=============================================================================

Purpose:
    Cross-match two large FITS tables based on sky coordinates (RA, Dec)
    with 1 arcsec tolerance, keeping closest matches. Performs LEFT JOIN.

Files:
    - Base table: RVS+DIST_42M (~42 million rows)
    - Lookup table: all_stellar_est (stellar parameters)

Matching:
    - 42M columns: ra_1_1, dec_1_1
    - all_stellar_est columns: ra, dec
    - Tolerance: 1 arcsec
    - Strategy: Closest match wins

Output:
    - All 42M rows with matched stellar_params_est, stellar_params_err
    - Unmatched 42M rows get NaN for stellar params

Optimizations:
    - Chunk-wise processing of base table
    - KD-tree spatial indexing (built once for lookup table)
    - Multiprocessing for parallel chunk processing
    - Memory-mapped FITS reading where possible
    - Garbage collection between chunks

Author: Generated for Sutirtha's stellar kinematics research
Usage:
    python crossmatch_stellar.py --base <42M.fits> --lookup <all_stellar_est.fits> --output <output.csv>
    
For cluster submission (SLURM example):
    sbatch submit_crossmatch.sh

=============================================================================
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial import cKDTree
import multiprocessing as mp
from multiprocessing import shared_memory
import argparse
import os
import sys
import gc
import time
import logging
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Cross-match parameters
MATCH_TOLERANCE_ARCSEC = 1.0  # 1 arcsec tolerance
MATCH_TOLERANCE_DEG = MATCH_TOLERANCE_ARCSEC / 3600.0

# Processing parameters
DEFAULT_CHUNK_SIZE = 500_000  # Rows per chunk (adjust based on available RAM)
DEFAULT_N_WORKERS = None  # None = use all available cores

# Column names - BASE TABLE (RVS+DIST_42M)
BASE_RA_COL = 'ra_1_1'
BASE_DEC_COL = 'dec_1_1'

# Column names - LOOKUP TABLE (all_stellar_est)
LOOKUP_RA_COL = 'ra'
LOOKUP_DEC_COL = 'dec'

# Columns to extract from lookup table
LOOKUP_EXTRACT_COLS = ['stellar_params_est', 'stellar_params_err', 'ra', 'dec']

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('crossmatch.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_memory_usage_gb() -> float:
    """Get current memory usage in GB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        return -1.0

def ra_dec_to_cartesian(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """
    Convert RA, Dec (in degrees) to unit Cartesian coordinates.
    This is essential for proper 3D KD-tree spatial queries.
    
    Parameters:
        ra: Right Ascension in degrees
        dec: Declination in degrees
    
    Returns:
        Array of shape (N, 3) with x, y, z unit vectors
    """
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    
    cos_dec = np.cos(dec_rad)
    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z = np.sin(dec_rad)
    
    return np.column_stack([x, y, z])

def angular_tolerance_to_cartesian(tolerance_deg: float) -> float:
    """
    Convert angular tolerance to Cartesian distance for KD-tree query.
    Uses chord length approximation: d = 2 * sin(theta/2)
    
    Parameters:
        tolerance_deg: Angular tolerance in degrees
    
    Returns:
        Cartesian distance threshold
    """
    tolerance_rad = np.deg2rad(tolerance_deg)
    return 2.0 * np.sin(tolerance_rad / 2.0)

def load_fits_columns(filepath: str, columns: Optional[List[str]] = None, 
                      ext: int = 1) -> pd.DataFrame:
    """
    Load specific columns from a FITS file efficiently.
    
    Parameters:
        filepath: Path to FITS file
        columns: List of column names to load (None = all columns)
        ext: FITS extension number
    
    Returns:
        DataFrame with requested columns
    """
    logger.info(f"Loading FITS file: {filepath}")
    
    with fits.open(filepath, memmap=True) as hdul:
        data = hdul[ext].data
        
        if columns is None:
            columns = data.dtype.names
        
        # Filter to only existing columns
        available_cols = set(data.dtype.names)
        valid_cols = [c for c in columns if c in available_cols]
        missing_cols = [c for c in columns if c not in available_cols]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Build dictionary of arrays
        result = {}
        for col in valid_cols:
            arr = data[col]
            # Handle variable-length arrays (object dtype)
            if arr.dtype == object or len(arr.shape) > 1:
                result[col] = list(arr)
            else:
                result[col] = np.array(arr)
        
        df = pd.DataFrame(result)
        logger.info(f"Loaded {len(df):,} rows, {len(valid_cols)} columns")
        
    return df

def load_fits_chunk(filepath: str, start_row: int, end_row: int,
                    columns: Optional[List[str]] = None, ext: int = 1) -> pd.DataFrame:
    """
    Load a chunk of rows from a FITS file.
    
    Parameters:
        filepath: Path to FITS file
        start_row: Starting row index
        end_row: Ending row index (exclusive)
        columns: List of column names to load
        ext: FITS extension number
    
    Returns:
        DataFrame with the chunk of data
    """
    with fits.open(filepath, memmap=True) as hdul:
        data = hdul[ext].data
        
        if columns is None:
            columns = list(data.dtype.names)
        
        available_cols = set(data.dtype.names)
        valid_cols = [c for c in columns if c in available_cols]
        
        result = {}
        for col in valid_cols:
            arr = data[col][start_row:end_row]
            if arr.dtype == object or len(arr.shape) > 1:
                result[col] = list(arr)
            else:
                result[col] = np.array(arr)
        
        return pd.DataFrame(result)

def get_fits_nrows(filepath: str, ext: int = 1) -> int:
    """Get the number of rows in a FITS table."""
    with fits.open(filepath, memmap=True) as hdul:
        return hdul[ext].data.shape[0]

def get_fits_columns(filepath: str, ext: int = 1) -> List[str]:
    """Get column names from a FITS table."""
    with fits.open(filepath, memmap=True) as hdul:
        return list(hdul[ext].data.dtype.names)

# =============================================================================
# KD-TREE BUILDER
# =============================================================================

class SpatialIndex:
    """
    Spatial index using KD-tree for efficient sky coordinate matching.
    Built on the lookup table for memory efficiency.
    """
    
    def __init__(self, ra: np.ndarray, dec: np.ndarray, 
                 tolerance_deg: float = MATCH_TOLERANCE_DEG):
        """
        Build KD-tree from coordinates.
        
        Parameters:
            ra: Right Ascension array in degrees
            dec: Declination array in degrees
            tolerance_deg: Match tolerance in degrees
        """
        self.n_points = len(ra)
        self.tolerance_deg = tolerance_deg
        self.cartesian_tolerance = angular_tolerance_to_cartesian(tolerance_deg)
        
        logger.info(f"Building KD-tree for {self.n_points:,} points...")
        logger.info(f"Tolerance: {tolerance_deg * 3600:.2f} arcsec = {self.cartesian_tolerance:.6f} cartesian units")
        
        start_time = time.time()
        
        # Convert to Cartesian
        self.coords_cartesian = ra_dec_to_cartesian(ra, dec)
        
        # Build KD-tree with leaf_size optimization for large datasets
        leaf_size = max(10, min(50, self.n_points // 10000))
        self.tree = cKDTree(self.coords_cartesian, leafsize=leaf_size)
        
        elapsed = time.time() - start_time
        logger.info(f"KD-tree built in {elapsed:.2f} seconds")
    
    def query_closest(self, ra: np.ndarray, dec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find closest match for each query point within tolerance.
        
        Parameters:
            ra: Query RA array in degrees
            dec: Query Dec array in degrees
        
        Returns:
            indices: Index of closest match (-1 if no match within tolerance)
            distances: Angular distance in arcsec (-1 if no match)
        """
        query_cartesian = ra_dec_to_cartesian(ra, dec)
        
        # Query KD-tree for closest point
        cart_distances, indices = self.tree.query(
            query_cartesian, 
            k=1, 
            distance_upper_bound=self.cartesian_tolerance,
            workers=-1  # Use all cores
        )
        
        # Convert Cartesian distance back to angular distance (arcsec)
        # d = 2 * sin(theta/2) => theta = 2 * arcsin(d/2)
        valid_mask = np.isfinite(cart_distances) & (indices < self.n_points)
        
        angular_distances = np.full(len(ra), -1.0)
        angular_distances[valid_mask] = np.rad2deg(
            2.0 * np.arcsin(cart_distances[valid_mask] / 2.0)
        ) * 3600  # Convert to arcsec
        
        # Set invalid indices to -1
        result_indices = np.where(valid_mask, indices, -1)
        
        return result_indices, angular_distances

# =============================================================================
# CHUNK PROCESSOR (for multiprocessing)
# =============================================================================

def process_chunk_worker(args: Tuple) -> Dict:
    """
    Worker function to process a single chunk.
    
    Parameters:
        args: Tuple of (chunk_id, base_filepath, start_row, end_row, 
                       lookup_coords_shm_name, lookup_shape, tolerance_deg,
                       base_columns)
    
    Returns:
        Dictionary with matched results for this chunk
    """
    (chunk_id, base_filepath, start_row, end_row, 
     lookup_ra_shm_name, lookup_dec_shm_name, lookup_shape,
     tolerance_deg, base_columns) = args
    
    try:
        # Access shared memory for lookup coordinates
        shm_ra = shared_memory.SharedMemory(name=lookup_ra_shm_name)
        shm_dec = shared_memory.SharedMemory(name=lookup_dec_shm_name)
        
        lookup_ra = np.ndarray(lookup_shape, dtype=np.float64, buffer=shm_ra.buf)
        lookup_dec = np.ndarray(lookup_shape, dtype=np.float64, buffer=shm_dec.buf)
        
        # Build local KD-tree (each worker builds its own to avoid pickle issues)
        spatial_index = SpatialIndex(lookup_ra, lookup_dec, tolerance_deg)
        
        # Load chunk from base table
        chunk_df = load_fits_chunk(base_filepath, start_row, end_row, base_columns)
        
        # Get coordinates for matching
        chunk_ra = chunk_df[BASE_RA_COL].values.astype(np.float64)
        chunk_dec = chunk_df[BASE_DEC_COL].values.astype(np.float64)
        
        # Perform matching
        match_indices, match_distances = spatial_index.query_closest(chunk_ra, chunk_dec)
        
        # Clean up shared memory reference (don't unlink, just close)
        shm_ra.close()
        shm_dec.close()
        
        # Return results
        return {
            'chunk_id': chunk_id,
            'start_row': start_row,
            'end_row': end_row,
            'match_indices': match_indices,
            'match_distances': match_distances,
            'n_matched': np.sum(match_indices >= 0),
            'n_total': len(chunk_df)
        }
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {e}")
        return {
            'chunk_id': chunk_id,
            'error': str(e)
        }

def process_chunk_serial(chunk_id: int, base_filepath: str, start_row: int, 
                         end_row: int, spatial_index: SpatialIndex,
                         base_columns: List[str]) -> Dict:
    """
    Process a single chunk (serial version, more memory efficient).
    
    Parameters:
        chunk_id: Chunk identifier
        base_filepath: Path to base FITS file
        start_row: Starting row
        end_row: Ending row
        spatial_index: Pre-built spatial index
        base_columns: Columns to load from base file
    
    Returns:
        Dictionary with match results
    """
    # Load chunk
    chunk_df = load_fits_chunk(base_filepath, start_row, end_row, base_columns)
    
    # Get coordinates
    chunk_ra = chunk_df[BASE_RA_COL].values.astype(np.float64)
    chunk_dec = chunk_df[BASE_DEC_COL].values.astype(np.float64)
    
    # Match
    match_indices, match_distances = spatial_index.query_closest(chunk_ra, chunk_dec)
    
    return {
        'chunk_id': chunk_id,
        'start_row': start_row,
        'end_row': end_row,
        'match_indices': match_indices,
        'match_distances': match_distances,
        'n_matched': np.sum(match_indices >= 0),
        'n_total': len(chunk_df)
    }

# =============================================================================
# MAIN CROSS-MATCH ENGINE
# =============================================================================

class CrossMatcher:
    """
    Main cross-matching engine with chunked processing and multiprocessing support.
    """
    
    def __init__(self, base_filepath: str, lookup_filepath: str,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 n_workers: Optional[int] = DEFAULT_N_WORKERS,
                 tolerance_arcsec: float = MATCH_TOLERANCE_ARCSEC):
        """
        Initialize cross-matcher.
        
        Parameters:
            base_filepath: Path to base FITS file (42M)
            lookup_filepath: Path to lookup FITS file (all_stellar_est)
            chunk_size: Number of rows per chunk
            n_workers: Number of parallel workers (None = all cores)
            tolerance_arcsec: Match tolerance in arcseconds
        """
        self.base_filepath = base_filepath
        self.lookup_filepath = lookup_filepath
        self.chunk_size = chunk_size
        self.n_workers = n_workers or mp.cpu_count()
        self.tolerance_deg = tolerance_arcsec / 3600.0
        
        # Get file info
        logger.info("=" * 60)
        logger.info("CROSS-MATCH INITIALIZATION")
        logger.info("=" * 60)
        
        self.base_nrows = get_fits_nrows(base_filepath)
        self.lookup_nrows = get_fits_nrows(lookup_filepath)
        self.base_columns = get_fits_columns(base_filepath)
        self.lookup_columns = get_fits_columns(lookup_filepath)
        
        logger.info(f"Base table: {base_filepath}")
        logger.info(f"  - Rows: {self.base_nrows:,}")
        logger.info(f"  - Columns: {len(self.base_columns)}")
        logger.info(f"Lookup table: {lookup_filepath}")
        logger.info(f"  - Rows: {self.lookup_nrows:,}")
        logger.info(f"  - Columns: {len(self.lookup_columns)}")
        logger.info(f"Chunk size: {chunk_size:,}")
        logger.info(f"Workers: {self.n_workers}")
        logger.info(f"Tolerance: {tolerance_arcsec} arcsec")
        
        # Calculate chunks
        self.n_chunks = (self.base_nrows + chunk_size - 1) // chunk_size
        logger.info(f"Total chunks: {self.n_chunks}")
        
        # Placeholders
        self.lookup_df = None
        self.spatial_index = None
        self.all_match_indices = None
        self.all_match_distances = None
    
    def load_lookup_table(self):
        """Load the lookup table (all_stellar_est) and build spatial index."""
        logger.info("=" * 60)
        logger.info("LOADING LOOKUP TABLE")
        logger.info("=" * 60)
        
        # Load only necessary columns for matching + columns to extract
        cols_to_load = list(set([LOOKUP_RA_COL, LOOKUP_DEC_COL] + LOOKUP_EXTRACT_COLS))
        cols_to_load = [c for c in cols_to_load if c in self.lookup_columns]
        
        self.lookup_df = load_fits_columns(self.lookup_filepath, cols_to_load)
        
        logger.info(f"Memory usage: {get_memory_usage_gb():.2f} GB")
        
        # Build spatial index
        lookup_ra = self.lookup_df[LOOKUP_RA_COL].values.astype(np.float64)
        lookup_dec = self.lookup_df[LOOKUP_DEC_COL].values.astype(np.float64)
        
        self.spatial_index = SpatialIndex(lookup_ra, lookup_dec, self.tolerance_deg)
        
        logger.info(f"Memory usage after KD-tree: {get_memory_usage_gb():.2f} GB")
    
    def run_matching_serial(self):
        """
        Run cross-matching in serial mode (more memory efficient).
        Processes one chunk at a time.
        """
        logger.info("=" * 60)
        logger.info("RUNNING CROSS-MATCH (SERIAL MODE)")
        logger.info("=" * 60)
        
        # Initialize result arrays
        self.all_match_indices = np.full(self.base_nrows, -1, dtype=np.int64)
        self.all_match_distances = np.full(self.base_nrows, -1.0, dtype=np.float64)
        
        total_matched = 0
        start_time = time.time()
        
        for chunk_id in range(self.n_chunks):
            start_row = chunk_id * self.chunk_size
            end_row = min((chunk_id + 1) * self.chunk_size, self.base_nrows)
            
            chunk_start = time.time()
            
            result = process_chunk_serial(
                chunk_id, self.base_filepath, start_row, end_row,
                self.spatial_index, self.base_columns
            )
            
            # Store results
            self.all_match_indices[start_row:end_row] = result['match_indices']
            self.all_match_distances[start_row:end_row] = result['match_distances']
            total_matched += result['n_matched']
            
            chunk_time = time.time() - chunk_start
            progress = (chunk_id + 1) / self.n_chunks * 100
            
            logger.info(
                f"Chunk {chunk_id + 1}/{self.n_chunks} ({progress:.1f}%) - "
                f"Matched: {result['n_matched']:,}/{result['n_total']:,} - "
                f"Time: {chunk_time:.1f}s - "
                f"Memory: {get_memory_usage_gb():.2f} GB"
            )
            
            # Force garbage collection
            gc.collect()
        
        elapsed = time.time() - start_time
        match_rate = total_matched / self.base_nrows * 100
        
        logger.info("=" * 60)
        logger.info("MATCHING COMPLETE")
        logger.info(f"Total matched: {total_matched:,}/{self.base_nrows:,} ({match_rate:.2f}%)")
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info("=" * 60)
    
    def run_matching_parallel(self):
        """
        Run cross-matching in parallel mode using shared memory.
        Faster but uses more memory.
        """
        logger.info("=" * 60)
        logger.info(f"RUNNING CROSS-MATCH (PARALLEL MODE - {self.n_workers} workers)")
        logger.info("=" * 60)
        
        # Create shared memory for lookup coordinates
        lookup_ra = self.lookup_df[LOOKUP_RA_COL].values.astype(np.float64)
        lookup_dec = self.lookup_df[LOOKUP_DEC_COL].values.astype(np.float64)
        
        shm_ra = shared_memory.SharedMemory(create=True, size=lookup_ra.nbytes)
        shm_dec = shared_memory.SharedMemory(create=True, size=lookup_dec.nbytes)
        
        # Copy data to shared memory
        shm_ra_arr = np.ndarray(lookup_ra.shape, dtype=np.float64, buffer=shm_ra.buf)
        shm_dec_arr = np.ndarray(lookup_dec.shape, dtype=np.float64, buffer=shm_dec.buf)
        shm_ra_arr[:] = lookup_ra[:]
        shm_dec_arr[:] = lookup_dec[:]
        
        try:
            # Prepare chunk arguments
            chunk_args = []
            for chunk_id in range(self.n_chunks):
                start_row = chunk_id * self.chunk_size
                end_row = min((chunk_id + 1) * self.chunk_size, self.base_nrows)
                
                chunk_args.append((
                    chunk_id, self.base_filepath, start_row, end_row,
                    shm_ra.name, shm_dec.name, lookup_ra.shape,
                    self.tolerance_deg, self.base_columns
                ))
            
            # Process with multiprocessing pool
            start_time = time.time()
            
            # Initialize result arrays
            self.all_match_indices = np.full(self.base_nrows, -1, dtype=np.int64)
            self.all_match_distances = np.full(self.base_nrows, -1.0, dtype=np.float64)
            
            with mp.Pool(processes=self.n_workers) as pool:
                results = pool.map(process_chunk_worker, chunk_args)
            
            # Collect results
            total_matched = 0
            for result in results:
                if 'error' in result:
                    logger.error(f"Chunk {result['chunk_id']} failed: {result['error']}")
                    continue
                
                start_row = result['start_row']
                end_row = result['end_row']
                self.all_match_indices[start_row:end_row] = result['match_indices']
                self.all_match_distances[start_row:end_row] = result['match_distances']
                total_matched += result['n_matched']
            
            elapsed = time.time() - start_time
            match_rate = total_matched / self.base_nrows * 100
            
            logger.info("=" * 60)
            logger.info("MATCHING COMPLETE")
            logger.info(f"Total matched: {total_matched:,}/{self.base_nrows:,} ({match_rate:.2f}%)")
            logger.info(f"Total time: {elapsed:.1f} seconds")
            logger.info("=" * 60)
            
        finally:
            # Clean up shared memory
            shm_ra.close()
            shm_ra.unlink()
            shm_dec.close()
            shm_dec.unlink()
    
    def write_output(self, output_filepath: str):
        """
        Write the final output CSV with LEFT JOIN.
        All 42M rows, with stellar params where matched.
        
        Parameters:
            output_filepath: Path to output CSV file
        """
        logger.info("=" * 60)
        logger.info("WRITING OUTPUT FILE")
        logger.info("=" * 60)
        
        # Prepare lookup data for fast access
        lookup_stellar_est = self.lookup_df['stellar_params_est'].values if 'stellar_params_est' in self.lookup_df.columns else None
        lookup_stellar_err = self.lookup_df['stellar_params_err'].values if 'stellar_params_err' in self.lookup_df.columns else None
        lookup_ra = self.lookup_df[LOOKUP_RA_COL].values
        lookup_dec = self.lookup_df[LOOKUP_DEC_COL].values
        
        # Write in chunks
        start_time = time.time()
        
        # First, write header by processing first chunk
        first_chunk = True
        total_written = 0
        
        for chunk_id in range(self.n_chunks):
            start_row = chunk_id * self.chunk_size
            end_row = min((chunk_id + 1) * self.chunk_size, self.base_nrows)
            
            # Load base chunk
            chunk_df = load_fits_chunk(self.base_filepath, start_row, end_row)
            
            # Get match indices for this chunk
            chunk_match_idx = self.all_match_indices[start_row:end_row]
            chunk_match_dist = self.all_match_distances[start_row:end_row]
            
            # Add matched columns
            chunk_df['match_distance_arcsec'] = chunk_match_dist
            chunk_df['matched'] = chunk_match_idx >= 0
            
            # Add stellar params from lookup table
            if lookup_stellar_est is not None:
                chunk_df['stellar_params_est'] = [
                    lookup_stellar_est[idx] if idx >= 0 else None 
                    for idx in chunk_match_idx
                ]
            if lookup_stellar_err is not None:
                chunk_df['stellar_params_err'] = [
                    lookup_stellar_err[idx] if idx >= 0 else None 
                    for idx in chunk_match_idx
                ]
            
            # Add matched RA/Dec from lookup
            chunk_df['ra_matched'] = [
                lookup_ra[idx] if idx >= 0 else np.nan 
                for idx in chunk_match_idx
            ]
            chunk_df['dec_matched'] = [
                lookup_dec[idx] if idx >= 0 else np.nan 
                for idx in chunk_match_idx
            ]
            
            # Write to CSV
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            chunk_df.to_csv(output_filepath, mode=mode, header=header, index=False)
            first_chunk = False
            total_written += len(chunk_df)
            
            progress = (chunk_id + 1) / self.n_chunks * 100
            logger.info(f"Written chunk {chunk_id + 1}/{self.n_chunks} ({progress:.1f}%) - Total rows: {total_written:,}")
            
            del chunk_df
            gc.collect()
        
        elapsed = time.time() - start_time
        
        # Get final file size
        file_size_gb = os.path.getsize(output_filepath) / (1024 ** 3)
        
        logger.info("=" * 60)
        logger.info("OUTPUT COMPLETE")
        logger.info(f"Output file: {output_filepath}")
        logger.info(f"Total rows: {total_written:,}")
        logger.info(f"File size: {file_size_gb:.2f} GB")
        logger.info(f"Write time: {elapsed:.1f} seconds")
        logger.info("=" * 60)
    
    def run(self, output_filepath: str, mode: str = 'serial'):
        """
        Run the complete cross-match pipeline.
        
        Parameters:
            output_filepath: Path to output CSV file
            mode: 'serial' or 'parallel'
        """
        total_start = time.time()
        
        # Step 1: Load lookup table and build index
        self.load_lookup_table()
        
        # Step 2: Run matching
        if mode == 'parallel':
            self.run_matching_parallel()
        else:
            self.run_matching_serial()
        
        # Step 3: Write output
        self.write_output(output_filepath)
        
        total_elapsed = time.time() - total_start
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Total time: {total_elapsed / 60:.1f} minutes")
        logger.info("=" * 60)

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cross-match FITS tables by sky coordinates (LEFT JOIN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serial mode (recommended for limited memory)
  python crossmatch_stellar.py --base RVS+DIST_42M.fits --lookup all_stellar_est.fits --output matched.csv

  # Parallel mode (faster, uses more memory)  
  python crossmatch_stellar.py --base RVS+DIST_42M.fits --lookup all_stellar_est.fits --output matched.csv --parallel

  # Custom chunk size and workers
  python crossmatch_stellar.py --base RVS+DIST_42M.fits --lookup all_stellar_est.fits --output matched.csv --chunk-size 200000 --workers 8
        """
    )
    
    parser.add_argument('--base', required=True, 
                        help='Path to base FITS file (RVS+DIST_42M)')
    parser.add_argument('--lookup', required=True,
                        help='Path to lookup FITS file (all_stellar_est)')
    parser.add_argument('--output', required=True,
                        help='Path to output CSV file')
    parser.add_argument('--tolerance', type=float, default=MATCH_TOLERANCE_ARCSEC,
                        help=f'Match tolerance in arcseconds (default: {MATCH_TOLERANCE_ARCSEC})')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Rows per chunk (default: {DEFAULT_CHUNK_SIZE:,})')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: all cores)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing mode')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.base):
        logger.error(f"Base file not found: {args.base}")
        sys.exit(1)
    if not os.path.exists(args.lookup):
        logger.error(f"Lookup file not found: {args.lookup}")
        sys.exit(1)
    
    # Run cross-match
    matcher = CrossMatcher(
        base_filepath=args.base,
        lookup_filepath=args.lookup,
        chunk_size=args.chunk_size,
        n_workers=args.workers,
        tolerance_arcsec=args.tolerance
    )
    
    mode = 'parallel' if args.parallel else 'serial'
    matcher.run(args.output, mode=mode)

if __name__ == '__main__':
    main()