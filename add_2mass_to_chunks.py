#!/usr/bin/env python3
"""
Add 2MASS J, H, K bands and their errors to Gaia catalogue chunks by matching source_id.
Processes chunks 01-08 and creates new files with .1 extension.
"""

import gzip
import pandas as pd
from astropy.io import fits
import numpy as np
from pathlib import Path
import re

# Configuration
INPUT_DIR = Path('/user/sutirtha/BallTree_Xmatch')
TABLE3_FILE = INPUT_DIR / 'BallTree_Xmatch' / 'table3.dat.gz'
CHUNKS = [f'{i:02d}' for i in range(1, 9)]

# Column names for table3.dat.gz (15 columns based on inspection)
TABLE3_COLS = ['source_id', 'gmag', 'plx_err', 'dec', 'tmass_id',
               'J', 'e_J', 'H', 'e_H', 'K', 'e_K', 'col12', 'col13', 'col14', 'col15']

def load_table3():
    """Load 2MASS crossmatch table and extract J, H, K data."""
    print("Loading table3.dat.gz...")
    df = pd.read_csv(TABLE3_FILE, sep='\s+', names=TABLE3_COLS, dtype={'source_id': 'int64'})

    # Keep only required columns
    df = df[['source_id', 'J', 'e_J', 'H', 'e_H', 'K', 'e_K']].copy()

    # Remove duplicates (keep first if source_id appears multiple times)
    df = df.drop_duplicates(subset=['source_id'], keep='first')
    df = df.set_index('source_id')

    print(f"  Loaded {len(df)} unique sources from table3")
    return df

def extract_gaia_source_id(all_survey_ids_str):
    """Extract Gaia DR3 source_id from all_survey_ids string."""
    if pd.isna(all_survey_ids_str):
        return None
    match = re.search(r'gdr3_source_id:(\d+)', all_survey_ids_str)
    if match:
        return int(match.group(1))
    return None

def process_chunk(chunk_num, table3_data):
    """Process a single chunk: add 2MASS data and save new file."""
    chunk_name = f'Entire_catalogue_chunk{chunk_num}.fits'
    input_path = INPUT_DIR / chunk_name
    output_name = f'Entire_catalogue_chunk{chunk_num}.1.fits'
    output_path = INPUT_DIR / output_name

    print(f"\n{'='*60}")
    print(f"Processing Chunk {chunk_num}")
    print(f"{'='*60}")
    print(f"  Reading: {input_path.name}")

    # Open the chunk file
    with fits.open(input_path) as hdul:
        # Work with the binary table HDU (index 1)
        table_hdu = hdul[1]
        data = table_hdu.data.copy()
        cols = table_hdu.columns

        print(f"  Current columns: {len(cols)}")
        print(f"  Current rows: {len(data)}")

        # Extract Gaia source IDs
        print("  Extracting Gaia source IDs from all_survey_ids...")
        source_ids = np.array([extract_gaia_source_id(sid) for sid in data['all_survey_ids']])

        matched_count = np.sum(~np.isnan(source_ids.astype('float')))
        print(f"  Found {matched_count} valid Gaia source IDs")

        # Create a DataFrame for easier merging
        # Convert source_ids, handling NaNs with nullable int type
        source_ids_clean = pd.array(source_ids, dtype='Int64')
        chunk_df = pd.DataFrame({
            'source_id': source_ids_clean,
            'original_index': np.arange(len(data))
        })

        # Merge with table3 data
        print("  Matching with 2MASS data...")
        chunk_df = chunk_df.merge(
            table3_data.reset_index(),
            on='source_id',
            how='left'
        )

        matched = chunk_df['J'].notna().sum()
        print(f"  Matched {matched} sources with 2MASS data ({matched/len(data)*100:.1f}%)")

        # Get the matched 2MASS values in original order
        J = chunk_df['J'].values
        e_J = chunk_df['e_J'].values
        H = chunk_df['H'].values
        e_H = chunk_df['e_H'].values
        K = chunk_df['K'].values
        e_K = chunk_df['e_K'].values

        # Create new columns with appropriate FITS format
        # Use float64 for magnitudes and errors, or appropriate type
        new_cols = [
            fits.Column(name='J_2MASS', array=J, format='D'),
            fits.Column(name='e_J_2MASS', array=e_J, format='D'),
            fits.Column(name='H_2MASS', array=H, format='D'),
            fits.Column(name='e_H_2MASS', array=e_H, format='D'),
            fits.Column(name='K_2MASS', array=K, format='D'),
            fits.Column(name='e_K_2MASS', array=e_K, format='D'),
        ]

        # Create new BinTableHDU by appending columns
        new_cols_hdu = fits.BinTableHDU.from_columns(list(cols) + new_cols, nrows=len(data))

        # Copy over the data
        for col in cols.names:
            new_cols_hdu.data[col] = data[col]
        for col in [c.name for c in new_cols]:
            new_cols_hdu.data[col] = chunk_df[col.replace('_2MASS', '')].values

        # Create new HDU list with modified header
        primary_hdu = hdul[0].copy()
        new_hdul = fits.HDUList([primary_hdu, new_cols_hdu])

        # Save the new file
        print(f"  Writing: {output_path.name}")
        new_hdul.writeto(output_path, overwrite=False)
        print(f"  ✓ Saved successfully")

        return matched, len(data)

def main():
    print("2MASS Data Addition Script")
    print("="*60)

    # Load table3 data
    table3_data = load_table3()

    # Process each chunk
    total_matched = 0
    total_rows = 0

    for chunk_num in CHUNKS:
        try:
            matched, rows = process_chunk(chunk_num, table3_data)
            total_matched += matched
            total_rows += rows
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows processed: {total_rows}")
    if total_rows > 0:
        print(f"Total matched with 2MASS: {total_matched} ({total_matched/total_rows*100:.1f}%)")
    else:
        print(f"Total matched with 2MASS: {total_matched}")
    print(f"Output files: Entire_catalogue_chunk[01-08].1.fits")
    print("="*60)

if __name__ == '__main__':
    main()
