import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances
import time
import os
import glob
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# --- CONFIGURATION: YOU MUST UPDATE THIS SECTION ---

# 1. FILE/FOLDER NAMES
# FOLDER CONTAINING YOUR MEMBER LISTS (e.g., 'antlia_2.txt')
MEMBER_CATALOGUE_FOLDER = 'member_catalogues'

# YOUR "DERIVED" STAR CATALOGUE (Script will generate a dummy one)
MY_STAR_CATALOGUE = 'my_star_catalogue.csv'

# OUTPUT SUMMARY FILE
OUTPUT_SUMMARY_FILE = 'match_summary.csv'
SPATial_PLOT_FILE = 'match_spatial_plot.png'
KINEMATIC_PLOT_FILE = 'match_kinematic_plot.png'

# 2. COLUMN NAMES
# These MUST match the headers in your files
RA_COL = 'ra'
DEC_COL = 'dec'
PMRA_COL = 'pmra'
PMDEC_COL = 'pmdec'
MEMBER_ID_COL = 'source_id' # Primary ID in your .txt files
MY_STAR_ID_COL = 'my_star_id'  # Primary ID in your derived catalogue
CLUSTER_NAME_COL = 'source_file' # Column in your .txt that names the cluster

# This is the exact header from your .txt files
MEMBER_FILE_HEADER_NAMES = [
    'source_id', 'ra', 'dec', 'phot_g_mean_mag', 'phot_rp_mean_mag',
    'phot_bp_mean_mag', 'ebv', 'pmra', 'pmra_error', 'pmdec',
    'pmdec_error', 'pmra_pmdec_corr', 'mem_fixed', 'mem_fixed_em',
    'mem_fixed_ep', 'mem_gauss', 'mem_gauss_em', 'mem_gauss_ep',
    'source_file'
]


# 3. MATCHING TOLERANCE (CRITICAL!)
# Max distance between your star and a member star to be a "match".
ANGULAR_TOLERANCE_DEG = 0.00027 # ~1 arcsecond
PM_TOLERANCE_MAS_YR = 0.1 # 0.1 mas/yr

# 4. MULTIPROCESSING
try:
    N_CORES = int(os.environ.get('PBS_NP', cpu_count()))
except (TypeError, ValueError):
    N_CORES = cpu_count()

# 5. DUMMY DATA PARAMETERS (for generation)
NUM_STARS_PER_CLUSTER = 200
NUM_FIELD_STARS = 1000 # Stars in your catalogue that won't match
CLUSTER_DEFS = {
    'antlia_2': (144.3, -39.7, -0.5, 0.4, 1.0, 0.5), # Using your example
    'Pleiades': (56.75, 24.11, 20.1, -45.5, 1.0, 0.5),
    'Hyades': (66.5, 15.8, 103.5, -28.0, 3.0, 1.0)
}
# Noise to add to your "derived" stars
RA_DEC_NOISE_DEG = 0.0001 # ~0.36 arcsec
PM_NOISE_MAS_YR = 0.05
# --- END OF CONFIGURATION ---


# --- Global variable for worker processes ---
worker_tree = None

def init_worker(tree_object):
    """Initializer for the multiprocessing Pool."""
    global worker_tree
    worker_tree = tree_object

def query_chunk(star_chunk):
    """Worker function to query a chunk of star data."""
    global worker_tree
    if worker_tree is None:
        raise ValueError("Worker tree not initialized!")
    # Query for the 1-nearest neighbor (k=1)
    return worker_tree.query(star_chunk, k=1)


def custom_distance_metric(my_star_props, member_star_props):
    """
    Calculates a single normalized 4D distance.
    A value < 1.0 means the star is "inside" the tolerance ellipse.
    Properties are: [ra, dec, pmra, pmdec]
    """
    # 1. Angular Distance (Haversine)
    star_rad = [np.radians(my_star_props[1]), np.radians(my_star_props[0])]
    member_rad = [np.radians(member_star_props[1]), np.radians(member_star_props[0])]
    angular_dist_rad = haversine_distances([star_rad], [member_rad])[0][0]
    angular_dist_deg = np.degrees(angular_dist_rad)

    # 2. Proper Motion Distance (Euclidean)
    pm_dist_sq = (my_star_props[2] - member_star_props[2])**2 + (my_star_props[3] - member_star_props[3])**2
    pm_dist = np.sqrt(pm_dist_sq)
    
    # 3. Normalization and Final Distance
    norm_ang_dist = angular_dist_deg / ANGULAR_TOLERANCE_DEG
    norm_pm_dist = pm_dist / PM_TOLERANCE_MAS_YR
    total_norm_dist = np.sqrt(norm_ang_dist**2 + norm_pm_dist**2)
    
    return total_norm_dist

# --- DATA GENERATION FUNCTIONS ---

def generate_dummy_member_files():
    """
    Generates dummy member .txt files in the correct
    TAB-SEPARATED format with header.
    """
    print(f"Generating dummy member files in '{MEMBER_CATALOGUE_FOLDER}'...")
    os.makedirs(MEMBER_CATALOGUE_FOLDER, exist_ok=True)
    all_members_for_derived_cat = []
    
    base_source_id = 5426537182448787584
    
    for cluster_id, props in CLUSTER_DEFS.items():
        center_ra, center_dec, center_pmra, center_pmdec, ang_size, pm_disp = props
        n = NUM_STARS_PER_CLUSTER
        
        # Generate core data
        ra = np.random.normal(center_ra, ang_size / 3.0, n)
        dec = np.random.normal(center_dec, ang_size / 3.0, n)
        pmra = np.random.normal(center_pmra, pm_disp, n)
        pmdec = np.random.normal(center_pmdec, pm_disp, n)
        
        # Generate realistic-looking extra data
        source_id = np.arange(base_source_id, base_source_id + n).astype(str) + '.000000000000000'
        base_source_id += n # Ensure unique IDs
        
        phot_g_mean_mag = np.random.normal(16, 2.5, n)
        bp_rp = np.random.normal(0.8, 0.3, n)
        phot_bp_mean_mag = phot_g_mean_mag + bp_rp / 2.0
        phot_rp_mean_mag = phot_g_mean_mag - bp_rp / 2.0
        ebv = np.abs(np.random.normal(0.4, 0.1, n))
        
        pmra_error = np.abs(np.random.normal(0.2, 0.05, n))
        pmdec_error = np.abs(np.random.normal(0.2, 0.05, n))
        pmra_pmdec_corr = np.random.uniform(-0.1, 0.1, n)
        
        mem_gauss = np.random.uniform(0.0001, 0.001, n)
        mem_gauss_em = mem_gauss - np.random.uniform(0.00001, 0.00005, n)
        mem_gauss_ep = mem_gauss + np.random.uniform(0.00001, 0.00005, n)
        mem_fixed = mem_gauss
        mem_fixed_em = mem_gauss_em
        mem_fixed_ep = mem_gauss_ep
        
        # Create DataFrame
        cluster_data = {
            'source_id': source_id, 'ra': ra, 'dec': dec,
            'phot_g_mean_mag': phot_g_mean_mag,
            'phot_rp_mean_mag': phot_rp_mean_mag,
            'phot_bp_mean_mag': phot_bp_mean_mag,
            'ebv': ebv, 'pmra': pmra, 'pmra_error': pmra_error,
            'pmdec': pmdec, 'pmdec_error': pmdec_error,
            'pmra_pmdec_corr': pmra_pmdec_corr, 'mem_fixed': mem_fixed,
            'mem_fixed_em': mem_fixed_em, 'mem_fixed_ep': mem_fixed_ep,
            'mem_gauss': mem_gauss, 'mem_gauss_em': mem_gauss_em,
            'mem_gauss_ep': mem_gauss_ep,
            'source_file': cluster_id
        }
        df = pd.DataFrame(cluster_data)

        # Save as TAB-SEPARATED .txt file with header
        txt_path = os.path.join(MEMBER_CATALOGUE_FOLDER, f"{cluster_id}.txt")
        try:
            df.to_csv(txt_path, columns=MEMBER_FILE_HEADER_NAMES, # Ensure order
                      sep='\t', index=False, header=True, float_format='%.15f')
            print(f"  Saved {txt_path}")
        except Exception as e:
            print(f"  Error saving {txt_path}: {e}")
            
        all_members_for_derived_cat.append(df)
        
    return pd.concat(all_members_for_derived_cat, ignore_index=True)

def generate_dummy_star_catalogue(member_stars_df):
    """
    Generates 'my_star_catalogue.csv' with "derived" stars.
    This is the file you will create yourself in the real run.
    """
    print(f"Generating dummy {MY_STAR_CATALOGUE}...")
    my_stars_list = []
    
    # 1. Add "derived" versions of the member stars
    for _, member_star in member_stars_df.iterrows():
        my_stars_list.append({
            MY_STAR_ID_COL: f"my_derived_star_{len(my_stars_list)}",
            RA_COL: member_star[RA_COL] + np.random.normal(0, RA_DEC_NOISE_DEG),
            DEC_COL: member_star[DEC_COL] + np.random.normal(0, RA_DEC_NOISE_DEG),
            PMRA_COL: member_star[PMRA_COL] + np.random.normal(0, PM_NOISE_MAS_YR),
            PMDEC_COL: member_star[PMDEC_COL] + np.random.normal(0, PM_NOISE_MAS_YR),
            'ra_error': np.abs(np.random.normal(0.1, 0.02)),
            'dec_error': np.abs(np.random.normal(0.1, 0.02)),
            'pmra_error': np.abs(np.random.normal(0.15, 0.05)),
            'pmdec_error': np.abs(np.random.normal(0.15, 0.05)),
            'true_member_source_id': member_star[MEMBER_ID_COL] # For validation
        })

    # 2. Add random field stars
    ra = np.random.uniform(40, 150, NUM_FIELD_STARS)
    dec = np.random.uniform(10, 30, NUM_FIELD_STARS)
    pmra = np.random.uniform(-50, 120, NUM_FIELD_STARS)
    pmdec = np.random.uniform(-60, 0, NUM_FIELD_STARS)

    for i in range(NUM_FIELD_STARS):
        my_stars_list.append({
            MY_STAR_ID_COL: f"my_field_star_{i}",
            RA_COL: ra[i],
            DEC_COL: dec[i],
            PMRA_COL: pmra[i],
            PMDEC_COL: pmdec[i],
            'ra_error': np.abs(np.random.normal(0.3, 0.1)),
            'dec_error': np.abs(np.random.normal(0.3, 0.1)),
            'pmra_error': np.abs(np.random.normal(0.5, 0.2)),
            'pmdec_error': np.abs(np.random.normal(0.5, 0.2)),
            'true_member_source_id': 'FIELD' # For validation
        })
        
    df = pd.DataFrame(my_stars_list)
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle
    df.to_csv(MY_STAR_CATALOGUE, index=False, float_format='%.15f')
    print(f"Saved {len(df)} total stars to {MY_STAR_CATALOGUE}.")
    return df

# --- DATA LOADING FUNCTION ---

def load_all_member_stars(folder_path):
    """
    Loads all .txt files from a folder and concatenates them.
    Assumes TAB-SEPARATED data with a header row.
    """
    print(f"\nLoading member stars from '{folder_path}'...")
    all_member_dfs = []
    # Find all files ending in .txt in the folder
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        print(f"ERROR: No .txt files found in '{folder_path}'.")
        print(f"Please check the MEMBER_CATALOGUE_FOLDER path in the config.")
        return None

    for f in txt_files:
        try:
            # Read TAB-SEPARATED file, assumes header is present
            df = pd.read_csv(f, sep='\t')
            
            if CLUSTER_NAME_COL not in df.columns:
                print(f"  ERROR: File {f} is missing the cluster name column '{CLUSTER_NAME_COL}'. Skipping.")
                continue
                
            # Rename for internal consistency
            df.rename(columns={CLUSTER_NAME_COL: 'cluster_name'}, inplace=True)
            
            all_member_dfs.append(df)
            
            # Get cluster name from the first row for logging
            cluster_name_found = df['cluster_name'].iloc[0]
            print(f"  Loaded {len(df)} members from {f} (Cluster: {cluster_name_found})")
        except Exception as e:
            print(f"  ERROR: Could not read file {f}. Skipping. Error: {e}")
            print("  Please ensure it is a TAB-SEPARATED file with the correct header.")
    
    if not all_member_dfs:
        print("ERROR: No member data was successfully loaded.")
        return None
        
    full_members_df = pd.concat(all_member_dfs, ignore_index=True)
    
    # Convert source_id to string to handle the .0000... format
    if MEMBER_ID_COL in full_members_df.columns:
         full_members_df[MEMBER_ID_COL] = full_members_df[MEMBER_ID_COL].astype(str)
            
    print(f"Total known members loaded: {len(full_members_df)}")
    return full_members_df

# --- PLOTTING FUNCTION ---

def plot_results(my_stars_df, all_members_df):
    """Generates and saves diagnostic plots."""
    print("Generating plots...")
    
    cluster_names = all_members_df['cluster_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_names)))
    color_map = {name: colors[i] for i, name in enumerate(cluster_names)}
    color_map['FIELD'] = (0.5, 0.5, 0.5, 0.1)
    
    matched_colors = my_stars_df['matched_cluster'].fillna('FIELD').map(color_map)
    
    # --- 1. Spatial Plot (RA vs Dec) ---
    plt.figure(figsize=(12, 10))
    plt.scatter(
        my_stars_df[RA_COL], my_stars_df[DEC_COL],
        c=matched_colors,
        s=10, alpha=0.7, label='My Stars (Colored by Match)'
    )
    member_colors = all_members_df['cluster_name'].map(color_map)
    plt.scatter(
        all_members_df[RA_COL], all_members_df[DEC_COL],
        s=30, c=member_colors, marker='x', label='True Member Catalogue'
    )
    plt.xlabel('Right Ascension (deg)')
    plt.ylabel('Declination (deg)')
    plt.title('Spatial Cross-Match Results (RA vs Dec)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().invert_xaxis()
    plt.savefig(SPATial_PLOT_FILE)
    print(f"Saved spatial plot to '{SPATial_PLOT_FILE}'")

    # --- 2. Kinematic Plot (pmRA vs pmDec) ---
    plt.figure(figsize=(12, 10))
    plt.scatter(
        my_stars_df[PMRA_COL], my_stars_df[PMDEC_COL],
        c=matched_colors,
        s=10, alpha=0.7, label='My Stars (Colored by Match)'
    )
    plt.scatter(
        all_members_df[PMRA_COL], all_members_df[PMDEC_COL],
        s=30, c=member_colors, marker='x', label='True Member Catalogue'
    )
    plt.xlabel('pmRA (mas/yr)')
    plt.ylabel('pmDec (mas/yr)')
    plt.title('Kinematic Cross-Match Results (pmRA vs pmDec)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(KINEMATIC_PLOT_FILE)
    print(f"Saved kinematic plot to '{KINEMATIC_PLOT_FILE}'")

# --- MAIN EXECUTION ---

def main():
    print("Starting star-to-star cross-match...")
    print(f"Using {N_CORES} cores for multiprocessing.")
    
    # --- 1. Generate Dummy Data ---
    # This section creates dummy files for testing.
    # COMMENT/DELETE this section when using your REAL data.
    print("--- GENERATING DUMMY DATA (FOR TESTING) ---")
    try:
        true_members_df = generate_dummy_member_files()
        my_stars_df_gen = generate_dummy_star_catalogue(true_members_df)
    except Exception as e:
        print(f"Error generating dummy data: {e}")
        return
    print("--- DUMMY DATA GENERATION COMPLETE ---")
    
    # --- 2. Load Real Data ---
    # UNCOMMENT these lines to use your REAL data.
    # print(f"Loading my star catalogue: {MY_STAR_CATALOGUE}")
    # my_stars_df = pd.read_csv(MY_STAR_CATALOGUE)
    # all_members_df = load_all_member_stars(MEMBER_CATALOGUE_FOLDER)
    
    # For this example, we use the generated data:
    my_stars_df = my_stars_df_gen
    all_members_df = load_all_member_stars(MEMBER_CATALOGUE_FOLDER)

    if all_members_df is None or my_stars_df is None:
        print("Halting due to data loading errors.")
        return

    # --- 3. Prepare Data for Tree ---
    cols_to_use = [RA_COL, DEC_COL, PMRA_COL, PMDEC_COL]
    if not all(col in all_members_df.columns for col in cols_to_use):
        print(f"ERROR: Member files are missing required columns: {cols_to_use}")
        return
    if not all(col in my_stars_df.columns for col in cols_to_use + [MY_STAR_ID_COL]):
        print(f"ERROR: Your star file '{MY_STAR_CATALOGUE}' is missing required columns: {cols_to_use + [MY_STAR_ID_COL]}")
        return

    member_data = all_members_df[cols_to_use].values
    my_star_data = my_stars_df[cols_to_use].values
    
    # --- 4. Build the BallTree ---
    print(f"\nBuilding BallTree on {len(member_data)} member stars...")
    start_time = time.time()
    tree = BallTree(member_data, metric=custom_distance_metric)
    end_time = time.time()
    print(f"Tree built in {end_time - start_time:.2f} seconds.")

    # --- 5. Query the Tree (in Parallel) ---
    print(f"Querying tree with {len(my_star_data)} stars using {N_CORES} cores...")
    start_time = time.time()
    star_chunks = np.array_split(my_star_data, N_CORES)
    
    try:
        with Pool(processes=N_CORES, initializer=init_worker, initargs=(tree,)) as pool:
            results = pool.map(query_chunk, star_chunks)
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        return
        
    distances = np.vstack([r[0] for r in results]).flatten()
    indices = np.vstack([r[1] for r in results]).flatten()
    end_time = time.time()
    print(f"Query completed in {end_time - start_time:.2f} seconds.")
    
    # --- 6. Process Results & Create Summary Table ---
    print("Processing results and building summary table...")
    
    my_stars_df['normalized_distance'] = distances
    my_stars_df['matched_member_index'] = indices
    
    # Add a suffix to member columns (e.g., 'ra' -> 'ra_member')
    member_props_df = all_members_df.add_suffix('_member')
    
    # Join 'my_stars_df' with 'member_props_df'
    summary_df = my_stars_df.merge(
        member_props_df,
        left_on='matched_member_index',
        right_index=True,
        how='left'
    )
    
    # --- 7. Filter non-matches ---
    no_match_mask = summary_df['normalized_distance'] > 1.0
    member_cols = member_props_df.columns.tolist()
    helper_cols = ['normalized_distance', 'matched_member_index']
    summary_df.loc[no_match_mask, member_cols + helper_cols] = np.nan
    
    # Rename cluster name column for clarity
    summary_df.rename(columns={'cluster_name_member': 'matched_cluster'}, inplace=True)
    
    # --- 8. Save Output ---
    try:
        # Reorder columns for clarity
        cols_order = [MY_STAR_ID_COL, RA_COL, DEC_COL, PMRA_COL, PMDEC_COL, 'matched_cluster']
        other_my_cols = [c for c in my_stars_df.columns if c not in cols_order and not c.startswith('matched_') and c != 'true_member_source_id']
        member_cols_ordered = [c for c in summary_df.columns if c.endswith('_member')]
        
        final_cols = cols_order + other_my_cols + member_cols_ordered + ['normalized_distance', 'true_member_source_id']
        final_cols = [c for c in final_cols if c in summary_df.columns] # Ensure all exist
        summary_df = summary_df[final_cols]
        
        summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False, float_format='%.15f')
    except Exception as e:
        print(f"Error saving output file: {e}")
        return
        
    # --- 9. Print Summary ---
    n_matched = summary_df['matched_cluster'].notna().sum()
    n_total = len(summary_df)
    print(f"\n--- Matching Complete ---")
    print(f"Matched {n_matched} out of {n_total} stars within tolerance.")
    print(f"Summary table of matching pairs saved to '{OUTPUT_SUMMARY_FILE}'")

    # --- 10. Generate Plots ---
    try:
        plot_results(summary_df, all_members_df)
    except Exception as e:
        print(f"Error generating plots: {e}")
        print("This may be due to a missing graphics backend on the cluster.")
        print("The CSV summary file was saved successfully.")

if __name__ == "__main__":
    if (ANGULAR_TOLERANCE_DEG <= 0 or PM_TOLERANCE_MAS_YR <= 0):
        print("ERROR: Search tolerances in the configuration must be positive values.")
    else:
        main()