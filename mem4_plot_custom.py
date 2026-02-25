#!/usr/bin/env python3
"""
================================================================================
ADAPTIVE MEMBERSHIP ANALYSIS V12-PLOTS — ENHANCED PUBLICATION PLOTS
================================================================================
Based on V11 (mem3.py). Internal machinery UNCHANGED.
Plot changes:
  1. Navy blue (#000080) for parallax/1/plx traces everywhere.
  2. Median lines are DASHED ('--') for both green and navy blue.
  3. MAD shaded bands more opaque (alpha=0.30) coordinated with each color.
  4. Red/crimson "all matched" histogram more visible (lw=2.5, alpha=0.7).
  5. Reference distance band always prominent (black line + gray fill).
  6. Both member-catalog and master-matched data shown in Dist & RV panels.
  7. Summary: 4 separate paginated series (Dist, RV, PM, RADEC).
     — 6×6 square panels per page, paginated as _1, _2, _3 …
     — Gold filter: >=15 green; per-type must have >=5 passing,
       else relax to >=10 green for that type.
  8. Grand summary updated accordingly.
Author: Sutirtha (V12-plots, publication-grade)
================================================================================
"""
import os, sys, json, time, logging, argparse, warnings, glob, re
import gc as gcmod
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde, median_abs_deviation, kstest, norm
from scipy.optimize import minimize_scalar
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
try:
    from astropy.coordinates import SkyCoord; from astropy.time import Time
    import astropy.units as u; HAS_ASTROPY = True
except ImportError: HAS_ASTROPY = False
try: import hdbscan; HAS_HDBSCAN = True
except ImportError: HAS_HDBSCAN = False

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
    MASTER_CATALOG=None; GC_MEMBERS_FILE=None; OC_MEMBERS_FILE=None
    SGR_MEMBERS_FILE=None; DWG_MEMBERS_FILE=None; GC_DIST_FILE=None
    OUTPUT_DIR='./outputs'; CHECKPOINT_DIR='./checkpoints'
    MASTER_COLS = {
        'ra':'RA_final','dec':'DEC_final','pmra':'pmra_final','pmdec':'pmdec_final',
        'pmra_err':'pmra_err_final','pmdec_err':'pmdec_err_final',
        'dist':'distance_final','dist_err':'distance_err_final',
        'rv':'Weighted_Avg_final','rv_err':'Weighted_Avg_err_final',
        'parallax':'parallax_final','parallax_err':'parallax_err_final',
        'params_est':'stellar_params_est','params_err':'stellar_params_err',
        'gmag':'Gmag','bpmag':'BPmag','rpmag':'RPmag','bp_rp':'BP-RP',
        'bp_g':'BP-G','g_rp':'G-RP','logg':'logg','feh':'feh','teff':'Teff',
        'ag':'AG','ebprp':'E(BP-RP)','a0':'A0','ruwe':'RUWE','gofal':'gofAL',
        'chi2al':'chi2AL','rv_sn':'RVS/N','rv_chi2':'RVchi2','rv_gof':'RVgof',
        'rvnper':'RVNper','pm_corr':'pmRApmDEcor','gal_l':'l','gal_b':'b',
        'pgal':'PGal','pqso':'PQSO','pss':'PSS','grvsm':'GRVSmag',
        'catwise_w1':'catwise_w1','catwise_w2':'catwise_w2',
        'logg_lo':'b_logg_x','logg_hi':'B_logg_xa','feh_lo':'b_[Fe/H]_x',
        'feh_hi':'B_[Fe/H]_xa','teff_lo':'b_Teff_x','teff_hi':'B_Teff_xa',
        'ag_lo':'b_AG_x','ag_hi':'B_AG_xa','ebprp_lo':'b_E(BP-RP)_x',
        'ebprp_hi':'B_E(BP-RP)_xa',
    }
    ALT_MASTER_COLS = {
        'rv':['Weighted_Avg_final','ZP_final','RV_final','Weighted_Avg','radial_velocity','RV'],
        'rv_err':['Weighted_Avg_err_final','ZP_err_final','RV_err_final',
                  'radial_velocity_error','RV_err'],
    }
    GC_MEM_COLS = {
        'key':'source','ra':'ra','dec':'dec','pmra':'pmra','pmdec':'pmdec',
        'pmra_err':'pmra_error','pmdec_err':'pmdec_error',
        'pmra_pmdec_corr':'pmra_pmdec_corr','parallax':'parallax',
        'membership_prob':'membership_probability',
        'rv':'RV_weighted_avg','rv_err':'e_RV_weighted_avg',
        'gmag':'g_mag','bp_rp':'bp_rp',
    }
    GC_DIST_COLS = {
        'name':'Name','lit_dist':'Lit. dist. (kpc)','lit_dist_err':'Lit. dist. Err+',
        'mean_dist':'Mean distance (kpc)','mean_dist_err':'Mean distance Err+',
    }
    OC_MEM_COLS = {
        'key':'Cluster','ra':'RAdeg','dec':'DEdeg','pmra':'pmRA','pmdec':'pmDE',
        'pmra_err':'e_pmRA','pmdec_err':'e_pmDE','pmra_pmdec_corr':'pmRApmDEcor',
        'parallax':'Plx','membership_prob':'Proba','rv':'RV','rv_err':'e_RV',
        'gmag':'Gmag','bp_rp':'BP-RP','plx_err':'e_Plx',
    }
    SGR_MEM_COLS = {
        'key':None,'ra':'ra','dec':'dec','pmra':'pmra','pmdec':'pmdec',
        'pmra_err':'pmraerr','pmdec_err':'pmdecerr',
        'parallax':'parallax','parallax_err':'plxerr',
        'dist':'dist','dist_err':'disterr','rv':'vlos','rv_err':'vloserr',
        'feh':'FeH','feh_err':'FeHerr','gmag':'g_mag','bp_rp':'bp_rp',
        'ebv':'ebv','lambda_sgr':'Lambda','beta_sgr':'Beta',
    }
    DWG_MEM_COLS = {
        'key':'name','ra':'ra_x','dec':'dec_x','pmra':'pmra','pmdec':'pmdec',
        'pmra_err':'pmra_error','pmdec_err':'pmdec_error',
        'distance':'distance','distance_err':'distance_error',
        'distance_modulus':'distance_modulus','distance_modulus_err':'distance_modulus_error',
        'rhalf':'rhalf','rv':'RV_km_s','rv_err':'e_RV_km_s',
        'rv_ref':'vlos_systemic','rv_ref_err':'vlos_systemic_error',
        'rv_sigma':'vlos_sigma','rv_sigma_err':'vlos_sigma_error',
        'ellipticity':'ellipticity','position_angle':'position_angle',
        'rcore':'rcore','rking':'rking','king_c':'king_concentration',
        'metallicity':'metallicity','metallicity_err':'metallicity_error',
        'metallicity_sigma':'metallicity_sigma',
        'edr3_pmra':'edr3_pmra','edr3_pmdec':'edr3_pmdec',
        'edr3_pmra_err':'edr3_pmra_error','edr3_pmdec_err':'edr3_pmdec_error',
        'gmax':'Gmax','rmax_deg':'Rmax',
    }
    CROSSMATCH_RADIUS_ARCSEC=1.0
    GMM_N_COMPONENTS=2; GMM_MAX_ITER=300; GMM_N_INIT=10; GMM_RANDOM_STATE=42
    DBSCAN_EPS_CLEANUP=0.3; DBSCAN_MIN_SAMPLES_CLEANUP=3
    DBSCAN_EPS_OC=0.25; DBSCAN_MIN_SAMPLES_OC=5
    DBSCAN_EPS_STREAM=0.4; DBSCAN_MIN_SAMPLES_STREAM=3
    P_MEM_HIGH=0.8; P_MEM_LOW=0.2
    MIN_STARS_FOR_ANALYSIS=10; MIN_SUMMARY_MATCH=5
    SGR_BIN_START_KPC=15.0; SGR_BIN_END_KPC=80.0; SGR_BIN_WIDTH_KPC=5.0
    SGR_MIN_STARS_PER_BIN=5
    P_MEM_PLOT_THRESHOLD=0.5
    EPOCH_DELTA=16.0; EPOCH_FROM=2000.0; EPOCH_TO=2016.0
    RUWE_GOOD_THRESHOLD=1.4; RUWE_PM_ERR_INFLATE=3.0
    RV_SN_MIN=3.0; RV_ERR_INFLATE_BAD_SN=5.0; GOFAL_THRESHOLD=3.0
    EM_MAX_ITER=60; EM_CONVERGENCE_TOL=1e-5; COV_REGULARIZE=1e-6
    ETA_INIT=0.3; BIC_TEST_COMPONENTS=[1,2,3]
    PLX_FOREGROUND_SIGMA=3.0; PLX_FOREGROUND_THRESHOLD=0.10
    SIGMA_INT_INIT=10.0; SIGMA_INT_FLOOR=0.5; FIELD_RV_SIGMA_INIT=80.0
    FIELD_INIT_RHALF_MULT=3.0; RHALF_FALLBACK_DEG=0.5
    CMD_KDE_BANDWIDTH=0.15; CMD_FIELD_ANNULUS_MULT=5.0; CMD_MIN_MEMBERS_FOR_KDE=15
    LOGG_GIANT_THRESHOLD=3.5; LOGG_GIANT_SIGMA=0.5
    FEH_FIELD_MEAN=-1.0; FEH_FIELD_SIGMA=0.8; FEH_MEMBER_SIGMA_DEFAULT=0.3
    DIST_FIELD_SIGMA_FRAC=0.5; DIST_MEMBER_SIGMA_FLOOR=0.5
    CONTROL_FIELD_INNER_MULT=5.0; CONTROL_FIELD_OUTER_MULT=10.0
    SUMMARY_MIN_HIGH_PMEM=15  # V12: raised to 15
    SUMMARY_MIN_HIGH_PMEM_RELAX=10  # V12: relaxed threshold
    SUMMARY_MIN_PER_TYPE=5  # V12: min objects per type at primary threshold
    SUMMARY_KS_ALPHA=0.05
    CMAP_PMEM='RdYlGn'; PLOT_DPI=150; SAVE_FORMAT='png'
    # ── V12 colour palette ──
    COL_MASTER='#DC143C'       # Crimson for all-matched
    COL_MASTER_LW=2.5          # V12: thicker red
    COL_MASTER_ALPHA=0.70      # V12: more visible red
    COL_MEMBER='#000080'       # Navy blue (member / parallax)
    COL_HIGHMEM='#006400'      # Dark green (high P_mem)
    COL_PLX='#000080'          # V12: navy blue parallax
    COL_GRAY_BG='#C0C0C0'
    GRAY_BG_ALPHA=1.0; GRAY_BG_SIZE=6
    # V12: Median & MAD styling
    MEDIAN_LS='--'             # Dashed median lines
    MEDIAN_LW=2.5
    MAD_ALPHA=0.18             # More opaque MAD bands
    REF_LW=3.0; REF_ALPHA=0.9; REF_BAND_ALPHA=0.28
    # V12: Summary panel grid
    SUMMARY_NCOLS=6; SUMMARY_NROWS=6  # 6x6 = 36 per page
    SUMMARY_PANEL_W=4.2; SUMMARY_PANEL_H=4.2  # square-ish

cfg = Config()

# ============================================================================
# UTILITIES  (unchanged from V11)
# ============================================================================
def normalize_name(name):
    if not isinstance(name, str): name = str(name)
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())

normalize_cluster_name = normalize_name

def _safe_float(row, col):
    if col is None: return None
    v = row.get(col)
    if v is None or pd.isna(v): return None
    try: return float(v)
    except: return None

def _safe_col(df, colname, dtype=float):
    if colname and colname in df.columns:
        return pd.to_numeric(df[colname], errors='coerce').values.astype(dtype)
    return np.full(len(df), np.nan, dtype=dtype)

def angular_separation_deg(ra1, dec1, ra2, dec2):
    r1=np.radians(np.asarray(ra1,dtype=float)); d1=np.radians(np.asarray(dec1,dtype=float))
    r2=np.radians(float(ra2)); d2=np.radians(float(dec2))
    a=np.sin((d2-d1)/2)**2+np.cos(d1)*np.cos(d2)*np.sin((r2-r1)/2)**2
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(a,0,1))))

def _compute_plx_distance(member_df, matched_df, cols):
    n = len(matched_df)
    mem_plx = np.full(n, np.nan)
    c = cols.get('parallax')
    if c and c in matched_df.columns:
        mem_plx = pd.to_numeric(matched_df[c], errors='coerce').values.astype(float)
    master_plx = np.full(n, np.nan)
    if 'plx_from_params_master' in matched_df.columns:
        master_plx = matched_df['plx_from_params_master'].values.astype(float)
    plx = np.where(np.isfinite(mem_plx)&(mem_plx>0), mem_plx,
                   np.where(np.isfinite(master_plx)&(master_plx>0), master_plx, np.nan))
    return np.where(plx>0, 1.0/plx, np.nan)

# ============================================================================
# EPOCH PROPAGATION  (unchanged)
# ============================================================================
def propagate_coords_to_gaia_epoch(ra, dec, pmra, pmdec, logger, epoch_delta=None):
    if epoch_delta is None: epoch_delta = cfg.EPOCH_DELTA
    ra=np.asarray(ra,dtype=float); dec=np.asarray(dec,dtype=float)
    pmra=np.asarray(pmra,dtype=float); pmdec=np.asarray(pmdec,dtype=float)
    ra_out=ra.copy(); dec_out=dec.copy()
    pos_ok=np.isfinite(ra)&np.isfinite(dec)
    prop_ok=pos_ok&np.isfinite(pmra)&np.isfinite(pmdec)
    n_prop=int(np.sum(prop_ok)); n_fb=int(np.sum(pos_ok&~prop_ok))
    if n_prop==0:
        logger.warning("      [epoch] No finite PM"); return ra_out,dec_out,0,n_fb
    if HAS_ASTROPY:
        try:
            t0=Time(f'J{cfg.EPOCH_FROM:.1f}'); t1=Time(f'J{cfg.EPOCH_TO:.1f}')
            c=SkyCoord(ra=ra[prop_ok]*u.deg, dec=dec[prop_ok]*u.deg,
                pm_ra_cosdec=pmra[prop_ok]*u.mas/u.yr, pm_dec=pmdec[prop_ok]*u.mas/u.yr,
                frame='icrs', obstime=t0)
            cp=c.apply_space_motion(new_obstime=t1)
            ra_out[prop_ok]=cp.ra.deg; dec_out[prop_ok]=cp.dec.deg
            logger.info(f"      [epoch] astropy: {n_prop} prop | {n_fb} fb")
            return ra_out,dec_out,n_prop,n_fb
        except Exception as e:
            logger.warning(f"      [epoch] astropy fail ({e}); linear")
    dec_r=np.radians(dec[prop_ok]); cd=np.cos(dec_r)
    cd=np.where(np.abs(cd)<1e-10,1e-10,cd)
    ra_out[prop_ok]+=pmra[prop_ok]*epoch_delta/(cd*3.6e6)
    dec_out[prop_ok]+=pmdec[prop_ok]*epoch_delta/3.6e6
    return ra_out,dec_out,n_prop,n_fb

def _get_query_coords(cdf, cols, logger, epoch_mode):
    ra=pd.to_numeric(cdf[cols['ra']],errors='coerce').values
    dec=pd.to_numeric(cdf[cols['dec']],errors='coerce').values
    if epoch_mode != '2000': return ra, dec
    pmra=_safe_col(cdf, cols.get('pmra')) if cols.get('pmra') else np.full(len(ra),np.nan)
    pmdec=_safe_col(cdf, cols.get('pmdec')) if cols.get('pmdec') else np.full(len(ra),np.nan)
    ra_p,dec_p,_,_=propagate_coords_to_gaia_epoch(ra,dec,pmra,pmdec,logger)
    return ra_p, dec_p

# ============================================================================
# APJ PAPER STYLE
# ============================================================================
def set_paper_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family':'serif',
        'font.serif':['Times New Roman','DejaVu Serif','Computer Modern Roman'],
        'font.size':14, 'axes.labelsize':16, 'axes.titlesize':18,
        'axes.labelweight':'bold', 'axes.titleweight':'bold',
        'axes.linewidth':1.8, 'axes.edgecolor':'black',
        'xtick.labelsize':13, 'ytick.labelsize':13,
        'xtick.major.size':7, 'ytick.major.size':7,
        'xtick.minor.size':4, 'ytick.minor.size':4,
        'xtick.major.width':1.4, 'ytick.major.width':1.4,
        'xtick.minor.width':1.0, 'ytick.minor.width':1.0,
        'xtick.direction':'in', 'ytick.direction':'in',
        'xtick.top':True, 'ytick.right':True,
        'xtick.minor.visible':True, 'ytick.minor.visible':True,
        'legend.fontsize':11, 'legend.framealpha':0.85,
        'legend.edgecolor':'black', 'legend.fancybox':False,
        'figure.facecolor':'white', 'axes.facecolor':'white',
        'savefig.facecolor':'white', 'savefig.bbox':'tight',
        'figure.dpi':150, 'savefig.dpi':300, 'text.usetex':False,
    })

# ============================================================================
# MASTER CATALOG  (unchanged)
# ============================================================================
class MasterCatalog:
    EXTRA_COLS = [
        'Gmag','BPmag','RPmag','BP-RP','BP-G','G-RP','logg','feh','Teff',
        'AG','E(BP-RP)','A0','b_logg_x','B_logg_xa','b_[Fe/H]_x','B_[Fe/H]_xa',
        'b_Teff_x','B_Teff_xa','b_AG_x','B_AG_xa','b_E(BP-RP)_x','B_E(BP-RP)_xa',
        'RUWE','gofAL','chi2AL','RVS/N','RVchi2','RVgof','RVNper',
        'pmRApmDEcor','l','b','PGal','PQSO','PSS','GRVSmag',
        'catwise_w1','catwise_w2',
    ]
    def __init__(self, logger):
        self.logger=logger; self.df=None; self.tree=None; self.coords_3d=None
        self.max_chord=None; self.nrows=0

    def _find_col(self, key, avail):
        primary = cfg.MASTER_COLS.get(key)
        if primary and primary in avail: return primary
        for alt in cfg.ALT_MASTER_COLS.get(key, []):
            if alt in avail: return alt
        return None

    def load(self, filepath, checkpoint_dir=None):
        from astropy.io import fits as afits
        self.logger.info("="*70+"\nLOADING MASTER CATALOG (V12-plots)\n"+"="*70)
        tree_cp = os.path.join(checkpoint_dir,'master_tree_v10.npz') if checkpoint_dir else None
        data_cp = os.path.join(checkpoint_dir,'master_data_v10.parquet') if checkpoint_dir else None
        if tree_cp and os.path.exists(tree_cp) and data_cp and os.path.exists(data_cp):
            self.logger.info("Loading from checkpoint...")
            try:
                self.df = pd.read_parquet(data_cp)
                self.coords_3d = np.load(tree_cp)['coords']
                self.tree = cKDTree(self.coords_3d)
                self.nrows = len(self.df); self._compute_max_chord()
                self.logger.info(f"Checkpoint: {self.nrows:,} rows")
                return True
            except Exception as e: self.logger.warning(f"Checkpoint failed: {e}")

        files = self._resolve_files(filepath)
        if not files: self.logger.error(f"No FITS: {filepath}"); return False
        self.logger.info(f"Loading {len(files)} file(s)...")
        t0 = time.time(); all_dfs = []
        for fi, fpath in enumerate(files):
            self.logger.info(f"  [{fi+1}/{len(files)}] {os.path.basename(fpath)}")
            try:
                with afits.open(fpath, memmap=True) as hdul:
                    dh = None
                    for hdu in hdul:
                        if hasattr(hdu,'columns') and hdu.columns is not None: dh=hdu; break
                    if dh is None: continue
                    cn = [c.name for c in dh.columns]; nc = dh.data.shape[0]; chunk = {}
                    rc=cfg.MASTER_COLS['ra']; dc=cfg.MASTER_COLS['dec']
                    if rc not in cn or dc not in cn: continue
                    chunk[rc]=np.array(dh.data[rc],dtype=np.float64)
                    chunk[dc]=np.array(dh.data[dc],dtype=np.float64)
                    for k in ['pmra','pmdec','pmra_err','pmdec_err']:
                        c=cfg.MASTER_COLS.get(k)
                        if c and c in cn: chunk[c]=np.array(dh.data[c],dtype=np.float64)
                    dist_c=cfg.MASTER_COLS['dist']
                    chunk['best_dist']=np.array(dh.data[dist_c],dtype=np.float64) if dist_c in cn else np.full(nc,np.nan)
                    de_c=cfg.MASTER_COLS.get('dist_err')
                    chunk['best_dist_err']=np.array(dh.data[de_c],dtype=np.float64) if de_c and de_c in cn else np.full(nc,np.nan)
                    rv_c=self._find_col('rv',cn); rve_c=self._find_col('rv_err',cn)
                    chunk['best_rv']=np.array(dh.data[rv_c],dtype=np.float64) if rv_c else np.full(nc,np.nan)
                    chunk['best_rv_err']=np.array(dh.data[rve_c],dtype=np.float64) if rve_c else np.full(nc,np.nan)
                    pc=cfg.MASTER_COLS.get('parallax','parallax_final')
                    pec=cfg.MASTER_COLS.get('parallax_err','parallax_err_final')
                    if pc in cn:
                        chunk['plx_from_params']=np.array(dh.data[pc],dtype=np.float64)
                    else:
                        pe=cfg.MASTER_COLS.get('params_est')
                        if pe and pe in cn:
                            try: chunk['plx_from_params']=dh.data[pe][:,4].copy().astype(np.float64)
                            except: chunk['plx_from_params']=np.full(nc,np.nan)
                        else: chunk['plx_from_params']=np.full(nc,np.nan)
                    if pec in cn:
                        chunk['plx_err_from_params']=np.array(dh.data[pec],dtype=np.float64)
                    else:
                        pee=cfg.MASTER_COLS.get('params_err')
                        if pee and pee in cn:
                            try: chunk['plx_err_from_params']=dh.data[pee][:,4].copy().astype(np.float64)
                            except: chunk['plx_err_from_params']=np.full(nc,np.nan)
                        else: chunk['plx_err_from_params']=np.full(nc,np.nan)
                    for ec in self.EXTRA_COLS:
                        if ec in cn:
                            try: chunk[ec]=np.array(dh.data[ec],dtype=np.float64)
                            except: chunk[ec]=np.full(nc,np.nan)
                    cdf=pd.DataFrame(chunk).dropna(subset=[rc,dc])
                    all_dfs.append(cdf)
                    self.logger.info(f"    -> {len(cdf):,} rows")
            except Exception as e:
                self.logger.warning(f"    [ERROR] {e}"); continue
        if not all_dfs: self.logger.error("No data!"); return False
        self.df=pd.concat(all_dfs,ignore_index=True); del all_dfs; gcmod.collect()
        self.nrows=len(self.df)
        self.logger.info(f"  Combined: {self.nrows:,} rows")
        self._build_kdtree()
        if checkpoint_dir:
            os.makedirs(checkpoint_dir,exist_ok=True)
            self.df.to_parquet(data_cp); np.savez_compressed(tree_cp,coords=self.coords_3d)
        self.logger.info(f"Master loaded in {time.time()-t0:.1f}s"); gcmod.collect()
        return True

    def _resolve_files(self, fp):
        if '*' in fp or '?' in fp: return sorted(glob.glob(fp))
        if os.path.isdir(fp):
            f=sorted(glob.glob(os.path.join(fp,'Entire_catalogue_chunk*.fits')))
            return f if f else sorted(glob.glob(os.path.join(fp,'*.fits')))
        return [fp] if os.path.exists(fp) else []

    def _build_kdtree(self):
        ra=np.radians(self.df[cfg.MASTER_COLS['ra']].values)
        dec=np.radians(self.df[cfg.MASTER_COLS['dec']].values)
        self.coords_3d=np.column_stack([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)])
        self.tree=cKDTree(self.coords_3d); self._compute_max_chord()

    def _compute_max_chord(self):
        self.max_chord=2*np.sin(np.radians(cfg.CROSSMATCH_RADIUS_ARCSEC/3600.0)/2)

    def query(self, ra, dec):
        rr=np.radians(ra); dr=np.radians(dec)
        c=np.column_stack([np.cos(dr)*np.cos(rr),np.cos(dr)*np.sin(rr),np.sin(dr)])
        d,i=self.tree.query(c,k=1,distance_upper_bound=self.max_chord)
        v=np.isfinite(d)
        return i[v], np.where(v)[0], np.degrees(2*np.arcsin(d[v]/2))*3600

    def get_matched_data(self, idx):
        return self.df.iloc[idx].reset_index(drop=True)

# ============================================================================
# QUALITY FILTERS  (unchanged)
# ============================================================================
def apply_quality_flags(mdf, logger):
    n=len(mdf)
    pe=mdf.get(f"{cfg.MASTER_COLS['pmra_err']}_master",pd.Series(np.full(n,1.0))).values.astype(float)
    pde=mdf.get(f"{cfg.MASTER_COLS['pmdec_err']}_master",pd.Series(np.full(n,1.0))).values.astype(float)
    ruwe=mdf.get('RUWE_master',pd.Series(np.full(n,np.nan))).values.astype(float)
    bad_ruwe=np.isfinite(ruwe)&(ruwe>cfg.RUWE_GOOD_THRESHOLD)
    inf_pm=np.where(bad_ruwe,cfg.RUWE_PM_ERR_INFLATE,1.0)
    mdf['pmra_err_adj']=pe*inf_pm; mdf['pmdec_err_adj']=pde*inf_pm
    rve=mdf.get('best_rv_err_master',pd.Series(np.full(n,5.0))).values.astype(float)
    rve=np.where(np.isfinite(rve)&(rve>0),rve,5.0)
    rvsn=mdf.get('RVS/N_master',pd.Series(np.full(n,np.nan))).values.astype(float)
    bad_rv=np.isfinite(rvsn)&(rvsn<cfg.RV_SN_MIN)
    mdf['rv_err_adj']=rve*np.where(bad_rv,cfg.RV_ERR_INFLATE_BAD_SN,1.0)
    nb=int(np.sum(bad_ruwe)); nr=int(np.sum(bad_rv))
    if nb>0 or nr>0:
        logger.info(f"      [quality] RUWE>{cfg.RUWE_GOOD_THRESHOLD}:{nb} | RV S/N<{cfg.RV_SN_MIN}:{nr}")
    return mdf

# ============================================================================
# SPATIAL PROFILES  (unchanged)
# ============================================================================
def elliptical_plummer_pdf(ra_deg, dec_deg, ra0, dec0, rhalf_deg,
                            ellipticity=0.0, position_angle_deg=0.0, r_max_deg=None):
    ra=np.asarray(ra_deg,dtype=float); dec=np.asarray(dec_deg,dtype=float)
    ra0_r=np.radians(float(ra0)); dec0_r=np.radians(float(dec0))
    ra_r=np.radians(ra); dec_r=np.radians(dec)
    dra=ra_r-ra0_r; cd=np.cos(dec_r); sd=np.sin(dec_r)
    cd0=np.cos(dec0_r); sd0=np.sin(dec0_r)
    den=sd0*sd+cd0*cd*np.cos(dra); den=np.where(np.abs(den)<1e-12,1e-12,den)
    xi=np.degrees(cd*np.sin(dra)/den)
    eta=np.degrees((cd0*sd-sd0*cd*np.cos(dra))/den)
    eps=float(np.clip(ellipticity,0,0.95)); q=1-eps
    th=np.radians(float(position_angle_deg)); ct,st=np.cos(th),np.sin(th)
    u_=xi*ct+eta*st; v_=-xi*st+eta*ct
    r_ell=np.sqrt((u_/q)**2+v_**2)
    a=float(rhalf_deg)/np.sqrt(2**(2/3)-1)
    if r_max_deg is None or not np.isfinite(r_max_deg) or r_max_deg<=0:
        r_max_deg=max(np.nanmax(r_ell),5*rhalf_deg)
    R=float(r_max_deg)
    norm_v=(a**2+R**2)/(np.pi*a**2*R**2*q)
    pdf=norm_v/(1+(r_ell/a)**2)**2; pdf[r_ell>R]=0
    return pdf, r_ell

def uniform_field_pdf(r_max, q=1.0):
    return 1.0/(np.pi*r_max**2*q)

# ============================================================================
# CMD + SPECTROSCOPIC + PM + RV LIKELIHOODS  (unchanged)
# ============================================================================
def build_cmd_template(gmag, bprp, dist_mod, ag=None, ebprp=None):
    g=np.asarray(gmag,dtype=float); c=np.asarray(bprp,dtype=float)
    if ag is not None:
        a=np.asarray(ag,dtype=float); ok=np.isfinite(a); g=np.where(ok,g-a,g)
    if ebprp is not None:
        e=np.asarray(ebprp,dtype=float); ok=np.isfinite(e); c=np.where(ok,c-e,c)
    MG=g-float(dist_mod) if dist_mod is not None and np.isfinite(dist_mod) else g
    ok=np.isfinite(MG)&np.isfinite(c)
    if np.sum(ok)<cfg.CMD_MIN_MEMBERS_FOR_KDE: return None
    try: return gaussian_kde(np.vstack([MG[ok],c[ok]]),bw_method=cfg.CMD_KDE_BANDWIDTH)
    except: return None

def cmd_likelihood(gmag, bprp, dist_mod, ag=None, ebprp=None, mem_kde=None, fld_kde=None):
    n=len(gmag); Lm=np.ones(n); Lf=np.ones(n)
    g=np.asarray(gmag,dtype=float); c=np.asarray(bprp,dtype=float)
    if ag is not None:
        a=np.asarray(ag,dtype=float); ok=np.isfinite(a); g=np.where(ok,g-a,g)
    if ebprp is not None:
        e=np.asarray(ebprp,dtype=float); ok=np.isfinite(e); c=np.where(ok,c-e,c)
    MG=g-float(dist_mod) if dist_mod is not None and np.isfinite(dist_mod) else g
    ok=np.isfinite(MG)&np.isfinite(c)
    if not np.any(ok): return Lm,Lf
    pts=np.vstack([MG[ok],c[ok]])
    if mem_kde is not None:
        try: Lm[ok]=np.clip(mem_kde(pts),1e-300,None)
        except: pass
    if fld_kde is not None:
        try: Lf[ok]=np.clip(fld_kde(pts),1e-300,None)
        except: pass
    else: Lf[ok]=1e-2
    return Lm, Lf

def logg_likelihood(logg_arr, obj_type, logg_thr=None):
    n=len(logg_arr); Lm=np.ones(n); Lf=np.ones(n)
    if obj_type in ['OC','SGR','STREAM']: return Lm,Lf
    if logg_thr is None: logg_thr=cfg.LOGG_GIANT_THRESHOLD
    lg=np.asarray(logg_arr,dtype=float); ok=np.isfinite(lg)
    if not np.any(ok): return Lm,Lf
    excess=np.clip(lg[ok]-logg_thr,0,None)
    Lm[ok]=np.exp(-0.5*(excess/cfg.LOGG_GIANT_SIGMA)**2)
    Lf[ok]=0.3+0.7*np.exp(-0.5*((lg[ok]-4.2)/0.8)**2)
    return Lm, Lf

def feh_likelihood(feh_arr, feh_sys, feh_sys_sigma=None, feh_mem_sigma=None):
    n=len(feh_arr); Lm=np.ones(n); Lf=np.ones(n)
    if feh_sys is None or not np.isfinite(feh_sys): return Lm,Lf
    feh=np.asarray(feh_arr,dtype=float); ok=np.isfinite(feh)
    if not np.any(ok): return Lm,Lf
    if feh_mem_sigma is None: feh_mem_sigma=cfg.FEH_MEMBER_SIGMA_DEFAULT
    sig=max(feh_sys_sigma,0.05) if feh_sys_sigma and np.isfinite(feh_sys_sigma) and feh_sys_sigma>0 else feh_mem_sigma
    Lm[ok]=norm.pdf(feh[ok],loc=feh_sys,scale=sig)
    Lf[ok]=norm.pdf(feh[ok],loc=cfg.FEH_FIELD_MEAN,scale=cfg.FEH_FIELD_SIGMA)
    return Lm, Lf

def distance_likelihood(dist_arr, de_arr, rd, rde):
    n=len(dist_arr); Lm=np.ones(n); Lf=np.ones(n)
    if rd is None or not np.isfinite(rd) or rd<=0: return Lm,Lf
    d=np.asarray(dist_arr,dtype=float); de=np.asarray(de_arr,dtype=float)
    de=np.where(np.isfinite(de)&(de>0),de,rd*0.2); ok=np.isfinite(d)&(d>0)
    if not np.any(ok): return Lm,Lf
    rde_v=rde if rde and np.isfinite(rde) and rde>0 else max(rd*0.1,cfg.DIST_MEMBER_SIGMA_FLOOR)
    sig_m=np.clip(np.sqrt(rde_v**2+de[ok]**2),cfg.DIST_MEMBER_SIGMA_FLOOR,None)
    Lm[ok]=norm.pdf(d[ok],loc=rd,scale=sig_m)
    Lf[ok]=np.clip(norm.pdf(d[ok],loc=10.0,scale=max(rd*cfg.DIST_FIELD_SIGMA_FRAC,20.0)),1e-10,None)
    return Lm, Lf

def pm_likelihood_vec(pmra, pmdec, mu, Sig, pe, pde, pcorr=None):
    n=len(pmra); L=np.ones(n)
    pmra_v=np.asarray(pmra,dtype=float); pmdec_v=np.asarray(pmdec,dtype=float)
    pe_v=np.asarray(pe,dtype=float); pde_v=np.asarray(pde,dtype=float)
    mu_v=np.asarray(mu,dtype=float); S=np.asarray(Sig,dtype=float)
    ok=np.isfinite(pmra_v)&np.isfinite(pmdec_v)
    if not np.any(ok): return L
    idx=np.where(ok)[0]; dx=pmra_v[idx]-mu_v[0]; dy=pmdec_v[idx]-mu_v[1]
    pe_ok=np.where(np.isfinite(pe_v[idx]),pe_v[idx],0.0)
    pde_ok=np.where(np.isfinite(pde_v[idx]),pde_v[idx],0.0)
    rho=np.zeros(len(idx))
    if pcorr is not None:
        corr=np.asarray(pcorr,dtype=float)
        rho=np.where(np.isfinite(corr[idx]),corr[idx],0.0)
    s11=S[0,0]+pe_ok**2+cfg.COV_REGULARIZE
    s22=S[1,1]+pde_ok**2+cfg.COV_REGULARIZE
    s12=S[0,1]+rho*pe_ok*pde_ok
    det=np.clip(s11*s22-s12**2,1e-20,None)
    maha=dx**2*(s22/det)+2*dx*dy*(-s12/det)+dy**2*(s11/det)
    L[idx]=np.clip(np.exp(-0.5*maha)/(2*np.pi*np.sqrt(det)),1e-300,None)
    return L

def rv_likelihood(rv, rve, mu, sigma):
    rv_v=np.asarray(rv,dtype=float); rve_v=np.asarray(rve,dtype=float)
    n=len(rv_v); L=np.ones(n); ok=np.isfinite(rv_v)&np.isfinite(rve_v)
    if not np.any(ok): return L
    st=np.sqrt(float(sigma)**2+rve_v[ok]**2)
    L[ok]=np.clip(norm.pdf(rv_v[ok],loc=float(mu),scale=st),1e-300,None)
    return L

def mle_sigma_int(rv, rve, w, vsys, floor=None):
    if floor is None: floor=cfg.SIGMA_INT_FLOOR
    rv_v=np.asarray(rv,dtype=float); re_v=np.asarray(rve,dtype=float); wv=np.asarray(w,dtype=float)
    ok=np.isfinite(rv_v)&np.isfinite(re_v)&(wv>1e-6)
    if np.sum(ok)<3: return cfg.SIGMA_INT_INIT
    rvo=rv_v[ok]; reo=re_v[ok]; wo=wv[ok]
    def nll(si):
        s2=si**2+reo**2; return -np.sum(-0.5*wo*(np.log(2*np.pi*s2)+(rvo-vsys)**2/s2))
    try:
        r=minimize_scalar(nll,bounds=(floor,200),method='bounded'); return max(r.x,floor)
    except:
        W=np.sum(wo); return max(np.sqrt(max(np.sum(wo*(rvo-vsys)**2)/W-np.sum(wo*reo**2)/W,0)),floor)

# ============================================================================
# WEIGHTED 2D COV + BAYESIAN EM  (unchanged)
# ============================================================================
def _weighted_cov_2d(x, y, w):
    W=np.sum(w)
    if W<=0 or np.sum(w>0)<3: return np.nanmean(x),np.nanmean(y),np.eye(2)*(cfg.COV_REGULARIZE+0.1)
    mx=np.sum(w*x)/W; my=np.sum(w*y)/W
    dx=x-mx; dy=y-my
    C=np.array([[np.sum(w*dx*dx)/W,np.sum(w*dx*dy)/W],[np.sum(w*dx*dy)/W,np.sum(w*dy*dy)/W]])
    return mx, my, C+np.eye(2)*cfg.COV_REGULARIZE

def algorithm_bayesian_em(pmra, pmdec, cp, cd, pmra_err=None, pmdec_err=None, pm_corr=None,
        ra_deg=None, dec_deg=None, ra_center=None, dec_center=None,
        rhalf_deg=None, ellipticity=0.0, position_angle=0.0,
        rv_obs=None, rv_err_obs=None, rv_sys_prior=None, sigma_int_prior=None,
        dist_obs=None, dist_err_obs=None, ref_dist=None, ref_dist_err=None,
        gmag_obs=None, bprp_obs=None, ag_obs=None, ebprp_obs=None,
        dist_modulus=None, member_cmd_kde=None, field_cmd_kde=None,
        logg_obs=None, feh_obs=None, feh_sys=None, feh_sys_sigma=None,
        obj_type='GC', ms=10, logger=None):
    n=len(pmra); pmra=np.asarray(pmra,dtype=float); pmdec=np.asarray(pmdec,dtype=float)
    ok_pm=np.isfinite(pmra)&np.isfinite(pmdec)
    if np.sum(ok_pm)<ms:
        return np.full(n,0.5),{'status':'insufficient_data','algorithm':'BayesianEM-V10'}
    def _e(a): return np.asarray(a,dtype=float) if a is not None else np.full(n,np.nan)
    ra_deg=_e(ra_deg); dec_deg=_e(dec_deg); rv_obs=_e(rv_obs); rv_err_obs=_e(rv_err_obs)
    dist_obs=_e(dist_obs); dist_err_obs=_e(dist_err_obs)
    gmag_obs=_e(gmag_obs); bprp_obs=_e(bprp_obs); ag_obs=_e(ag_obs); ebprp_obs=_e(ebprp_obs)
    logg_obs=_e(logg_obs); feh_obs=_e(feh_obs)
    pmra_err=_e(pmra_err); pmdec_err=_e(pmdec_err)
    if pm_corr is not None: pm_corr=np.asarray(pm_corr,dtype=float)
    if sigma_int_prior is None: sigma_int_prior=cfg.SIGMA_INT_INIT
    use_spatial=(ra_center is not None and dec_center is not None
                 and np.isfinite(ra_center) and np.isfinite(dec_center))
    if rhalf_deg is None or not np.isfinite(rhalf_deg) or rhalf_deg<=0:
        rhalf_deg=cfg.RHALF_FALLBACK_DEG
    r_ell=np.full(n,np.nan); r_max=5*rhalf_deg
    if use_spatial:
        _,r_ell=elliptical_plummer_pdf(ra_deg,dec_deg,ra_center,dec_center,
                                        rhalf_deg,ellipticity,position_angle)
        rm=np.nanmax(r_ell)
        if np.isfinite(rm) and rm>0: r_max=rm
    ok_rv=np.isfinite(rv_obs)&np.isfinite(rv_err_obs); n_rv=int(np.sum(ok_rv))
    n_dist=int(np.sum(np.isfinite(dist_obs)&(dist_obs>0)))
    n_cmd=int(np.sum(np.isfinite(gmag_obs)&np.isfinite(bprp_obs)))
    n_feh=int(np.sum(np.isfinite(feh_obs))); n_logg=int(np.sum(np.isfinite(logg_obs)))
    mu_mem=np.array([cp,cd])
    if use_spatial and np.any(np.isfinite(r_ell)):
        inner=ok_pm&(r_ell<2*rhalf_deg)
        Sig_mem=_weighted_cov_2d(pmra[inner],pmdec[inner],np.ones(np.sum(inner)))[2] if np.sum(inner)>=5 else np.eye(2)*0.5
    else: Sig_mem=np.eye(2)*0.5
    outer=ok_pm&(r_ell>cfg.FIELD_INIT_RHALF_MULT*rhalf_deg) if use_spatial and np.any(np.isfinite(r_ell)) else ok_pm
    if np.sum(outer)<ms: outer=ok_pm
    if np.sum(outer)>=3:
        mx_f,my_f,Sig_field=_weighted_cov_2d(pmra[outer],pmdec[outer],np.ones(np.sum(outer)))
        mu_field=np.array([mx_f,my_f])
    else: mu_field=np.array([0,0]); Sig_field=np.eye(2)*4
    rv_sys=rv_sys_prior if rv_sys_prior is not None and np.isfinite(rv_sys_prior) else (float(np.median(rv_obs[ok_rv])) if n_rv>=5 else 0)
    sigma_int=float(sigma_int_prior)
    rv_field_mu=float(np.median(rv_obs[ok_rv])) if n_rv>=5 else 0
    rv_field_sig=max(float(median_abs_deviation(rv_obs[ok_rv],nan_policy='omit')*1.4826),cfg.FIELD_RV_SIGMA_INIT) if n_rv>=5 else cfg.FIELD_RV_SIGMA_INIT
    eta=cfg.ETA_INIT
    Ld_m,Ld_f=distance_likelihood(dist_obs,dist_err_obs,ref_dist,ref_dist_err)
    if member_cmd_kde: Lc_m,Lc_f=cmd_likelihood(gmag_obs,bprp_obs,dist_modulus,ag_obs,ebprp_obs,member_cmd_kde,field_cmd_kde)
    else: Lc_m=np.ones(n); Lc_f=np.ones(n)
    Lfeh_m,Lfeh_f=feh_likelihood(feh_obs,feh_sys,feh_sys_sigma)
    Llg_m,Llg_f=logg_likelihood(logg_obs,obj_type)
    static_m=Ld_m*Lc_m*Lfeh_m*Llg_m; static_f=Ld_f*Lc_f*Lfeh_f*Llg_f
    P=np.full(n,eta); logL_prev=-np.inf; q_ell=max(1-ellipticity,0.05)
    for em_it in range(cfg.EM_MAX_ITER):
        if use_spatial and np.any(np.isfinite(r_ell)):
            pm,_=elliptical_plummer_pdf(ra_deg,dec_deg,ra_center,dec_center,rhalf_deg,ellipticity,position_angle,r_max)
            pf=np.where(np.isfinite(r_ell),uniform_field_pdf(r_max,q_ell),1.0)
            pm=np.clip(pm,1e-300,None); pf=np.clip(pf,1e-300,None)
        else: pm=np.ones(n); pf=np.ones(n)
        Lpm_m=pm_likelihood_vec(pmra,pmdec,mu_mem,Sig_mem,pmra_err,pmdec_err,pm_corr)
        Lpm_f=pm_likelihood_vec(pmra,pmdec,mu_field,Sig_field,pmra_err,pmdec_err,pm_corr)
        Lrv_m=rv_likelihood(rv_obs,rv_err_obs,rv_sys,sigma_int)
        Lrv_f=rv_likelihood(rv_obs,rv_err_obs,rv_field_mu,rv_field_sig)
        num=eta*pm*Lpm_m*Lrv_m*static_m
        den=num+(1-eta)*pf*Lpm_f*Lrv_f*static_f
        den=np.where((den<=0)|~np.isfinite(den),1e-300,den)
        P_new=np.clip(np.where(ok_pm,num/den,np.nan),1e-6,1-1e-6)
        logL=np.nansum(np.log(np.clip(den[ok_pm],1e-300,None)))
        Pv=np.where(np.isfinite(P_new),P_new,0.0)
        eta=float(np.clip(np.mean(Pv[ok_pm]),0.01,0.99))
        wm=Pv.copy(); wm[~ok_pm]=0
        if np.sum(wm)>=3:
            mx,my,Sig_mem=_weighted_cov_2d(pmra,pmdec,wm); mu_mem=np.array([mx,my])
        if np.linalg.norm(mu_mem-np.array([cp,cd]))>3: mu_mem=0.5*mu_mem+0.5*np.array([cp,cd])
        wf=1-Pv; wf[~ok_pm]=0
        if np.sum(wf)>=3:
            mx,my,Sig_field=_weighted_cov_2d(pmra,pmdec,wf); mu_field=np.array([mx,my])
        if n_rv>=5:
            wr=Pv[ok_rv]; Wr=np.sum(wr); rvo=rv_obs[ok_rv]; reo=rv_err_obs[ok_rv]
            if Wr>0:
                st=np.sqrt(sigma_int**2+reo**2); iv=1/(st**2+1e-10)
                rv_sys=float(np.sum(wr*iv*rvo)/(np.sum(wr*iv)+1e-10))
                sigma_int=mle_sigma_int(rvo,reo,wr,rv_sys)
            wrf=(1-Pv)[ok_rv]; Wrf=np.sum(wrf)
            if Wrf>0:
                rv_field_mu=float(np.sum(wrf*rvo)/Wrf)
                rv_field_sig=max(float(np.sqrt(np.sum(wrf*(rvo-rv_field_mu)**2)/Wrf+1e-4)),cfg.FIELD_RV_SIGMA_INIT/2)
        dl=abs(logL-logL_prev)/(abs(logL_prev)+1e-10)
        if em_it>2 and dl<cfg.EM_CONVERGENCE_TOL: break
        logL_prev=logL; P=P_new
    Lpm_m=pm_likelihood_vec(pmra,pmdec,mu_mem,Sig_mem,pmra_err,pmdec_err,pm_corr)
    Lpm_f=pm_likelihood_vec(pmra,pmdec,mu_field,Sig_field,pmra_err,pmdec_err,pm_corr)
    Lrv_m=rv_likelihood(rv_obs,rv_err_obs,rv_sys,sigma_int)
    Lrv_f=rv_likelihood(rv_obs,rv_err_obs,rv_field_mu,rv_field_sig)
    if use_spatial and np.any(np.isfinite(r_ell)):
        pm,_=elliptical_plummer_pdf(ra_deg,dec_deg,ra_center,dec_center,rhalf_deg,ellipticity,position_angle,r_max)
        pf=np.where(np.isfinite(r_ell),uniform_field_pdf(r_max,q_ell),1.0)
        pm=np.clip(pm,1e-300,None); pf=np.clip(pf,1e-300,None)
    else: pm=np.ones(n); pf=np.ones(n)
    num=eta*pm*Lpm_m*Lrv_m*static_m
    den=num+(1-eta)*pf*Lpm_f*Lrv_f*static_f
    den=np.where((den<=0)|~np.isfinite(den),1e-300,den)
    Pf=np.where(ok_pm,np.clip(num/den,0,1),np.nan)
    nv=int(np.sum(ok_pm)); bic=-2*logL_prev+11*np.log(max(nv,1))
    info={'status':'success','algorithm':'BayesianEM-V10','obj_type':obj_type,
        'n_em_iterations':em_it+1,'eta':float(eta),
        'mu_cluster':mu_mem.tolist(),'Sigma_cluster':Sig_mem.tolist(),
        'mu_field':mu_field.tolist(),'Sigma_field':Sig_field.tolist(),
        'pm_dispersion':float(np.sqrt(np.trace(Sig_mem))),
        'center_pmra':cp,'center_pmdec':cd,
        'v_sys_recovered':float(rv_sys),'sigma_int_km_s':float(sigma_int),
        'rv_field_mu':float(rv_field_mu),'rv_field_sigma':float(rv_field_sig),
        'rhalf_deg_used':float(rhalf_deg),'ellipticity':float(ellipticity),
        'r_max_deg':float(r_max),'n_rv_used':n_rv,'n_dist_used':n_dist,
        'n_cmd_used':n_cmd,'n_feh_used':n_feh,'n_logg_used':n_logg,
        'use_spatial':bool(use_spatial),'logL_final':float(logL_prev),
        'BIC':float(bic),'ref_dist':ref_dist,'ref_dist_err':ref_dist_err,'feh_sys':feh_sys}
    return Pf, info

# ============================================================================
# LEGACY DBSCAN  (unchanged)
# ============================================================================
def algorithm_dbscan(pmra, pmdec, cp, cd, eps=0.25, minsamp=5, use_h=False, ms=10):
    n=len(pmra); vm=~(np.isnan(pmra)|np.isnan(pmdec)); nv=int(np.sum(vm))
    if nv<ms: return np.full(n,0.5),{'status':'insufficient_data','algorithm':'DBSCAN'}
    X=np.column_stack([pmra[vm],pmdec[vm]]); sc=StandardScaler(); Xs=sc.fit_transform(X)
    rs=sc.transform([[cp,cd]])[0]
    try:
        if use_h and HAS_HDBSCAN:
            cl=hdbscan.HDBSCAN(min_cluster_size=5,min_samples=3); labels=cl.fit_predict(Xs); an='HDBSCAN'
        else: cl=DBSCAN(eps=eps,min_samples=minsamp); labels=cl.fit_predict(Xs); an='DBSCAN'
        ul=set(labels)-{-1}
        if len(ul)==0:
            d=np.linalg.norm(Xs-rs,axis=1); P=np.exp(-d**2/2)
            Pm=np.full(n,np.nan); Pm[vm]=P
            return Pm,{'status':'no_cluster','algorithm':an,'center_pmra':cp,'center_pmdec':cd}
        bc=min(ul,key=lambda l:np.linalg.norm(Xs[labels==l].mean(axis=0)-rs))
        cm=labels==bc; P=np.zeros(nv)
        if use_h and HAS_HDBSCAN and hasattr(cl,'probabilities_'): P[cm]=cl.probabilities_[cm]
        else: P[cm]=1.0
        cc=Xs[cm].mean(axis=0); cs=max(Xs[cm].std(axis=0).mean(),0.1)
        nc2=~cm
        if np.any(nc2): P[nc2]=np.exp(-np.linalg.norm(Xs[nc2]-cc,axis=1)**2/(2*cs**2))*0.3
        Pm=np.full(n,np.nan); Pm[vm]=P
        cpm=X[cm]; mu=cpm.mean(axis=0) if len(cpm)>0 else np.array([cp,cd])
        Sig=np.cov(cpm.T) if len(cpm)>2 else np.eye(2)*0.1
        return Pm,{'status':'success','algorithm':an,'n_cluster_members':int(np.sum(cm)),
            'mu_cluster':mu.tolist(),'Sigma_cluster':Sig.tolist(),
            'pm_dispersion':float(np.sqrt(np.trace(Sig))) if len(cpm)>2 else np.nan,
            'eta':float(np.sum(cm)/nv),'center_pmra':cp,'center_pmdec':cd}
    except Exception as e: return np.full(n,0.5),{'status':f'error:{e}','algorithm':'DBSCAN'}

def algorithm_stream_dbscan(pmra, pmdec, cp, cd, eps=0.4, minsamp=3, ms=10):
    n=len(pmra); vm=~(np.isnan(pmra)|np.isnan(pmdec)); nv=int(np.sum(vm))
    if nv<ms: return np.full(n,0.5),{'status':'insufficient_data','algorithm':'Stream-DBSCAN'}
    X=np.column_stack([pmra[vm],pmdec[vm]]); sc=StandardScaler(); Xs=sc.fit_transform(X)
    rs=sc.transform([[cp,cd]])[0]
    try:
        db=DBSCAN(eps=eps,min_samples=minsamp); labels=db.fit_predict(Xs)
        ul=set(labels)-{-1}
        if len(ul)==0:
            d=np.linalg.norm(Xs-rs,axis=1); P=np.exp(-d**2/(2*eps**2))
            Pm=np.full(n,np.nan); Pm[vm]=P
            return Pm,{'status':'no_cluster','algorithm':'Stream-DBSCAN','center_pmra':cp,'center_pmdec':cd}
        bc=min(ul,key=lambda l:np.min(np.linalg.norm(Xs[labels==l]-rs,axis=1)))
        cm=labels==bc; cp2=Xs[cm]; P=np.zeros(nv)
        if len(cp2)>0:
            tr=cKDTree(cp2)
            for idx in np.where(cm)[0]:
                d,_=tr.query(Xs[idx],k=min(3,len(cp2)))
                P[idx]=np.clip(1-np.mean(d[1:] if len(d)>1 else [0])/eps,0.5,1.0)
        nc2=~cm
        if np.any(nc2):
            for idx in np.where(nc2)[0]:
                P[idx]=0.3*np.exp(-np.min(np.linalg.norm(cp2-Xs[idx],axis=1))**2/(2*eps**2))
        Pm=np.full(n,np.nan); Pm[vm]=P
        cpm=X[cm]; mu=cpm.mean(axis=0) if len(cpm)>0 else np.array([cp,cd])
        Sig=np.cov(cpm.T) if len(cpm)>2 else np.eye(2)*0.1
        return Pm,{'status':'success','algorithm':'Stream-DBSCAN',
            'n_stream_members':int(np.sum(cm)),'mu_cluster':mu.tolist(),'Sigma_cluster':Sig.tolist(),
            'pm_dispersion':float(np.sqrt(np.trace(Sig))) if len(cpm)>2 else np.nan,
            'eta':float(np.sum(cm)/nv),'center_pmra':cp,'center_pmdec':cd}
    except Exception as e: return np.full(n,0.5),{'status':f'error:{e}','algorithm':'Stream-DBSCAN'}

# ============================================================================
# DIAGNOSTICS + HELPERS  (unchanged)
# ============================================================================
def compute_diagnostics(mdf, P, rd, rrv, ot, algo, logger=None):
    diag={'quality_flag':'GOOD'}; P=np.asarray(P,dtype=float)
    hi=np.isfinite(P)&(P>cfg.P_MEM_PLOT_THRESHOLD); nh=int(np.sum(hi))
    diag['n_high_pmem']=nh
    if nh<cfg.SUMMARY_MIN_HIGH_PMEM: diag['quality_flag']='LOW_N'
    if rd and np.isfinite(rd) and rd>0 and 'best_dist' in mdf.columns:
        hd=mdf.loc[hi,'best_dist'].values; hd=hd[np.isfinite(hd)&(hd>0)]
        if len(hd)>=5:
            re=algo.get('ref_dist_err',rd*0.15)
            if re is None or not np.isfinite(re) or re<=0: re=rd*0.15
            s,p=kstest(hd,'norm',args=(rd,max(re,0.5)))
            diag.update({'ks_dist_stat':float(s),'ks_dist_pval':float(p),
                'dist_median':float(np.median(hd)),'dist_mad':float(median_abs_deviation(hd))})
            if p<cfg.SUMMARY_KS_ALPHA: diag['quality_flag']='KS_DIST_FAIL'
    if rrv and np.isfinite(rrv) and 'best_rv' in mdf.columns:
        hr=mdf.loc[hi,'best_rv'].values; hr=hr[np.isfinite(hr)]
        if len(hr)>=5:
            si=algo.get('sigma_int_km_s',20); si=si if si and np.isfinite(si) else 20
            s,p=kstest(hr,'norm',args=(rrv,max(si,5)))
            diag.update({'ks_rv_stat':float(s),'ks_rv_pval':float(p),
                'rv_median':float(np.median(hr)),'rv_mad':float(median_abs_deviation(hr))})
    if algo.get('use_spatial',False): diag['contamination_est']=1-algo.get('eta',0.5)
    return diag

def estimate_control_field_contamination(master, rac, decc, rh, im=None, om=None, logger=None):
    if im is None: im=cfg.CONTROL_FIELD_INNER_MULT
    if om is None: om=cfg.CONTROL_FIELD_OUTER_MULT
    ri=im*rh; ro=om*rh; cd=max(np.cos(np.radians(decc)),0.1)
    mdf=master.df; rc=cfg.MASTER_COLS['ra']; dc=cfg.MASTER_COLS['dec']
    box=(mdf[rc]>=rac-ro/cd)&(mdf[rc]<=rac+ro/cd)&(mdf[dc]>=decc-ro)&(mdf[dc]<=decc+ro)
    bdf=mdf[box]
    if len(bdf)==0: return None
    seps=angular_separation_deg(bdf[rc].values,bdf[dc].values,rac,decc)
    na=int(np.sum((seps>=ri)&(seps<=ro))); area=np.pi*(ro**2-ri**2)
    if area>0 and na>0:
        d=na/area
        if logger: logger.info(f"      [control] {ri:.3f}-{ro:.3f}: {na} stars, dens={d:.1f}/deg2")
        return d
    return None

def _standard_match_columns(mdf):
    mdf['pmra']=mdf.get(f"{cfg.MASTER_COLS['pmra']}_master",pd.Series(dtype=float))
    mdf['pmdec']=mdf.get(f"{cfg.MASTER_COLS['pmdec']}_master",pd.Series(dtype=float))
    mdf['ra']=mdf.get(f"{cfg.MASTER_COLS['ra']}_master",pd.Series(dtype=float))
    mdf['dec']=mdf.get(f"{cfg.MASTER_COLS['dec']}_master",pd.Series(dtype=float))
    mdf['best_dist']=mdf.get('best_dist_master',pd.Series(dtype=float))
    mdf['best_rv']=mdf.get('best_rv_master',pd.Series(dtype=float))
    return mdf

def _build_cmd_kde_from_members(mdf, cols, dm):
    gc=cols.get('gmag'); bc=cols.get('bp_rp')
    if gc is None or bc is None or gc not in mdf.columns or bc not in mdf.columns: return None
    return build_cmd_template(pd.to_numeric(mdf[gc],errors='coerce').values,
                              pd.to_numeric(mdf[bc],errors='coerce').values, dm)

def _build_field_cmd_kde(mdf, r_ell=None, rh=None):
    gm=mdf.get('Gmag_master',pd.Series(dtype=float)).values
    bp=mdf.get('BP-RP_master',pd.Series(dtype=float)).values
    ag=mdf.get('AG_master',pd.Series(dtype=float)).values
    eb=mdf.get('E(BP-RP)_master',pd.Series(dtype=float)).values
    if r_ell is not None and rh is not None:
        o=r_ell>cfg.CMD_FIELD_ANNULUS_MULT*rh
        if np.sum(o)>=cfg.CMD_MIN_MEMBERS_FOR_KDE:
            return build_cmd_template(gm[o],bp[o],None,ag[o],eb[o])
    ok=np.isfinite(gm)&np.isfinite(bp)
    if np.sum(ok)>=cfg.CMD_MIN_MEMBERS_FOR_KDE: return build_cmd_template(gm,bp,None,ag,eb)
    return None

def load_gc_reference_distances(fp, logger):
    if not fp or not os.path.exists(fp): logger.warning("GC_dist.csv not found"); return {}
    logger.info(f"Loading GC ref distances from {fp}")
    df=pd.read_csv(fp,encoding='utf-8-sig'); df.columns=df.columns.str.strip()
    gc_d={}
    def empty(v):
        if pd.isna(v): return True
        s=str(v).strip(); return s in ['','-','–','—','nan','NaN','N/A','n/a'] or len(s)==0
    for _,row in df.iterrows():
        name=str(row[cfg.GC_DIST_COLS['name']]).strip(); dist=err=np.nan
        try:
            v=row[cfg.GC_DIST_COLS['lit_dist']]
            if not empty(v):
                dist=float(v)
                ev=row[cfg.GC_DIST_COLS['lit_dist_err']]
                err=float(str(ev).replace('+','').strip()) if not empty(ev) else np.nan
        except: pass
        if not np.isfinite(dist):
            try:
                v=row[cfg.GC_DIST_COLS['mean_dist']]
                if not empty(v):
                    dist=float(v)
                    ev=row[cfg.GC_DIST_COLS['mean_dist_err']]
                    err=float(str(ev).replace('+','').strip()) if not empty(ev) else np.nan
            except: pass
        if np.isfinite(dist):
            nn=normalize_name(name)
            for k in [nn,name,name.lower(),name.upper()]: gc_d[k]=(dist,err)
    logger.info(f"  Loaded {len(gc_d)} distances")
    return gc_d

def _get_member_rv(mdf, cols):
    rc=cols.get('rv'); rec=cols.get('rv_err')
    rv=None
    if rc and rc in mdf.columns: rv=pd.to_numeric(mdf[rc],errors='coerce').values
    else:
        for a in ['RV_weighted_avg','RV_km_s','vlos','radial_velocity','RV']:
            if a in mdf.columns: rv=pd.to_numeric(mdf[a],errors='coerce').values; break
    rve=pd.to_numeric(mdf[rec],errors='coerce').values if rec and rec in mdf.columns else None
    return rv, rve

def _compute_rv_reference(rv_arr):
    if rv_arr is None: return None,None
    rv=np.asarray(rv_arr,dtype=float); rv=rv[np.isfinite(rv)]
    if len(rv)<3: return None,None
    return float(np.median(rv)),float(median_abs_deviation(rv,nan_policy='omit'))

# ============================================================================
# PROCESS GC / OC / SGR / DW  (unchanged from V11)
# ============================================================================
def _make_result(cn, ot, cdf, mdf, algo, cols, midx, rd, rde, rrv, rrve,
                 md, memd, mrv, memrv, plxd, pra, pdec, ppmra, ppmdec, diag):
    return {'cluster_name':cn,'obj_type':ot,'member_df':cdf,'matched_df':mdf,
            'algo_info':algo,'mem_cols':cols,'n_members':len(cdf),'n_matched':len(midx),
            'ref_dist':rd,'ref_dist_err':rde,'ref_rv':rrv,'ref_rv_err':rrve,
            'master_dist':md,'member_dist':memd,'master_rv':mrv,'member_rv':memrv,
            'plx_dist':plxd,'prematch_ra':pra,'prematch_dec':pdec,
            'prematch_pmra':ppmra,'prematch_pmdec':ppmdec,'diagnostics':diag}

def _run_em_and_build(cdf, mdf, mm, cols, c_pmra, c_pmdec, rd, rde, ref_rv, ref_rv_err,
                       dist_mod, obj_type, logger, ra_c=None, dec_c=None, rhalf=None,
                       ellip=0, pa=0, feh_sys=None, feh_sigma=None, sig_int_prior=None):
    mdf=_standard_match_columns(mdf); mdf=apply_quality_flags(mdf,logger)
    mcmd=_build_cmd_kde_from_members(cdf,cols,dist_mod); fcmd=_build_field_cmd_kde(mdf)
    kw=dict(pmra=mdf['pmra'].values,pmdec=mdf['pmdec'].values,cp=c_pmra,cd=c_pmdec,
        pmra_err=mdf['pmra_err_adj'].values,pmdec_err=mdf['pmdec_err_adj'].values,
        pm_corr=mdf.get('pmRApmDEcor_master',pd.Series(dtype=float)).values,
        rv_obs=mdf['best_rv'].values,rv_err_obs=mdf['rv_err_adj'].values,
        rv_sys_prior=ref_rv,sigma_int_prior=sig_int_prior or cfg.SIGMA_INT_INIT,
        dist_obs=mdf['best_dist'].values,
        dist_err_obs=mdf.get('best_dist_err_master',pd.Series(dtype=float)).values,
        ref_dist=rd,ref_dist_err=rde,
        gmag_obs=mdf.get('Gmag_master',pd.Series(dtype=float)).values,
        bprp_obs=mdf.get('BP-RP_master',pd.Series(dtype=float)).values,
        ag_obs=mdf.get('AG_master',pd.Series(dtype=float)).values,
        ebprp_obs=mdf.get('E(BP-RP)_master',pd.Series(dtype=float)).values,
        dist_modulus=dist_mod,member_cmd_kde=mcmd,field_cmd_kde=fcmd,
        feh_obs=mdf.get('feh_master',pd.Series(dtype=float)).values,
        feh_sys=feh_sys,feh_sys_sigma=feh_sigma,
        logg_obs=mdf.get('logg_master',pd.Series(dtype=float)).values,
        obj_type=obj_type,ms=cfg.MIN_STARS_FOR_ANALYSIS,logger=logger)
    if ra_c is not None:
        kw.update(ra_deg=mdf['ra'].values,dec_deg=mdf['dec'].values,
                  ra_center=ra_c,dec_center=dec_c,rhalf_deg=rhalf,
                  ellipticity=ellip,position_angle=pa)
    P,algo=algorithm_bayesian_em(**kw)
    mdf['P_mem']=P
    return mdf, P, algo

def process_gc_members(master, gc_dists, logger, epoch_mode='2016', eod=None):
    if not cfg.GC_MEMBERS_FILE or not os.path.exists(cfg.GC_MEMBERS_FILE): return []
    logger.info(f"\n{'─'*50}\nGLOBULAR CLUSTERS [{epoch_mode}] (V12)\n{'─'*50}")
    df=pd.read_csv(cfg.GC_MEMBERS_FILE); cols=cfg.GC_MEM_COLS; kc=cols['key']
    clusters=sorted(df[kc].unique()) if kc in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters"); results=[]
    for i,cn in enumerate(clusters):
        logger.info(f"\n  [{i+1}/{len(clusters)}] {cn} [{epoch_mode}]")
        cdf=df[df[kc]==cn].copy()
        if len(cdf)<cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra=pd.to_numeric(cdf[cols['ra']],errors='coerce').values
        dec=pd.to_numeric(cdf[cols['dec']],errors='coerce').values
        vm=np.isfinite(ra)&np.isfinite(dec); cdf=cdf[vm].reset_index(drop=True)
        ra=ra[vm]; dec=dec[vm]
        if len(ra)<cfg.MIN_STARS_FOR_ANALYSIS: continue
        raq,decq=_get_query_coords(cdf,cols,logger,epoch_mode)
        cpmra=pd.to_numeric(cdf[cols['pmra']],errors='coerce').median()
        cpmdec=pd.to_numeric(cdf[cols['pmdec']],errors='coerce').median()
        rd=rde=None; nn=normalize_name(cn)
        for nt in [nn,cn,cn.lower(),cn.upper()]:
            if nt in gc_dists: rd,rde=gc_dists[nt]; break
        dm=5*np.log10(max(rd,0.01)*1000)-5 if rd and np.isfinite(rd) else None
        algo={'status':'no_matches','algorithm':'None','center_pmra':cpmra,'center_pmdec':cpmdec}
        mdf=None; md=memd=mrv=memrv=plxd=np.array([]); rrv=rrve=None; diag={}
        pra=pd.to_numeric(cdf[cols['ra']],errors='coerce').values
        pdec=pd.to_numeric(cdf[cols['dec']],errors='coerce').values
        ppmra=_safe_col(cdf,cols.get('pmra')); ppmdec=_safe_col(cdf,cols.get('pmdec'))
        midx,memidx,seps=master.query(raq,decq)
        logger.info(f"    Members:{len(ra)} | Matched:{len(midx)} ({100*len(midx)/max(len(ra),1):.1f}%)")
        if len(midx)>=cfg.MIN_STARS_FOR_ANALYSIS:
            mm=cdf.iloc[memidx].reset_index(drop=True)
            mst=master.get_matched_data(midx); mst.columns=[f"{c}_master" for c in mst.columns]
            mdf=pd.concat([mm,mst],axis=1); mdf['xmatch_sep_arcsec']=seps
            rac=float(np.median(raq)); decc=float(np.median(decq))
            mrv_a,_=_get_member_rv(mm,cols); memrv=mrv_a if mrv_a is not None else np.array([])
            rrv,rrve=_compute_rv_reference(memrv)
            mdf,P,algo=_run_em_and_build(cdf,mdf,mm,cols,cpmra,cpmdec,rd,rde,rrv,rrve,dm,'GC',logger,
                                          ra_c=rac,dec_c=decc)
            md=mdf['best_dist'].values; mrv=mdf['best_rv'].values
            plxd=_compute_plx_distance(mm,mdf,cols)
            plx_a=_safe_col(mm,cols.get('parallax'))
            memd=np.where(plx_a>0,1.0/plx_a,np.nan)
            diag=compute_diagnostics(mdf,P,rd,rrv,'GC',algo,logger)
            nh=int(np.sum(np.where(np.isfinite(P),P,0)>cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"    EM: {algo.get('n_em_iterations',0)}it eta={algo.get('eta',0):.3f} n_hi={nh}")
        results.append(_make_result(cn,'GC',cdf,mdf,algo,cols,midx,rd,rde,rrv,rrve,
                                     md,memd,mrv,memrv,plxd,pra,pdec,ppmra,ppmdec,diag))
        gcmod.collect()
    return results

def process_oc_members(master, logger, epoch_mode='2016', eod=None):
    if not cfg.OC_MEMBERS_FILE or not os.path.exists(cfg.OC_MEMBERS_FILE): return []
    logger.info(f"\n{'─'*50}\nOPEN CLUSTERS [{epoch_mode}] (V12)\n{'─'*50}")
    df=pd.read_csv(cfg.OC_MEMBERS_FILE); cols=cfg.OC_MEM_COLS; kc=cols['key']
    clusters=sorted(df[kc].unique()) if kc in df.columns else ['ALL']
    logger.info(f"Found {len(clusters)} clusters"); results=[]
    for i,cn in enumerate(clusters):
        logger.info(f"\n  [{i+1}/{len(clusters)}] {cn} [{epoch_mode}]")
        cdf=df[df[kc]==cn].copy()
        if len(cdf)<cfg.MIN_STARS_FOR_ANALYSIS: continue
        ra=pd.to_numeric(cdf[cols['ra']],errors='coerce').values
        dec=pd.to_numeric(cdf[cols['dec']],errors='coerce').values
        vm=np.isfinite(ra)&np.isfinite(dec); cdf=cdf[vm].reset_index(drop=True)
        ra=ra[vm]; dec=dec[vm]
        if len(ra)<cfg.MIN_STARS_FOR_ANALYSIS: continue
        plx=pd.to_numeric(cdf[cols['parallax']],errors='coerce').values; vp=plx[plx>0]
        rd=float(np.mean(1.0/vp)) if len(vp)>0 else None
        rde=float(np.std(1.0/vp)) if len(vp)>0 else None
        dm=5*np.log10(max(rd,0.01)*1000)-5 if rd and np.isfinite(rd) else None
        raq,decq=_get_query_coords(cdf,cols,logger,epoch_mode)
        cpmra=pd.to_numeric(cdf[cols['pmra']],errors='coerce').median()
        cpmdec=pd.to_numeric(cdf[cols['pmdec']],errors='coerce').median()
        algo={'status':'no_matches','algorithm':'None','center_pmra':cpmra,'center_pmdec':cpmdec}
        mdf=None; md=memd=mrv=memrv=plxd=np.array([]); rrv=rrve=None; diag={}
        pra=pd.to_numeric(cdf[cols['ra']],errors='coerce').values
        pdec=pd.to_numeric(cdf[cols['dec']],errors='coerce').values
        ppmra=_safe_col(cdf,cols.get('pmra')); ppmdec=_safe_col(cdf,cols.get('pmdec'))
        midx,memidx,seps=master.query(raq,decq)
        logger.info(f"    Members:{len(ra)} | Matched:{len(midx)}")
        if len(midx)>=cfg.MIN_STARS_FOR_ANALYSIS:
            mm=cdf.iloc[memidx].reset_index(drop=True)
            mst=master.get_matched_data(midx); mst.columns=[f"{c}_master" for c in mst.columns]
            mdf=pd.concat([mm,mst],axis=1); mdf['xmatch_sep_arcsec']=seps
            mrv_a,_=_get_member_rv(mm,cols); memrv=mrv_a if mrv_a is not None else np.array([])
            rrv,rrve=_compute_rv_reference(memrv)
            mdf,P,algo=_run_em_and_build(cdf,mdf,mm,cols,cpmra,cpmdec,rd,rde,rrv,rrve,dm,'OC',logger)
            md=mdf['best_dist'].values; mrv=mdf['best_rv'].values
            plxd=_compute_plx_distance(mm,mdf,cols)
            memd=np.where(pd.to_numeric(mm[cols['parallax']],errors='coerce').values>0,
                          1.0/pd.to_numeric(mm[cols['parallax']],errors='coerce').values,np.nan)
            diag=compute_diagnostics(mdf,P,rd,rrv,'OC',algo,logger)
            nh=int(np.sum(np.where(np.isfinite(P),P,0)>cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"    EM: {algo.get('n_em_iterations',0)}it eta={algo.get('eta',0):.3f} n_hi={nh}")
        results.append(_make_result(cn,'OC',cdf,mdf,algo,cols,midx,rd,rde,rrv,rrve,
                                     md,memd,mrv,memrv,plxd,pra,pdec,ppmra,ppmdec,diag))
        gcmod.collect()
    return results

def process_sgr_members(master, logger, epoch_mode='2016', eod=None):
    if not cfg.SGR_MEMBERS_FILE or not os.path.exists(cfg.SGR_MEMBERS_FILE): return []
    logger.info(f"\n{'─'*50}\nSGR STREAM [{epoch_mode}] (V12 Bayesian EM)\n{'─'*50}")
    df=pd.read_csv(cfg.SGR_MEMBERS_FILE); cols=cfg.SGR_MEM_COLS
    dc=cols.get('dist','dist')
    if dc not in df.columns: logger.error(f"  SGR dist col '{dc}' missing!"); return []
    df[dc]=pd.to_numeric(df[dc],errors='coerce'); df=df.dropna(subset=[dc])
    be=np.arange(cfg.SGR_BIN_START_KPC,cfg.SGR_BIN_END_KPC+cfg.SGR_BIN_WIDTH_KPC,cfg.SGR_BIN_WIDTH_KPC)
    bl=[f'{be[j]:.0f}-{be[j+1]:.0f} kpc' for j in range(len(be)-1)]
    logger.info(f"  SGR bins: {len(bl)} from {cfg.SGR_BIN_START_KPC:.0f} to {cfg.SGR_BIN_END_KPC:.0f} kpc")
    df['dist_bin']=pd.cut(df[dc],bins=be,labels=bl,right=False); df=df.dropna(subset=['dist_bin'])
    bc=df['dist_bin'].value_counts()
    vb=bc[bc>=cfg.SGR_MIN_STARS_PER_BIN].index.tolist()
    bs=sorted([(b,df[df['dist_bin']==b][dc].mean()) for b in vb],key=lambda x:x[1])
    bins_list=[x[0] for x in bs]
    if not bins_list: logger.warning("  No valid SGR bins!"); return []
    logger.info(f"  Valid bins: {len(bins_list)}"); results=[]
    for i,bln in enumerate(bins_list):
        logger.info(f"\n  [{i+1}/{len(bins_list)}] SGR {bln} [{epoch_mode}]")
        bdf=df[df['dist_bin']==bln].copy()
        ra=pd.to_numeric(bdf[cols['ra']],errors='coerce').values
        dec=pd.to_numeric(bdf[cols['dec']],errors='coerce').values
        vm=np.isfinite(ra)&np.isfinite(dec); bdf=bdf[vm].reset_index(drop=True)
        ra=ra[vm]; dec=dec[vm]
        if len(ra)<cfg.MIN_STARS_FOR_ANALYSIS: continue
        raq,decq=_get_query_coords(bdf,cols,logger,epoch_mode)
        cpmra=pd.to_numeric(bdf[cols['pmra']],errors='coerce').median()
        cpmdec=pd.to_numeric(bdf[cols['pmdec']],errors='coerce').median()
        bp=bln.replace(' kpc','').split('-'); blo=float(bp[0]); bhi=float(bp[1])
        rd_sgr=(blo+bhi)/2; rde_sgr=(bhi-blo)/2
        dm_sgr=5*np.log10(max(rd_sgr,0.01)*1000)-5
        feh_a=_safe_col(bdf,cols.get('feh','FeH')); fv=feh_a[np.isfinite(feh_a)]
        feh_sys=float(np.median(fv)) if len(fv)>=3 else None
        mrv_f,_=_get_member_rv(bdf,cols); rrv,rrve=_compute_rv_reference(mrv_f)
        algo={'status':'no_matches','algorithm':'None'}
        mdf=None; md=memd=mrv=memrv=plxd=np.array([]); diag={}
        pra=pd.to_numeric(bdf[cols['ra']],errors='coerce').values
        pdec=pd.to_numeric(bdf[cols['dec']],errors='coerce').values
        ppmra=_safe_col(bdf,cols.get('pmra')); ppmdec=_safe_col(bdf,cols.get('pmdec'))
        midx,memidx,seps=master.query(raq,decq)
        logger.info(f"    Members:{len(ra)} | Matched:{len(midx)}")
        if len(midx)>=cfg.MIN_STARS_FOR_ANALYSIS:
            mm=bdf.iloc[memidx].reset_index(drop=True)
            mst=master.get_matched_data(midx); mst.columns=[f"{c}_master" for c in mst.columns]
            mdf=pd.concat([mm,mst],axis=1); mdf['xmatch_sep_arcsec']=seps
            mdf=_standard_match_columns(mdf); mdf=apply_quality_flags(mdf,logger)
            mcmd=_build_cmd_kde_from_members(bdf,cols,dm_sgr); fcmd=_build_field_cmd_kde(mdf)
            try:
                P,algo=algorithm_bayesian_em(
                    pmra=mdf['pmra'].values,pmdec=mdf['pmdec'].values,cp=cpmra,cd=cpmdec,
                    pmra_err=mdf['pmra_err_adj'].values,pmdec_err=mdf['pmdec_err_adj'].values,
                    pm_corr=mdf.get('pmRApmDEcor_master',pd.Series(dtype=float)).values,
                    ra_deg=None,dec_deg=None,ra_center=None,dec_center=None,rhalf_deg=None,
                    rv_obs=mdf['best_rv'].values,rv_err_obs=mdf['rv_err_adj'].values,
                    rv_sys_prior=rrv,sigma_int_prior=cfg.SIGMA_INT_INIT,
                    dist_obs=mdf['best_dist'].values,
                    dist_err_obs=mdf.get('best_dist_err_master',pd.Series(dtype=float)).values,
                    ref_dist=rd_sgr,ref_dist_err=rde_sgr,
                    gmag_obs=mdf.get('Gmag_master',pd.Series(dtype=float)).values,
                    bprp_obs=mdf.get('BP-RP_master',pd.Series(dtype=float)).values,
                    ag_obs=mdf.get('AG_master',pd.Series(dtype=float)).values,
                    ebprp_obs=mdf.get('E(BP-RP)_master',pd.Series(dtype=float)).values,
                    dist_modulus=dm_sgr,member_cmd_kde=mcmd,field_cmd_kde=fcmd,
                    feh_obs=mdf.get('feh_master',pd.Series(dtype=float)).values,
                    feh_sys=feh_sys,obj_type='SGR',ms=cfg.MIN_STARS_FOR_ANALYSIS,logger=logger)
                if algo.get('status')!='success': raise RuntimeError(algo.get('status'))
                logger.info(f"    SGR EM: {algo.get('n_em_iterations',0)}it eta={algo.get('eta',0):.3f} ref_d={rd_sgr:.1f}")
            except Exception as ee:
                logger.warning(f"    SGR EM failed ({ee}), fallback to Stream-DBSCAN")
                P,algo=algorithm_stream_dbscan(mdf['pmra'].values,mdf['pmdec'].values,
                    cpmra,cpmdec,cfg.DBSCAN_EPS_STREAM,cfg.DBSCAN_MIN_SAMPLES_STREAM,cfg.MIN_STARS_FOR_ANALYSIS)
            mdf['P_mem']=P; md=mdf['best_dist'].values; mrv=mdf['best_rv'].values
            memd=_safe_col(mm,dc); plxd=_compute_plx_distance(mm,mdf,cols)
            mrv_a,_=_get_member_rv(mm,cols); memrv=mrv_a if mrv_a is not None else np.array([])
            diag=compute_diagnostics(mdf,P,rd_sgr,rrv,'SGR',algo,logger)
        results.append(_make_result(bln,'SGR',bdf,mdf,algo,cols,midx,rd_sgr,rde_sgr,rrv,rrve,
                                     md,memd,mrv,memrv,plxd,pra,pdec,ppmra,ppmdec,diag))
        gcmod.collect()
    return results

def process_dwg_members(master, logger, epoch_mode='2016', eod=None):
    if not cfg.DWG_MEMBERS_FILE or not os.path.exists(cfg.DWG_MEMBERS_FILE): return []
    logger.info(f"\n{'─'*50}\nDWARF GALAXIES [{epoch_mode}] (V12)\n{'─'*50}")
    df=pd.read_csv(cfg.DWG_MEMBERS_FILE); cols=cfg.DWG_MEM_COLS; kc=cols['key']
    for c in [cols['ra'],cols['dec']]:
        if c not in df.columns: logger.error(f"  Col '{c}' missing!"); return []
    gals=df[kc].unique() if kc in df.columns else ['ALL']; results=[]
    for i,gn in enumerate(gals):
        logger.info(f"\n  [{i+1}/{len(gals)}] {gn} [{epoch_mode}]")
        gdf=df[df[kc]==gn].copy()
        if len(gdf)<1: continue
        gr=gdf.iloc[0]
        rd=_safe_float(gr,cols.get('distance')); rde=_safe_float(gr,cols.get('distance_err'))
        rrv=_safe_float(gr,cols.get('rv_ref')); rrve=_safe_float(gr,cols.get('rv_ref_err'))
        rh=_safe_float(gr,cols.get('rhalf')); el=_safe_float(gr,cols.get('ellipticity')) or 0
        pa=_safe_float(gr,cols.get('position_angle')) or 0
        feh_sys=_safe_float(gr,cols.get('metallicity'))
        feh_sig=_safe_float(gr,cols.get('metallicity_sigma'))
        sip=_safe_float(gr,cols.get('rv_sigma')) or cfg.SIGMA_INT_INIT
        dm=_safe_float(gr,cols.get('distance_modulus'))
        epmra=_safe_float(gr,cols.get('edr3_pmra')); epmdec=_safe_float(gr,cols.get('edr3_pmdec'))
        ra=pd.to_numeric(gdf[cols['ra']],errors='coerce').values
        dec=pd.to_numeric(gdf[cols['dec']],errors='coerce').values
        vm=np.isfinite(ra)&np.isfinite(dec); gdf=gdf[vm].reset_index(drop=True)
        ra=ra[vm]; dec=dec[vm]
        if len(ra)<cfg.MIN_STARS_FOR_ANALYSIS: continue
        raq,decq=_get_query_coords(gdf,cols,logger,epoch_mode)
        cpmra=epmra if epmra and np.isfinite(epmra) else (pd.to_numeric(gdf[cols['pmra']],errors='coerce').median() if cols['pmra'] in gdf.columns else 0)
        cpmdec=epmdec if epmdec and np.isfinite(epmdec) else (pd.to_numeric(gdf[cols['pmdec']],errors='coerce').median() if cols['pmdec'] in gdf.columns else 0)
        algo={'status':'no_matches','algorithm':'BayesianEM-V10'}
        mdf=None; md=memd=mrv=memrv=plxd=np.array([]); nplx=0; diag={}
        pra=pd.to_numeric(gdf[cols['ra']],errors='coerce').values
        pdec=pd.to_numeric(gdf[cols['dec']],errors='coerce').values
        ppmra=_safe_col(gdf,cols.get('pmra')); ppmdec=_safe_col(gdf,cols.get('pmdec'))
        midx,memidx,seps=master.query(raq,decq)
        logger.info(f"    Matched:{len(midx)}/{len(ra)}")
        if len(midx)>=cfg.MIN_STARS_FOR_ANALYSIS:
            mm=gdf.iloc[memidx].reset_index(drop=True)
            mst=master.get_matched_data(midx); mst.columns=[f"{c}_master" for c in mst.columns]
            mdf=pd.concat([mm,mst],axis=1); mdf['xmatch_sep_arcsec']=seps
            mdf=_standard_match_columns(mdf); mdf=apply_quality_flags(mdf,logger)
            pv=mdf.get('plx_from_params_master',pd.Series(dtype=float)).values.astype(float)
            pe=mdf.get('plx_err_from_params_master',pd.Series(dtype=float)).values.astype(float)
            plo=np.where(np.isfinite(pe),pv-cfg.PLX_FOREGROUND_SIGMA*pe,pv)
            fg=np.isfinite(pv)&np.isfinite(pe)&(plo>cfg.PLX_FOREGROUND_THRESHOLD)
            nplx=int(np.sum(fg))
            if nplx>0:
                logger.info(f"    Plx cut: removed {nplx} fg")
                mdf=mdf[~fg].reset_index(drop=True); mm=mm[~fg].reset_index(drop=True)
            if len(mdf)<cfg.MIN_STARS_FOR_ANALYSIS:
                results.append(_make_result(gn,'DW',gdf,None,
                    {'status':'too_few_after_plx','n_parallax_removed':nplx},cols,midx,
                    rd,rde,rrv,rrve,np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),
                    pra,pdec,ppmra,ppmdec,{})); continue
            rac=float(np.nanmedian(mdf['ra'].values)); decc=float(np.nanmedian(mdf['dec'].values))
            if rh and np.isfinite(rh):
                estimate_control_field_contamination(master,rac,decc,rh,logger=logger)
            fcmd=_build_field_cmd_kde(mdf,None,rh)
            P,algo=algorithm_bayesian_em(
                pmra=mdf['pmra'].values,pmdec=mdf['pmdec'].values,cp=cpmra,cd=cpmdec,
                pmra_err=mdf['pmra_err_adj'].values,pmdec_err=mdf['pmdec_err_adj'].values,
                pm_corr=mdf.get('pmRApmDEcor_master',pd.Series(dtype=float)).values,
                ra_deg=mdf['ra'].values,dec_deg=mdf['dec'].values,
                ra_center=rac,dec_center=decc,rhalf_deg=rh,ellipticity=el,position_angle=pa,
                rv_obs=mdf['best_rv'].values,rv_err_obs=mdf['rv_err_adj'].values,
                rv_sys_prior=rrv,sigma_int_prior=sip,
                dist_obs=mdf['best_dist'].values,
                dist_err_obs=mdf.get('best_dist_err_master',pd.Series(dtype=float)).values,
                ref_dist=rd,ref_dist_err=rde,
                gmag_obs=mdf.get('Gmag_master',pd.Series(dtype=float)).values,
                bprp_obs=mdf.get('BP-RP_master',pd.Series(dtype=float)).values,
                ag_obs=mdf.get('AG_master',pd.Series(dtype=float)).values,
                ebprp_obs=mdf.get('E(BP-RP)_master',pd.Series(dtype=float)).values,
                dist_modulus=dm,member_cmd_kde=None,field_cmd_kde=fcmd,
                logg_obs=mdf.get('logg_master',pd.Series(dtype=float)).values,
                feh_obs=mdf.get('feh_master',pd.Series(dtype=float)).values,
                feh_sys=feh_sys,feh_sys_sigma=feh_sig,
                obj_type='DW',ms=cfg.MIN_STARS_FOR_ANALYSIS,logger=logger)
            algo['n_parallax_removed']=nplx; mdf['P_mem']=P
            md=mdf['best_dist'].values; mrv=mdf['best_rv'].values
            plxd=_compute_plx_distance(mm,mdf,cols)
            rc=cols.get('rv','RV_km_s')
            memrv=_safe_col(mm,rc) if rc in mm.columns else np.array([])
            diag=compute_diagnostics(mdf,P,rd,rrv,'DW',algo,logger)
            nh=int(np.sum(np.where(np.isfinite(P),P,0)>cfg.P_MEM_PLOT_THRESHOLD))
            logger.info(f"    EM: {algo.get('n_em_iterations',0)}it eta={algo.get('eta',0):.3f} "
                        f"v_sys={algo.get('v_sys_recovered',0):.1f} sig={algo.get('sigma_int_km_s',0):.1f} n_hi={nh}")
        results.append(_make_result(gn,'DW',gdf,mdf,algo,cols,midx,rd,rde,rrv,rrve,
                                     md,memd,mrv,memrv,plxd,pra,pdec,ppmra,ppmdec,diag))
        gcmod.collect()
    return results

# ============================================================================
# V12: PLOTTING HELPERS  (MODIFIED)
# ============================================================================
def _clean(arr, plo=1, phi=99, min_n=3):
    a=np.asarray(arr,dtype=float); a=a[np.isfinite(a)]
    if len(a)<min_n: return a
    lo,hi=np.percentile(a,[plo,phi]); return a[(a>=lo)&(a<=hi)]

def _safe_kde(ax, d, bins, color, ls='--', alpha=0.7):
    d2=d[np.isfinite(d)]
    if len(d2)<4: return
    try:
        kde=gaussian_kde(d2); x=np.linspace(bins[0],bins[-1],200)
        ax.plot(x,kde(x)*len(d2)*(bins[1]-bins[0]),color=color,ls=ls,alpha=alpha,lw=2)
    except: pass

def _med_mad_box(ax, data, color, loc='upper right', prefix=''):
    d=data[np.isfinite(data)]
    if len(d)<2: return
    med=np.median(d); mad=median_abs_deviation(d,nan_policy='omit')
    txt=f"{prefix}Med={med:.2f}\nMAD={mad:.2f}\nN={len(d)}"
    locs={'upper right':(0.97,0.97,'right','top'),'upper left':(0.03,0.97,'left','top'),
          'lower right':(0.97,0.03,'right','bottom'),'lower left':(0.03,0.03,'left','bottom')}
    x,y,ha,va=locs.get(loc,(0.97,0.97,'right','top'))
    ax.text(x,y,txt,transform=ax.transAxes,fontsize=9,fontweight='bold',ha=ha,va=va,family='monospace',
            bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor=color,alpha=0.9,lw=1.5))

def _hist_step_kde(ax, data, color, label, br=None, nb=30):
    """V12: Step histogram + KDE overlay."""
    d=_clean(data)
    if len(d)<3: return np.array([]),np.array([])
    if br is None:
        lo,hi=np.percentile(d,[0.5,99.5]); rng=max(hi-lo,1); br=[lo-0.1*rng,hi+0.1*rng]
    c_,b_,_=ax.hist(d,bins=nb,range=br,alpha=0)
    ax.step(b_[:-1],c_,where='post',color=color,lw=2.5,label=label)
    _safe_kde(ax,d,b_,color); ax.axvline(np.median(d),color=color,ls=':',lw=2.5,alpha=0.9)
    return c_,b_

def _bins_range_with_ref(arrays, ref=None, pos=False):
    all_d=[_clean(a) for a in arrays if a is not None and len(a)>0]
    if pos: all_d=[a[(a>0)&(a<300)] for a in all_d]
    all_d=[a for a in all_d if len(a)>0]
    if not all_d and ref is None: return [0,100]
    if not all_d: rng=max(abs(ref)*0.3,10); return [ref-rng,ref+rng]
    comb=np.concatenate(all_d); lo,hi=np.percentile(comb,[1,99]); rng=max(hi-lo,1)
    if ref is not None and np.isfinite(ref):
        lo=min(lo,ref-0.05*rng); hi=max(hi,ref+0.05*rng); rng=max(hi-lo,1)
    pad=0.15*rng; blo=max(0,lo-pad) if pos else lo-pad; return [blo,hi+pad]

def _filter_by_pmem(mdf, col, thr=None):
    if thr is None: thr=cfg.P_MEM_PLOT_THRESHOLD
    if col not in mdf.columns: return np.array([]),np.array([])
    all_a=mdf[col].values.astype(float)
    if 'P_mem' not in mdf.columns: return all_a,all_a
    pm=mdf['P_mem'].values.astype(float); m=(pm>=thr)&np.isfinite(pm)
    return all_a[m],all_a

def _draw_median_mad_lines(ax, data, color, zorder=8):
    """V12: DASHED median + more opaque MAD band."""
    d=np.asarray(data,dtype=float); d=d[np.isfinite(d)]
    if len(d)<3: return
    med=np.median(d); mad=median_abs_deviation(d,nan_policy='omit')
    ax.axvline(med,color=color,ls=cfg.MEDIAN_LS,lw=cfg.MEDIAN_LW,alpha=0.9,zorder=zorder)
    if mad>0: ax.axvspan(med-mad,med+mad,alpha=cfg.MAD_ALPHA,color=color,zorder=zorder-1)

def _draw_ref_band(ax, rv, re, zorder=20):
    """V12: Reference band ON TOP, prominent."""
    if rv is None or not np.isfinite(rv): return
    ax.axvline(rv,color='black',lw=cfg.REF_LW,alpha=cfg.REF_ALPHA,
               zorder=zorder,label=f'Ref={rv:.1f}')
    if re is not None and np.isfinite(re) and re>0:
        ax.axvspan(rv-re,rv+re,alpha=cfg.REF_BAND_ALPHA,color='gray',zorder=zorder-1)

def _draw_all_matched_hist(ax, data, br, nb=25, zorder=3):
    """V12: Red all-matched histogram — more visible."""
    d=_clean(data); d=d[(d>-9999)]
    if len(d)<3: return
    c_,b_,_=ax.hist(d,bins=nb,range=br,alpha=0)
    ax.step(b_[:-1],c_,where='post',color=cfg.COL_MASTER,
            lw=cfg.COL_MASTER_LW,alpha=cfg.COL_MASTER_ALPHA,ls='--',
            label=f'All matched (n={len(d)})',zorder=zorder)

def _draw_all_matched_hist_pos(ax, data, br, nb=25, zorder=3):
    """V12: Red all-matched for positive data (dist)."""
    d=_clean(data); d=d[(d>0)&(d<300)]
    if len(d)<3: return
    c_,b_,_=ax.hist(d,bins=nb,range=br,alpha=0)
    ax.step(b_[:-1],c_,where='post',color=cfg.COL_MASTER,
            lw=cfg.COL_MASTER_LW,alpha=cfg.COL_MASTER_ALPHA,ls='--',
            label=f'All matched (n={len(d)})',zorder=zorder)

# ============================================================================
# V12: INDIVIDUAL PLOTS (4-panel: PM, Sky, Distance, RV)
# ============================================================================
def plot_individual_panels(mdf, name, ot, algo, rd=None, rde=None, rrv=None, rrve=None,
        md=None, memd=None, mrv_a=None, memrv=None, plx_dist=None,
        pra=None, pdec=None, ppmra=None, ppmdec=None,
        save_dir=None, epoch_mode='2016', nplx=0, diag=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    set_paper_style()
    if save_dir is None: save_dir=os.path.join(cfg.OUTPUT_DIR,'individual_plots')
    os.makedirs(save_dir,exist_ok=True)
    thr=cfg.P_MEM_PLOT_THRESHOLD; et='[J2000->2016]' if epoch_mode=='2000' else '[J2016]'
    fig,axes=plt.subplots(2,2,figsize=(16,14)); af=axes.flatten()
    title=f'{ot}: {name} {et}'
    if rd and np.isfinite(rd): title+=f' ($d_{{ref}}$={rd:.1f} kpc)'
    fig.suptitle(title,fontsize=16,fontweight='bold',family='serif',y=0.98)
    hid,alld=_filter_by_pmem(mdf,'best_dist',thr); hirv,allrv=_filter_by_pmem(mdf,'best_rv',thr)
    # ── Panel 0: PM ──
    ax=af[0]
    if ppmra is not None and ppmdec is not None:
        vm=np.isfinite(ppmra)&np.isfinite(ppmdec)
        if np.sum(vm)>0:
            ax.scatter(ppmra[vm],ppmdec[vm],c=cfg.COL_GRAY_BG,s=cfg.GRAY_BG_SIZE,
                alpha=cfg.GRAY_BG_ALPHA,zorder=1,edgecolors='none',label=f'All mem (n={int(np.sum(vm))})')
    if 'P_mem' in mdf.columns:
        v=mdf['P_mem'].notna()
        sc=ax.scatter(mdf.loc[v,'pmra'],mdf.loc[v,'pmdec'],c=mdf.loc[v,'P_mem'],
            cmap=cfg.CMAP_PMEM,s=25,edgecolors='k',lw=0.3,vmin=0,vmax=1,zorder=5)
        plt.colorbar(sc,ax=ax).set_label('$P_{mem}$',fontsize=14,fontweight='bold')
    if algo.get('Sigma_cluster') and ot not in ['SGR','STREAM']:
        mu=algo['mu_cluster']; sig=np.array(algo['Sigma_cluster'])
        w,v2=np.linalg.eigh(sig); ang=np.degrees(np.arctan2(v2[1,0],v2[0,0]))
        for ns in [1,2]:
            ell=Ellipse(xy=mu,width=2*ns*np.sqrt(max(w[0],0)),height=2*ns*np.sqrt(max(w[1],0)),
                angle=ang,fill=False,ec='lime',lw=2.5 if ns==1 else 2,ls='-' if ns==1 else '--')
            ax.add_patch(ell)
        ax.scatter(*mu,marker='x',s=120,c='lime',linewidths=3,zorder=10)
    ax.set_xlabel(r'$\mu_{\alpha}\cos\delta$ (mas yr$^{-1}$)',fontsize=16,fontweight='bold')
    ax.set_ylabel(r'$\mu_{\delta}$ (mas yr$^{-1}$)',fontsize=16,fontweight='bold')
    ax.set_title('Proper Motion',fontsize=18,fontweight='bold'); ax.grid(True,alpha=0.2); ax.set_aspect('equal','datalim')
    # ── Panel 1: Sky ──
    ax=af[1]
    if pra is not None and pdec is not None:
        vm=np.isfinite(pra)&np.isfinite(pdec)
        if np.sum(vm)>0:
            ax.scatter(pra[vm],pdec[vm],c=cfg.COL_GRAY_BG,s=cfg.GRAY_BG_SIZE,
                alpha=cfg.GRAY_BG_ALPHA,zorder=1,edgecolors='none',label=f'All mem (n={int(np.sum(vm))})')
    if 'P_mem' in mdf.columns:
        v=mdf['P_mem'].notna()
        sc=ax.scatter(mdf.loc[v,'ra'],mdf.loc[v,'dec'],c=mdf.loc[v,'P_mem'],
            cmap=cfg.CMAP_PMEM,s=25,edgecolors='k',lw=0.3,vmin=0,vmax=1,zorder=5)
        plt.colorbar(sc,ax=ax).set_label('$P_{mem}$',fontsize=14,fontweight='bold')
    ax.set_xlabel('RA (deg)',fontsize=16,fontweight='bold')
    ax.set_ylabel('Dec (deg)',fontsize=16,fontweight='bold')
    ax.set_title('Sky Position',fontsize=18,fontweight='bold'); ax.invert_xaxis(); ax.grid(True,alpha=0.2)
    # ── Panel 2: Distance ── V12: member + master + plx
    ax=af[2]
    hd=_clean(hid); hd=hd[(hd>0)&(hd<300)]; ad=_clean(alld); ad=ad[(ad>0)&(ad<300)]
    plxc=np.array([])
    if plx_dist is not None and len(plx_dist)>0:
        plxc=_clean(plx_dist); plxc=plxc[(plxc>0)&(plxc<300)]
    # V12: also plot member catalog distances if available
    memd_c=np.array([])
    if memd is not None and len(memd)>0:
        memd_c=_clean(memd); memd_c=memd_c[(memd_c>0)&(memd_c<300)]
    br=_bins_range_with_ref([ad,hd,plxc,memd_c],rd,pos=True)
    # Red all-matched (more visible)
    _draw_all_matched_hist_pos(ax,alld,br,25,zorder=3)
    # Green high-P
    if len(hd)>=3:
        _hist_step_kde(ax,hd,cfg.COL_HIGHMEM,f'P>{thr:.1f} (n={len(hd)})',br,25)
        _draw_median_mad_lines(ax,hd,cfg.COL_HIGHMEM,zorder=8)
        _med_mad_box(ax,hd,cfg.COL_HIGHMEM,'upper right','Green ')
    # Navy blue parallax
    if len(plxc)>=3:
        _hist_step_kde(ax,plxc,cfg.COL_PLX,f'Parallax dist (n={len(plxc)})',br,25)
        _draw_median_mad_lines(ax,plxc,cfg.COL_PLX,zorder=7)
        _med_mad_box(ax,plxc,cfg.COL_PLX,'upper left','Plx ')
    # Reference band on top
    _draw_ref_band(ax,rd,rde,zorder=20)
    if diag and diag.get('ks_dist_pval') is not None:
        pv=diag['ks_dist_pval']
        ax.text(0.03,0.03,f"KS p={pv:.3f}",transform=ax.transAxes,fontsize=9,
            color='green' if pv>0.05 else 'red',fontweight='bold',
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
    ax.set_xlabel('Distance (kpc)',fontsize=16,fontweight='bold')
    ax.set_ylabel('Count',fontsize=16,fontweight='bold')
    ax.set_title('Distance Distribution',fontsize=18,fontweight='bold')
    ax.legend(fontsize=9,loc='best'); ax.grid(True,alpha=0.2,axis='y')
    # ── Panel 3: RV ── V12: member + master
    ax=af[3]; hrc=_clean(hirv); arc=_clean(allrv)
    # V12: also member catalog RV
    memrv_c=np.array([])
    if memrv is not None and len(memrv)>0:
        memrv_c=np.asarray(memrv,dtype=float); memrv_c=memrv_c[np.isfinite(memrv_c)]
    if len(hrc)>=3 or len(arc)>=3 or len(memrv_c)>=3:
        br=_bins_range_with_ref([arc,hrc,memrv_c],rrv)
        # Red all-matched (more visible)
        _draw_all_matched_hist(ax,allrv,br,25,zorder=3)
        # Green high-P
        if len(hrc)>=3:
            _hist_step_kde(ax,hrc,cfg.COL_HIGHMEM,f'P>{thr:.1f} (n={len(hrc)})',br,25)
            _draw_median_mad_lines(ax,hrc,cfg.COL_HIGHMEM,zorder=8)
            _med_mad_box(ax,hrc,cfg.COL_HIGHMEM,'upper right','Green ')
        # Navy blue member-catalog RV (if distinct from master)
        if len(memrv_c)>=3:
            _hist_step_kde(ax,memrv_c,cfg.COL_PLX,f'Member RV (n={len(memrv_c)})',br,25)
            _draw_median_mad_lines(ax,memrv_c,cfg.COL_PLX,zorder=7)
        # Reference band on top
        _draw_ref_band(ax,rrv,rrve,zorder=20)
        vs=algo.get('v_sys_recovered',np.nan)
        if np.isfinite(vs):
            ax.axvline(vs,color='darkblue',lw=2.5,ls='-.',alpha=0.9,label=f'$v_{{sys}}$={vs:.1f}',zorder=15)
        if diag and diag.get('ks_rv_pval') is not None:
            pv=diag['ks_rv_pval']
            ax.text(0.03,0.03,f"KS p={pv:.3f}",transform=ax.transAxes,fontsize=9,
                color='green' if pv>0.05 else 'red',fontweight='bold',
                bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
        ax.set_xlabel('RV (km s$^{-1}$)',fontsize=16,fontweight='bold')
        ax.set_ylabel('Count',fontsize=16,fontweight='bold')
        ax.set_title('Radial Velocity',fontsize=18,fontweight='bold')
        ax.legend(fontsize=9,loc='best'); ax.grid(True,alpha=0.2,axis='y')
    else:
        ax.text(0.5,0.5,'No RV data',ha='center',va='center',transform=ax.transAxes,fontsize=16,color='gray')
        ax.set_title('Radial Velocity',fontsize=18,fontweight='bold'); ax.axis('off')
    plt.tight_layout(rect=[0,0,1,0.95])
    safe=str(name).replace(' ','_').replace('/','_').replace('-','_')
    out=os.path.join(save_dir,f"{ot}_{safe}.{cfg.SAVE_FORMAT}")
    plt.savefig(out,dpi=cfg.PLOT_DPI); plt.close(); return out

# ============================================================================
# V12: SMART GOLD SAMPLE FILTER
# ============================================================================
def _gold_sample(all_results):
    """V12: >=15 green primary, relax to >=10 per type if type has <5 passing."""
    thr=cfg.P_MEM_PLOT_THRESHOLD
    primary=cfg.SUMMARY_MIN_HIGH_PMEM        # 15
    relaxed=cfg.SUMMARY_MIN_HIGH_PMEM_RELAX  # 10
    min_per_type=cfg.SUMMARY_MIN_PER_TYPE     # 5

    # First pass: count green for each valid result
    scored=[]
    for r in all_results:
        if r['matched_df'] is None: continue
        if r['n_matched']<cfg.MIN_SUMMARY_MATCH: continue
        if 'P_mem' not in r['matched_df'].columns: continue
        P=r['matched_df']['P_mem'].values
        nh=int(np.sum(np.where(np.isfinite(P),P,0)>thr))
        r['_n_hi']=nh; scored.append(r)

    # Second pass: per type, try primary threshold; relax if needed
    gold=[]
    types_seen={}
    for r in scored: types_seen.setdefault(r['obj_type'],[]).append(r)
    for otype,rlist in types_seen.items():
        primary_pass=[r for r in rlist if r['_n_hi']>=primary]
        if len(primary_pass)>=min_per_type:
            gold.extend(primary_pass)
        else:
            # Relax to >=10
            relaxed_pass=[r for r in rlist if r['_n_hi']>=relaxed]
            gold.extend(relaxed_pass)

    to={'GC':0,'OC':1,'DW':2,'SGR':3}
    gold.sort(key=lambda x:(to.get(x['obj_type'],99),-x['_n_hi']))
    return gold

# ============================================================================
# V12: PAGINATED 6×6 SUMMARY PANELS
# ============================================================================
def _paginated_summary(gold, panel_func, series_name, logger, out_dir, em):
    """Generic paginator: splits gold into 6×6 pages, calls panel_func per page."""
    import matplotlib.pyplot as plt
    set_paper_style()
    nc=cfg.SUMMARY_NCOLS; nr=cfg.SUMMARY_NROWS; per_page=nc*nr
    n=len(gold); npages=max(1,int(np.ceil(n/per_page)))
    et='[J2000\u21922016]' if em=='2000' else '[J2016]'
    thr=cfg.P_MEM_PLOT_THRESHOLD
    tc={'GC':'#006400','OC':'#FF8C00','DW':'#00008B','SGR':'#8B0000'}
    for pg in range(npages):
        i0=pg*per_page; i1=min(i0+per_page,n); batch=gold[i0:i1]
        nb=len(batch)
        act_rows=int(np.ceil(nb/nc)); act_cols=min(nc,nb)
        fig,axes=plt.subplots(act_rows,act_cols,
            figsize=(cfg.SUMMARY_PANEL_W*act_cols, cfg.SUMMARY_PANEL_H*act_rows),squeeze=False)
        fig.suptitle(f'V12 {series_name} Summary (P$_{{mem}}$>{thr:.1f}) {et}  '
                     f'[page {pg+1}/{npages}]',fontsize=16,fontweight='bold')
        for j,r in enumerate(batch):
            ax=axes[j//act_cols, j%act_cols]
            panel_func(ax,r,thr,tc)
        for j in range(nb,act_rows*act_cols):
            axes[j//act_cols, j%act_cols].set_visible(False)
        plt.tight_layout(rect=[0,0,1,0.96])
        sfx=f'_{pg+1}' if npages>1 else ''
        plt.savefig(os.path.join(out_dir,f'SUMMARY_{series_name}{sfx}.{cfg.SAVE_FORMAT}'),dpi=cfg.PLOT_DPI)
        plt.close()
    logger.info(f"  Saved: SUMMARY_{series_name} ({npages} page(s), {n} panels)")

# ── panel functions for each series ──

def _panel_dist(ax, r, thr, tc):
    mdf=r['matched_df']; rd=r.get('ref_dist'); rde=r.get('ref_dist_err')
    ot=r['obj_type']; cn=r['cluster_name']
    hid,alld=_filter_by_pmem(mdf,'best_dist',thr)
    hd=_clean(hid); hd=hd[(hd>0)&(hd<300)]; ad=_clean(alld); ad=ad[(ad>0)&(ad<300)]
    plxd=r.get('plx_dist',np.array([])); plxc=np.array([])
    if plxd is not None and len(plxd)>0: plxc=_clean(plxd); plxc=plxc[(plxc>0)&(plxc<300)]
    br=_bins_range_with_ref([ad,hd,plxc],rd,pos=True)
    _draw_all_matched_hist_pos(ax,alld,br,20,zorder=3)
    if len(hd)>=3:
        _hist_step_kde(ax,hd,cfg.COL_HIGHMEM,f'P>{thr:.1f} (n={len(hd)})',br,20)
        _draw_median_mad_lines(ax,hd,cfg.COL_HIGHMEM,zorder=8)
    if len(plxc)>=3:
        _hist_step_kde(ax,plxc,cfg.COL_PLX,f'Parallax (n={len(plxc)})',br,20)
        _draw_median_mad_lines(ax,plxc,cfg.COL_PLX,zorder=7)
    _draw_ref_band(ax,rd,rde,zorder=20)
    ax.set_xlabel('Dist (kpc)',fontsize=10,fontweight='bold')
    ax.set_ylabel('N',fontsize=10,fontweight='bold')
    ax.set_title(f'{ot}: {cn}',fontsize=10,fontweight='bold',color=tc.get(ot,'k'))
    ax.legend(fontsize=6,loc='best'); ax.grid(True,alpha=0.2,axis='y')

def _panel_rv(ax, r, thr, tc):
    mdf=r['matched_df']; rrv=r.get('ref_rv'); rrve=r.get('ref_rv_err')
    ot=r['obj_type']; cn=r['cluster_name']
    hirv,allrv=_filter_by_pmem(mdf,'best_rv',thr); hrc=_clean(hirv); arc=_clean(allrv)
    memrv=r.get('member_rv',np.array([])); memrv_c=np.array([])
    if memrv is not None and len(memrv)>0:
        memrv_c=np.asarray(memrv,dtype=float); memrv_c=memrv_c[np.isfinite(memrv_c)]
    if len(hrc)>=3 or len(arc)>=3 or len(memrv_c)>=3:
        br=_bins_range_with_ref([arc,hrc,memrv_c],rrv)
        _draw_all_matched_hist(ax,allrv,br,20,zorder=3)
        if len(hrc)>=3:
            _hist_step_kde(ax,hrc,cfg.COL_HIGHMEM,f'P>{thr:.1f} (n={len(hrc)})',br,20)
            _draw_median_mad_lines(ax,hrc,cfg.COL_HIGHMEM,zorder=8)
        if len(memrv_c)>=3:
            _hist_step_kde(ax,memrv_c,cfg.COL_PLX,f'Mem RV (n={len(memrv_c)})',br,20)
            _draw_median_mad_lines(ax,memrv_c,cfg.COL_PLX,zorder=7)
        _draw_ref_band(ax,rrv,rrve,zorder=20)
        vs=r['algo_info'].get('v_sys_recovered',np.nan)
        if np.isfinite(vs): ax.axvline(vs,color='darkblue',lw=1.8,ls='-.',alpha=0.8,zorder=15)
    else: ax.text(0.5,0.5,'No RV',ha='center',va='center',transform=ax.transAxes,fontsize=10,color='gray')
    ax.set_xlabel('RV (km/s)',fontsize=10,fontweight='bold')
    ax.set_ylabel('N',fontsize=10,fontweight='bold')
    ax.set_title(f'{ot}: {cn}',fontsize=10,fontweight='bold',color=tc.get(ot,'k'))
    ax.legend(fontsize=6,loc='best'); ax.grid(True,alpha=0.2,axis='y')

def _panel_pm(ax, r, thr, tc):
    from matplotlib.patches import Ellipse
    mdf=r['matched_df']; algo=r['algo_info']; ot=r['obj_type']; cn=r['cluster_name']
    pp=r.get('prematch_pmra'); ppd=r.get('prematch_pmdec')
    if pp is not None and ppd is not None:
        vm=np.isfinite(pp)&np.isfinite(ppd)
        if np.sum(vm)>0: ax.scatter(pp[vm],ppd[vm],c=cfg.COL_GRAY_BG,s=4,alpha=cfg.GRAY_BG_ALPHA,zorder=1,edgecolors='none')
    if 'P_mem' in mdf.columns:
        v=mdf['P_mem'].notna()
        ax.scatter(mdf.loc[v,'pmra'],mdf.loc[v,'pmdec'],c=mdf.loc[v,'P_mem'],
            cmap=cfg.CMAP_PMEM,s=12,edgecolors='k',lw=0.2,vmin=0,vmax=1,zorder=5)
    if algo.get('Sigma_cluster') and ot not in ['SGR','STREAM']:
        mu=algo['mu_cluster']; sig=np.array(algo['Sigma_cluster'])
        w,v2=np.linalg.eigh(sig); ang=np.degrees(np.arctan2(v2[1,0],v2[0,0]))
        for ns in [1,2]:
            ell=Ellipse(xy=mu,width=2*ns*np.sqrt(max(w[0],0)),height=2*ns*np.sqrt(max(w[1],0)),
                angle=ang,fill=False,ec='lime',lw=2.0 if ns==1 else 1.5,ls='-' if ns==1 else '--')
            ax.add_patch(ell)
        ax.scatter(*mu,marker='x',s=60,c='lime',linewidths=2,zorder=10)
    ax.set_xlabel(r'$\mu_\alpha\cos\delta$',fontsize=10,fontweight='bold')
    ax.set_ylabel(r'$\mu_\delta$',fontsize=10,fontweight='bold')
    ax.set_title(f'{ot}: {cn}',fontsize=10,fontweight='bold',color=tc.get(ot,'k'))
    ax.grid(True,alpha=0.2); ax.set_aspect('equal','datalim')

def _panel_radec(ax, r, thr, tc):
    mdf=r['matched_df']; ot=r['obj_type']; cn=r['cluster_name']
    pra=r.get('prematch_ra'); pdec=r.get('prematch_dec')
    if pra is not None and pdec is not None:
        vm=np.isfinite(pra)&np.isfinite(pdec)
        if np.sum(vm)>0: ax.scatter(pra[vm],pdec[vm],c=cfg.COL_GRAY_BG,s=4,alpha=cfg.GRAY_BG_ALPHA,zorder=1,edgecolors='none')
    if 'P_mem' in mdf.columns:
        v=mdf['P_mem'].notna()
        ax.scatter(mdf.loc[v,'ra'],mdf.loc[v,'dec'],c=mdf.loc[v,'P_mem'],
            cmap=cfg.CMAP_PMEM,s=12,edgecolors='k',lw=0.2,vmin=0,vmax=1,zorder=5)
    ax.set_xlabel('RA (deg)',fontsize=10,fontweight='bold')
    ax.set_ylabel('Dec (deg)',fontsize=10,fontweight='bold')
    ax.set_title(f'{ot}: {cn}',fontsize=10,fontweight='bold',color=tc.get(ot,'k'))
    ax.invert_xaxis(); ax.grid(True,alpha=0.2); ax.set_aspect('equal','datalim')

# ============================================================================
# V12: GRAND SUMMARY — 4 cols x N rows, adaptive, pub quality
# ============================================================================
def generate_grand_summary(gold, logger, out_dir, em='2016'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    set_paper_style()
    n=len(gold)
    if n==0: logger.warning("  GRAND_SUMMARY: no objects!"); return
    et='[J2000\u21922016]' if em=='2000' else '[J2016]'
    thr=cfg.P_MEM_PLOT_THRESHOLD; tc={'GC':'#006400','OC':'#FF8C00','DW':'#00008B','SGR':'#8B0000'}
    pw=4.5; ph=4.0; ncols=4
    fig,axes=plt.subplots(n,ncols,figsize=(pw*ncols,ph*n),squeeze=False)
    for ri,r in enumerate(gold):
        _panel_radec(axes[ri,0],r,thr,tc)
        _panel_pm(axes[ri,1],r,thr,tc)
        _panel_dist(axes[ri,2],r,thr,tc)
        _panel_rv(axes[ri,3],r,thr,tc)
    fig.suptitle(f'V12 Grand Summary (P$_{{mem}}$>{thr:.1f}, N$_{{green}}$>={cfg.SUMMARY_MIN_HIGH_PMEM_RELAX}) {et}',
                 fontsize=20,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(os.path.join(out_dir,f'GRAND_SUMMARY.{cfg.SAVE_FORMAT}'),dpi=300); plt.close()
    logger.info(f"  Saved: GRAND_SUMMARY ({n} objects x 4 cols)")

# ============================================================================
# ORCHESTRATE SUMMARIES + SAVE + EPOCH + CLI + MAIN
# ============================================================================
def generate_summary_plots(all_results, logger, eod=None, em='2016'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    set_paper_style(); out_dir=eod or cfg.OUTPUT_DIR; os.makedirs(out_dir,exist_ok=True)
    logger.info(f"\n{'='*70}\nV12 SUMMARY + GRAND PLOTS\n{'='*70}")
    gold=_gold_sample(all_results)
    if not gold:
        logger.warning("No objects pass gold filter! Trying fallback...")
        # Ultimate fallback: anything with >=3 green
        thr=cfg.P_MEM_PLOT_THRESHOLD
        gold=[]
        for r in all_results:
            if r['matched_df'] is None: continue
            if 'P_mem' not in r['matched_df'].columns: continue
            P=r['matched_df']['P_mem'].values
            nh=int(np.sum(np.where(np.isfinite(P),P,0)>thr))
            if nh>=3: r['_n_hi']=nh; gold.append(r)
        gold.sort(key=lambda x:-x['_n_hi'])
    if not gold: logger.warning("No valid results for summary!"); return
    logger.info(f"  Gold sample: {len(gold)} objects")
    for r in gold: logger.info(f"    {r['obj_type']} {r['cluster_name']}: n_green={r['_n_hi']}")
    # 4 paginated summary series
    _paginated_summary(gold,_panel_dist,'Dist',logger,out_dir,em)
    _paginated_summary(gold,_panel_rv,'RV',logger,out_dir,em)
    _paginated_summary(gold,_panel_pm,'PM',logger,out_dir,em)
    _paginated_summary(gold,_panel_radec,'RADEC',logger,out_dir,em)
    generate_grand_summary(gold,logger,out_dir,em)

def save_results(all_results, logger, eod=None, em='2016'):
    out_dir=eod or cfg.OUTPUT_DIR; os.makedirs(out_dir,exist_ok=True)
    logger.info(f"\n{'─'*50}\nSaving results [epoch{em}]\n{'─'*50}")
    summary_data=[]; master_dfs=[]
    for r in all_results:
        nh=0
        if r['matched_df'] is not None and 'P_mem' in r['matched_df'].columns:
            nh=int(np.sum(r['matched_df']['P_mem'].fillna(0)>cfg.P_MEM_HIGH))
            t=r['matched_df'].copy(); t.insert(0,'Epoch',em); t.insert(1,'Object_Type',r['obj_type'])
            t.insert(2,'Cluster_Name',r['cluster_name']); master_dfs.append(t)
        diag=r.get('diagnostics',{})
        row={'Epoch':em,'Object':r['cluster_name'],'Type':r['obj_type'],
            'N_members':r['n_members'],'N_matched':r['n_matched'],
            'Match_pct':f"{100*r['n_matched']/r['n_members']:.1f}" if r['n_members']>0 else '0',
            'N_high_prob':nh,
            'Ref_dist_kpc':r.get('ref_dist',np.nan),'Ref_RV_kms':r.get('ref_rv',np.nan),
            'Algorithm':r['algo_info'].get('algorithm','None'),
            'Status':r['algo_info'].get('status','N/A'),
            'BIC':r['algo_info'].get('BIC',np.nan),
            'Quality_flag':diag.get('quality_flag','N/A'),
            'KS_dist_pval':diag.get('ks_dist_pval',np.nan),
            'KS_rv_pval':diag.get('ks_rv_pval',np.nan),
            'N_terms':sum(1 for k in ['n_cmd_used','n_feh_used','n_logg_used','n_dist_used','n_rv_used']
                          if r['algo_info'].get(k,0)>0)+2}
        if r['obj_type']=='DW':
            row.update({'v_sys_rec':r['algo_info'].get('v_sys_recovered',np.nan),
                'sigma_int':r['algo_info'].get('sigma_int_km_s',np.nan),
                'n_plx_removed':r['algo_info'].get('n_parallax_removed',0),
                'eta':r['algo_info'].get('eta',np.nan),'feh_sys':r['algo_info'].get('feh_sys',np.nan)})
        summary_data.append(row)
    if master_dfs:
        full=pd.concat(master_dfs,ignore_index=True)
        fp=os.path.join(out_dir,f'V12_full_membership_epoch{em}.csv')
        full.to_csv(fp,index=False); logger.info(f"  Saved: {fp} ({len(full):,} rows)")
    sdf=pd.DataFrame(summary_data).sort_values('N_matched',ascending=False).reset_index(drop=True)
    sf=os.path.join(out_dir,f'V12_summary_epoch{em}.csv'); sdf.to_csv(sf,index=False)
    logger.info(f"  Saved: {sf}")
    af2=os.path.join(out_dir,f'V12_algorithm_results_epoch{em}.json')
    with open(af2,'w') as f:
        json.dump({f"{r['obj_type']}_{r['cluster_name']}":r['algo_info'] for r in all_results},f,indent=2,default=str)
    logger.info(f"  Saved: {af2}")

def run_epoch_analysis(master, gc_dists, logger, em, eo_dir, ec_dir, skip_plots=False):
    logger.info(f"\n{'='*70}\nV12 EPOCH {em}\n{'='*70}")
    os.makedirs(eo_dir,exist_ok=True); os.makedirs(os.path.join(eo_dir,'individual_plots'),exist_ok=True)
    t0=time.time(); all_results=[]
    for fn,name in [(lambda:process_gc_members(master,gc_dists,logger,em,eo_dir),'GC'),
                    (lambda:process_oc_members(master,logger,em,eo_dir),'OC'),
                    (lambda:process_sgr_members(master,logger,em,eo_dir),'SGR'),
                    (lambda:process_dwg_members(master,logger,em,eo_dir),'DW')]:
        try: all_results.extend(fn())
        except Exception as e:
            logger.error(f"Error {name}: {e}"); import traceback; traceback.print_exc()
    if not skip_plots:
        idir=os.path.join(eo_dir,'individual_plots')
        for r in all_results:
            if r['matched_df'] is None or r['n_matched']<cfg.MIN_STARS_FOR_ANALYSIS: continue
            try:
                plot_individual_panels(r['matched_df'],r['cluster_name'],r['obj_type'],r['algo_info'],
                    r.get('ref_dist'),r.get('ref_dist_err'),r.get('ref_rv'),r.get('ref_rv_err'),
                    r.get('master_dist'),r.get('member_dist'),r.get('master_rv'),r.get('member_rv'),
                    plx_dist=r.get('plx_dist'),pra=r.get('prematch_ra'),pdec=r.get('prematch_dec'),
                    ppmra=r.get('prematch_pmra'),ppmdec=r.get('prematch_pmdec'),
                    save_dir=idir,epoch_mode=em,
                    nplx=r['algo_info'].get('n_parallax_removed',0),diag=r.get('diagnostics',{}))
            except Exception as e: logger.warning(f"  Plot err {r['cluster_name']}: {e}")
        generate_summary_plots(all_results,logger,eo_dir,em)
    save_results(all_results,logger,eo_dir,em)
    logger.info(f"\nEPOCH {em} DONE: {len(all_results)} objects ({time.time()-t0:.1f}s)")
    return all_results

def parse_args():
    p=argparse.ArgumentParser(description='Adaptive Membership V12-Plots')
    p.add_argument('--master',required=True,help='Master catalog')
    p.add_argument('--gc',default=None); p.add_argument('--oc',default=None)
    p.add_argument('--sgr',default=None); p.add_argument('--dwg',default=None)
    p.add_argument('--gc-dist',default=None); p.add_argument('--output',default='./outputs')
    p.add_argument('--checkpoint',default='./checkpoints'); p.add_argument('--log',default=None)
    p.add_argument('--skip-plots',action='store_true')
    p.add_argument('--pmem-threshold',type=float,default=0.5)
    p.add_argument('--epoch-delta',type=float,default=16.0)
    p.add_argument('--epochs',nargs='+',default=['2016','2000'],choices=['2016','2000'])
    p.add_argument('--em-maxiter',type=int,default=60)
    p.add_argument('--plx-sigma',type=float,default=3.0)
    p.add_argument('--plx-threshold',type=float,default=0.10)
    p.add_argument('--sgr-bin-start',type=float,default=15.0)
    p.add_argument('--sgr-bin-end',type=float,default=80.0)
    p.add_argument('--sgr-bin-width',type=float,default=5.0)
    return p.parse_args()

def main():
    args=parse_args()
    cfg.MASTER_CATALOG=args.master; cfg.GC_MEMBERS_FILE=args.gc; cfg.OC_MEMBERS_FILE=args.oc
    cfg.SGR_MEMBERS_FILE=args.sgr; cfg.DWG_MEMBERS_FILE=args.dwg; cfg.GC_DIST_FILE=args.gc_dist
    cfg.OUTPUT_DIR=args.output; cfg.CHECKPOINT_DIR=args.checkpoint
    cfg.P_MEM_PLOT_THRESHOLD=args.pmem_threshold; cfg.EPOCH_DELTA=args.epoch_delta
    cfg.EPOCH_FROM=2000.0; cfg.EPOCH_TO=2000.0+args.epoch_delta
    cfg.EM_MAX_ITER=args.em_maxiter; cfg.PLX_FOREGROUND_SIGMA=args.plx_sigma
    cfg.PLX_FOREGROUND_THRESHOLD=args.plx_threshold
    cfg.SGR_BIN_START_KPC=args.sgr_bin_start; cfg.SGR_BIN_END_KPC=args.sgr_bin_end
    cfg.SGR_BIN_WIDTH_KPC=args.sgr_bin_width
    os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
    log_path=args.log or os.path.join(cfg.OUTPUT_DIR,'adaptive_membership_v12.log')
    logger=setup_logging(log_path)
    logger.info("="*70)
    logger.info("ADAPTIVE MEMBERSHIP ANALYSIS V12-PLOTS")
    logger.info(f"  7-term: Spatial+PM+RV+Distance+CMD+[Fe/H]+logg")
    logger.info(f"  V12: Navy blue plx | Dashed medians | Opaque MAD | Visible red")
    logger.info(f"        6x6 paginated summaries (Dist/RV/PM/RADEC) + Grand")
    logger.info(f"        Gold: >={cfg.SUMMARY_MIN_HIGH_PMEM} green, relax >={cfg.SUMMARY_MIN_HIGH_PMEM_RELAX}")
    logger.info(f"  SGR bins: {cfg.SGR_BIN_START_KPC:.0f}-{cfg.SGR_BIN_END_KPC:.0f} kpc (w={cfg.SGR_BIN_WIDTH_KPC:.0f})")
    logger.info(f"  Epochs: {args.epochs} | astropy:{HAS_ASTROPY} | HDBSCAN:{HAS_HDBSCAN}")
    logger.info("="*70)
    t_global=time.time()
    master=MasterCatalog(logger)
    cp16=os.path.join(cfg.CHECKPOINT_DIR,'epoch2016'); cp00=os.path.join(cfg.CHECKPOINT_DIR,'epoch2000')
    if '2016' in args.epochs:
        os.makedirs(cp16,exist_ok=True)
        if not master.load(cfg.MASTER_CATALOG,cp16): logger.error("Master fail!"); sys.exit(1)
    elif '2000' in args.epochs:
        os.makedirs(cp00,exist_ok=True)
        if not master.load(cfg.MASTER_CATALOG,cp00): logger.error("Master fail!"); sys.exit(1)
    if '2016' in args.epochs and '2000' in args.epochs:
        os.makedirs(cp00,exist_ok=True)
        dc=os.path.join(cp00,'master_data_v10.parquet'); tc=os.path.join(cp00,'master_tree_v10.npz')
        if not os.path.exists(dc):
            master.df.to_parquet(dc); np.savez_compressed(tc,coords=master.coords_3d)
    gc_dists=load_gc_reference_distances(cfg.GC_DIST_FILE,logger)
    combined={}
    for em in args.epochs:
        eo=os.path.join(cfg.OUTPUT_DIR,f'epoch{em}'); ec=cp16 if em=='2016' else cp00
        combined[em]=run_epoch_analysis(master,gc_dists,logger,em,eo,ec,args.skip_plots)
    if len(combined)==2:
        logger.info("\nCombined summary...")
        rows=[]
        for em,res in combined.items():
            for r in res:
                nh=0
                if r['matched_df'] is not None and 'P_mem' in r['matched_df'].columns:
                    nh=int(np.sum(r['matched_df']['P_mem'].fillna(0)>cfg.P_MEM_HIGH))
                rows.append({'Epoch':em,'Object':r['cluster_name'],'Type':r['obj_type'],
                    'N_members':r['n_members'],'N_matched':r['n_matched'],'N_high_prob':nh,
                    'Algorithm':r['algo_info'].get('algorithm','None'),
                    'Quality':r.get('diagnostics',{}).get('quality_flag','N/A')})
        pd.DataFrame(rows).to_csv(os.path.join(cfg.OUTPUT_DIR,'V12_summary_COMBINED.csv'),index=False)
    el=(time.time()-t_global)/60
    logger.info(f"\n{'='*70}\nV12-PLOTS COMPLETE — {el:.1f} min\n{'='*70}")

if __name__ == '__main__':
    main()