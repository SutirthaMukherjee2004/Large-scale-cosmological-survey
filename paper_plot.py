#!/usr/bin/env python3
"""
================================================================================
PAPER PLOT — RV vs DISTANCE PHASE-SPACE + ERROR ANALYSIS
================================================================================
Self-contained.  Matches V12b column conventions & Config class exactly.

Reads:
  1. Master catalog FITS chunks   — 32-52M stars
  2. Raw member CSVs              — GC, OC, SGR, DW (with V12b column maps)
  3. GC reference distances       — GC_dist.csv
  4. V12 membership CSV           — V12_full_membership_epoch2016.csv
     (for high-P_mem overlay with best_dist, best_rv)

Outputs:
  FIGURE 1 — RV vs Distance (single panel, no error sub-panel)
    • 2D density histogram (WHITE background, log scale, magma-no-black)
    • Median ZP overlay (gold dashed)
    • Median Weighted_Avg overlay (cyan dashed)
    • Raw member CSV scatter: GC ○  OC ◇  SGR △  DW □
    • V12 high-P_mem members overlaid (filled, semi-transparent)
    • KDE contours for overdensity identification

  FIGURE 2 — Error Analysis (3 stacked rows, single column)
    Row 1: Normalised error histograms (all 3 overlaid on one panel)
    Row 2: N (log) vs distance — Raw RV count, Σn_meas, Outlier count
    Row 3: Mean error vs distance — Weighted_Avg_err, ZP_err, RV_err

  CACHING — .npz for master arrays, .pkl for member data
    --replot flag: loads cache, skips 10-min FITS read.

Author: Sutirtha (V12b publication-grade)
================================================================================
"""
import os, sys, warnings, glob, time, pickle, re
import gc as gcmod
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, median_abs_deviation
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS — edit these to match your cluster layout
# ============================================================================
WORK_DIR = '/user/sutirtha/BallTree_Xmatch'

# Master catalog
MASTER_GLOB = os.path.join(WORK_DIR, 'Entire_catalogue_chunk*.fits')

# Raw member CSVs (same paths as V12b PBS script)
GC_MEMBERS  = os.path.join(WORK_DIR, 'GC_members_with_RV.csv')
OC_MEMBERS  = os.path.join(WORK_DIR, 'OC_members.csv')
SGR_MEMBERS = os.path.join(WORK_DIR, 'Sgr_stream_members.csv')
DWG_MEMBERS = os.path.join(WORK_DIR, 'DWG_members_with_RV.csv')
GC_DIST_FILE = os.path.join(WORK_DIR, 'GC_dist.csv')

# V12 membership CSV (from your latest run)
MEMBERSHIP_CSV = os.path.join(WORK_DIR, 'outputs_v10_2526830.pawna', 'epoch2016',
                              'V12_full_membership_epoch2016.csv')

# Output
OUTPUT_DIR = os.path.join(WORK_DIR, 'rv_results')
CACHE_NPZ  = os.path.join(OUTPUT_DIR, 'master_cache.npz')
CACHE_MEM  = os.path.join(OUTPUT_DIR, 'member_cache.pkl')

# Plot settings
DIST_MAX   = 150.0
RV_MAX     = 500.0
NBINS_DIST = 300
NBINS_RV   = 200
PMEM_THR   = 0.5

# Toggle contour layers on Figure 1
PLOT_CLUSTER_CONTOURS = True   # per-type filled overdensity contours
PLOT_WHITE_CONTOURS   = True   # white dashed combined-member contours

# ============================================================================
# COLUMN MAPS — identical to V12b Config
# ============================================================================
COL_DIST     = 'distance_final'
COL_DIST_ERR = 'distance_err_final'
COL_WA       = 'Weighted_Avg_final'
COL_WA_ERR   = 'Weighted_Avg_err_final'
COL_ZP       = 'ZP_final'
COL_ZP_ERR   = 'ZP_err_final'
COL_RV       = 'RV_final'
COL_RV_ERR   = 'RV_err_final'
COL_NMEAS    = 'n_measurements'
COL_RVSN     = 'RVS/N'
COL_RUWE     = 'RUWE'

# GC member CSV columns (V12b GC_MEM_COLS)
GC_KEY = 'source'; GC_RV = 'RV_weighted_avg'; GC_PLX = 'parallax'
# GC_dist.csv columns (V12b GC_DIST_COLS)
GC_DIST_NAME = 'Name'; GC_DIST_LIT = 'Lit. dist. (kpc)'
GC_DIST_LIT_ERR = 'Lit. dist. Err+'
GC_DIST_MEAN = 'Mean distance (kpc)'; GC_DIST_MEAN_ERR = 'Mean distance Err+'

# OC member CSV columns (V12b OC_MEM_COLS)
OC_KEY = 'Cluster'; OC_RV = 'RV'; OC_PLX = 'Plx'

# SGR member CSV columns (V12b SGR_MEM_COLS)
SGR_RV = 'vlos'; SGR_DIST = 'dist'

# DW member CSV columns (V12b DWG_MEM_COLS)
DW_KEY = 'name'; DW_RV = 'RV_km_s'; DW_DIST = 'distance'

# ============================================================================
# STYLING
# ============================================================================
OBJ_STYLE = {
    'GC':  {'mk': 'o', 'c': '#00CC44', 'ec': 'black', 'sz': 50,
            'lab': 'Globular Clusters', 'zo': 15},
    'OC':  {'mk': 'D', 'c': '#FF8C00', 'ec': 'black', 'sz': 42,
            'lab': 'Open Clusters',     'zo': 14},
    'SGR': {'mk': '^', 'c': '#FF1493', 'ec': 'black', 'sz': 48,
            'lab': 'Sgr Stream',        'zo': 13},
    'DW':  {'mk': 's', 'c': '#00BFFF', 'ec': 'black', 'sz': 44,
            'lab': 'Dwarf Galaxies',    'zo': 12},
}

MEM_STYLE = {
    'GC':  {'mk': 'o', 'c': '#006400', 'sz': 18, 'alpha': 0.4, 'zo': 11},
    'OC':  {'mk': 'D', 'c': '#CC6600', 'sz': 14, 'alpha': 0.4, 'zo': 10},
    'SGR': {'mk': '^', 'c': '#CC1177', 'sz': 16, 'alpha': 0.35, 'zo': 9},
    'DW':  {'mk': 's', 'c': '#0088CC', 'sz': 14, 'alpha': 0.35, 'zo': 8},
}


def set_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'axes.linewidth': 2.0, 'axes.edgecolor': 'black',
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'xtick.major.size': 8, 'ytick.major.size': 8,
        'xtick.minor.size': 5, 'ytick.minor.size': 5,
        'xtick.major.width': 1.6, 'ytick.major.width': 1.6,
        'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'legend.fontsize': 11, 'legend.framealpha': 0.90,
        'legend.edgecolor': 'black', 'legend.fancybox': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.bbox': 'tight',
        'figure.dpi': 150, 'savefig.dpi': 300, 'text.usetex': False,
    })


def make_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    base = plt.cm.get_cmap('magma')
    cols = base(np.linspace(0.13, 0.97, 256))
    cm = LinearSegmentedColormap.from_list('magma_white', cols, 256)
    cm.set_under('white'); cm.set_bad('white')
    return cm


def normalize_name(name):
    if not isinstance(name, str): name = str(name)
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())


# ============================================================================
# LOAD MASTER CATALOG → dict of numpy arrays (with NPZ caching)
# ============================================================================
def load_master(pattern, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading master from cache: {cache_path}", flush=True)
        d = np.load(cache_path, allow_pickle=False)
        out = {k: d[k] for k in d.files}
        # Backward compat: old caches may have 'RVNper' instead of 'n_measurements'
        if COL_NMEAS not in out and 'RVNper' in out:
            out[COL_NMEAS] = out.pop('RVNper')
            print("  [INFO] Mapped cached 'RVNper' → 'n_measurements'")
        print(f"  {len(out[COL_DIST]):,} rows from cache\n")
        return out

    from astropy.io import fits as afits
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[FATAL] No FITS files matching: {pattern}"); sys.exit(1)
    print(f"Loading {len(files)} FITS chunk(s)...", flush=True)

    cols_want = [COL_DIST, COL_DIST_ERR, COL_WA, COL_WA_ERR,
                 COL_ZP, COL_ZP_ERR, COL_RV, COL_RV_ERR,
                 COL_NMEAS, COL_RVSN, COL_RUWE]
    arrs = {c: [] for c in cols_want}

    for fi, fp in enumerate(files):
        print(f"  [{fi+1}/{len(files)}] {os.path.basename(fp)}", flush=True)
        try:
            with afits.open(fp, memmap=True) as hdul:
                dh = None
                for hdu in hdul:
                    if hasattr(hdu, 'columns') and hdu.columns is not None:
                        dh = hdu; break
                if dh is None: continue
                cn = [c.name for c in dh.columns]; nc = dh.data.shape[0]
                if COL_DIST not in cn:
                    print(f"    [SKIP] no {COL_DIST}"); continue
                for c in cols_want:
                    if c in cn:
                        arrs[c].append(np.array(dh.data[c], dtype=np.float64))
                    else:
                        arrs[c].append(np.full(nc, np.nan, dtype=np.float64))
                print(f"    -> {nc:,} rows")
        except Exception as e:
            print(f"    [ERR] {e}"); continue

    if not arrs[COL_DIST]:
        print("[FATAL] No data loaded!"); sys.exit(1)

    result = {c: np.concatenate(arrs[c]) for c in cols_want}
    del arrs; gcmod.collect()
    print(f"  Master total: {len(result[COL_DIST]):,} rows")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **result)
    print(f"  Cached to: {cache_path}\n")
    return result


# ============================================================================
# LOAD GC REFERENCE DISTANCES — same logic as V12b
# ============================================================================
def load_gc_dists(fp):
    if not fp or not os.path.exists(fp):
        print(f"  [WARN] GC_dist.csv not found: {fp}"); return {}
    df = pd.read_csv(fp, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    gc_d = {}
    def empty(v):
        if pd.isna(v): return True
        s = str(v).strip()
        return s in ['', '-', '\u2013', '\u2014', 'nan', 'NaN', 'N/A', 'n/a'] or len(s) == 0

    for _, row in df.iterrows():
        name = str(row[GC_DIST_NAME]).strip(); dist = err = np.nan
        try:
            v = row[GC_DIST_LIT]
            if not empty(v):
                dist = float(v)
                ev = row[GC_DIST_LIT_ERR]
                err = float(str(ev).replace('+', '').strip()) if not empty(ev) else np.nan
        except: pass
        if not np.isfinite(dist):
            try:
                v = row[GC_DIST_MEAN]
                if not empty(v):
                    dist = float(v)
                    ev = row[GC_DIST_MEAN_ERR]
                    err = float(str(ev).replace('+', '').strip()) if not empty(ev) else np.nan
            except: pass
        if np.isfinite(dist):
            nn = normalize_name(name)
            for k in [nn, name, name.lower(), name.upper()]:
                gc_d[k] = (dist, err)
    print(f"  Loaded {len(gc_d)} GC reference distances")
    return gc_d


# ============================================================================
# LOAD RAW MEMBER CSVs + V12 MEMBERSHIP (with pkl caching)
# ============================================================================
def load_raw_members(gc_dists_dict, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading members from cache: {cache_path}", flush=True)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    result = {}

    # ── GC: dist from GC_dist.csv, RV from RV_weighted_avg ──
    if os.path.exists(GC_MEMBERS):
        print(f"Loading GC members: {GC_MEMBERS}")
        df = pd.read_csv(GC_MEMBERS)
        rv = pd.to_numeric(df.get(GC_RV, pd.Series(dtype=float)), errors='coerce').values
        dist = np.full(len(df), np.nan)
        if GC_KEY in df.columns:
            for i, row in df.iterrows():
                cn = str(row[GC_KEY]).strip(); nn = normalize_name(cn)
                for nt in [nn, cn, cn.lower(), cn.upper()]:
                    if nt in gc_dists_dict:
                        dist[i] = gc_dists_dict[nt][0]; break
        names = df[GC_KEY].values if GC_KEY in df.columns else np.full(len(df), 'GC')
        ok = np.isfinite(dist) & (dist > 0) & np.isfinite(rv)
        result['GC'] = pd.DataFrame({'dist': dist[ok], 'rv': rv[ok], 'name': names[ok]})
        print(f"  GC: {len(result['GC'])} stars with valid (dist, rv)")

    # ── OC: dist = 1/Plx, RV from RV column ──
    if os.path.exists(OC_MEMBERS):
        print(f"Loading OC members: {OC_MEMBERS}")
        df = pd.read_csv(OC_MEMBERS)
        rv = pd.to_numeric(df.get(OC_RV, pd.Series(dtype=float)), errors='coerce').values
        plx = pd.to_numeric(df.get(OC_PLX, pd.Series(dtype=float)), errors='coerce').values
        dist = np.where(plx > 0, 1.0 / plx, np.nan)
        names = df[OC_KEY].values if OC_KEY in df.columns else np.full(len(df), 'OC')
        ok = np.isfinite(dist) & (dist > 0) & (dist < 300) & np.isfinite(rv)
        result['OC'] = pd.DataFrame({'dist': dist[ok], 'rv': rv[ok], 'name': names[ok]})
        print(f"  OC: {len(result['OC'])} stars with valid (dist, rv)")

    # ── SGR: dist and vlos directly from CSV ──
    if os.path.exists(SGR_MEMBERS):
        print(f"Loading SGR members: {SGR_MEMBERS}")
        df = pd.read_csv(SGR_MEMBERS)
        rv = pd.to_numeric(df.get(SGR_RV, pd.Series(dtype=float)), errors='coerce').values
        dist = pd.to_numeric(df.get(SGR_DIST, pd.Series(dtype=float)), errors='coerce').values
        ok = np.isfinite(dist) & (dist > 0) & np.isfinite(rv)
        result['SGR'] = pd.DataFrame({'dist': dist[ok], 'rv': rv[ok],
                                       'name': np.full(np.sum(ok), 'Sgr')})
        print(f"  SGR: {len(result['SGR'])} stars with valid (dist, rv)")

    # ── DW: distance and RV_km_s directly from CSV ──
    if os.path.exists(DWG_MEMBERS):
        print(f"Loading DW members: {DWG_MEMBERS}")
        df = pd.read_csv(DWG_MEMBERS)
        rv = pd.to_numeric(df.get(DW_RV, pd.Series(dtype=float)), errors='coerce').values
        dist = pd.to_numeric(df.get(DW_DIST, pd.Series(dtype=float)), errors='coerce').values
        names = df[DW_KEY].values if DW_KEY in df.columns else np.full(len(df), 'DW')
        ok = np.isfinite(dist) & (dist > 0) & np.isfinite(rv)
        result['DW'] = pd.DataFrame({'dist': dist[ok], 'rv': rv[ok], 'name': names[ok]})
        print(f"  DW: {len(result['DW'])} stars with valid (dist, rv)")

    # ── V12 membership CSV (high-P_mem from EM analysis) ──
    if os.path.exists(MEMBERSHIP_CSV):
        print(f"Loading V12 membership CSV: {MEMBERSHIP_CSV}")
        mdf = pd.read_csv(MEMBERSHIP_CSV, low_memory=False)
        if 'P_mem' in mdf.columns and 'Object_Type' in mdf.columns:
            hi = mdf['P_mem'].astype(float) >= PMEM_THR
            sub = mdf[hi].copy()
            d = pd.to_numeric(sub.get('best_dist', pd.Series(dtype=float)), errors='coerce').values
            v = pd.to_numeric(sub.get('best_rv', pd.Series(dtype=float)), errors='coerce').values
            ot = sub['Object_Type'].values.astype(str)
            cn = sub.get('Cluster_Name', pd.Series(dtype=str)).values.astype(str)
            ok = np.isfinite(d) & (d > 0) & np.isfinite(v)
            result['V12_mem'] = pd.DataFrame({'dist': d[ok], 'rv': v[ok],
                                               'obj_type': ot[ok], 'name': cn[ok]})
            print(f"  V12 high-P_mem: {len(result['V12_mem'])} stars")
        else:
            print("  [WARN] V12 CSV missing P_mem or Object_Type columns")
    else:
        print(f"  [WARN] V12 membership CSV not found: {MEMBERSHIP_CSV}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached to: {cache_path}\n")
    return result


# ============================================================================
# HELPERS
# ============================================================================
def _binned_stat(x, y, edges, min_n=5, clip_pct=98):
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(len(centres), np.nan)
    bidx = np.digitize(x, edges) - 1
    for bi in range(len(centres)):
        m = bidx == bi
        if np.sum(m) < min_n: continue
        vals = y[m]
        if clip_pct < 100:
            clip = np.percentile(vals, clip_pct)
            vals = vals[vals <= clip]
        if len(vals) >= 2:
            out[bi] = np.median(vals)
    return centres, out


def _binned_count(x, mask, edges):
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(len(centres))
    x_ok = x[mask]
    bidx = np.digitize(x_ok, edges) - 1
    for bi in range(len(centres)):
        counts[bi] = np.sum(bidx == bi)
    return centres, counts


def _binned_sum(x, y, mask, edges):
    """Sum values of y in each distance bin (for Σn_meas)."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    sums = np.zeros(len(centres))
    x_ok = x[mask]; y_ok = y[mask]
    bidx = np.digitize(x_ok, edges) - 1
    for bi in range(len(centres)):
        m = bidx == bi
        if np.sum(m) > 0:
            sums[bi] = np.nansum(y_ok[m])
    return centres, sums


# ============================================================================
# FIGURE 1 — RV vs DISTANCE (single panel)
# ============================================================================
def figure_rv_dist(master, members, outdir):
    print("\u2500" * 60)
    print("FIGURE 1: RV vs Distance phase-space")
    print("\u2500" * 60, flush=True)
    set_paper_style(); cmap = make_cmap()
    DR = (0, DIST_MAX); RR = (-RV_MAX, RV_MAX)

    dist = master[COL_DIST]; wa = master[COL_WA]; zp = master[COL_ZP]
    ok_wa = np.isfinite(dist) & np.isfinite(wa) & (dist > 0)
    ok_zp = np.isfinite(dist) & np.isfinite(zp) & (dist > 0)
    d_wa = dist[ok_wa]; r_wa = wa[ok_wa]
    d_zp = dist[ok_zp]; z_zp = zp[ok_zp]
    print(f"  Weighted_Avg pairs: {len(d_wa):,}  |  ZP pairs: {len(d_zp):,}")

    fig, ax = plt.subplots(figsize=(14, 10))

    # 2D density
    H, xe, ye = np.histogram2d(d_wa, r_wa, bins=[NBINS_DIST, NBINS_RV],
                                range=[list(DR), list(RR)])
    H = H.T; Hp = H.copy(); Hp[Hp == 0] = np.nan
    im = ax.pcolormesh(xe, ye, Hp, cmap=cmap,
                       norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(Hp)),
                       rasterized=True)
    cb = plt.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cb.set_label('Star Count (Log Scale)', fontsize=14, fontweight='bold')
    cb.ax.tick_params(labelsize=12)

    # Median overlays
    edges150 = np.linspace(DR[0], DR[1], 150)
    if len(d_zp) > 200:
        cx, my = _binned_stat(d_zp, z_zp, edges150, min_n=5, clip_pct=100)
        ok = np.isfinite(my)
        if np.sum(ok) > 5:
            ax.plot(cx[ok], my[ok], color='#FFD700', lw=2.5, ls='--',
                    alpha=0.95, label='Median ZP', zorder=25)
    if len(d_wa) > 200:
        cx2, my2 = _binned_stat(d_wa, r_wa, edges150, min_n=5, clip_pct=100)
        ok2 = np.isfinite(my2)
        if np.sum(ok2) > 5:
            ax.plot(cx2[ok2], my2[ok2], color='#00FFFF', lw=2.0, ls='--',
                    alpha=0.80, label='Median Weighted Avg', zorder=24)

    # ── LOCAL OVERDENSITY CONTOURS per object type ──
    # No scatter points. Members only contribute to contour density field.
    from matplotlib.patches import Patch as _Patch
    CONTOUR_NBINS_D = 200; CONTOUR_NBINS_R = 160; SMOOTH_SIGMA = 2.5
    contour_legend = []
    all_mem_d = []; all_mem_r = []  # collect for combined white contour

    for ot, sty in OBJ_STYLE.items():
        if ot not in members: continue
        mdf = members[ot]; d_m = mdf['dist'].values; r_m = mdf['rv'].values
        valid = (np.isfinite(d_m) & np.isfinite(r_m) &
                 (d_m >= DR[0]) & (d_m <= DR[1]) &
                 (r_m >= RR[0]) & (r_m <= RR[1]))
        d_ok = d_m[valid]; r_ok = r_m[valid]
        all_mem_d.append(d_ok); all_mem_r.append(r_ok)
        if not PLOT_CLUSTER_CONTOURS: continue
        if len(d_ok) < 15:
            print(f"    [SKIP] {ot}: {len(d_ok)} stars"); continue
        H_mem, xe_m, ye_m = np.histogram2d(
            d_ok, r_ok, bins=[CONTOUR_NBINS_D, CONTOUR_NBINS_R],
            range=[list(DR), list(RR)])
        H_mem = H_mem.T
        H_sm = gaussian_filter(H_mem.astype(float), sigma=SMOOTH_SIGMA)
        occupied = H_sm[H_sm > 0]
        if len(occupied) < 5: continue
        levs = sorted(set(np.percentile(occupied, [60, 80, 95])))
        levs = [l for l in levs if l > 0]
        if len(levs) < 1: continue
        xc = 0.5 * (xe_m[:-1] + xe_m[1:]); yc = 0.5 * (ye_m[:-1] + ye_m[1:])
        Xm, Ym = np.meshgrid(xc, yc)
        levs_fill = levs + [H_sm.max() * 1.1]
        ax.contourf(Xm, Ym, H_sm, levels=levs_fill,
                    colors=sty['c'], alpha=0.20, zorder=sty['zo'] - 4)
        ax.contour(Xm, Ym, H_sm, levels=levs,
                   colors=[sty['c']], linewidths=1.0,
                   alpha=0.55, linestyles='-', zorder=sty['zo'] - 3)
        contour_legend.append(_Patch(facecolor=sty['c'], alpha=0.30,
                                     edgecolor=sty['c'],
                                     label=f"{sty['lab']} ({len(d_ok)} stars)"))
        print(f"    {ot}: {len(levs)} contour levels, {len(d_ok)} stars")

    # ── WHITE DASHED CONTOURS — all member types combined ──
    if PLOT_WHITE_CONTOURS:
        all_d = np.concatenate(all_mem_d) if all_mem_d else np.array([])
        all_r = np.concatenate(all_mem_r) if all_mem_r else np.array([])
        if len(all_d) > 50:
            H_all, xe_a, ye_a = np.histogram2d(
                all_d, all_r, bins=[CONTOUR_NBINS_D, CONTOUR_NBINS_R],
                range=[list(DR), list(RR)])
            H_all = H_all.T
            H_all_sm = gaussian_filter(H_all.astype(float), sigma=3.0)
            occ_all = H_all_sm[H_all_sm > 0]
            if len(occ_all) > 10:
                levs_w = sorted(set(np.percentile(occ_all, [40, 60, 78, 90, 97])))
                levs_w = [l for l in levs_w if l > 0]
                if len(levs_w) >= 2:
                    xc_a = 0.5 * (xe_a[:-1] + xe_a[1:])
                    yc_a = 0.5 * (ye_a[:-1] + ye_a[1:])
                    Xma, Yma = np.meshgrid(xc_a, yc_a)
                    ax.contour(Xma, Yma, H_all_sm, levels=levs_w,
                               colors='white', linewidths=0.8,
                               alpha=0.65, linestyles='--', zorder=30)
                    print(f"    White combined: {len(levs_w)} levels, {len(all_d)} stars")

    # Legend
    extra = [Line2D([0], [0], color='#FFD700', ls='--', lw=2.5, label='Median ZP'),
             Line2D([0], [0], color='#00FFFF', ls='--', lw=2.0, label='Median Weighted Avg')]
    if PLOT_WHITE_CONTOURS:
        extra.append(Line2D([0], [0], color='white', ls='--', lw=1.0, label='Combined Member Contours'))
    ax.legend(handles=extra + contour_legend, loc='upper right',
              fontsize=8, framealpha=0.92, edgecolor='gray', fancybox=False, ncol=1)

    ax.set_xlabel('Distance (kpc)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'Radial Velocity (km s$^{-1}$)', fontsize=16, fontweight='bold')
    ax.set_title('RV vs Distance Phase-Space', fontsize=18, fontweight='bold')
    ax.set_xlim(DR); ax.set_ylim(RR)

    out = os.path.join(outdir, 'RV_vs_Distance_phaseSpace.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  \u2713 Saved: {out}\n"); gcmod.collect()


# ============================================================================
# FIGURE 2 — ERROR ANALYSIS (3 stacked rows)
# ============================================================================
def figure_errors(master, outdir):
    print("\u2500" * 60)
    print("FIGURE 2: Error distributions + N(log) + Mean error")
    print("\u2500" * 60, flush=True)
    set_paper_style()

    dist   = master[COL_DIST]
    wa_err = master[COL_WA_ERR]
    rv_err = master[COL_RV_ERR]
    zp_err = master[COL_ZP_ERR]
    wa     = master[COL_WA]
    rv_raw = master[COL_RV]
    nmeas  = master[COL_NMEAS]
    rvsn   = master[COL_RVSN]
    ruwe   = master[COL_RUWE]

    fig, axes = plt.subplots(3, 1, figsize=(14, 16),
                             gridspec_kw={'height_ratios': [3, 2, 2], 'hspace': 0.28})

    # ════════════════════════════════════════════════════════
    # ROW 1: All 3 error distributions overlaid (normalised)
    # ════════════════════════════════════════════════════════
    ax = axes[0]
    err_cfgs = [
        (wa_err, '#4169E1', 'Weighted Avg err', '-'),
        (rv_err, '#DC143C', 'Raw RV err',       '--'),
        (zp_err, '#FFB700', 'ZP err',           '-.'),
    ]
    for data, color, label, ls in err_cfgs:
        ok = np.isfinite(data) & (data > 0)
        d = data[ok]
        if len(d) < 10: continue
        d = d[d <= 50.0]  # hard clip at 50 km/s for resolution
        nb = 500          # many bins → smooth structure in 0–10 range
        c_, b_, _ = ax.hist(d, bins=nb, density=True, alpha=0, zorder=1)
        ax.step(b_[:-1], c_, where='post', color=color, lw=2.2,
                ls=ls, label=label, zorder=5)
        try:
            kde = gaussian_kde(d, bw_method='silverman')
            xk = np.linspace(0, 50, 600)
            ax.fill_between(xk, kde(xk), alpha=0.12, color=color, zorder=3)
        except: pass
        med = np.median(d)
        ax.axvline(med, color=color, ls=':', lw=1.8, alpha=0.8, zorder=7)
        mad = median_abs_deviation(d, nan_policy='omit')
        print(f"    {label}: N={len(d):,}  Med={med:.2f}  MAD={mad:.2f}")

    ax.set_xlabel(r'Error (km s$^{-1}$)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalised Density', fontsize=16, fontweight='bold')
    ax.set_title('Radial Velocity Error Distributions', fontsize=18, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.92, edgecolor='gray')
    ax.grid(True, alpha=0.15, axis='y'); ax.minorticks_on()

    # ════════════════════════════════════════════════════════
    # ROW 2: N (log) vs Distance
    # ════════════════════════════════════════════════════════
    ax = axes[1]
    ok_dist = np.isfinite(dist) & (dist > 0)
    edges_n = np.linspace(0, DIST_MAX, 150)

    # Raw RV count = stars with finite RV_final per dist bin
    ok_rv = ok_dist & np.isfinite(rv_raw)
    cx, cnt_rv = _binned_count(dist, ok_rv, edges_n)
    cnt_rv[cnt_rv == 0] = np.nan
    ax.plot(cx, cnt_rv, color='#DC143C', lw=2.0, alpha=0.9,
            label='Raw RV count', zorder=5)

    # Σn_meas = sum of n_measurements column per dist bin (total RV epochs)
    ok_nm = ok_dist & np.isfinite(nmeas) & (nmeas > 0)
    cx, sum_nm = _binned_sum(dist, nmeas, ok_nm, edges_n)
    sum_nm[sum_nm == 0] = np.nan
    ax.plot(cx, sum_nm, color='#6495ED', lw=2.0, alpha=0.9,
            label=r'$\Sigma n_{meas}$', zorder=6)

    # Outlier count = bad quality (RUWE > 1.4 OR RV S/N < 3)
    bad_ruwe = np.isfinite(ruwe) & (ruwe > 1.4)
    bad_sn = np.isfinite(rvsn) & (rvsn < 3.0)
    ok_out = ok_dist & (bad_ruwe | bad_sn)
    cx, cnt_out = _binned_count(dist, ok_out, edges_n)
    cnt_out[cnt_out == 0] = np.nan
    ax.plot(cx, cnt_out, color='#77DD77', lw=1.8, alpha=0.85,
            label='Outlier count', zorder=4)

    ax.set_yscale('log')
    ax.set_xlabel('Distance (kpc)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$N$ (log)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, DIST_MAX)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.92, edgecolor='gray')
    ax.grid(True, alpha=0.2, which='both'); ax.minorticks_on()

    # ════════════════════════════════════════════════════════
    # ROW 3: Mean Error vs Distance
    # ════════════════════════════════════════════════════════
    ax = axes[2]
    edges_e = np.linspace(0, DIST_MAX, 120)

    ok_we = np.isfinite(dist) & np.isfinite(wa_err) & (dist > 0) & (wa_err > 0)
    if np.sum(ok_we) > 200:
        cx, me = _binned_stat(dist[ok_we], wa_err[ok_we], edges_e, min_n=5)
        ok = np.isfinite(me)
        ax.plot(cx[ok], me[ok], color='#4169E1', lw=2.0, alpha=0.9,
                label='Median Weighted Avg err')

    ok_ze = np.isfinite(dist) & np.isfinite(zp_err) & (dist > 0) & (zp_err > 0)
    if np.sum(ok_ze) > 200:
        cx, me = _binned_stat(dist[ok_ze], zp_err[ok_ze], edges_e, min_n=5)
        ok = np.isfinite(me)
        ax.plot(cx[ok], me[ok], color='#FFD700', lw=2.0, ls='--', alpha=0.9,
                label='Median ZP err')

    ok_re = np.isfinite(dist) & np.isfinite(rv_err) & (dist > 0) & (rv_err > 0)
    if np.sum(ok_re) > 200:
        cx, me = _binned_stat(dist[ok_re], rv_err[ok_re], edges_e, min_n=5)
        ok = np.isfinite(me)
        ax.plot(cx[ok], me[ok], color='#DC143C', lw=1.8, ls='-.', alpha=0.85,
                label='Median RV err')

    ax.set_xlabel('Distance (kpc)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'Mean Error (km s$^{-1}$)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, DIST_MAX)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.92, edgecolor='gray')
    ax.grid(True, alpha=0.2, axis='y'); ax.minorticks_on()

    out = os.path.join(outdir, 'Error_Distributions.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  \u2713 Saved: {out}\n"); gcmod.collect()


# ============================================================================
# MAIN
# ============================================================================
def main():
    import argparse
    p = argparse.ArgumentParser(description='Paper Plot \u2014 RV vs Distance + Errors (V12b)')
    p.add_argument('--replot', action='store_true',
                   help='Replot from cached .npz/.pkl (skip FITS loading)')
    args = p.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("PAPER PLOT \u2014 RV vs Distance + Error Analysis (V12b style)")
    print("=" * 70, flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.replot:
        if not os.path.exists(CACHE_NPZ):
            print(f"[FATAL] No cache at {CACHE_NPZ} \u2014 run full analysis first!")
            sys.exit(1)
        print("\u2500\u2500 REPLOT MODE: loading from cache \u2500\u2500\n")

    master = load_master(MASTER_GLOB, CACHE_NPZ)
    gc_dists = load_gc_dists(GC_DIST_FILE)
    members = load_raw_members(gc_dists, CACHE_MEM)

    figure_rv_dist(master, members, OUTPUT_DIR)
    figure_errors(master, OUTPUT_DIR)

    el = time.time() - t0
    print(f"{'=' * 70}")
    print(f"ALL DONE \u2014 {el / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()