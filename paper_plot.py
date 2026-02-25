import os
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d, gaussian_filter
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CHUNK_PATTERN = 'Entire_catalogue_chunk*.fits'
OUTPUT_PLOT   = 'RV_Analysis_Dashboard_v5.png'
CACHE_FILE    = 'RV_Analysis_Cache.npz'

COLS = {
    'dist':      'distance_final',
    'rv_zp':     'ZP_final',
    'err_zp':    'ZP_err_final',
    'out_zp':    'ZP_outliers',
    'rv_weight': 'Weighted_Avg_final',
    'err_weight':'Weighted_Avg_err_final',
    'out_weight':'Weighted_Avg_outliers',
    'rv_raw':    'RV_final',
    'err_raw':   'RV_err_final',
    'out_raw':   'RV_outliers',
    'n_meas':    'n_measurements',
}

DIST_LIMITS     = (0, 250)
RV_LIMITS       = (-600, 600)
ZP_LIMITS       = (-600, 600)
BINS_DIST       = 300
BINS_RV         = 300
ERROR_THRESHOLD = 300
BINS_STATS      = 250

# ==============================================================================
# 2. PARSERS
# ==============================================================================
def parse_string_to_floats(s):
    if isinstance(s, (bytes, np.bytes_)):
        s = s.decode('utf-8')
    s = str(s).strip()
    if not s or s.lower() in ['nan', 'none', 'null', '']:
        return []
    s = s.replace(',', ' ').replace(';', ' ').replace('|', ' ')
    values = []
    for val in s.split():
        val = val.strip()
        if val and val.lower() not in ['nan', 'none', 'null']:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                pass
    return values

def smart_column_to_array(column):
    if np.issubdtype(column.dtype, np.number):
        return [[float(v) if np.isfinite(v) else np.nan] for v in column], False
    values, max_count = [], 0
    for entry in column:
        parsed = parse_string_to_floats(entry)
        if not parsed:
            parsed = [np.nan]
        values.append(parsed)
        max_count = max(max_count, len(parsed))
    return values, max_count > 1

def expand_measurements(dist, *columns):
    parsed_cols = []
    for col in columns:
        parsed, _ = smart_column_to_array(col)
        parsed_cols.append(parsed)
    dist_parsed, _ = smart_column_to_array(dist)
    expanded_dist, expanded_cols = [], [[] for _ in columns]
    for i in range(len(dist_parsed)):
        d_vals = dist_parsed[i]
        if not d_vals or not np.isfinite(d_vals[0]):
            continue
        d      = d_vals[0]
        n_meas = max(len(parsed_cols[j][i]) for j in range(len(columns)))
        if n_meas == 0:
            continue
        expanded_dist.extend([d] * n_meas)
        for j in range(len(columns)):
            vals = parsed_cols[j][i]
            if len(vals) == 1:
                expanded_cols[j].extend([vals[0]] * n_meas)
            elif len(vals) == n_meas:
                expanded_cols[j].extend(vals)
            else:
                expanded_cols[j].extend(vals + [np.nan] * (n_meas - len(vals)))
    return np.array(expanded_dist), tuple(np.array(col) for col in expanded_cols)

# ==============================================================================
# 3. DATA PROCESSING
# ==============================================================================
def process_all_data():
    files = sorted(glob.glob(CHUNK_PATTERN))
    print(f"Processing {len(files)} chunks...")

    weighted_dist, weighted_rv = [], []
    bin_edges = np.linspace(DIST_LIMITS[0], DIST_LIMITS[1], BINS_STATS + 1)
    stats = {k: np.zeros(BINS_STATS) for k in [
        'raw_count', 'n_meas_sum', 'outlier_count',
        'sum_zp_corr', 'count_zp_corr',
        'sum_err_raw', 'sum_err_weight', 'sum_err_zp',
        'count_err_raw', 'count_err_weight', 'count_err_zp',
    ]}

    for i, f in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}")
        try:
            with fits.open(f, memmap=True) as hdu:
                data = hdu[1].data
                try:
                    d_col=data[COLS['dist']];      v_z_col=data[COLS['rv_zp']];    e_z_col=data[COLS['err_zp']]
                    v_w_col=data[COLS['rv_weight']]; e_w_col=data[COLS['err_weight']]; v_r_col=data[COLS['rv_raw']]
                    e_r_col=data[COLS['err_raw']];   o_w_col=data[COLS['out_weight']]; n_m_col=data[COLS['n_meas']]
                except KeyError as e:
                    print(f"    [SKIP] Missing column: {e}"); continue

                d_raw, (v_r,) = expand_measurements(d_col, v_r_col)
                mask_raw = (np.isfinite(d_raw)&np.isfinite(v_r)&
                            (d_raw>=DIST_LIMITS[0])&(d_raw<=DIST_LIMITS[1])&
                            (v_r>=RV_LIMITS[0])&(v_r<=RV_LIMITS[1]))
                if mask_raw.any():
                    idx=np.digitize(d_raw[mask_raw],bin_edges)-1; v=((idx>=0)&(idx<BINS_STATS))
                    np.add.at(stats['raw_count'],idx[v],1)

                d_weight,(v_w,e_w,o_w)=expand_measurements(d_col,v_w_col,e_w_col,o_w_col)
                has_outlier=np.isfinite(o_w)
                mask_no=(np.isfinite(d_weight)&np.isfinite(v_w)&np.isfinite(e_w)&
                         (d_weight>=DIST_LIMITS[0])&(d_weight<=DIST_LIMITS[1])&
                         (v_w>=RV_LIMITS[0])&(v_w<=RV_LIMITS[1])&
                         (e_w<ERROR_THRESHOLD)&~has_outlier)
                if mask_no.any():
                    weighted_dist.extend(d_weight[mask_no]); weighted_rv.extend(v_w[mask_no])

                mask_out=(np.isfinite(d_weight)&np.isfinite(o_w)&
                          (d_weight>=DIST_LIMITS[0])&(d_weight<=DIST_LIMITS[1])&
                          (o_w>=RV_LIMITS[0])&(o_w<=RV_LIMITS[1]))
                if mask_out.any():
                    idx=np.digitize(d_weight[mask_out],bin_edges)-1; v=((idx>=0)&(idx<BINS_STATS))
                    np.add.at(stats['outlier_count'],idx[v],1)

                d_zp,(v_z,v_r_zp)=expand_measurements(d_col,v_z_col,v_r_col)
                mask_zp=(np.isfinite(d_zp)&np.isfinite(v_z)&np.isfinite(v_r_zp)&
                         (d_zp>=DIST_LIMITS[0])&(d_zp<=DIST_LIMITS[1]))
                if mask_zp.any():
                    zp_corr=v_z[mask_zp]-v_r_zp[mask_zp]
                    idx=np.digitize(d_zp[mask_zp],bin_edges)-1; v=((idx>=0)&(idx<BINS_STATS))
                    np.add.at(stats['sum_zp_corr'],idx[v],zp_corr[v])
                    np.add.at(stats['count_zp_corr'],idx[v],1)

                d_nm,(n_m,)=expand_measurements(d_col,n_m_col)
                mask_nm=(np.isfinite(d_nm)&np.isfinite(n_m)&
                         (d_nm>=DIST_LIMITS[0])&(d_nm<=DIST_LIMITS[1]))
                if mask_nm.any():
                    idx=np.digitize(d_nm[mask_nm],bin_edges)-1; v=((idx>=0)&(idx<BINS_STATS))
                    np.add.at(stats['n_meas_sum'],idx[v],n_m[mask_nm][v])

                for col_k,s_k,c_k in [(e_r_col,'sum_err_raw','count_err_raw'),
                                       (e_w_col,'sum_err_weight','count_err_weight'),
                                       (e_z_col,'sum_err_zp','count_err_zp')]:
                    d_e,(e_a,)=expand_measurements(d_col,col_k)
                    m=(np.isfinite(d_e)&np.isfinite(e_a)&
                       (d_e>=DIST_LIMITS[0])&(d_e<=DIST_LIMITS[1]))
                    if m.any():
                        idx=np.digitize(d_e[m],bin_edges)-1; v=((idx>=0)&(idx<BINS_STATS))
                        np.add.at(stats[s_k],idx[v],e_a[m][v])
                        np.add.at(stats[c_k],idx[v],1)

        except Exception as e:
            print(f"    [ERROR] {e}"); import traceback; traceback.print_exc()

    result={'weighted_dist':np.array(weighted_dist),'weighted_rv':np.array(weighted_rv),'bin_edges':bin_edges}
    result.update(stats)
    print(f"\n{'='*70}\n  Weighted Avg (error<{ERROR_THRESHOLD}, non-outliers): {len(result['weighted_dist']):,}\n{'='*70}\n")
    return result

def save_cache(data):
    print(f"Saving cache → {CACHE_FILE}")
    np.savez_compressed(CACHE_FILE, **data)
    print(f"  Saved ({os.path.getsize(CACHE_FILE)/1e6:.1f} MB)\n")

def load_cache():
    print(f"Loading cache from {CACHE_FILE}...")
    data=dict(np.load(CACHE_FILE, allow_pickle=True))
    print(f"  Weighted Avg: {len(data['weighted_dist']):,}\n")
    return data

# ==============================================================================
# 4. PLOTTING — v5
#    KEY FIXES vs v4:
#      • Gaussian-smoothed histogram (sigma=2) used ONLY for contouring
#        → closed, smooth contour lines instead of fragmented pixel-noise
#      • 20 log-spaced levels spanning 1 % → 99 % of occupied density
#        → dense, evenly-distributed contour bands
#      • No arbitrary gap filter (the old 1.15× rule caused missing levels)
#      • imshow still uses the RAW histogram (maximum sharpness for the heatmap)
#      • Colormap: 'inferno' with dark-navy (#07090F) panel background
#        → matches the reference blue-dominant dark scheme while keeping
#          the yellow/orange hotspot at the galactic centre
# ==============================================================================
def create_dashboard(data):
    print("Generating Dashboard v5 …")

    weighted_dist = data['weighted_dist']
    weighted_rv   = data['weighted_rv']
    bin_edges     = data['bin_edges']
    bin_centers   = 0.5*(bin_edges[:-1]+bin_edges[1:])

    # ---- Derived statistics ----
    with np.errstate(invalid='ignore', divide='ignore'):
        def safe_mean(s, c):
            cnt = c.copy(); cnt[cnt == 0] = 1.0
            m = s / cnt; m[c == 0] = np.nan; return m
        avg_zp_corr = safe_mean(data['sum_zp_corr'],   data['count_zp_corr'])
        mean_err_r  = safe_mean(data['sum_err_raw'],    data['count_err_raw'])
        mean_err_w  = safe_mean(data['sum_err_weight'], data['count_err_weight'])
        mean_err_z  = safe_mean(data['sum_err_zp'],     data['count_err_zp'])

    # ---- Global rcParams ----
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':  'stix',
        'axes.linewidth':    1.8,
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.top':         True,
        'ytick.right':       True,
        'xtick.major.width': 1.8,
        'ytick.major.width': 1.8,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.major.size':  8,
        'ytick.major.size':  8,
        'xtick.minor.size':  5,
        'ytick.minor.size':  5,
    })

    LABEL_SIZE  = 20
    TICK_SIZE   = 14
    LEGEND_SIZE = 12

    # Colors
    ZP_COLOR  = '#FFD700'   # gold
    COL_RAW   = '#E05C5C'
    COL_NM    = '#64BFEE'
    COL_OUT   = '#7ECA7E'
    COL_ERR_R = '#E05C5C'
    COL_ERR_W = '#F5A623'
    COL_ERR_Z = '#64BFEE'

    # ================================================================
    # FIGURE
    # ================================================================
    fig = plt.figure(figsize=(15, 18), facecolor='white')
    gs  = plt.GridSpec(3, 1,
                       height_ratios=[5, 1.5, 1.5],
                       hspace=0.04,
                       left=0.09, right=0.87,
                       top=0.98, bottom=0.06)

    # ==========================================================================
    #  PANEL 1 — Phase Space
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0])
    # Dark navy — matches the reference blue-dominant dark scheme
    BG_COLOR = '#07090F'
    ax1.set_facecolor(BG_COLOR)

    if len(weighted_dist) > 0:
        # ── RAW 2-D histogram (used for imshow — keeps pixel sharpness) ────────
        H, xedges, yedges = np.histogram2d(
            weighted_dist, weighted_rv,
            bins=[BINS_DIST, BINS_RV],
            range=[DIST_LIMITS, RV_LIMITS],
        )
        H_plot = H.T   # shape (BINS_RV, BINS_DIST)

        H_disp = np.where(H_plot > 0, H_plot, np.nan)
        vmin_use = 1
        vmax_use = max(H_plot.max(), 2)

        im = ax1.imshow(
            H_disp,
            extent=[DIST_LIMITS[0], DIST_LIMITS[1], RV_LIMITS[0], RV_LIMITS[1]],
            origin='lower', aspect='auto',
            cmap='inferno',
            norm=mcolors.LogNorm(vmin=vmin_use, vmax=vmax_use),
            interpolation='nearest',
            zorder=2,
        )

        # ── Colourbar ──────────────────────────────────────────────────────────
        cbar_ax = fig.add_axes([0.882, 0.355, 0.016, 0.622])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Star Count (log scale)', fontsize=13,
                       fontweight='bold', labelpad=8)
        cbar.ax.tick_params(labelsize=11, width=1.5, length=5)
        cbar.outline.set_linewidth(1.5)

        # ── SMOOTH histogram for contours ──────────────────────────────────────
        # FIX #1: Apply Gaussian blur so contours form closed, smooth curves
        #         instead of fractured pixel-noise outlines.
        #         sigma=(sy, sx) — y-axis (RV) needs more smoothing than x (dist)
        #         because the histogram is denser along x.
        SIGMA_X = 2.5   # blur in the distance direction
        SIGMA_Y = 2.5   # blur in the RV direction
        H_smooth = gaussian_filter(H_plot.astype(float),
                                   sigma=[SIGMA_Y, SIGMA_X])

        Xc = 0.5*(xedges[:-1]+xedges[1:])
        Yc = 0.5*(yedges[:-1]+yedges[1:])
        Xm, Ym = np.meshgrid(Xc, Yc)

        occupied_sm = H_smooth[H_smooth > 0]
        if len(occupied_sm) >= 10:
            # FIX #2: log-spaced levels across the *smoothed* density range
            #         — guarantees dense, evenly-spaced contour bands.
            # FIX #3: removed the old "1.15× gap" filter that discarded levels
            lvl_lo = max(np.percentile(occupied_sm, 1), 0.3)
            lvl_hi = H_smooth.max() * 0.97
            N_LEVELS = 22
            lvls = np.unique(
                np.logspace(np.log10(lvl_lo), np.log10(lvl_hi), N_LEVELS)
            )
            lvls = lvls[lvls < lvl_hi]          # safety cap
            lvls = lvls[np.diff(lvls, prepend=0) > 0]  # strictly increasing

            if len(lvls) >= 2:
                CS = ax1.contour(Xm, Ym, H_smooth,
                                 levels=lvls,
                                 colors='white',
                                 linewidths=0.9,
                                 linestyles='solid',
                                 alpha=0.80,
                                 zorder=4)
                print(f"  Contours: {len(lvls)} levels  "
                      f"min={lvls[0]:.2f}  max={lvls[-1]:.2f}")
            else:
                print("  [WARN] Not enough contour levels after filtering.")

        print(f"  Phase space: {len(weighted_dist):,} stars")

    # ── ZP correction twin-axis (gold) ─────────────────────────────────────────
    if np.sum(np.isfinite(avg_zp_corr)) > 0:
        zp_sm = uniform_filter1d(
            np.where(np.isfinite(avg_zp_corr), avg_zp_corr, 0), size=5)
        zp_sm[~np.isfinite(avg_zp_corr)] = np.nan

        ax1t = ax1.twinx()
        ax1t.plot(bin_centers, zp_sm,
                  color=ZP_COLOR, lw=2.0, alpha=0.95,
                  label='Mean ZP Correction', zorder=5)
        ax1t.axhline(0, color=ZP_COLOR, ls='--', lw=1.0, alpha=0.45, zorder=5)

        ax1t.set_ylabel(r'$\Delta\mathrm{RV}_{ZP}\ (\mathrm{km\,s^{-1}})$',
                        fontsize=LABEL_SIZE-3, fontweight='bold',
                        color=ZP_COLOR, labelpad=6)
        ax1t.set_ylim(ZP_LIMITS)
        ax1t.tick_params(axis='y', labelcolor=ZP_COLOR, labelsize=TICK_SIZE,
                         width=1.8, length=8, which='major', direction='in')
        ax1t.tick_params(axis='y', width=1.0, length=5,
                         which='minor', direction='in')
        ax1t.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1t.spines['right'].set_edgecolor(ZP_COLOR)
        ax1t.spines['right'].set_linewidth(1.8)
        ax1t.legend(loc='upper right', fontsize=LEGEND_SIZE,
                    framealpha=0.85, edgecolor='gray', fancybox=False,
                    facecolor='#111111', labelcolor=ZP_COLOR)

    # ── Axis styling (Panel 1) ──────────────────────────────────────────────────
    ax1.set_ylabel(r'Weighted Avg RV (km s$^{-1}$)',
                   fontsize=LABEL_SIZE, fontweight='bold',
                   color='white', labelpad=6)
    ax1.set_xlim(DIST_LIMITS)
    ax1.set_ylim(RV_LIMITS)
    ax1.grid(True, alpha=0.08, ls='--', color='lightgray', lw=0.5, zorder=1)

    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE,
                    colors='white', width=1.8, length=8, direction='in',
                    top=True, right=False)
    ax1.tick_params(axis='both', which='minor', width=1.0, length=5,
                    direction='in', top=True, right=False, colors='white')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    for spine in ax1.spines.values():
        spine.set_edgecolor('white'); spine.set_linewidth(1.5)
    ax1.yaxis.label.set_color('white')
    ax1.tick_params(colors='white', which='both')

    legend_els = [
        Patch(facecolor='#7B3F00', edgecolor='white', alpha=0.85,
              label=f'Weighted Avg  (N = {len(weighted_dist):,})'),
        Line2D([0],[0], color='white', lw=1.5, ls='-',
               label='Density contours'),
    ]
    ax1.legend(handles=legend_els, loc='upper left',
               fontsize=LEGEND_SIZE, framealpha=0.85,
               edgecolor='gray', fancybox=False,
               facecolor='#111111', labelcolor='white')

    plt.setp(ax1.get_xticklabels(), visible=False)

    # ==========================================================================
    #  PANEL 2 — Measurement Counts  (WHITE background)
    # ==========================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor('white')

    ax2.semilogy(bin_centers, data['raw_count']    + 1, color=COL_RAW, lw=1.8,
                 label='Raw RV count',               alpha=0.9)
    ax2.semilogy(bin_centers, data['n_meas_sum']   + 1, color=COL_NM,  lw=1.8,
                 label=r'$\Sigma\,n_\mathrm{meas}$', alpha=0.9)
    ax2.semilogy(bin_centers, data['outlier_count']+ 1, color=COL_OUT, lw=1.8,
                 label='Outlier count',              alpha=0.9)

    ax2.fill_between(bin_centers, 1, data['raw_count']  + 1,
                     color=COL_RAW, alpha=0.12)
    ax2.fill_between(bin_centers, 1, data['n_meas_sum'] + 1,
                     color=COL_NM,  alpha=0.12)

    ax2.set_ylabel(r'$N\ \mathrm{(log)}$',
                   fontsize=LABEL_SIZE-4, fontweight='bold')
    ax2.legend(loc='upper right', ncol=3, fontsize=LEGEND_SIZE-1,
               framealpha=0.92, edgecolor='silver', fancybox=False,
               columnspacing=0.8, handletextpad=0.4)
    ax2.grid(True, alpha=0.25, ls='--', which='both', color='gray', lw=0.5)
    ax2.set_xlim(DIST_LIMITS); ax2.set_ylim(1, None)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE-1,
                    width=1.8, length=8, direction='in', top=True, right=True)
    ax2.tick_params(axis='both', which='minor', width=1.0, length=5,
                    direction='in', top=True, right=True)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ==========================================================================
    #  PANEL 3 — Error Distribution  (WHITE background)
    # ==========================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor('white')

    ax3.plot(bin_centers, mean_err_r, color=COL_ERR_R, lw=1.8, ls='--',
             label='Raw RV error',       alpha=0.9)
    ax3.plot(bin_centers, mean_err_w, color=COL_ERR_W, lw=1.8, ls='-.',
             label='Weighted avg error', alpha=0.9)
    ax3.plot(bin_centers, mean_err_z, color=COL_ERR_Z, lw=2.2, ls='-',
             label='ZP corrected error', alpha=0.95)
    ax3.fill_between(bin_centers, 0, mean_err_z,
                     color=COL_ERR_Z, alpha=0.15,
                     where=np.isfinite(mean_err_z))

    all_e = np.concatenate([mean_err_r[np.isfinite(mean_err_r)],
                            mean_err_w[np.isfinite(mean_err_w)],
                            mean_err_z[np.isfinite(mean_err_z)]])
    if len(all_e) > 0:
        ax3.set_ylim(0, min(np.percentile(all_e, 95)*1.5, 50))

    ax3.set_ylabel(r'$\sigma_\mathrm{RV}\ (\mathrm{km\,s^{-1}})$',
                   fontsize=LABEL_SIZE-4, fontweight='bold')
    ax3.set_xlabel(r'Distance (kpc)', fontsize=LABEL_SIZE, fontweight='bold')
    ax3.legend(loc='upper right', ncol=3, fontsize=LEGEND_SIZE-1,
               framealpha=0.92, edgecolor='silver', fancybox=False,
               columnspacing=0.8, handletextpad=0.4)
    ax3.grid(True, alpha=0.25, ls='--', color='gray', lw=0.5)
    ax3.set_xlim(DIST_LIMITS)
    ax3.tick_params(axis='both', which='major', labelsize=TICK_SIZE-1,
                    width=1.8, length=8, direction='in', top=True, right=True)
    ax3.tick_params(axis='both', which='minor', width=1.0, length=5,
                    direction='in', top=True, right=True)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(5))

    # ── Save ───────────────────────────────────────────────────────────────────
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n  Dashboard saved → {os.path.abspath(OUTPUT_PLOT)}")


# ==============================================================================
# 5. MAIN
# ==============================================================================
if __name__ == "__main__":
    print("\n"+"="*70)
    print("  RV ANALYSIS DASHBOARD  v5")
    print("  inferno | gaussian-smoothed contours | gold ZP | dark-navy panel")
    print("="*70+"\n")

    replot_only = '--replot' in sys.argv or '-r' in sys.argv

    if replot_only and os.path.exists(CACHE_FILE):
        data = load_cache()
    else:
        if os.path.exists(CACHE_FILE):
            print(f"Cache exists. Pass --replot / -r to skip reprocessing.\n")
        data = process_all_data()
        save_cache(data)

    create_dashboard(data)

    print("\n"+"="*70)
    print("  COMPLETE!")
    print("="*70)
    print(f"\n  python {sys.argv[0]}           # full reprocess + plot")
    print(f"  python {sys.argv[0]} --replot  # replot from cache only")
    print("="*70+"\n")



    