#!/usr/bin/env python3
"""
================================================================================
STELLAR CATALOG VISUALIZATION — PAPER-QUALITY EDITION v5.0
================================================================================
Generates FOUR publication figures (in this order):
  1.  Multi-panel Sky Grid   (All, SOS, 125M, 33M, LAMOST, DESI, SDSS,
                               GC, OC, Sgr Stream, DWG)
  2.  Extinction Sky Map     (E(BP-RP) + A_G side-by-side, +/-5 deg lines)
  3.  Combined HR Diagram    (unfilt gray + filt turbo + contours;
                               NO extra scatter; SPACIOUS Teff top axis)
  4.  Kiel Diagram           (unfilt gray + filt turbo + contours)

Survey auto-detection (case-insensitive):
  SOS      → survey_name matches /sos/
  LAMOST   → survey_name matches /lamost/
  SDSS     → survey_name matches /sdss|boss/
  Gaia125M → survey_name matches /gaia.*125|125.*gaia|125m/
  Gaia33M  → survey_name matches /gaia.*33|33.*rv|33m|rv.*33/
  DESI     → survey_name matches /desi/

Usage:
  # Compute from FITS chunks, then plot
  python stellar_plots_paper_v5.py \
      --input /path/to/fits_dir_or_file \
      --compute \
      --output-data stellar_data_v5.npz \
      --plot \
      --output-dir ./paper_figs \
      --chunk-size 1000000

  # Plot only from existing NPZ
  python stellar_plots_paper_v5.py \
      --load-data stellar_data_v5.npz \
      --plot \
      --output-dir ./paper_figs

Author: Sutirtha (v5 refactor)
================================================================================
"""

import os
import sys
import re
import gc as gcmod
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import AutoMinorLocator, NullLocator, MultipleLocator
import matplotlib.gridspec as gridspec
from matplotlib import patches
import warnings
warnings.filterwarnings('ignore')

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False
    print("WARNING: healpy not available -- sky maps disabled")

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available -- contour smoothing disabled")

try:
    from astropy.io import fits
    from astropy.table import Table
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("WARNING: astropy not available -- FITS reading disabled")

# =============================================================================
# PUBLICATION STYLE
# =============================================================================
plt.rcParams.update({
    'text.usetex':          False,
    'text.parse_math':      True,
    'mathtext.fontset':     'dejavusans',
    'font.family':          'serif',
    'font.serif':           ['DejaVu Serif', 'Times', 'Palatino'],
    'font.size':            18,
    'axes.labelsize':       28,
    'axes.titlesize':       26,
    'xtick.labelsize':      20,
    'ytick.labelsize':      20,
    'legend.fontsize':      13,
    'legend.title_fontsize': 14,
    'xtick.major.size':     12,
    'xtick.major.width':    2.2,
    'ytick.major.size':     12,
    'ytick.major.width':    2.2,
    'xtick.minor.size':     6,
    'xtick.minor.width':    1.5,
    'ytick.minor.size':     6,
    'ytick.minor.width':    1.5,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.top':            True,
    'ytick.right':          True,
    'xtick.minor.visible':  True,
    'ytick.minor.visible':  True,
    'axes.linewidth':       2.2,
    'axes.grid':            False,
    'axes.facecolor':       'white',
    'figure.facecolor':     'white',
    'savefig.facecolor':    'white',
    'figure.dpi':           100,
    'savefig.dpi':          200,
})

# =============================================================================
# CONSTANTS
# =============================================================================
NSIDE = 64

HR_BPRP_BINS    = 600
HR_MG_BINS      = 600
HR_BPRP_RANGE   = (-1.5, 5.5)
HR_MG_RANGE     = (-8.0, 16.0)

KIEL_TEFF_BINS  = 400
KIEL_LOGG_BINS  = 400
KIEL_TEFF_RANGE = (3000, 10000)
KIEL_LOGG_RANGE = (-0.5, 5.5)

# Survey name regex patterns (case-insensitive)
SURVEY_PATTERNS = {
    'SOS':      re.compile(r'sos',                          re.IGNORECASE),
    'LAMOST':   re.compile(r'lamost',                       re.IGNORECASE),
    'SDSS':     re.compile(r'sdss|boss',                    re.IGNORECASE),
    'Gaia125M': re.compile(r'gaia.{0,10}125|125.{0,10}gaia|125m_bp', re.IGNORECASE),
    'Gaia33M':  re.compile(r'gaia.{0,10}33|33.{0,10}gaia|33m_rv|rv.{0,10}33', re.IGNORECASE),
    'DESI':     re.compile(r'desi',                         re.IGNORECASE),
}

# Member catalog CSV definitions
MEMBER_CATALOGS = {
    'GC':  {'filename': 'GC_members_with_RV.csv',
             'ra_col': 'ra',    'dec_col': 'dec',
             'label': 'Globular Clusters', 'abbrev': 'GC'},
    'OC':  {'filename': 'OC_members.csv',
             'ra_col': 'RAdeg', 'dec_col': 'DEdeg',
             'label': 'Open Clusters',     'abbrev': 'OC'},
    'SGR': {'filename': 'Sgr_stream_members.csv',
             'ra_col': 'ra',    'dec_col': 'dec',
             'label': 'Sgr Stream',        'abbrev': 'Sgr'},
    'DWG': {'filename': 'DWG_members.csv',
             'ra_col': 'ra_x',  'dec_col': 'dec_x',
             'label': 'Dwarf Galaxies',    'abbrev': 'DWG'},
}

# Sky panel definitions — (label, npz_key_or_None, source_type)
SKY_PANELS = [
    ('All Surveys',        'sky_full_filt',          'npz'),
    ('Gaia — SOS',         'sky_SOS',                'npz'),
    ('Gaia — 125M BP-RP',  'sky_Gaia125M',           'npz'),
    ('Gaia — 33M RV',      'sky_Gaia33M',            'npz'),
    ('LAMOST',             'sky_LAMOST',             'npz'),
    ('DESI',               'sky_DESI',               'npz'),
    ('SDSS/BOSS',          'sky_SDSS',               'npz'),
    ('Globular Clusters',  None,                      'member_GC'),
    ('Open Clusters',      None,                      'member_OC'),
    ('Sgr Stream',         None,                      'member_SGR'),
    ('Dwarf Galaxies',     None,                      'member_DWG'),
]


# =============================================================================
# STELLAR POPULATION BOUNDARIES
# =============================================================================
def _population_defs():
    P = {}
    P['Main Sequence'] = {
        'vertices': np.array([
            [-0.2,15.5],[0.0,13.0],[0.3,10.0],[0.6,8.0],[1.0,6.5],[1.5,5.0],
            [2.0,4.0],[2.5,3.5],[3.0,3.3],[3.5,3.2],[4.0,3.2],
            [4.0,16.0],[-0.2,16.0]]),
        'color':'#DDAA00','abbrev':'MS','ls':':','lw':2.2}
    P['RGB'] = {
        'vertices': np.array([
            [0.8,2.5],[1.0,1.5],[1.2,0.5],[1.4,-0.5],[1.6,-1.5],[1.9,-2.5],
            [2.2,-3.0],[2.5,-2.5],[2.7,-1.5],[2.5,-0.5],[2.2,0.5],
            [1.8,1.5],[1.4,2.5]]),
        'color':'#DC143C','abbrev':'RGB','ls':'-','lw':2.2}
    P['AGB'] = {
        'vertices': np.array([
            [2.0,-1.0],[2.5,-1.5],[3.0,-1.3],[3.5,-0.8],[4.0,0.0],
            [4.3,0.8],[4.0,1.5],[3.5,1.2],[3.0,0.5],[2.5,0.0]]),
        'color':'#8B4513','abbrev':'AGB','ls':'-','lw':2.2}
    P['White Dwarfs'] = {
        'vertices': np.array([[-0.6,9.0],[0.8,9.0],[0.8,16.0],[-0.6,16.0]]),
        'color':'#9370DB','abbrev':'WD','ls':'--','lw':1.8}
    P['MS Turnoff'] = {
        'vertices': np.array([[0.2,2.0],[0.9,2.0],[1.1,4.5],[0.6,4.5]]),
        'color':'#FFD700','abbrev':'MSTO','ls':'--','lw':1.8}
    P['Subgiants'] = {
        'vertices': np.array([[0.8,2.5],[1.8,2.5],[2.0,5.0],[1.2,5.0]]),
        'color':'#FF8C00','abbrev':'SG','ls':'--','lw':1.8}
    P['RGB Bump'] = {
        'vertices': np.array([[0.9,0.2],[1.5,0.2],[1.5,1.8],[0.9,1.8]]),
        'color':'#FF6347','abbrev':'RGB Bump','ls':':','lw':2.0}
    P['Red Clump'] = {
        'vertices': np.array([[0.5,-0.5],[1.3,-0.5],[1.3,1.5],[0.5,1.5]]),
        'color':'#B22222','abbrev':'RC/HB','ls':'-','lw':2.0}
    P['TRGB'] = {
        'vertices': np.array([[1.0,-5.5],[3.0,-5.5],[3.0,-3.5],[1.0,-3.5]]),
        'color':'#8B0000','abbrev':'TRGB','ls':':','lw':2.0}
    P['Blue HB'] = {
        'vertices': np.array([[-0.5,-1.5],[0.4,-1.5],[0.4,2.0],[-0.5,2.0]]),
        'color':'#1E90FF','abbrev':'BHB','ls':'-','lw':2.0}
    P['EHB/sdO'] = {
        'vertices': np.array([[-0.6,2.0],[0.0,2.0],[0.0,7.0],[-0.6,7.0]]),
        'color':'#4682B4','abbrev':'EHB','ls':'--','lw':1.8}
    return P


# =============================================================================
# UTILITY
# =============================================================================
def log(msg):
    import datetime
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def detect_survey(name):
    """Return canonical survey key for a given survey_name string, or None."""
    for key, pat in SURVEY_PATTERNS.items():
        if pat.search(name):
            return key
    return None


def ra_dec_to_galactic(ra, dec):
    ra_r  = np.radians(ra);  dec_r = np.radians(dec)
    x = np.cos(dec_r)*np.cos(ra_r)
    y = np.cos(dec_r)*np.sin(ra_r)
    z = np.sin(dec_r)
    R = np.array([
        [-0.054875539390, -0.873437104725, -0.483834991775],
        [ 0.494109453633, -0.444829594298,  0.746982248696],
        [-0.867666135681, -0.198076389622,  0.455983794523]])
    xyz = R @ np.array([x, y, z])
    l = np.degrees(np.arctan2(xyz[1], xyz[0]))
    b = np.degrees(np.arcsin(np.clip(xyz[2], -1, 1)))
    return np.mod(l, 360.0), b


def _apply_style(ax, xlabel=None, ylabel=None, title=None,
                 xlim=None, ylim=None, minor=True, grid=True):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=28, fontweight='bold', labelpad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=28, fontweight='bold', labelpad=12)
    if title:
        ax.set_title(title, fontsize=24, fontweight='bold', pad=16)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(which='major', length=12, width=2.2, direction='in',
                   top=True, right=True, labelsize=20)
    ax.tick_params(which='minor', length=6,  width=1.5, direction='in',
                   top=True, right=True)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    if grid:
        ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.5, zorder=1)


# =============================================================================
# SPACIOUS Teff top-axis  — enforces minimum angular separation
# =============================================================================
def _add_teff_axis(ax, bprp_edges, teff_bprp_mean):
    """
    Temperature axis on top of HR diagram.
    Ticks are placed only when they are >0.45 mag apart in BP-RP
    so labels never overlap, even for dense sequences.
    Outward tick direction keeps labels above the frame.
    """
    if teff_bprp_mean is None or not np.any(np.isfinite(teff_bprp_mean)):
        return None

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    bprp_c = (bprp_edges[:-1] + bprp_edges[1:]) / 2.0
    valid  = np.isfinite(teff_bprp_mean)
    if not np.any(valid):
        return None

    # Candidate temperatures — well spread
    teff_cands = [8500, 7500, 6500, 5500, 4800, 4200, 3700, 3400]
    MIN_SEP = 0.50    # minimum BP-RP separation between adjacent ticks [mag]

    t_pos, t_lbl = [], []
    last_pos = -999.0

    for t in teff_cands:
        idx = np.argmin(np.abs(teff_bprp_mean[valid] - t))
        pos = float(bprp_c[valid][idx])
        diff = abs(teff_bprp_mean[valid][idx] - t)
        in_range = (HR_BPRP_RANGE[0] + 0.2) <= pos <= (HR_BPRP_RANGE[1] - 0.2)
        if in_range and diff < 600 and (pos - last_pos) >= MIN_SEP:
            t_pos.append(pos)
            t_lbl.append(f'{t:,}')
            last_pos = pos

    if not t_pos:
        return None

    ax_top.set_xticks(t_pos)
    ax_top.set_xticklabels(t_lbl, fontsize=15, fontweight='bold', rotation=0)
    ax_top.tick_params(which='major', length=10, width=1.8,
                       direction='out', pad=8, labelsize=15)
    ax_top.tick_params(which='minor', length=0)
    ax_top.xaxis.set_minor_locator(NullLocator())
    ax_top.set_xlabel(r'$T_{\rm eff}$  [K]',
                      fontsize=22, fontweight='bold', labelpad=28)
    ax_top.spines['top'].set_linewidth(1.5)
    ax_top.spines['top'].set_color('#555555')
    return ax_top


# =============================================================================
# POPULATION BOUNDARIES
# =============================================================================
def _draw_pop_boundaries(ax, pop_counts=None, zorder_base=20):
    pops = _population_defs()
    for name, p in pops.items():
        v = p['vertices']
        poly = patches.Polygon(v, fill=False, edgecolor=p['color'],
                               linewidth=p['lw'], linestyle=p['ls'],
                               alpha=0.90, zorder=zorder_base)
        ax.add_patch(poly)
        x_mid = v[:,0].mean()
        y_top = v[:,1].min()
        cnt_str = ''
        if pop_counts:
            c = pop_counts.get(name, 0)
            if c > 0:
                cnt_str = f'\n{c:,}'
        ax.text(x_mid, y_top, f"{p['abbrev']}{cnt_str}",
                fontsize=10, fontweight='bold', color=p['color'],
                ha='center', va='bottom', zorder=zorder_base+2,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=p['color'], alpha=0.92, linewidth=1.4))


def _add_density_contours(ax, hist2d, x_edges, y_edges,
                          n_levels=5, color='white', linewidths=0.9,
                          smooth_sigma=3.0):
    if not SCIPY_AVAILABLE:
        return
    h = hist2d.copy().astype(float)
    h[~np.isfinite(h)] = 0.0
    smoothed  = gaussian_filter(h, sigma=smooth_sigma)
    nonzero   = smoothed[smoothed > 0]
    if len(nonzero) < 10:
        return
    levels = np.unique(np.percentile(nonzero, np.linspace(20, 95, n_levels)))
    if len(levels) < 2:
        return
    x_c = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_c = (y_edges[:-1] + y_edges[1:]) / 2.0
    ax.contour(x_c, y_c, smoothed, levels=levels,
               colors=color, linewidths=linewidths, alpha=0.75, zorder=12)


# =============================================================================
# MOLLWEIDE HELPERS
# =============================================================================
def _mollweide_grid(nside, n_lon=1440, n_lat=720):
    l_grid = np.linspace(-180, 180, n_lon)
    b_grid = np.linspace(-90,  90,  n_lat)
    L, B   = np.meshgrid(l_grid, b_grid)
    theta  = np.radians(90.0 - B.ravel())
    phi    = np.radians(np.mod(L.ravel(), 360.0))
    pix    = hp.ang2pix(nside, theta, phi)
    return L, B, -np.radians(L), np.radians(B), pix


def _style_mollweide(ax, title, n_stars):
    ax.grid(True, alpha=0.30, linestyle='-', linewidth=0.5, color='gray')
    lon_ticks = np.array([150,120,90,60,30,0,-30,-60,-90,-120,-150])
    ax.set_xticks(np.radians(lon_ticks))
    ax.set_xticklabels([f'{int(np.mod(-ld,360))}\u00b0' for ld in lon_ticks],
                       fontsize=12, fontweight='bold')
    lat_ticks = [-60,-30,0,30,60]
    ax.set_yticks(np.radians(lat_ticks))
    ax.set_yticklabels([f'{ld}\u00b0' for ld in lat_ticks],
                       fontsize=12, fontweight='bold')
    n_str = f'  N = {n_stars:,}' if n_stars > 0 else ''
    ax.set_title(f'{title}{n_str}', fontsize=16, fontweight='bold', pad=6)


def _galactic_annotations(ax, l_grid):
    ax.plot(-np.radians(l_grid), np.zeros_like(l_grid),
            '-', color='black', lw=0.8, alpha=0.35, zorder=9)
    ax.plot(0, 0, '+', color='red', ms=12, mew=2.2, zorder=10)


# =============================================================================
# SKY MAP ACCUMULATION HELPERS
# =============================================================================
def compute_healpix_sky(ra, dec, nside, values=None):
    """
    Accumulate star counts (or sum of values) into HEALPix map.
    Returns count_map and (optionally) value_sum_map.
    """
    if not HEALPY_AVAILABLE:
        return None, None
    l, b   = ra_dec_to_galactic(ra, dec)
    theta  = np.radians(90.0 - b)
    phi    = np.radians(l)
    npix   = hp.nside2npix(nside)
    cnt    = np.zeros(npix, dtype=np.float64)
    pix    = hp.ang2pix(nside, theta, phi)
    np.add.at(cnt, pix, 1)
    if values is not None:
        vsum = np.zeros(npix, dtype=np.float64)
        ok   = np.isfinite(values)
        np.add.at(vsum, pix[ok], values[ok])
        return cnt, vsum
    return cnt, None


# =============================================================================
# COMPUTE  — read FITS chunks, build histograms & sky maps
# =============================================================================

def _iter_fits_files(input_path):
    """Yield FITS file paths from a file or a directory."""
    if os.path.isfile(input_path):
        yield input_path
    elif os.path.isdir(input_path):
        patterns = ['*.fits', '*.fit', '*.fits.gz', '*.FITS']
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(input_path, pat)))
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No FITS files found in {input_path}")
        log(f"Found {len(files)} FITS files in {input_path}")
        for f in files:
            yield f
    else:
        raise ValueError(f"Input path not found: {input_path}")


def compute_and_save(input_path, output_npz, chunk_size=1_000_000, nside=NSIDE):
    """
    Read all FITS files in input_path in chunks, accumulate histograms & sky maps,
    then save to output_npz.

    Expected FITS columns (adapt column names below to your schema):
      ra, dec           — sky coordinates (degrees)
      bp_rp_0           — dereddened BP-RP colour
      mg                — absolute magnitude M_G
      teff_gspphot      — Teff (for top axis & Kiel)
      logg_gspphot      — log g (Kiel)
      mh_gspphot        — [Fe/H]
      ebpminrp_gspphot  — E(BP-RP) extinction
      ag_gspphot        — A_G extinction
      survey_name       — string survey label
      filtered          — boolean / 0-1 flag for quality filter
    """
    if not ASTROPY_AVAILABLE:
        log("ERROR: astropy required for --compute mode")
        sys.exit(1)

    log("=" * 70)
    log(f"COMPUTE MODE  — input: {input_path}")
    log(f"Chunk size: {chunk_size:,}  |  NSIDE: {nside}")
    log("=" * 70)

    npix = hp.nside2npix(nside) if HEALPY_AVAILABLE else 0

    # Bin edges
    bprp_edges = np.linspace(HR_BPRP_RANGE[0],  HR_BPRP_RANGE[1],  HR_BPRP_BINS+1)
    mg_edges   = np.linspace(HR_MG_RANGE[0],    HR_MG_RANGE[1],    HR_MG_BINS+1)
    teff_edges = np.linspace(KIEL_TEFF_RANGE[0], KIEL_TEFF_RANGE[1], KIEL_TEFF_BINS+1)
    logg_edges = np.linspace(KIEL_LOGG_RANGE[0], KIEL_LOGG_RANGE[1], KIEL_LOGG_BINS+1)

    # Accumulators — unfiltered and filtered
    hr_unfilt   = np.zeros((HR_MG_BINS,   HR_BPRP_BINS),   dtype=np.float64)
    hr_filt     = np.zeros((HR_MG_BINS,   HR_BPRP_BINS),   dtype=np.float64)
    kiel_unfilt = np.zeros((KIEL_LOGG_BINS, KIEL_TEFF_BINS), dtype=np.float64)
    kiel_filt   = np.zeros((KIEL_LOGG_BINS, KIEL_TEFF_BINS), dtype=np.float64)

    # For Teff mean along BP-RP (HR top axis)
    teff_bprp_sum = np.zeros(HR_BPRP_BINS, dtype=np.float64)
    teff_bprp_cnt = np.zeros(HR_BPRP_BINS, dtype=np.float64)

    # For [Fe/H] mean on Kiel
    feh_kiel_sum  = np.zeros((KIEL_LOGG_BINS, KIEL_TEFF_BINS), dtype=np.float64)
    feh_kiel_cnt  = np.zeros((KIEL_LOGG_BINS, KIEL_TEFF_BINS), dtype=np.float64)

    # Sky maps per survey + full + extinction
    sky_full_unfilt = np.zeros(npix, dtype=np.float64) if npix else None
    sky_full_filt   = np.zeros(npix, dtype=np.float64) if npix else None
    sky_survey      = {}   # key: canonical survey name → count map

    ext_ebprp_sum = np.zeros(npix, dtype=np.float64) if npix else None
    ext_ag_sum    = np.zeros(npix, dtype=np.float64) if npix else None
    ext_cnt       = np.zeros(npix, dtype=np.float64) if npix else None

    n_total = 0
    n_filt  = 0

    fits_files = list(_iter_fits_files(input_path))
    log(f"Processing {len(fits_files)} FITS file(s) ...")

    for fits_idx, fits_file in enumerate(fits_files):
        log(f"  [{fits_idx+1}/{len(fits_files)}] {os.path.basename(fits_file)}")
        try:
            with fits.open(fits_file, memmap=True) as hdul:
                data_hdu = hdul[1]
                total_rows = data_hdu.header['NAXIS2']
                log(f"    Total rows: {total_rows:,}")

                for start in range(0, total_rows, chunk_size):
                    end    = min(start + chunk_size, total_rows)
                    chunk  = data_hdu.data[start:end]
                    nrows  = end - start

                    # ---- Read columns (with fallbacks) ----
                    def _col(names, default=None):
                        for n in names:
                            if n in chunk.dtype.names:
                                return chunk[n].copy()
                        return default

                    ra   = _col(['ra', 'RA', 'RAJ2000'], np.full(nrows, np.nan))
                    dec  = _col(['dec', 'DEC', 'DEJ2000', 'de'], np.full(nrows, np.nan))
                    bprp = _col(['bp_rp_0', 'bp_rp', 'BP_RP_0', 'BP_RP'],
                                np.full(nrows, np.nan))
                    mg   = _col(['mg', 'MG', 'M_G', 'm_g'],
                                np.full(nrows, np.nan))
                    teff = _col(['teff_gspphot', 'teff', 'TEFF', 'Teff'],
                                np.full(nrows, np.nan))
                    logg = _col(['logg_gspphot', 'logg', 'LOGG', 'log_g'],
                                np.full(nrows, np.nan))
                    feh  = _col(['mh_gspphot', 'feh', 'FEH', '[Fe/H]', 'met'],
                                np.full(nrows, np.nan))
                    ebp  = _col(['ebpminrp_gspphot', 'ebpminrp', 'E_BP_RP',
                                 'ebprp'], np.full(nrows, np.nan))
                    ag   = _col(['ag_gspphot', 'ag', 'AG', 'A_G'],
                                np.full(nrows, np.nan))

                    # Filter flag
                    filt_col = _col(['filtered', 'FILTERED', 'flag_filt',
                                     'quality_flag'], None)
                    if filt_col is not None:
                        mask_filt = filt_col.astype(bool)
                    else:
                        mask_filt = np.ones(nrows, dtype=bool)

                    # Survey name column
                    srv_col = _col(['survey_name', 'SURVEY_NAME', 'survey',
                                    'Survey', 'SOURCE'], None)

                    n_total += nrows
                    n_filt  += int(np.sum(mask_filt))

                    ra_ok  = np.isfinite(ra)  & np.isfinite(dec)

                    # ---- HR histogram ----
                    ok_hr = np.isfinite(bprp) & np.isfinite(mg)
                    if np.any(ok_hr):
                        h, _, _ = np.histogram2d(
                            bprp[ok_hr], mg[ok_hr],
                            bins=[bprp_edges, mg_edges])
                        hr_unfilt += h.T
                    ok_hr_f = ok_hr & mask_filt
                    if np.any(ok_hr_f):
                        h, _, _ = np.histogram2d(
                            bprp[ok_hr_f], mg[ok_hr_f],
                            bins=[bprp_edges, mg_edges])
                        hr_filt += h.T

                    # Teff mean along BP-RP axis
                    ok_t = ok_hr_f & np.isfinite(teff)
                    if np.any(ok_t):
                        idx_bprp = np.searchsorted(bprp_edges, bprp[ok_t]) - 1
                        valid_idx = (idx_bprp >= 0) & (idx_bprp < HR_BPRP_BINS)
                        np.add.at(teff_bprp_sum, idx_bprp[valid_idx],
                                  teff[ok_t][valid_idx])
                        np.add.at(teff_bprp_cnt, idx_bprp[valid_idx], 1)

                    # ---- Kiel histogram ----
                    ok_k = np.isfinite(teff) & np.isfinite(logg)
                    if np.any(ok_k):
                        h, _, _ = np.histogram2d(
                            teff[ok_k], logg[ok_k],
                            bins=[teff_edges, logg_edges])
                        kiel_unfilt += h.T
                    ok_k_f = ok_k & mask_filt
                    if np.any(ok_k_f):
                        h, _, _ = np.histogram2d(
                            teff[ok_k_f], logg[ok_k_f],
                            bins=[teff_edges, logg_edges])
                        kiel_filt += h.T
                        # [Fe/H] mean
                        ok_fe = ok_k_f & np.isfinite(feh)
                        if np.any(ok_fe):
                            ix = np.clip(
                                np.searchsorted(teff_edges, teff[ok_fe])-1,
                                0, KIEL_TEFF_BINS-1)
                            iy = np.clip(
                                np.searchsorted(logg_edges, logg[ok_fe])-1,
                                0, KIEL_LOGG_BINS-1)
                            flat = iy * KIEL_TEFF_BINS + ix
                            np.add.at(feh_kiel_sum.ravel(), flat, feh[ok_fe])
                            np.add.at(feh_kiel_cnt.ravel(), flat, 1)

                    # ---- Sky maps ----
                    if HEALPY_AVAILABLE and np.any(ra_ok):
                        ra_v  = ra[ra_ok].astype(np.float64)
                        dec_v = dec[ra_ok].astype(np.float64)
                        filt_v = mask_filt[ra_ok]

                        l, b  = ra_dec_to_galactic(ra_v, dec_v)
                        theta = np.radians(90.0 - b)
                        phi   = np.radians(l)
                        pix   = hp.ang2pix(nside, theta, phi)

                        np.add.at(sky_full_unfilt, pix, 1)
                        np.add.at(sky_full_filt,   pix[filt_v], 1)

                        # Extinction
                        ebp_v = ebp[ra_ok]
                        ag_v  = ag[ra_ok]
                        ok_e  = np.isfinite(ebp_v)
                        if np.any(ok_e):
                            np.add.at(ext_ebprp_sum, pix[ok_e], ebp_v[ok_e])
                            np.add.at(ext_cnt,        pix[ok_e], 1)
                        ok_ag = np.isfinite(ag_v)
                        if np.any(ok_ag):
                            np.add.at(ext_ag_sum, pix[ok_ag], ag_v[ok_ag])

                        # Per-survey sky maps
                        if srv_col is not None:
                            srvs_v = srv_col[ra_ok]
                            unique_srvs = np.unique(srvs_v)
                            for sname in unique_srvs:
                                sname_str = str(sname)
                                canon = detect_survey(sname_str)
                                if canon is None:
                                    continue
                                log(f"      Detected survey: '{sname_str}' → {canon}")
                                smask = (srvs_v == sname) & filt_v
                                if not np.any(smask):
                                    continue
                                if canon not in sky_survey:
                                    sky_survey[canon] = np.zeros(npix,
                                                                  dtype=np.float64)
                                np.add.at(sky_survey[canon], pix[smask], 1)

                    log(f"      chunk {start//chunk_size}: rows {start:,}-{end:,}"
                        f"  (total so far: {n_total:,}, filt: {n_filt:,})")
                    gcmod.collect()

        except Exception as e:
            log(f"  ERROR reading {fits_file}: {e}")
            import traceback; traceback.print_exc()
            continue

    log(f"Finished reading.  Total rows: {n_total:,}  |  Filtered: {n_filt:,}")

    # ---- Compute means ----
    with np.errstate(invalid='ignore', divide='ignore'):
        teff_mean = np.where(teff_bprp_cnt > 0,
                             teff_bprp_sum / teff_bprp_cnt, np.nan)
        feh_mean  = np.where(feh_kiel_cnt  > 0,
                             feh_kiel_sum  / feh_kiel_cnt,  np.nan)
        ext_ebprp_mean = np.where(ext_cnt > 0,
                                  ext_ebprp_sum / ext_cnt, np.nan) \
                         if HEALPY_AVAILABLE else None
        ext_ag_mean    = np.where(ext_cnt > 0,
                                  ext_ag_sum    / ext_cnt, np.nan) \
                         if HEALPY_AVAILABLE else None

    # ---- Pack and save ----
    save_dict = dict(
        _bprp_edges   = bprp_edges,
        _mg_edges     = mg_edges,
        _teff_edges   = teff_edges,
        _logg_edges   = logg_edges,
        _nside        = np.array([nside]),
        _n_rows       = np.array([n_total]),
        _n_filtered   = np.array([n_filt]),
        hr_full_unfilt  = hr_unfilt.astype(np.float32),
        hr_full_filt    = hr_filt.astype(np.float32),
        kiel_full_unfilt= kiel_unfilt.astype(np.float32),
        kiel_full_filt  = kiel_filt.astype(np.float32),
        teff_bprp_mean  = teff_mean.astype(np.float32),
        kiel_feh_mean   = feh_mean.astype(np.float32),
    )
    if HEALPY_AVAILABLE:
        save_dict.update(dict(
            sky_full_unfilt   = sky_full_unfilt.astype(np.float32),
            sky_full_filt     = sky_full_filt.astype(np.float32),
            ext_ebprp_mean    = ext_ebprp_mean.astype(np.float32),
            ext_ag_mean       = ext_ag_mean.astype(np.float32),
        ))
        for k, v in sky_survey.items():
            save_dict[f'sky_{k}'] = v.astype(np.float32)

    np.savez_compressed(output_npz, **save_dict)
    sz = os.path.getsize(output_npz) / 1e6
    log(f"NPZ saved: {output_npz}  ({sz:.1f} MB)")
    return save_dict


# =============================================================================
# LOAD NPZ
# =============================================================================
def load_npz(npz_file):
    log(f"Loading {npz_file} ...")
    data = dict(np.load(npz_file, allow_pickle=False))
    log(f"  Keys: {sorted(data.keys())}")
    # Reconstruct pop count dicts if present
    if 'pop_names_filt' in data:
        data['pop_counts_filt_dict'] = dict(
            zip(data['pop_names_filt'], data['pop_counts_filt']))
    log("NPZ loaded.")
    return data


# =============================================================================
# MEMBER CATALOG HELPERS
# =============================================================================
def load_member_csv(member_dir, catalog_key):
    cfg      = MEMBER_CATALOGS[catalog_key]
    filepath = os.path.join(member_dir, cfg['filename'])
    if not os.path.isfile(filepath):
        log(f"  WARNING: {filepath} not found -- skipping {catalog_key}")
        return None, None
    import pandas as pd
    df      = pd.read_csv(filepath)
    ra_col  = cfg['ra_col']
    dec_col = cfg['dec_col']
    if ra_col not in df.columns or dec_col not in df.columns:
        log(f"  WARNING: columns {ra_col}/{dec_col} not in {filepath}")
        return None, None
    ra  = df[ra_col].values.astype(np.float64)
    dec = df[dec_col].values.astype(np.float64)
    ok  = np.isfinite(ra) & np.isfinite(dec)
    log(f"  {catalog_key}: {ok.sum():,} valid / {len(ra):,} total")
    return ra[ok], dec[ok]


def load_member_sky(data, member_dir, nside):
    member_sky = {}
    member_n   = {}
    for key in MEMBER_CATALOGS:
        npz_key = f'sky_member_{key}'
        if npz_key in data:
            member_sky[key] = data[npz_key]
            member_n[key]   = int(np.sum(data[npz_key]))
            log(f"  {key}: from NPZ ({member_n[key]:,} stars)")
        elif member_dir:
            ra, dec = load_member_csv(member_dir, key)
            if ra is not None and HEALPY_AVAILABLE:
                sky, _ = compute_healpix_sky(ra, dec, nside)
                member_sky[key] = sky
                member_n[key]   = len(ra)
            else:
                member_sky[key] = None; member_n[key] = 0
        else:
            member_sky[key] = None; member_n[key] = 0
    return member_sky, member_n


# =============================================================================
# FIGURE 1 — MULTI-PANEL SKY GRID  (3 × 4)
# =============================================================================
def plot_sky_grid(data, member_sky, member_n, output_file, dpi=200):
    if not HEALPY_AVAILABLE:
        log("  healpy unavailable -- skipping sky grid."); return
    log("[FIG 1] Multi-panel sky grid ...")

    nside = int(data.get('_nside', np.array([NSIDE]))[0])
    L, B, lon_rad, lat_rad, pix_grid = _mollweide_grid(nside)
    l_grid = L[0, :]

    nrows, ncols = 3, 4
    fig = plt.figure(figsize=(9.5*ncols, 6*nrows), facecolor='white')
    fig.suptitle('Sky Coverage — Survey & Member Catalog Overview',
                 fontsize=32, fontweight='bold', y=1.01)

    panels = list(SKY_PANELS)
    while len(panels) < nrows*ncols:
        panels.append(None)

    for idx, panel in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, idx+1, projection='mollweide')
        ax.set_facecolor('#f0f0eb')

        if panel is None:
            ax.set_visible(False); continue

        label, npz_key, source_type = panel
        sky_d   = None
        n_stars = 0

        if source_type == 'npz':
            sky_d = data.get(npz_key)
            if npz_key == 'sky_full_filt':
                n_stars = int(data.get('_n_filtered', np.array([0]))[0])
            elif sky_d is not None:
                n_stars = int(np.nansum(sky_d))
        elif source_type.startswith('member_'):
            cat_key = source_type.replace('member_', '')
            sky_d   = member_sky.get(cat_key)
            n_stars = member_n.get(cat_key, 0)

        if sky_d is None or not np.any(sky_d > 0):
            ax.text(0.5, 0.5, f'{label}\nNo data',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=16, fontweight='bold', color='#888888')
            _style_mollweide(ax, label, 0)
            continue

        p = sky_d.copy().astype(float)
        p[p == 0] = np.nan

        if np.any(np.isfinite(p)):
            vmax  = np.nanpercentile(p[np.isfinite(p)], 99.5)
            grid  = p[pix_grid].reshape(L.shape)
            cmap  = 'viridis' if source_type.startswith('member_') else 'YlOrRd'
            im    = ax.pcolormesh(lon_rad, lat_rad, grid,
                                  cmap=cmap,
                                  norm=LogNorm(vmin=1, vmax=max(vmax, 2)),
                                  shading='auto', rasterized=True)
            cbar  = fig.colorbar(im, ax=ax, orientation='horizontal',
                                 pad=0.08, aspect=35, shrink=0.65)
            cbar.set_label('Stars / pixel', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=11)

        _galactic_annotations(ax, l_grid)
        _style_mollweide(ax, label, n_stars)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"  Saved: {output_file}")


# =============================================================================
# FIGURE 2 — EXTINCTION SKY MAP  (side-by-side)
# =============================================================================
def plot_extinction(data, output_file, dpi=200):
    if not HEALPY_AVAILABLE:
        log("  healpy unavailable -- skipping extinction."); return
    log("[FIG 2] Extinction sky map ...")

    nside = int(data.get('_nside', np.array([NSIDE]))[0])
    ext_ebprp = data.get('ext_ebprp_mean')
    ext_ag    = data.get('ext_ag_mean')
    if ext_ebprp is None or ext_ag is None:
        log("  Extinction data missing -- skipping."); return

    L, B, lon_rad, lat_rad, pix_grid = _mollweide_grid(nside)
    l_grid = L[0, :]

    fig = plt.figure(figsize=(30, 12), facecolor='white')
    fig.suptitle(
        'Stellar Extinction Map — Galactic Coordinates\n'
        r'Mean $E(G_{\rm BP}-G_{\rm RP})$ (left)   $\cdot$   Mean $A_G$ (right)',
        fontsize=26, fontweight='bold', y=1.01)

    gs      = gridspec.GridSpec(1, 4, figure=fig,
                                width_ratios=[30,1.2,30,1.2],
                                hspace=0.10, wspace=0.12)
    axes_m  = [fig.add_subplot(gs[0,0], projection='mollweide'),
               fig.add_subplot(gs[0,2], projection='mollweide')]
    axes_cb = [fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,3])]

    panels = [
        (ext_ebprp, r'Mean $E(G_{\rm BP}-G_{\rm RP})$ [mag]', 'afmhot',  '#C94B00'),
        (ext_ag,    r'Mean $A_G$ [mag]',                        'plasma',  '#205090'),
    ]

    for i, (sky_map, label, cmap_name, accent) in enumerate(panels):
        ax    = axes_m[i];  ax_cb = axes_cb[i]
        ax.set_facecolor('#f5f5f0')
        finite = sky_map[np.isfinite(sky_map)]
        if len(finite) == 0:
            ax.set_title(f'No data — {label}', fontsize=20, fontweight='bold')
            ax_cb.set_visible(False); continue

        vmax = max(float(np.percentile(finite, 99.0)), 1e-6)
        norm = Normalize(vmin=0, vmax=vmax)
        grid = sky_map[pix_grid].reshape(L.shape)
        ax.pcolormesh(lon_rad, lat_rad, grid, cmap=cmap_name, norm=norm,
                      shading='auto', rasterized=True)

        ax.grid(True, alpha=0.30, linestyle='-', linewidth=0.5, color='gray')
        ax.plot(-np.radians(l_grid), np.zeros_like(l_grid),
                '-', color='white', lw=1.0, alpha=0.6, zorder=9)
        for boff in [5, -5]:
            ax.plot(-np.radians(l_grid),
                    np.radians(np.full_like(l_grid, boff)),
                    '--', color='cyan', lw=0.9, alpha=0.55, zorder=9)
        ax.plot(0, 0, '+', color='lime', ms=14, mew=2.5, zorder=10)

        ax.set_title(label, fontsize=22, fontweight='bold', pad=10)
        lon_ticks = np.array([150,120,90,60,30,0,-30,-60,-90,-120,-150])
        ax.set_xticks(np.radians(lon_ticks))
        ax.set_xticklabels([f'{int(np.mod(-ld,360))}\u00b0' for ld in lon_ticks],
                           fontsize=13, fontweight='bold')
        ax.set_yticks(np.radians([-60,-30,0,30,60]))
        ax.set_yticklabels([f'{ld}\u00b0' for ld in [-60,-30,0,30,60]],
                           fontsize=13, fontweight='bold')

        med  = float(np.nanmedian(finite))
        mean = float(np.nanmean(finite))
        npx  = int(np.sum(np.isfinite(sky_map)))
        ax.text(0.01, 0.02,
                f'Median={med:.3f}  Mean={mean:.3f}\n'
                f'Max={float(np.nanmax(finite)):.3f}  N_pix={npx:,}',
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=accent, alpha=0.90, linewidth=1.8))

        cb = ColorbarBase(ax_cb, cmap=plt.get_cmap(cmap_name),
                          norm=norm, orientation='vertical')
        cb.set_label(label, fontsize=15, fontweight='bold', labelpad=10)
        cb.ax.tick_params(labelsize=13, width=1.5)

    plt.savefig(output_file, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    log(f"  Saved: {output_file}")


# =============================================================================
# FIGURE 3 — COMBINED HR DIAGRAM  (unfilt gray + filt turbo overlaid)
# NO extra scatter markers.  Population boundaries + contours only.
# =============================================================================
def plot_combined_hr(data, output_file, dpi=200):
    log("[FIG 3] Combined HR Diagram (overlay unfilt+filt, no scatter) ...")

    bprp_edges = data['_bprp_edges']
    mg_edges   = data['_mg_edges']
    hr_unfilt  = data['hr_full_unfilt'].copy().astype(float)
    hr_filt    = data['hr_full_filt'].copy().astype(float)
    hr_unfilt[hr_unfilt == 0] = np.nan
    hr_filt[hr_filt   == 0]   = np.nan
    teff_mean  = data.get('teff_bprp_mean')

    if not np.any(np.isfinite(hr_unfilt)):
        log("  No valid HR data -- skipping."); return

    pop_counts = data.get('pop_counts_filt_dict', {})
    n_total    = int(data.get('_n_rows',     np.array([0]))[0])
    n_filt     = int(data.get('_n_filtered', np.array([0]))[0])

    # Extra tall figure: ~22% top margin for Teff label + ticks
    fig, ax = plt.subplots(figsize=(14, 20), facecolor='white')
    ax.set_facecolor('white')

    # --- Unfiltered background (gray) ---
    vmax_u = np.nanpercentile(hr_unfilt[np.isfinite(hr_unfilt)], 99.5)
    ax.pcolormesh(bprp_edges, mg_edges, hr_unfilt,
                  cmap='Greys',
                  norm=LogNorm(vmin=1, vmax=max(vmax_u, 2)),
                  shading='flat', rasterized=True, alpha=0.35, zorder=2)

    # --- Filtered foreground (turbo = rainbow-like) ---
    im = None
    if np.any(np.isfinite(hr_filt)):
        vmax_f = np.nanpercentile(hr_filt[np.isfinite(hr_filt)], 99.5)
        im = ax.pcolormesh(bprp_edges, mg_edges, hr_filt,
                           cmap='turbo',
                           norm=LogNorm(vmin=1, vmax=max(vmax_f, 2)),
                           shading='flat', rasterized=True, alpha=0.80, zorder=3)
        _add_density_contours(ax, hr_filt, bprp_edges, mg_edges,
                              n_levels=6, color='white',
                              linewidths=1.0, smooth_sigma=4.0)

    ax.invert_yaxis()

    # --- Population boundaries ---
    _draw_pop_boundaries(ax, pop_counts, zorder_base=20)

    # --- Axes ---
    _apply_style(ax,
        xlabel=r'$(G_{\rm BP} - G_{\rm RP})_0$  [mag]',
        ylabel=r'$M_G$  [mag]',
        title=(f'Full Catalog — HR Diagram\n'
               f'Gray: All N={n_total:,}   |   Colour: Filtered N={n_filt:,}'),
        xlim=HR_BPRP_RANGE,
        ylim=(HR_MG_RANGE[1], HR_MG_RANGE[0]))

    if im is not None:
        cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=42)
        cbar.set_label('Stars (Filtered, turbo)', fontsize=22, fontweight='bold')
        cbar.ax.tick_params(labelsize=16)

    # --- Spacious Teff top axis ---
    _add_teff_axis(ax, bprp_edges, teff_mean)

    # Reserve 22% headroom at top for Teff label
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)

    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"  Saved: {output_file}")


# =============================================================================
# FIGURE 4 — KIEL DIAGRAM  (unfilt gray + filt turbo + contours)
# =============================================================================
def plot_kiel(data, output_file, dpi=200):
    log("[FIG 4] Kiel Diagram (overlay unfilt+filt, turbo, contours) ...")

    teff_edges   = data['_teff_edges']
    logg_edges   = data['_logg_edges']
    kiel_unfilt  = data.get('kiel_full_unfilt')
    kiel_filt    = data.get('kiel_full_filt')
    feh_mean     = data.get('kiel_feh_mean')

    if kiel_unfilt is None or not np.any(kiel_unfilt > 0):
        log("  No Kiel data -- skipping."); return

    ku = kiel_unfilt.copy().astype(float)
    kf = kiel_filt.copy().astype(float) if kiel_filt is not None else None
    ku[ku == 0] = np.nan
    if kf is not None:
        kf[kf == 0] = np.nan

    n_total = int(data.get('_n_rows',     np.array([0]))[0])
    n_filt  = int(data.get('_n_filtered', np.array([0]))[0])

    # Two-panel: left = counts overlay, right = [Fe/H] mean
    has_feh = feh_mean is not None and np.any(np.isfinite(feh_mean))
    ncols   = 2 if has_feh else 1
    fig, axes = plt.subplots(1, ncols, figsize=(14*ncols, 13), facecolor='white')
    if ncols == 1:
        axes = [axes]

    # ---- Panel A: density overlay ----
    ax = axes[0]
    ax.set_facecolor('white')

    # Unfilt background (gray)
    vmax_u = np.nanpercentile(ku[np.isfinite(ku)], 99.5)
    ax.pcolormesh(teff_edges, logg_edges, ku,
                  cmap='Greys',
                  norm=LogNorm(vmin=1, vmax=max(vmax_u, 2)),
                  shading='flat', rasterized=True, alpha=0.35, zorder=2)

    # Filt foreground (turbo)
    im_kf = None
    if kf is not None and np.any(np.isfinite(kf)):
        vmax_f = np.nanpercentile(kf[np.isfinite(kf)], 99.5)
        im_kf  = ax.pcolormesh(teff_edges, logg_edges, kf,
                               cmap='turbo',
                               norm=LogNorm(vmin=1, vmax=max(vmax_f, 2)),
                               shading='flat', rasterized=True, alpha=0.80, zorder=3)
        _add_density_contours(ax, kf, teff_edges, logg_edges,
                              n_levels=6, color='white',
                              linewidths=0.9, smooth_sigma=3.5)

    ax.invert_xaxis(); ax.invert_yaxis()
    _apply_style(ax,
        xlabel=r'$T_{\rm eff}$  [K]',
        ylabel=r'$\log g$  [dex]',
        title=(f'Kiel Diagram\n'
               f'Gray: All N={n_total:,}  |  Colour: Filt N={n_filt:,}'),
        xlim=KIEL_TEFF_RANGE, ylim=(KIEL_LOGG_RANGE[1], KIEL_LOGG_RANGE[0]))

    if im_kf is not None:
        cb = plt.colorbar(im_kf, ax=ax, pad=0.02, aspect=42)
        cb.set_label('Stars (Filtered, turbo)', fontsize=20, fontweight='bold')
        cb.ax.tick_params(labelsize=15)

    # ---- Panel B: [Fe/H] mean ----
    if has_feh:
        ax2 = axes[1]
        ax2.set_facecolor('white')
        fm  = feh_mean.copy().astype(float)
        fin = fm[np.isfinite(fm)]
        vmin_fe, vmax_fe = np.percentile(fin, 3), np.percentile(fin, 97)
        im_fe = ax2.pcolormesh(teff_edges, logg_edges, fm,
                               cmap='RdYlBu_r',
                               norm=Normalize(vmin=vmin_fe, vmax=vmax_fe),
                               shading='flat', rasterized=True)
        if kf is not None:
            _add_density_contours(ax2, kf, teff_edges, logg_edges,
                                  n_levels=6, color='black',
                                  linewidths=0.8, smooth_sigma=3.5)
        ax2.invert_xaxis(); ax2.invert_yaxis()
        _apply_style(ax2,
            xlabel=r'$T_{\rm eff}$  [K]',
            ylabel=r'$\log g$  [dex]',
            title='Kiel Diagram — Mean [Fe/H]',
            xlim=KIEL_TEFF_RANGE, ylim=(KIEL_LOGG_RANGE[1], KIEL_LOGG_RANGE[0]))
        cb2 = plt.colorbar(im_fe, ax=ax2, pad=0.02, aspect=42)
        cb2.set_label('[Fe/H]  [dex]', fontsize=20, fontweight='bold')
        cb2.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    log(f"  Saved: {output_file}")


# =============================================================================
# GENERATE ALL — in the requested order
# =============================================================================
def generate_all(data, member_sky, member_n, output_dir, prefix, dpi):
    os.makedirs(output_dir, exist_ok=True)
    log("\n" + "="*70)
    log("PAPER-QUALITY PLOT SUITE v5.0  (4 figures)")
    log("  Order: Sky Grid → Extinction → HR Diagram → Kiel Diagram")
    log("="*70)

    plot_sky_grid(data, member_sky, member_n,
        os.path.join(output_dir, f'{prefix}_1_sky_grid.pdf'), dpi)

    plot_extinction(data,
        os.path.join(output_dir, f'{prefix}_2_extinction.pdf'), dpi)

    plot_combined_hr(data,
        os.path.join(output_dir, f'{prefix}_3_HR_combined.pdf'), dpi)

    plot_kiel(data,
        os.path.join(output_dir, f'{prefix}_4_Kiel.pdf'), dpi)

    log("\n" + "="*70)
    log(f"ALL 4 FIGURES SAVED TO: {output_dir}")
    log("="*70)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Stellar Catalog Paper-Quality Plots v5.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # Data source
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--input', '-i', type=str,
                     help='FITS file or directory with FITS chunks (for --compute)')
    grp.add_argument('--load-data', '-l', type=str,
                     help='Pre-computed NPZ file to load directly')

    # Compute
    parser.add_argument('--compute', '-c', action='store_true',
                        help='Read FITS files and compute histograms/sky maps')
    parser.add_argument('--output-data', '-d', type=str,
                        default='stellar_data_v5.npz',
                        help='Output NPZ path when --compute is used')
    parser.add_argument('--chunk-size', type=int, default=1_000_000,
                        help='Rows per chunk when reading FITS (default 1M)')

    # Plot
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Generate all paper figures')
    parser.add_argument('--output-dir', '-O', type=str, default='./paper_figs',
                        help='Output directory')
    parser.add_argument('--prefix', type=str, default='stellar',
                        help='Filename prefix')
    parser.add_argument('--dpi', type=int, default=200,
                        help='Figure DPI')

    # Members
    parser.add_argument('--member-dir', '-m', type=str, default=None,
                        help='Directory containing member CSV files')
    parser.add_argument('--bake-members', action='store_true',
                        help='Compute member sky maps and bake into NPZ')

    args = parser.parse_args()

    nside = NSIDE
    data  = None

    # ---- Compute mode ----
    if args.compute:
        if not args.input:
            parser.error("--input required with --compute")
        data = compute_and_save(args.input, args.output_data,
                                chunk_size=args.chunk_size, nside=nside)
        npz_to_load = args.output_data
    else:
        npz_to_load = args.load_data

    # ---- Load (if not already in memory from compute) ----
    if args.plot or args.bake_members:
        if data is None:
            data = load_npz(npz_to_load)
        nside = int(data.get('_nside', np.array([NSIDE]))[0])

        # Bake members if requested
        if args.bake_members:
            if not args.member_dir:
                parser.error("--member-dir required with --bake-members")
            for key in MEMBER_CATALOGS:
                ra, dec = load_member_csv(args.member_dir, key)
                if ra is not None and HEALPY_AVAILABLE:
                    sky, _ = compute_healpix_sky(ra, dec, nside)
                    data[f'sky_member_{key}'] = sky
                    log(f"  Baked {key}: {len(ra):,} stars")
            np.savez_compressed(npz_to_load, **data)
            log(f"NPZ updated with member maps: {npz_to_load}")

        # Member sky maps
        member_sky, member_n = load_member_sky(data, args.member_dir, nside)

        # Generate plots
        if args.plot:
            generate_all(data, member_sky, member_n,
                         args.output_dir, args.prefix, args.dpi)

    if not args.compute and not args.plot and not args.bake_members:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())