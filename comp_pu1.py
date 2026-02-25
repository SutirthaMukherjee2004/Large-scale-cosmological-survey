#!/usr/bin/env python3
"""
COMPLETENESS & PURITY v8 — FINAL
- LN collapse fixed (absolute separation from data robust_std)
- Gamma primary for dist/RV, LN for comparison
- 3 completeness: mixture-model, retention, GC/OC/SGR (Eq.22)
- Saves all params to JSON for plot reproduction
- More completeness cuts for GC/OC/SGR
- Bold styling, major+minor ticks everywhere
"""
import os,sys,gc as gcmod,json,time,logging,argparse,warnings,pickle
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass,field
from typing import Dict,List,Optional,Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma,polygamma
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
warnings.filterwarnings('ignore')
import numpy as np
import json
import os


@dataclass
class Cfg:
    master_catalog_dir:str="";gc_file:str="";oc_file:str="";sgr_file:str=""
    output_dir:str="./v8_results"
    master_cols:Dict[str,str]=field(default_factory=lambda:{'ra':'RA_final','dec':'DEC_final',
        'distance':'distance_final','distance_err':'distance_err_final',
        'rv':'RV_final','rv_err':'RV_err_final','parallax':'parallax_final',
        'parallax_err':'parallax_err_final','ruwe':'RUWE','gmag':'Gmag'})
    alt_master_cols:Dict[str,List[str]]=field(default_factory=lambda:{
        'ra':['RA_final','RA_all','ra'],'dec':['DEC_final','DEC_all','dec'],
        'distance':['distance_final','DIST','Dist_x'],'distance_err':['distance_err_final','DISTERR'],
        'rv':['RV_final','radial_velocity','RV'],'rv_err':['RV_err_final','radial_velocity_error'],
        'parallax':['parallax_final','parallax'],'parallax_err':['parallax_err_final','parallax_error'],
        'ruwe':['RUWE','ruwe'],'gmag':['Gmag','phot_g_mean_mag']})
    xmatch_radius_arcsec:float=1.0;gmm_max_iter:int=500;gmm_tol:float=1e-6;gmm_n_init:int=10
    eta_prior_alpha:float=5.0;eta_prior_beta:float=2.0;min_eta:float=0.40;max_eta:float=0.90
    min_sep_frac:float=0.8;min_tail_width_ratio:float=2.0;min_core_sigma_floor:float=0.15
    n_radial_bins:int=80;rgal_max:float=300.0
    # More cuts for better completeness resolution
    rel_dist_err_cuts:List[float]=field(default_factory=lambda:[
        .005,.01,.015,.02,.03,.04,.05,.07,.1,.12,.15,.2,.25,.3,.4,.5,.6,.75,1,1.5,2])
    rel_rv_err_cuts:List[float]=field(default_factory=lambda:[
        .005,.01,.015,.02,.03,.04,.05,.07,.1,.12,.15,.2,.25,.3,.4,.5,.6,.75,1,1.5,2])
    ruwe_cuts:List[float]=field(default_factory=lambda:[
        .7,.8,.9,.95,1,1.05,1.1,1.15,1.2,1.3,1.4,1.5,1.6,1.8,2,2.5,3,4,5])
    rgal_bins:List[float]=field(default_factory=lambda:[0,5,10,20,50,100,200,500])
    p_mem_high:float=0.8;plot_dpi:int=150
    max_chunks:Optional[int]=None;max_rows_per_chunk:Optional[int]=None;max_samples:int=2_000_000

def setup_log(lf=None):
    L=logging.getLogger('V8');L.setLevel(logging.INFO);L.handlers=[]
    fmt=logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    ch=logging.StreamHandler(sys.stdout);ch.setFormatter(fmt);L.addHandler(ch)
    if lf:fh=logging.FileHandler(lf);fh.setFormatter(fmt);L.addHandler(fh)
    return L

def style_ax(ax,minor=True):
    """Apply bold styling, major+minor ticks to any axis."""
    for spine in ax.spines.values():spine.set_linewidth(1.8)
    ax.tick_params(axis='both',which='major',labelsize=12,width=1.8,length=7,direction='in',top=True,right=True)
    ax.tick_params(axis='both',which='minor',width=1.2,length=4,direction='in',top=True,right=True)
    if minor:ax.minorticks_on()
    for item in [ax.xaxis.label,ax.yaxis.label]:item.set_fontsize(14);item.set_fontweight('bold')
    if ax.get_title():ax.title.set_fontsize(15);ax.title.set_fontweight('bold')

# ======================= LOG-NORMAL MIXTURE (v8 FIXED) =======================
class LogNormalMix:
    """Log-Normal mixture with ABSOLUTE separation constraint from data robust_std."""
    def __init__(s,L,C):s.L=L;s.C=C;s.converged=False;s.eta=s.mu_c=s.sig_c=s.mu_t=s.sig_t=None;s.params={}
    @staticmethod
    def _gll(y,mu,sig,delta=None):
        tv=sig**2+(delta**2 if delta is not None else 0);tv=np.maximum(tv,1e-10)
        return -0.5*np.log(2*np.pi*tv)-0.5*(y-mu)**2/tv
    def fit(s,data,me=None,name="data",ms=None):
        s.L.info(f"\n{'='*60}\nLOG-NORMAL (v8): {name}\n{'='*60}")
        v=np.isfinite(data)&(data>0)
        if me is not None:me=np.asarray(me,dtype=np.float64);v&=np.isfinite(me)&(me>0)
        x=data[v].astype(np.float64);nt=len(x);s.L.info(f"  Valid: {nt:,}")
        if nt<100:return s
        C=s.C
        if ms and nt>ms:
            rng=np.random.default_rng(42);idx=rng.choice(nt,size=ms,replace=False)
            xf,ef=x[idx],(me[v][idx] if me is not None else None)
        else:xf,ef=x,(me[v] if me is not None else None)
        nf=len(xf);y=np.log(xf);delta=np.clip(ef/xf,.01,2) if ef is not None else None
        p10,p25,p50,p75,p90=np.percentile(y,[10,25,50,75,90])
        std_y=1.4826*np.median(np.abs(y-p50))
        ams=C.min_sep_frac*std_y;sf=max(C.min_core_sigma_floor,.1*std_y)
        s.L.info(f"  robust_std={std_y:.4f}, abs_min_sep={ams:.4f}, sig_floor={sf:.4f}")
        best=-np.inf;bp=None
        for ii in range(C.gmm_n_init):
            if ii==0:eta,mc,sc,mt,st=.70,p25,(p50-p10)/1.28,p75,(p90-p50)/1.28
            elif ii==1:eta,mc,sc,mt,st=.60,p50-.5*std_y,.3*std_y,p50+.8*std_y,std_y
            elif ii==2:eta,mc,sc,mt,st=.55,p10,.25*std_y,p90,1.5*std_y
            else:
                eta=np.random.uniform(C.min_eta+.05,C.max_eta-.05)
                mc=p50+.3*std_y*np.random.randn();sc=std_y*np.random.uniform(.15,.5)
                mt=mc+std_y*np.random.uniform(.8,2);st=sc*np.random.uniform(C.min_tail_width_ratio,C.min_tail_width_ratio+2)
            sc=max(sc,sf);st=max(st,sc*C.min_tail_width_ratio)
            if mt<mc:mc,mt=mt,mc
            if mt-mc<ams:mt=mc+ams
            for _ in range(C.gmm_max_iter):
                lc=s._gll(y,mc,sc,delta);lt=s._gll(y,mt,st,delta)
                lwc=np.log(eta+1e-10)+lc;lwt=np.log(1-eta+1e-10)+lt
                lt2=np.logaddexp(lwc,lwt);gc_=np.exp(lwc-lt2);gt_=1-gc_
                Nc=np.sum(gc_)+1e-10;Nt=np.sum(gt_)+1e-10
                en=np.clip((Nc+C.eta_prior_alpha-1)/(nf+C.eta_prior_alpha+C.eta_prior_beta-2),C.min_eta,C.max_eta)
                mcn=np.sum(gc_*y)/Nc;mtn=np.sum(gt_*y)/Nt
                if delta is not None:
                    scn=np.sqrt(max(np.sum(gc_*(y-mcn)**2)/Nc-np.sum(gc_*delta**2)/Nc,sf**2))
                    stn=np.sqrt(max(np.sum(gt_*(y-mtn)**2)/Nt-np.sum(gt_*delta**2)/Nt,sf**2))
                else:
                    scn=max(np.sqrt(np.sum(gc_*(y-mcn)**2)/Nc),sf);stn=max(np.sqrt(np.sum(gt_*(y-mtn)**2)/Nt),sf)
                if mtn<mcn:mcn,mtn=mtn,mcn;scn,stn=stn,scn;en=np.clip(1-en,C.min_eta,C.max_eta)
                if mtn-mcn<ams:mtn=mcn+ams
                stn=max(stn,C.min_tail_width_ratio*scn)
                dp=abs(en-eta)+abs(mcn-mc)+abs(scn-sc)+abs(mtn-mt)
                eta,mc,sc,mt,st=en,mcn,scn,mtn,stn
                if dp<C.gmm_tol:break
            lc=s._gll(y,mc,sc,delta);lt=s._gll(y,mt,st,delta)
            ll=np.sum(np.logaddexp(np.log(eta+1e-10)+lc,np.log(1-eta+1e-10)+lt))
            lp=((C.eta_prior_alpha-1)*np.log(eta)+(C.eta_prior_beta-1)*np.log(1-eta)) if 0<eta<1 else -np.inf
            if ll+lp>best:best=ll+lp;bp=(eta,mc,sc,mt,st)
        s.eta,s.mu_c,s.sig_c,s.mu_t,s.sig_t=bp;s.converged=True
        s.params={'eta':float(s.eta),'mu_core':float(s.mu_c),'sigma_core':float(s.sig_c),
            'mu_tail':float(s.mu_t),'sigma_tail':float(s.sig_t),
            'median_core':float(np.exp(s.mu_c)),'median_tail':float(np.exp(s.mu_t)),'data_robust_std':float(std_y)}
        r=np.exp(s.mu_t)/np.exp(s.mu_c)
        s.L.info(f"  CONVERGED: eta={s.eta:.4f}, Core_med={np.exp(s.mu_c):.4f}, Tail_med={np.exp(s.mu_t):.4f}, ratio={r:.2f}x")
        return s
    def predict_purity(s,data,me=None):
        if not s.converged:return np.full(len(data),.5)
        p=np.full(len(data),.5);v=np.isfinite(data)&(data>0)
        if np.sum(v)==0:return p
        x=data[v];yv=np.log(x);d=None
        if me is not None and len(me)==len(data):
            e=me[v];d=np.where((e>0)&np.isfinite(e),np.clip(e/x,0,2),0.)
        lc=s._gll(yv,s.mu_c,s.sig_c,d);lt=s._gll(yv,s.mu_t,s.sig_t,d)
        lwc=np.log(s.eta+1e-10)+lc;lwt=np.log(1-s.eta+1e-10)+lt
        pp=np.exp(lwc-np.logaddexp(lwc,lwt));p[v]=np.where(np.isfinite(pp),pp,.5);return p
    def core_pdf(s,x):
        o=np.zeros_like(x,dtype=np.float64);v=x>0
        o[v]=s.eta*stats.lognorm.pdf(x[v],s=s.sig_c,scale=np.exp(s.mu_c));return o
    def tail_pdf(s,x):
        o=np.zeros_like(x,dtype=np.float64);v=x>0
        o[v]=(1-s.eta)*stats.lognorm.pdf(x[v],s=s.sig_t,scale=np.exp(s.mu_t));return o
    def total_pdf(s,x):return s.core_pdf(x)+s.tail_pdf(x)

# =========================== GAMMA MIXTURE ===================================
class GammaMix:
    def __init__(s,L,C):s.L=L;s.C=C;s.converged=False;s.eta=s.ac=s.bc=s.at=s.bt=None;s.params={}
    @staticmethod
    def _gll(x,a,b):return stats.gamma.logpdf(x,a=a,scale=1./b)
    @staticmethod
    def _fitc(x,w):
        w=w/(np.sum(w)+1e-30);mx=np.sum(w*x);mlx=np.sum(w*np.log(x+1e-30))
        s=np.log(max(mx,1e-10))-mlx
        if s<=0:s=.01
        a=(3-s+np.sqrt(max((s-3)**2+24*s,0)))/(12*s);a=np.clip(a,.5,200)
        for _ in range(3):
            pv=float(digamma(a));p1=float(polygamma(1,a))
            num=np.log(a)-pv-s;den=1./a-p1
            if abs(den)>1e-12:a=a-num/den
            a=np.clip(a,.1,500)
        return float(a),float(a/max(mx,1e-10))
    def fit(s,data,me=None,name="data",ms=None):
        s.L.info(f"\n{'='*60}\nGAMMA: {name}\n{'='*60}")
        v=np.isfinite(data)&(data>0);xa=data[v].astype(np.float64);nt=len(xa)
        s.L.info(f"  Valid: {nt:,}")
        if nt<100:return s
        C=s.C
        if ms and nt>ms:rng=np.random.default_rng(42);x=xa[rng.choice(nt,size=ms,replace=False)]
        else:x=xa
        n=len(x);p25,p50,p75,mx,sx=np.percentile(x,25),np.median(x),np.percentile(x,75),np.mean(x),np.std(x)
        s.L.info(f"  P25={p25:.4f} P50={p50:.4f} P75={p75:.4f} mean={mx:.4f}")
        best=-np.inf;bp=None
        for ii in range(C.gmm_n_init):
            if ii==0:eta,ac,bc,at,bt=.65,4.,4./max(p25,.01),2.,2./max(p75,.01)
            elif ii==1:eta,ac,bc,at,bt=.55,6.,6./max(p50,.01),1.5,1.5/max(mx+sx,.01)
            else:
                eta=np.random.uniform(C.min_eta+.05,C.max_eta-.05)
                ac=np.random.uniform(2,10);bc=ac/max(p50*np.random.uniform(.3,1),.01)
                at=np.random.uniform(1,5);bt=at/max(mx*np.random.uniform(1.5,3),.01)
            for _ in range(C.gmm_max_iter):
                lc=GammaMix._gll(x,ac,bc);lt=GammaMix._gll(x,at,bt)
                lwc=np.log(eta+1e-10)+lc;lwt=np.log(1-eta+1e-10)+lt;lt2=np.logaddexp(lwc,lwt)
                gc_=np.exp(lwc-lt2);gt_=1-gc_
                en=np.clip(np.sum(gc_)/n,C.min_eta,C.max_eta)
                acn,bcn=GammaMix._fitc(x,gc_);atn,btn=GammaMix._fitc(x,gt_)
                if atn/btn<acn/bcn:acn,atn=atn,acn;bcn,btn=btn,bcn;en=np.clip(1-en,C.min_eta,C.max_eta)
                d=abs(en-eta)+abs(acn-ac)+abs(bcn-bc)
                eta,ac,bc,at,bt=en,acn,bcn,atn,btn
                if d<C.gmm_tol:break
            lc=GammaMix._gll(x,ac,bc);lt=GammaMix._gll(x,at,bt)
            ll=np.sum(np.logaddexp(np.log(eta+1e-10)+lc,np.log(1-eta+1e-10)+lt))
            if ll>best:best=ll;bp=(eta,ac,bc,at,bt)
        s.eta,s.ac,s.bc,s.at,s.bt=bp;s.converged=True
        mc=s.ac/s.bc;mt=s.at/s.bt
        s.params={'eta':float(s.eta),'alpha_core':float(s.ac),'beta_core':float(s.bc),
            'alpha_tail':float(s.at),'beta_tail':float(s.bt),'mean_core':float(mc),'mean_tail':float(mt)}
        s.L.info(f"  CONVERGED: eta={s.eta:.4f}, Core={mc:.4f}, Tail={mt:.4f}, ratio={mt/mc:.2f}x")
        return s
    def predict_purity(s,data,me=None):
        if not s.converged:return np.full(len(data),.5)
        p=np.full(len(data),.5);v=np.isfinite(data)&(data>0);x=data[v]
        lc=GammaMix._gll(x,s.ac,s.bc);lt=GammaMix._gll(x,s.at,s.bt)
        lwc=np.log(s.eta+1e-10)+lc;lwt=np.log(1-s.eta+1e-10)+lt
        pp=np.exp(lwc-np.logaddexp(lwc,lwt));p[v]=np.where(np.isfinite(pp),pp,.5);return p
    def core_pdf(s,x):
        o=np.zeros_like(x,dtype=np.float64);v=x>0
        o[v]=s.eta*stats.gamma.pdf(x[v],a=s.ac,scale=1./s.bc);return o
    def tail_pdf(s,x):
        o=np.zeros_like(x,dtype=np.float64);v=x>0
        o[v]=(1-s.eta)*stats.gamma.pdf(x[v],a=s.at,scale=1./s.bt);return o
    def total_pdf(s,x):return s.core_pdf(x)+s.tail_pdf(x)

# ======================= REG GAUSSIAN (RUWE) =================================
class RegGaussMix:
    def __init__(s,L,C):s.L=L;s.C=C;s.converged=False;s.eta=s.mu_c=s.sig_c=s.mu_t=s.sig_t=None;s.params={}
    @staticmethod
    def _gll(x,mu,sig):v=max(sig**2,1e-10);return -0.5*np.log(2*np.pi*v)-0.5*(x-mu)**2/v
    def fit(s,data,name="RUWE",ms=None):
        s.L.info(f"\n{'='*60}\nREG GAUSSIAN: {name}\n{'='*60}")
        v=np.isfinite(data);xa=data[v].astype(np.float64);C=s.C
        if ms and len(xa)>ms:rng=np.random.default_rng(42);x=xa[rng.choice(len(xa),size=ms,replace=False)]
        else:x=xa
        n=len(x);med=np.median(x);rs=1.4826*np.median(np.abs(x-med))
        s.L.info(f"  N={n:,}, median={med:.4f}, robust_sig={rs:.4f}")
        best=-np.inf;bp=None
        for ii in range(C.gmm_n_init):
            if ii==0:eta,mc,sc,mt,st=.75,med,rs*.5,med+2*rs,rs*3
            else:
                eta=np.random.uniform(C.min_eta+.05,C.max_eta-.05)
                mc=med+.2*rs*np.random.randn();sc=rs*np.random.uniform(.3,.7)
                mt=mc+rs*np.random.uniform(1.5,4);st=rs*np.random.uniform(2,5)
            for _ in range(C.gmm_max_iter):
                lc=s._gll(x,mc,sc);lt=s._gll(x,mt,st)
                lwc=np.log(eta+1e-10)+lc;lwt=np.log(1-eta+1e-10)+lt
                lt2=np.logaddexp(lwc,lwt);gc_=np.exp(lwc-lt2);gt_=1-gc_
                Nc=np.sum(gc_)+1e-10;Nt=np.sum(gt_)+1e-10
                en=np.clip((Nc+C.eta_prior_alpha-1)/(n+C.eta_prior_alpha+C.eta_prior_beta-2),C.min_eta,C.max_eta)
                mcn=np.sum(gc_*x)/Nc;mtn=np.sum(gt_*x)/Nt
                scn=np.sqrt(np.sum(gc_*(x-mcn)**2)/Nc);stn=np.sqrt(np.sum(gt_*(x-mtn)**2)/Nt)
                if stn<scn:mcn,mtn=mtn,mcn;scn,stn=stn,scn;en=np.clip(1-en,C.min_eta,C.max_eta)
                stn=max(stn,1.5*scn)
                d=abs(en-eta)+abs(mcn-mc)+abs(scn-sc)
                eta,mc,sc,mt,st=en,mcn,scn,mtn,stn
                if d<C.gmm_tol:break
            lc=s._gll(x,mc,sc);lt=s._gll(x,mt,st)
            ll=np.sum(np.logaddexp(np.log(eta)+lc,np.log(1-eta)+lt))
            lp=(C.eta_prior_alpha-1)*np.log(eta)+(C.eta_prior_beta-1)*np.log(1-eta)
            if ll+lp>best:best=ll+lp;bp=(eta,mc,sc,mt,st)
        s.eta,s.mu_c,s.sig_c,s.mu_t,s.sig_t=bp;s.converged=True
        s.params={'eta':float(s.eta),'mu_core':float(s.mu_c),'sigma_core':float(s.sig_c),
            'mu_tail':float(s.mu_t),'sigma_tail':float(s.sig_t)}
        s.L.info(f"  CONVERGED: eta={s.eta:.4f}, Core:mu={s.mu_c:.4f},sig={s.sig_c:.4f}, Tail:mu={s.mu_t:.4f},sig={s.sig_t:.4f}")
        return s
    def predict_purity(s,data):
        if not s.converged:return np.full(len(data),.5)
        lc=s._gll(data,s.mu_c,s.sig_c);lt=s._gll(data,s.mu_t,s.sig_t)
        lwc=np.log(s.eta+1e-10)+lc;lwt=np.log(1-s.eta+1e-10)+lt
        p=np.exp(lwc-np.logaddexp(lwc,lwt));return np.where(np.isfinite(p),p,.5)
    def core_pdf(s,x):return s.eta*stats.norm.pdf(x,s.mu_c,s.sig_c)
    def tail_pdf(s,x):return(1-s.eta)*stats.norm.pdf(x,s.mu_t,s.sig_t)
    def total_pdf(s,x):return s.core_pdf(x)+s.tail_pdf(x)

# ======================= CATALOG LOADER ======================================
class CatLoader:
    def __init__(s,C,L):s.C=C;s.L=L;s.df=None;s.cm={};s.tree=None;s.max_chord=0
    def _fc(s,k,av):
        pr=s.C.master_cols.get(k)
        if pr and pr in av:return pr
        for a in s.C.alt_master_cols.get(k,[]):
            if a in av:return a
        return None
    def load(s):
        s.L.info("="*70+"\nLOADING CATALOG\n"+"="*70)
        p=Path(s.C.master_catalog_dir)
        if p.is_file() and str(p).endswith('.fits'):files=[str(p)]
        elif p.is_dir():
            files=[]
            for pat in ['Entire_catalogue_chunk*.fits','*_chunk*.fits','*.fits']:
                files=sorted(p.glob(pat))
                if files:files=[str(f) for f in files];break
        else:s.L.error(f"Not found: {p}");return False
        if not files:s.L.error("No FITS!");return False
        if s.C.max_chunks:files=files[:s.C.max_chunks]
        s.L.info(f"Found {len(files)} files")
        dfs=[]
        for i,fp in enumerate(files):
            s.L.info(f"[{i+1}/{len(files)}] {Path(fp).name}")
            try:
                with fits.open(fp,memmap=True) as hdu:
                    data=hdu[1].data;cols=[c.name for c in hdu[1].columns]
                    if i==0:
                        for k in s.C.master_cols:
                            f=s._fc(k,cols)
                            if f:s.cm[k]=f;s.L.info(f"  {k} -> {f}")
                    nr=len(data)
                    if s.C.max_rows_per_chunk:nr=min(nr,s.C.max_rows_per_chunk)
                    ch={}
                    for k,c in s.cm.items():
                        try:ch[k]=np.array(data[c][:nr],dtype=np.float64)
                        except:ch[k]=np.full(nr,np.nan)
                    dfs.append(pd.DataFrame(ch));s.L.info(f"  {nr:,} rows")
            except Exception as e:s.L.error(f"  {e}")
            gcmod.collect()
        if not dfs:return False
        s.df=pd.concat(dfs,ignore_index=True);del dfs;gcmod.collect()
        s._derived();s.df=s.df.dropna(subset=['ra','dec'])
        s._build_tree()
        s.L.info(f"Total: {len(s.df):,} stars");return True
    def _derived(s):
        df=s.df
        if 'distance' in df and 'distance_err' in df:
            with np.errstate(divide='ignore',invalid='ignore'):df['rel_dist_err']=np.abs(df['distance_err']/df['distance'])
        if 'rv' in df and 'rv_err' in df:
            with np.errstate(divide='ignore',invalid='ignore'):df['rel_rv_err']=np.abs(df['rv_err']/np.abs(df['rv']))
        if 'ra' in df and 'dec' in df:
            try:
                c=SkyCoord(ra=df['ra'].values*u.deg,dec=df['dec'].values*u.deg,frame='icrs')
                g=c.galactic;df['l']=g.l.deg;df['b']=g.b.deg
            except:pass
        if all(c in df for c in ['distance','l','b']):
            try:
                d=df['distance'].values;lr=np.radians(df['l'].values);br=np.radians(df['b'].values)
                x=d*np.cos(br)*np.cos(lr)-8.2;y=d*np.cos(br)*np.sin(lr);z=d*np.sin(br)
                df['R_gal']=np.sqrt(x**2+y**2+z**2)
            except:pass
    def _build_tree(s):
        ra_r=np.radians(s.df['ra'].values);dec_r=np.radians(s.df['dec'].values)
        xyz=np.column_stack([np.cos(dec_r)*np.cos(ra_r),np.cos(dec_r)*np.sin(ra_r),np.sin(dec_r)])
        s.tree=cKDTree(xyz);s.max_chord=2*np.sin(np.radians(s.C.xmatch_radius_arcsec/3600)/2)
    def query(s,ra,dec):
        ra_r=np.radians(ra);dec_r=np.radians(dec)
        xyz=np.column_stack([np.cos(dec_r)*np.cos(ra_r),np.cos(dec_r)*np.sin(ra_r),np.sin(dec_r)])
        dists,idxs=s.tree.query(xyz,k=1,distance_upper_bound=s.max_chord)
        v=np.isfinite(dists);return idxs[v],np.where(v)[0]

# ===================== TRUTH MEMBER LOADER ===================================
class TruthLoader:
    """Load GC/OC/SGR member lists, cross-match, compute Eq.22 completeness."""
    def __init__(s,C,L,cat):s.C=C;s.L=L;s.cat=cat;s.truth={}
    def load_all(s):
        s.L.info("\n"+"="*70+"\nLOADING TRUTH MEMBERS\n"+"="*70)
        for label,fpath in [('GC',s.C.gc_file),('OC',s.C.oc_file),('SGR',s.C.sgr_file)]:
            if not fpath or not os.path.isfile(fpath):
                s.L.info(f"  {label}: not provided or not found, skipping");continue
            s.L.info(f"  Loading {label}: {fpath}")
            try:
                ext=Path(fpath).suffix.lower()
                if ext=='.csv':df=pd.read_csv(fpath)
                elif ext in ('.fits','.fit'):
                    with fits.open(fpath) as hdu:
                        df=pd.DataFrame({c.name:hdu[1].data[c.name] for c in hdu[1].columns})
                else:df=pd.read_csv(fpath)
                ra_col=dec_col=None
                for rc in ['ra','RA','RAdeg','_RAJ2000']:
                    if rc in df.columns:ra_col=rc;break
                for dc in ['dec','DEC','DEdeg','_DEJ2000']:
                    if dc in df.columns:dec_col=dc;break
                if ra_col is None or dec_col is None:
                    s.L.warning(f"  Cannot find RA/DEC in {label}: {df.columns.tolist()[:10]}");continue
                ra=df[ra_col].values.astype(np.float64);dec=df[dec_col].values.astype(np.float64)
                v=np.isfinite(ra)&np.isfinite(dec);ra=ra[v];dec=dec[v]
                s.L.info(f"  {label}: {len(ra):,} members with valid coords")
                s.truth[label]={'ra':ra,'dec':dec,'n_total':len(ra)}
            except Exception as e:s.L.error(f"  Failed {label}: {e}")
        return len(s.truth)>0
    def xmatch(s):
        if not s.truth:return
        s.L.info("\n--- Cross-matching truth members ---")
        for label,td in s.truth.items():
            matched_idx,query_idx=s.cat.query(td['ra'],td['dec'])
            td['matched_cat_idx']=matched_idx
            td['n_matched']=len(matched_idx)
            td['completeness_raw']=td['n_matched']/td['n_total'] if td['n_total']>0 else 0
            s.L.info(f"  {label}: {td['n_matched']:,}/{td['n_total']:,} matched = {td['completeness_raw']*100:.1f}%")
    def completeness_vs_cuts(s,metric_col,cuts):
        """Eq.22: C(cut) = N_truth_passing_cut / N_truth_total"""
        results={}
        for label,td in s.truth.items():
            if 'matched_cat_idx' not in td:continue
            cidx=td['matched_cat_idx'];ntot=td['n_total']
            if metric_col not in s.cat.df.columns:continue
            vals=s.cat.df[metric_col].values
            cl=[]
            for cut in cuts:
                n_pass=np.sum(np.isfinite(vals[cidx])&(vals[cidx]<cut))
                cl.append(n_pass/ntot if ntot>0 else 0)
            results[label]=cl
        return results

# ======================= PURITY ANALYZER =====================================
class PurityAnalyzer:
    def __init__(s,C,L,cat):s.C=C;s.L=L;s.cat=cat;s.models={};s.results={}
    def run(s):
        s.L.info("\n"+"="*70+"\nPURITY ANALYSIS\n"+"="*70)
        s._ruwe();s._dist();s._rv();s._combined();s._by_radius()
        return s.results
    def _ruwe(s):
        if 'ruwe' not in s.cat.df:return
        r=s.cat.df['ruwe'].values;v=np.isfinite(r)&(r>0)&(r<100)
        m=RegGaussMix(s.L,s.C);m.fit(r[v],name="RUWE",ms=s.C.max_samples)
        s.models['ruwe']=m;p=m.predict_purity(r);s.cat.df['purity_ruwe']=p
        s.results['ruwe']={**m.params,'mean_purity':float(np.nanmean(p))}
    def _dist(s):
        if 'rel_dist_err' not in s.cat.df:return
        re=s.cat.df['rel_dist_err'].values;v=np.isfinite(re)&(re>0)&(re<10)
        de=None
        if 'distance_err' in s.cat.df and 'distance' in s.cat.df:
            d=s.cat.df['distance'].values;derr=s.cat.df['distance_err'].values
            with np.errstate(divide='ignore',invalid='ignore'):
                de=np.abs(derr/d);de=np.where(np.isfinite(de)&(de>0),de,np.nan)
        ms=s.C.max_samples
        ln=LogNormalMix(s.L,s.C);ln.fit(re[v],me=de[v] if de is not None else None,name="sig_d/d(LN)",ms=ms)
        s.models['dist_ln']=ln;pln=ln.predict_purity(re,de);s.cat.df['purity_dist_ln']=pln
        gm=GammaMix(s.L,s.C);gm.fit(re[v],name="sig_d/d(Gamma)",ms=ms)
        s.models['dist_gm']=gm;pgm=gm.predict_purity(re);s.cat.df['purity_dist_gm']=pgm
        s.cat.df['purity_dist']=pgm  # Gamma primary
        s.results['dist']={'lognormal':ln.params,'gamma':gm.params,
            'mean_purity_ln':float(np.nanmean(pln)),'mean_purity_gm':float(np.nanmean(pgm))}
    def _rv(s):
        if 'rel_rv_err' not in s.cat.df:return
        re=s.cat.df['rel_rv_err'].values;v=np.isfinite(re)&(re>0)&(re<10)
        rve=None
        if 'rv_err' in s.cat.df and 'rv' in s.cat.df:
            rv=np.abs(s.cat.df['rv'].values);rverr=s.cat.df['rv_err'].values
            with np.errstate(divide='ignore',invalid='ignore'):
                rve=np.abs(rverr/rv);rve=np.where(np.isfinite(rve)&(rve>0),rve,np.nan)
        ms=s.C.max_samples
        ln=LogNormalMix(s.L,s.C);ln.fit(re[v],me=rve[v] if rve is not None else None,name="sig_v/v(LN)",ms=ms)
        s.models['rv_ln']=ln;pln=ln.predict_purity(re,rve);s.cat.df['purity_rv_ln']=pln
        gm=GammaMix(s.L,s.C);gm.fit(re[v],name="sig_v/v(Gamma)",ms=ms)
        s.models['rv_gm']=gm;pgm=gm.predict_purity(re);s.cat.df['purity_rv_gm']=pgm
        s.cat.df['purity_rv']=pgm
        s.results['rv']={'lognormal':ln.params,'gamma':gm.params,
            'mean_purity_ln':float(np.nanmean(pln)),'mean_purity_gm':float(np.nanmean(pgm))}
    def _combined(s):
        c=np.ones(len(s.cat.df))
        for col in ['purity_ruwe','purity_dist','purity_rv']:
            if col in s.cat.df:c*=s.cat.df[col].fillna(1.).values
        s.cat.df['purity_combined']=c
        s.results['combined']={'mean':float(np.nanmean(c)),'median':float(np.nanmedian(c)),
            'n_gt_0.8':int(np.sum(c>.8)),'n_gt_0.95':int(np.sum(c>.95))}
        s.L.info(f"  Combined: mean={s.results['combined']['mean']:.4f}, P>0.8={s.results['combined']['n_gt_0.8']:,}")
    def _by_radius(s):
        if 'R_gal' not in s.cat.df:return
        s.results['by_radius']={}
        for i in range(len(s.C.rgal_bins)-1):
            rl,rh=s.C.rgal_bins[i],s.C.rgal_bins[i+1]
            mask=(s.cat.df['R_gal']>=rl)&(s.cat.df['R_gal']<rh);n=np.sum(mask)
            if n<10:continue
            pc=s.cat.df.loc[mask,'purity_combined'].values;lab=f"{rl}-{rh} kpc"
            s.results['by_radius'][lab]={'n':int(n),'mean':float(np.nanmean(pc)),
                'frac_gt_0.8':float(np.mean(pc>.8))}

# ======================= TRADE-OFF / COMPLETENESS ============================
class TradeoffAnalyzer:
    """Retention, model completeness, purity trade-off + radial profiles + Eq.22 truth completeness."""
    def __init__(s,C,L,cat,pa,truth=None):
        s.C=C;s.L=L;s.cat=cat;s.pa=pa;s.truth=truth;s.results={}
    def run(s):
        s.L.info("\n"+"="*70+"\nTRADE-OFF & COMPLETENESS\n"+"="*70)
        for mcol,pcol,cuts,key in [
            ('rel_dist_err','purity_dist',s.C.rel_dist_err_cuts,'dist'),
            ('rel_rv_err','purity_rv',s.C.rel_rv_err_cuts,'rv'),
            ('ruwe','purity_ruwe',s.C.ruwe_cuts,'ruwe')]:
            if mcol not in s.cat.df:continue
            s._sweep(mcol,pcol,cuts,key)
        return s.results
    def _sweep(s,mcol,pcol,cuts,key):
        m=s.cat.df[mcol].values;p=s.cat.df.get(pcol,pd.Series(np.ones(len(s.cat.df)))).values
        v=np.isfinite(m);nv=np.sum(v)
        gc_all=p;gc_total=np.nansum(gc_all[v])
        r={'cuts':[],'retention':[],'mean_purity':[],'n_stars':[],'model_completeness':[]}
        for cut in cuts:
            mask=v&(m<cut);ns=np.sum(mask)
            r['cuts'].append(cut);r['retention'].append(ns/nv if nv>0 else 0)
            r['mean_purity'].append(float(np.nanmean(p[mask])) if ns>0 else np.nan)
            r['n_stars'].append(int(ns))
            mc=np.nansum(gc_all[mask])/gc_total if gc_total>0 else 0
            r['model_completeness'].append(float(mc))
        s.results[key+'_global']=r
        # GC/OC/SGR truth completeness (Eq.22)
        if s.truth and s.truth.truth:
            tc=s.truth.completeness_vs_cuts(mcol,cuts)
            for label,cvals in tc.items():
                r[f'truth_completeness_{label}']=cvals
        # Radial profiles
        if 'R_gal' in s.cat.df:
            rg=s.cat.df['R_gal'].values
            re=np.logspace(np.log10(0.1),np.log10(s.C.rgal_max),s.C.n_radial_bins+1)
            rc=.5*(re[:-1]+re[1:]);bc={}
            for cut in cuts:
                rp=[];pp=[];mcp=[]
                for j in range(len(rc)):
                    bm=(rg>=re[j])&(rg<re[j+1])&v;nb=np.sum(bm)
                    sm=bm&(m<cut);ns2=np.sum(sm)
                    rp.append(ns2/nb if nb>0 and ns2>0 else np.nan)
                    pp.append(float(np.nanmean(p[sm])) if ns2>0 else np.nan)
                    gc_bin=np.nansum(gc_all[bm])
                    mcp.append(np.nansum(gc_all[sm])/gc_bin if gc_bin>0 else np.nan)
                bc[str(cut)]={'retention':rp,'purity':pp,'model_completeness':mcp}
            s.results[key+'_radial']={'r_edges':re.tolist(),'r_centers':rc.tolist(),'by_cut':bc}

# ======================= PLOTTER =============================================
COL_RUWE='#EE6677';COL_DIST='#228833';COL_RV='#4477AA';COL_COMB='#222222'

class Plotter:
    def __init__(s,C,L,od):
        s.C=C;s.L=L;s.od=od;os.makedirs(od,exist_ok=True)
        plt.rcParams.update({'font.size':13,'axes.titlesize':16,'axes.labelsize':15,
            'legend.fontsize':10,'xtick.labelsize':12,'ytick.labelsize':12,
            'figure.dpi':100,'savefig.dpi':C.plot_dpi,'savefig.bbox':'tight','figure.facecolor':'white'})
    def _sv(s,fig,n):
        p=os.path.join(s.od,n);fig.savefig(p);plt.close(fig);s.L.info(f"  Saved: {n}")

    def plot_all(s,cat,pa,pr,tr,truth):
        s.L.info("\n"+"="*70+"\nPLOTTING\n"+"="*70)
        s._p1(cat,pa)
        s._p2(cat,pa)
        s._p3(cat,pa,tr)
        s._p4_eta(pr)
        s._p5_purity_by_method(pr)
        s._p6_gc_oc_sgr(tr,truth)

    # ── PLOT 1: 3×2 mixture comparison (LN top, Gamma bottom) ──
    def _p1(s,cat,pa):
        fig,axes=plt.subplots(2,3,figsize=(24,14))
        fig.suptitle('Mixture Decomposition: Log-Normal (top) vs Gamma (bottom)',fontsize=20,fontweight='bold',y=.98)
        for row in range(2):
            ax=axes[row,0]
            if 'ruwe' in cat.df and 'ruwe' in pa.models:
                r=cat.df['ruwe'].dropna();r=r[(r>0)&(r<5)]
                ax.hist(r,bins=200,density=True,alpha=.5,color='steelblue',label='Data')
                m=pa.models['ruwe']
                if m.converged:
                    x=np.linspace(.5,5,2000)
                    ax.fill_between(x,m.core_pdf(x),alpha=.3,color='green',label=f'Core ($\\eta$={m.eta:.2f})')
                    ax.fill_between(x,m.tail_pdf(x),alpha=.3,color='red',label='Tail')
                    ax.plot(x,m.total_pdf(x),'k--',lw=2,label='Total')
                ax.axvline(1.4,color='orange',ls=':',lw=2,label='RUWE=1.4')
                ax.set_xlabel('RUWE');ax.set_ylabel('Density');ax.set_xlim(.5,5)
                ax.legend(loc='upper right');ax.set_title('RUWE — Reg. Gaussian')
            style_ax(ax)
        for ci,(dc,lk,gk,xl,xlm,rl,rlab) in enumerate([
            ('rel_dist_err','dist_ln','dist_gm',r'$\sigma_d/d$',(0,1.5),.2,'20%'),
            ('rel_rv_err','rv_ln','rv_gm',r'$\sigma_v/v$',(0,1.5),None,None)]):
            if dc not in cat.df:continue
            err=cat.df[dc].dropna();err=err[(err>0)&(err<xlm[1]*1.5)]
            for row,(k,mn) in enumerate([(lk,'Log-Normal'),(gk,'Gamma')]):
                ax=axes[row,ci+1]
                ax.hist(err,bins=200,density=True,alpha=.5,color='steelblue',label='Data')
                if k in pa.models and pa.models[k].converged:
                    mdl=pa.models[k];x=np.linspace(.001,xlm[1]*1.2,2000)
                    cy=mdl.core_pdf(x);ty=mdl.tail_pdf(x)
                    ax.fill_between(x,cy,alpha=.3,color='green',label=f'Core ($\\eta$={mdl.eta:.2f})')
                    ax.fill_between(x,ty,alpha=.3,color='red',label='Tail')
                    ax.plot(x,cy+ty,'k--',lw=2,label='Total')
                    cr=np.where((cy[1:]<ty[1:])&(cy[:-1]>=ty[:-1]))[0]
                    if len(cr)>0:ax.axvline(x[cr[0]],color='purple',ls='-.',lw=2,label=f'Cross={x[cr[0]]:.3f}')
                if rl is not None:ax.axvline(rl,color='orange',ls=':',lw=2,label=rlab)
                ax.set_xlabel(xl);ax.set_ylabel('Density');ax.set_xlim(xlm)
                ax.legend(loc='upper right');ax.set_title(f'{xl} — {mn}')
                style_ax(ax)
        plt.tight_layout(rect=[0,0,1,.95]);s._sv(fig,'plot1_mixture_comparison.png')

    # ── PLOT 2: LEFT=P_core curves; RIGHT=purity histograms (count/1e6, xlim 2) ──
    def _p2(s,cat,pa):
        fig,axes=plt.subplots(1,2,figsize=(22,9))
        # LEFT: P_core(x) curves
        ax=axes[0]
        x_ruwe=np.linspace(.5,5,1000);x_err=np.linspace(.001,1.5,1000)
        if 'ruwe' in pa.models and pa.models['ruwe'].converged:
            m=pa.models['ruwe']
            lc=m._gll(x_ruwe,m.mu_c,m.sig_c);lt=m._gll(x_ruwe,m.mu_t,m.sig_t)
            lwc=np.log(m.eta)+lc;lwt=np.log(1-m.eta)+lt
            ax.plot(x_ruwe,np.exp(lwc-np.logaddexp(lwc,lwt)),color=COL_RUWE,lw=2.5,ls='-',label='RUWE (RegGauss)')
        for k,clr,ls,lab in [('dist_ln',COL_DIST,'--',r'Dist (LN)'),('dist_gm',COL_DIST,'-',r'Dist ($\Gamma$)')]:
            if k in pa.models and pa.models[k].converged:
                mdl=pa.models[k];c=mdl.core_pdf(x_err);t=mdl.tail_pdf(x_err);tot=c+t
                ax.plot(x_err,np.where(tot>1e-30,c/tot,.5),color=clr,ls=ls,lw=2.5,label=lab)
        for k,clr,ls,lab in [('rv_ln',COL_RV,'--',r'RV (LN)'),('rv_gm',COL_RV,'-',r'RV ($\Gamma$)')]:
            if k in pa.models and pa.models[k].converged:
                mdl=pa.models[k];c=mdl.core_pdf(x_err);t=mdl.tail_pdf(x_err);tot=c+t
                ax.plot(x_err,np.where(tot>1e-30,c/tot,.5),color=clr,ls=ls,lw=2.5,label=lab)
        ax.axhline(.5,color='gray',ls='--',alpha=.4)
        ax.set_xlabel('Error metric value');ax.set_ylabel(r'$P_{\rm core}$ (Purity)')
        ax.set_title('Purity Curves — All Metrics');ax.set_ylim(-.05,1.05);ax.legend(fontsize=10);ax.grid(True,alpha=.2)
        style_ax(ax)
        # RIGHT: purity histograms count/1e6
        ax=axes[1]
        bins=np.linspace(0,1,60)
        for col,clr,lab in [('purity_ruwe',COL_RUWE,'RUWE'),('purity_dist',COL_DIST,r'Distance ($\Gamma$)'),
            ('purity_rv',COL_RV,r'RV ($\Gamma$)'),('purity_combined',COL_COMB,'Combined')]:
            if col in cat.df:
                p=cat.df[col].dropna().values
                lw=3 if col=='purity_combined' else 1.8
                counts,_=np.histogram(p,bins=bins)
                ax.step(bins[:-1],counts/1e6,where='post',color=clr,lw=lw,label=f'{lab} ($\\mu$={np.mean(p):.3f})')
        ax.set_xlabel('Purity');ax.set_ylabel(r'Count [$\times 10^6$]');ax.set_title('Purity Distributions')
        ax.set_xlim(0,1);ax.legend(fontsize=10);ax.grid(True,alpha=.2)
        style_ax(ax)
        plt.tight_layout();s._sv(fig,'plot2_purity_curves_and_distributions.png')

    # ── PLOT 3: 3 figures (dist/rv/ruwe), each 2-row (purity+retention) with insets ──
    def _p3(s,cat,pa,tr):
        for key,mcol,pcol,title,xlbl in [
            ('dist','rel_dist_err','purity_dist',r'$\sigma_d/d$','Distance Error'),
            ('rv','rel_rv_err','purity_rv',r'$\sigma_v/v$','RV Error'),
            ('ruwe','ruwe','purity_ruwe','RUWE','RUWE')]:
            rk=key+'_radial';gk=key+'_global'
            if rk not in tr or gk not in tr:continue
            fig,(ax_pur,ax_ret)=plt.subplots(2,1,figsize=(16,14),sharex=True)
            rd=tr[rk];gd=tr[gk]
            rc=np.array(rd['r_centers']);bc=rd['by_cut']
            cut_keys=sorted(bc.keys(),key=float)
            clrs=plt.cm.viridis(np.linspace(0,.9,len(cut_keys)))

            # TOP: Radial purity profile
            for ck,clr in zip(cut_keys,clrs):
                pp=np.array(bc[ck]['purity']);vm=np.isfinite(pp)
                if np.sum(vm)>3:ax_pur.plot(rc[vm],pp[vm]*100,'-',color=clr,lw=1.5,label=f'{title}<{float(ck)}')
            ax_pur.set_ylabel('Mean Purity [%]');ax_pur.set_title(f'Radial Purity Profile — {xlbl} Cuts',fontsize=18,fontweight='bold')
            ax_pur.set_xscale('log');ax_pur.set_ylim(0,105);ax_pur.grid(True,alpha=.2)
            ax_pur.legend(loc='lower left',fontsize=7,ncol=3)
            style_ax(ax_pur)

            # INSET on purity: trade-off (retention+purity+model_completeness)
            ax_ins1=inset_axes(ax_pur,width="38%",height="35%",loc='upper right',borderpad=2.5)
            ax_ins1.patch.set_facecolor('white');ax_ins1.patch.set_alpha(1.0)
            ax_ins1.plot(gd['cuts'],[r*100 for r in gd['retention']],'o-',color='#2196F3',lw=2,ms=4,label='Retention')
            ax_ins1.plot(gd['cuts'],[p*100 if not np.isnan(p) else 0 for p in gd['mean_purity']],'s--',color='#F44336',lw=2,ms=4,label='Purity')
            ax_ins1.plot(gd['cuts'],[c*100 for c in gd['model_completeness']],'D:',color='#228833',lw=2,ms=4,label='Model Compl.')
            ax_ins1.set_xlabel(f'{title} cut',fontsize=9,fontweight='bold');ax_ins1.set_ylabel('%',fontsize=9,fontweight='bold')
            ax_ins1.set_title('Trade-off',fontsize=10,fontweight='bold');ax_ins1.set_ylim(0,105)
            ax_ins1.legend(fontsize=7,loc='center right');ax_ins1.grid(True,alpha=.3)
            style_ax(ax_ins1)

            # BOTTOM: Radial retention profile
            for ck,clr in zip(cut_keys,clrs):
                rr=np.array(bc[ck]['retention']);vm=np.isfinite(rr)
                if np.sum(vm)>3:ax_ret.plot(rc[vm],rr[vm]*100,'--',color=clr,lw=1.5,label=f'{title}<{float(ck)}')
            ax_ret.set_xlabel('Galactocentric Distance R [kpc]');ax_ret.set_ylabel('Retention [%]')
            ax_ret.set_title(f'Radial Retention Profile — {xlbl} Cuts',fontsize=18,fontweight='bold')
            ax_ret.set_xscale('log');ax_ret.set_ylim(0,105);ax_ret.grid(True,alpha=.2)
            ax_ret.legend(loc='lower left',fontsize=7,ncol=3)
            style_ax(ax_ret)

            # INSET on retention: radial error profile
            ax_ins2=inset_axes(ax_ret,width="38%",height="35%",loc='upper right',borderpad=2.5)
            ax_ins2.patch.set_facecolor('white');ax_ins2.patch.set_alpha(1.0)
            if mcol in cat.df and 'R_gal' in cat.df:
                rg=cat.df['R_gal'].values;vals=cat.df[mcol].values
                re2=np.logspace(np.log10(.1),np.log10(300),40);rcm=.5*(re2[:-1]+re2[1:])
                meds=[];p16s=[];p84s=[]
                for j in range(len(rcm)):
                    mask2=(rg>=re2[j])&(rg<re2[j+1])&np.isfinite(vals);n2=np.sum(mask2)
                    if n2>10:vv=vals[mask2];meds.append(np.median(vv));p16s.append(np.percentile(vv,16));p84s.append(np.percentile(vv,84))
                    else:meds.append(np.nan);p16s.append(np.nan);p84s.append(np.nan)
                meds=np.array(meds);p16s=np.array(p16s);p84s=np.array(p84s);vm2=np.isfinite(meds)
                if np.sum(vm2)>2:
                    ax_ins2.plot(rcm[vm2],meds[vm2],'b-',lw=1.5)
                    ax_ins2.fill_between(rcm[vm2],p16s[vm2],p84s[vm2],alpha=.25,color='blue')
                ax_ins2.set_xscale('log');ax_ins2.set_xlabel('R [kpc]',fontsize=9,fontweight='bold')
                ax_ins2.set_ylabel(f'Median {title}',fontsize=9,fontweight='bold')
                ax_ins2.set_title('Error Profile (16-84th)',fontsize=10,fontweight='bold')
                ax_ins2.grid(True,alpha=.3)
            style_ax(ax_ins2)
            plt.tight_layout();s._sv(fig,f'plot3_radial_{key}.png')

    # ── PLOT 4: Core fraction η bar chart ──
    def _p4_eta(s,pr):
        fig,ax=plt.subplots(figsize=(12,7))
        labels=[];vals=[];clrs=[]
        if 'ruwe' in pr:labels.append('RUWE');vals.append(pr['ruwe'].get('eta',0));clrs.append('#9b59b6')
        if 'dist' in pr:
            labels.append('Dist (LN)');vals.append(pr['dist'].get('lognormal',{}).get('eta',0));clrs.append('#2196F3')
            labels.append(r'Dist ($\Gamma$)');vals.append(pr['dist'].get('gamma',{}).get('eta',0));clrs.append('#F44336')
        if 'rv' in pr:
            labels.append('RV (LN)');vals.append(pr['rv'].get('lognormal',{}).get('eta',0));clrs.append('#00BCD4')
            labels.append(r'RV ($\Gamma$)');vals.append(pr['rv'].get('gamma',{}).get('eta',0));clrs.append('#FF9800')
        if not labels:return
        bars=ax.bar(labels,vals,color=clrs,edgecolor='black',alpha=.85)
        for b,v in zip(bars,vals):
            if isinstance(v,(int,float)):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,f'{v:.3f}',ha='center',fontsize=13,fontweight='bold')
        ax.set_ylim(0,1.05);ax.axhline(.9,color='red',ls='--',alpha=.4,label=r'$\eta$=0.9')
        ax.axhline(.5,color='gray',ls='--',alpha=.3,label=r'$\eta$=0.5')
        ax.set_ylabel(r'Core Fraction $\eta$');ax.set_title(r'Core Fraction $\eta$ by Model',fontsize=18,fontweight='bold')
        ax.legend(fontsize=11);style_ax(ax)
        plt.tight_layout();s._sv(fig,'plot4_core_fraction_eta.png')

    # ── PLOT 5: Mean purity by method bar chart ──
    def _p5_purity_by_method(s,pr):
        fig,ax=plt.subplots(figsize=(12,7))
        labels=[];vals=[];clrs=[]
        if 'ruwe' in pr:labels.append('RUWE');vals.append(pr['ruwe'].get('mean_purity',0));clrs.append('#9b59b6')
        if 'dist' in pr:
            labels.append('Dist (LN)');vals.append(pr['dist'].get('mean_purity_ln',0));clrs.append('#2196F3')
            labels.append(r'Dist ($\Gamma$)');vals.append(pr['dist'].get('mean_purity_gm',0));clrs.append('#F44336')
        if 'rv' in pr:
            labels.append('RV (LN)');vals.append(pr['rv'].get('mean_purity_ln',0));clrs.append('#00BCD4')
            labels.append(r'RV ($\Gamma$)');vals.append(pr['rv'].get('mean_purity_gm',0));clrs.append('#FF9800')
        if 'combined' in pr:
            labels.append('Combined');vals.append(pr['combined'].get('mean',0));clrs.append('#222222')
        if not labels:return
        bars=ax.bar(labels,vals,color=clrs,edgecolor='black',alpha=.85)
        for b,v in zip(bars,vals):
            if isinstance(v,(int,float)):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.015,f'{v:.3f}',ha='center',fontsize=13,fontweight='bold')
        ax.set_ylim(0,1.05);ax.axhline(.8,color='orange',ls='--',alpha=.4,label='P=0.8')
        ax.set_ylabel('Mean Purity');ax.set_title('Mean Purity by Method',fontsize=18,fontweight='bold')
        ax.legend(fontsize=11);style_ax(ax)
        plt.tight_layout();s._sv(fig,'plot5_purity_by_method.png')

    # ── PLOT 6: GC/OC/SGR completeness vs quality cut (Eq.22) ──
    def _p6_gc_oc_sgr(s,tr,truth):
        if not truth or not truth.truth:s.L.info("  No truth data for plot6");return
        fig,axes=plt.subplots(1,3,figsize=(24,8))
        fig.suptitle('GC/OC/SGR Completeness vs Quality Cut (Eq. 22)',fontsize=20,fontweight='bold',y=1.0)
        truth_colors={'GC':'#2ecc71','OC':'#3498db','SGR':'#e74c3c'}
        truth_markers={'GC':'o','OC':'s','SGR':'D'}
        for axi,(gk,title,xlabel) in enumerate([
            ('dist_global',r'$\sigma_d/d$',r'$\sigma_d/d$ cut'),
            ('rv_global',r'$\sigma_v/v$',r'$\sigma_v/v$ cut'),
            ('ruwe_global','RUWE','RUWE cut')]):
            ax=axes[axi]
            if gk not in tr:ax.text(.5,.5,'No data',ha='center',va='center',transform=ax.transAxes);style_ax(ax);continue
            d=tr[gk];has_any=False
            for label in truth.truth:
                tk=f'truth_completeness_{label}'
                if tk in d:
                    ax.plot(d['cuts'],[c*100 for c in d[tk]],f'{truth_markers.get(label,"o")}-',
                        color=truth_colors.get(label,'gray'),lw=2.5,ms=6,label=f'{label} (N={truth.truth[label]["n_total"]:,})')
                    has_any=True
            # Also plot retention and model completeness for reference
            ax.plot(d['cuts'],[r*100 for r in d['retention']],'k--',lw=1.5,alpha=.5,label='Retention')
            ax.plot(d['cuts'],[c*100 for c in d['model_completeness']],'k:',lw=1.5,alpha=.5,label='Model Compl.')
            ax.set_xlabel(xlabel);ax.set_ylabel('Completeness [%]')
            ax.set_title(f'{title} Cuts',fontsize=16,fontweight='bold')
            ax.set_ylim(0,105);ax.grid(True,alpha=.3);ax.legend(fontsize=9)
            style_ax(ax)
        plt.tight_layout();s._sv(fig,'plot6_gc_oc_sgr_completeness.png')

# ======================= MAIN PIPELINE =======================================
def to_json(o):
    if isinstance(o,np.ndarray):return o.tolist()
    elif isinstance(o,(np.integer,np.floating)):return float(o)
    elif isinstance(o,dict):return{k:to_json(v) for k,v in o.items()}
    elif isinstance(o,list):return[to_json(i) for i in o]
    return o

def run(cfg):
    os.makedirs(cfg.output_dir,exist_ok=True)
    L=setup_log(os.path.join(cfg.output_dir,'analysis_v8.log'))
    L.info("="*70+f"\nCOMPLETENESS & PURITY v8\n"+"="*70)
    L.info(f"Started: {datetime.now().isoformat()}")
    L.info(f"eta:[{cfg.min_eta},{cfg.max_eta}] sep_frac={cfg.min_sep_frac} tail_ratio={cfg.min_tail_width_ratio}")
    start=time.time();results={}
    try:
        cat=CatLoader(cfg,L)
        if not cat.load():L.error("Load failed!");return{}
        truth=TruthLoader(cfg,L,cat)
        has_truth=truth.load_all()
        if has_truth:truth.xmatch()
        pa=PurityAnalyzer(cfg,L,cat);pr=pa.run();results['purity']=pr
        ta=TradeoffAnalyzer(cfg,L,cat,pa,truth if has_truth else None)
        tr=ta.run();results['tradeoff']=tr
        if has_truth:
            results['truth_completeness']={lab:{'n_total':td['n_total'],'n_matched':td['n_matched'],
                'completeness_raw':td['completeness_raw']} for lab,td in truth.truth.items()}
        jf=os.path.join(cfg.output_dir,'results_v8.json')
        with open(jf,'w') as f:json.dump(to_json(results),f,indent=2,default=str)
        L.info(f"  Saved JSON: {jf}")
        mp={'models':{}}
        for k,m in pa.models.items():mp['models'][k]=m.params
        with open(os.path.join(cfg.output_dir,'model_params.json'),'w') as f:
            json.dump(to_json(mp),f,indent=2,default=str)
        save_extra_plot_data(cfg, cat, cfg.output_dir)    
        pl=Plotter(cfg,L,cfg.output_dir)
        pl.plot_all(cat,pa,pr,tr,truth if has_truth else None)
        elapsed=time.time()-start
        L.info(f"\n{'='*70}\nDONE in {elapsed/60:.1f} min\n{'='*70}")
    except Exception as e:
        L.error(f"FAILED: {e}");import traceback;traceback.print_exc();raise
    return results
"""
ADD THIS FUNCTION to your pipeline script (completeness_purity_v8.py),
then call it inside run() just before the Plotter section.

Usage inside run():
    save_extra_plot_data(cfg, cat, OUTPUT_JSON_PATH)
"""



def save_extra_plot_data(cfg, cat, out_dir):
    """
    Saves histogram data needed for full plot reproduction from JSON.
    Call this inside run() after PurityAnalyzer.run() and before Plotter.
    Output: <out_dir>/extra_plot_data.json
    """
    extra = {}

    # ── 1. Metric histograms (for Plot 1 blue bars) ──────────────────────────
    metric_hists = {}
    for col, xlim, nb in [
        ('ruwe',         (0.5, 5.0), 200),
        ('rel_dist_err', (0.0, 1.5), 200),
        ('rel_rv_err',   (0.0, 1.5), 200),
    ]:
        if col not in cat.df.columns:
            continue
        vals = cat.df[col].dropna().values
        vals = vals[(vals > xlim[0]) & (vals < xlim[1])]
        counts, edges = np.histogram(vals, bins=nb, range=xlim, density=True)
        metric_hists[col] = {
            'counts': counts.tolist(),
            'edges':  edges.tolist(),
            'xlim':   list(xlim),
        }
    extra['metric_histograms'] = metric_hists

    # ── 2. Purity histograms (for Plot 2 right panel) ────────────────────────
    purity_hists = {}
    for col in ['purity_ruwe', 'purity_dist', 'purity_rv', 'purity_combined']:
        if col not in cat.df.columns:
            continue
        vals = cat.df[col].dropna().values
        counts, edges = np.histogram(vals, bins=60, range=(0, 1))
        purity_hists[col] = {
            'counts': counts.tolist(),
            'edges':  edges.tolist(),
            'mean':   float(np.nanmean(vals)),
        }
    extra['purity_histograms'] = purity_hists

    # ── 3. Radial error profiles (for Plot 3 bottom insets) ──────────────────
    radial_error_profiles = {}
    if 'R_gal' in cat.df.columns:
        rg   = cat.df['R_gal'].values
        re2  = np.logspace(np.log10(0.1), np.log10(300), 40)
        rcm  = (0.5 * (re2[:-1] + re2[1:])).tolist()
        for col in ['rel_dist_err', 'rel_rv_err', 'ruwe']:
            if col not in cat.df.columns:
                continue
            vals2 = cat.df[col].values
            meds, p16s, p84s = [], [], []
            for j in range(len(rcm)):
                mask = (rg >= re2[j]) & (rg < re2[j+1]) & np.isfinite(vals2)
                if np.sum(mask) > 10:
                    vv = vals2[mask]
                    meds.append(float(np.median(vv)))
                    p16s.append(float(np.percentile(vv, 16)))
                    p84s.append(float(np.percentile(vv, 84)))
                else:
                    meds.append(None); p16s.append(None); p84s.append(None)
            radial_error_profiles[col] = {
                'r_centers': rcm,
                'median': meds,
                'p16': p16s,
                'p84': p84s,
            }
    extra['radial_error_profiles'] = radial_error_profiles

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, 'extra_plot_data.json')
    with open(out_path, 'w') as f:
        json.dump(extra, f, indent=2)
    print(f"Saved extra plot data: {out_path}")
    return extra
def main():
    p=argparse.ArgumentParser(description='Completeness & Purity v8')
    p.add_argument('--master','-m',required=True)
    p.add_argument('--gc',default='',help='GC members CSV/FITS')
    p.add_argument('--oc',default='',help='OC members CSV/FITS')
    p.add_argument('--sgr',default='',help='SGR members CSV/FITS')
    p.add_argument('--output','-o',default='./v8_results')
    p.add_argument('--eta-min',type=float,default=.40)
    p.add_argument('--eta-max',type=float,default=.90)
    p.add_argument('--min-sep-frac',type=float,default=.8)
    p.add_argument('--min-tail-width-ratio',type=float,default=2.0)
    p.add_argument('--min-core-sigma-floor',type=float,default=.15)
    p.add_argument('--n-radial-bins',type=int,default=80)
    p.add_argument('--rgal-max',type=float,default=300.)
    p.add_argument('--max-chunks',type=int,default=None)
    p.add_argument('--max-rows-per-chunk',type=int,default=None)
    p.add_argument('--max-samples',type=int,default=2_000_000)
    p.add_argument('--dpi',type=int,default=150)
    p.add_argument('--xmatch-radius',type=float,default=1.0)
    a=p.parse_args()
    cfg=Cfg(master_catalog_dir=a.master,gc_file=a.gc,oc_file=a.oc,sgr_file=a.sgr,
        output_dir=a.output,min_eta=a.eta_min,max_eta=a.eta_max,
        min_sep_frac=a.min_sep_frac,min_tail_width_ratio=a.min_tail_width_ratio,
        min_core_sigma_floor=a.min_core_sigma_floor,
        n_radial_bins=a.n_radial_bins,rgal_max=a.rgal_max,
        max_chunks=a.max_chunks,max_rows_per_chunk=a.max_rows_per_chunk,
        max_samples=a.max_samples,plot_dpi=a.dpi,xmatch_radius_arcsec=a.xmatch_radius)
    r=run(cfg);return 0 if r else 1

if __name__=='__main__':sys.exit(main())