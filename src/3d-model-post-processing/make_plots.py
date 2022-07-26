import logging
from pickle import FALSE
from typing import overload

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                    )

logging.info('Importing standard python libraries')
from pathlib import Path
from os import cpu_count

logging.info('Importing third party python libraries')
import numpy as np
from scipy.sparse.linalg import eigs
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import font_manager as fm
from matplotlib.ticker import EngFormatter, ScalarFormatter
import xarray as xr
import cmocean.cm as cmo
from joblib import Parallel, delayed
from MITgcmutils.mds import rdmds
import xrft
from cycler import cycler


logging.info('Importing custom python libraries')


logging.info('Setting paths')
base_path = Path('../../').absolute().resolve()
figure_path = base_path / 'figures'
figure_path.mkdir(exist_ok=True)
data_path = base_path / 'data'
raw_path = data_path / 'raw'
processed_path = data_path / 'processed'
interim_path = data_path / 'interim'
run_path = interim_path / '2DStandardNoSlip'


logging.info('Setting plotting defaults')
# fonts
fpath = Path('/System/Library/Fonts/Supplemental/PTSans.ttc')
if fpath.exists():
    font_prop = fm.FontProperties(fname=fpath)
    plt.rcParams['font.family'] = font_prop.get_family()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

# font size
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
plt.rc('text', usetex=False)
plt.rcParams['axes.titlesize'] = 12

# output
dpi = 600

# Set plots to make
pv_50m = True
vorticity_comparison = True
pv_slice = True
vorticity_correlations = True
power_spectra = True
formation = True

run_names = ['StandardNoSlip',
             'StandardFreeSlip',
             'ExtraViscousNoSlip',
             'ExtraViscousFreeSlip']

plot_titles = {'StandardNoSlip': 'Standard no-slip',
               'StandardFreeSlip': 'Standard free-slip',
               'ExtraViscousNoSlip': 'Viscous no-slip',
               'ExtraViscousFreeSlip': 'Viscous free-slip'}


def plot_pv_50m(run_name):
    clim = 2e-8
    run_path = processed_path / ('3DPV50m' + run_name + '.zarr')
    assert run_path.exists()
    ds = xr.open_dataset(run_path, engine='zarr')

    height = 9
    fig = plt.figure(figsize=[6, height])
    gs = gridspec.GridSpec(2, 3,
                           height_ratios=[height, 4 / 14])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds['Xp1'] * 1e-3,
                          ds['Yp1'] * 1e-3,
                          ds['potVort'].isel(T=0),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds['Xp1'] * 1e-3,
                          ds['Yp1'] * 1e-3,
                          ds['potVort'].isel(T=1),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = ax2.pcolormesh(ds['Xp1'] * 1e-3,
                          ds['Yp1'] * 1e-3,
                          ds['potVort'].isel(T=2),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax0.set_ylim(-512 + 40, 2176 - 350)
    ax1.set_ylim(-512 + 40, 2176 - 350)
    ax2.set_ylim(-512 + 40, 2176 - 350)

    ax0.set_facecolor('grey')
    ax1.set_facecolor('grey')
    ax2.set_facecolor('grey')

    ax0.set_xlim(0, 400)
    ax1.set_xlim(0, 400)
    ax2.set_xlim(0, 400)

    ax0.set_title('26 days')
    ax1.set_title('34 days')
    ax2.set_title('42 days')
    
    ax0.set_title('(a)', loc='left')
    ax1.set_title('(b)', loc='left')
    ax2.set_title('(c)', loc='left')

    ax0.set_aspect('equal')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    fig.suptitle(plot_titles[run_name])

    ax0.set_ylabel('Latitude (km)')
    ax1.set_xlabel('Longitude (km)')

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(cax0, cax=cbax, orientation='horizontal',
                      label='$Q$ (s$^{-3}$)', format=fmt)

    fig.tight_layout()

    figure_name = '3DPV50m' + run_name + '.pdf'
    fig.savefig(figure_path / figure_name, dpi=dpi)

pv_50m = False
if pv_50m:
    for run in run_names: plot_pv_50m(run)


def plot_pv_slice(run_name):
    clim = 2e-9
    run_path = processed_path / ('3DPV750km' + run_name + '.zarr')
    assert run_path.exists()
    ds = xr.open_dataset(run_path, engine='zarr')

    fig = plt.figure(figsize=[6, 4])
    gs = gridspec.GridSpec(2, 3,
                           height_ratios=[14, 1])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds['Xp1'] * 1e-3,
                          -ds['Zl'],
                          ds['potVort'].isel(T=0),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds['Xp1'] * 1e-3,
                          -ds['Zl'],
                          ds['potVort'].sel(T=2 * 7 * 24 * 60 * 60, method='nearest'),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = ax2.pcolormesh(ds['Xp1'] * 1e-3,
                          -ds['Zl'],
                          ds['potVort'].sel(T=4 * 7 * 24 * 60 * 60, method='nearest'),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax0.set_ylim(1200, 0)
    ax1.set_ylim(1200, 0)
    ax2.set_ylim(1200, 0)

    ax0.set_facecolor('grey')
    ax1.set_facecolor('grey')
    ax2.set_facecolor('grey')

    ax0.set_xlim(0, 200)
    ax1.set_xlim(0, 200)
    ax2.set_xlim(0, 200)

    ax0.set_title('0 weeks')
    ax1.set_title('2 weeks')
    ax2.set_title('4 weeks')
    
    ax0.set_title('(a)', loc='left')
    ax1.set_title('(b)', loc='left')
    ax2.set_title('(c)', loc='left')

    fig.suptitle(plot_titles[run_name])

    ax0.set_ylabel('Depth (m)')
    ax1.set_xlabel('Longitude (km)')

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(cax0, cax=cbax, orientation='horizontal',
                      label='$Q$ (s$^{-3}$)', format=fmt)

    fig.tight_layout()
    
    figure_name = '3DPV750km' + run_name + '.pdf'
    fig.savefig(figure_path / figure_name, dpi=dpi)

pv_slice = False
if pv_slice:
    for run in run_names: plot_pv_slice(run)

vorticity_comparison = False
if vorticity_comparison:
    rclim, qclim = 2.2e-5, 2e-8

    ds = xr.open_dataset(processed_path / '3DVorticityComparison50mStandardNoSlip.zarr',
                         engine='zarr')

    height = 9
    fig = plt.figure(figsize=[6, height])
    gs = gridspec.GridSpec(2, 2,
                           height_ratios=[height, 4 / 14])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds['Xp1'] * 1e-3,
                          ds['Yp1'] * 1e-3,
                          ds['momVort3'].squeeze(),
                          cmap=cmo.balance, shading='nearest',
                          vmin=-rclim, vmax=rclim, rasterized=True)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds['X'] * 1e-3,
                          ds['Y'] * 1e-3,
                          ds['potVort'].squeeze(),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-qclim, vmax=qclim, rasterized=True)

    ax0.set_ylim(0, 1250)
    ax1.set_ylim(0, 1250)

    ax0.set_facecolor('grey')
    ax1.set_facecolor('grey')

    ax0.set_xlim(0, 400)
    ax1.set_xlim(0, 400)

    ax0.set_title('Relative vorticity')
    ax1.set_title('Potential vortticity')
    
    ax0.set_title('(a)', loc='left')
    ax1.set_title('(b)', loc='left')

    ax0.set_aspect('equal')
    ax1.set_aspect('equal')

    #fig.suptitle(plot_titles[run_name])

    ax0.set_ylabel('Latitude (km)')
    ax0.set_xlabel('Longitude (km)')
    ax1.set_xlabel('Longitude (km)')

    ax1.set_yticklabels([])
    
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    qcbax = fig.add_subplot(gs[1, 1])
    qcb = fig.colorbar(cax1, cax=qcbax, orientation='horizontal',
                       label='$Q$ (s$^{-3}$)', format=fmt)

    rcbax = fig.add_subplot(gs[1, 0])
    rcb = fig.colorbar(cax0, cax=rcbax, orientation='horizontal',
                       label='$\\xi$ (s$^{-3}$)', format=fmt)

    fig.tight_layout()

    figure_name = 'VorticityComparison.pdf'
    fig.savefig(figure_path / figure_name, dpi=dpi)

if power_spectra:
    n = 8
    color = cmo.tempo(np.linspace(0.1, 0.9, n))
    plt.rcParams['axes.prop_cycle'] = cycler(color=color)
    wvels = ['3DStandard', '3DExtraViscous', '2DStandard']#, '2DViscous']
    i = 0
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))
    axs= axs.flatten()
    for wvel in wvels:

        run_path = processed_path / (wvel[:2] + 'WVEL50m' + wvel[2:] + 'NoSlip.zarr')    

        print(run_path)
        assert run_path.exists()
        ds = xr.open_dataset(run_path, engine='zarr')
        if wvel.startswith('3D'):
            ds = ds.sel(X=slice(0, 200e3), Y=slice(250e3, 750e3))
            wvel_spec = xrft.power_spectrum(0.5 * ds['WVEL'],
                                            dim=['X'],
                                            detrend='constant')
            wvel_spec = wvel_spec.mean(dim='Y').squeeze()
        
        elif wvel.startswith('2D') and wvel.endswith('Standard'):
            ds = ds.sel(X=slice(0, 200e3))
            wvel_spec = xrft.power_spectrum(0.5 * ds['WVEL'],
                                            dim=['X'],
                                            detrend='constant')
    
        lambda_x = 2 * np.pi / wvel_spec['freq_X'] * 1e-3
        for nt in range(wvel_spec.sizes['T']):
            axs[i].loglog(lambda_x, wvel_spec.isel(T=nt))

        i += 1
