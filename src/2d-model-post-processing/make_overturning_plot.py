import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                    )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import font_manager as fm
from matplotlib.ticker import ScalarFormatter
import xarray as xr
import cmocean.cm as cmo


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

potential_vorticity = False
overturning_streamfunction = False

ds = xr.open_mfdataset(run_path.glob('proc*.nc'))
ds = ds.drop_vars(['XC', 'XG'])
ds = ds.rename_dims({'X': 'XC', 'Xp1': 'XG', 'T': 'time'})
ds = ds.rename_vars({'X': 'XC', 'Xp1': 'XG', 'T': 'time'})
ds = ds.assign_coords({'YC': [1]})
ds = ds.assign_coords({'YG': [1]})
ds['Zp1'] = ('Zp1', np.append(ds['Z'].values, -1500))
ds['Zu'] = ('Zu', ds['Zp1'].values[1:])
ds = ds.isel(XG=slice(1, None))

if overturning_streamfunction:
    ds['psir'] = ds['psi'].rolling(dim={'time': 44}, center=True).mean()

    clim = 3
    
    fig = plt.figure(figsize=[6, 4])
    gs = gridspec.GridSpec(2, 3,
                           height_ratios=[14, 1])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds['XC'] * 1e-3,
                          -ds['Zl'],
                          ds['psir'].sel(time=1 * 7 * 24 * 60 * 60, method='nearest'),
                          cmap=cmo.balance, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds['XC'] * 1e-3,
                          -ds['Zl'],
                          ds['psir'].sel(time=3 * 7 * 24 * 60 * 60, method='nearest'),
                          cmap=cmo.balance, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = ax2.pcolormesh(ds['XC'] * 1e-3,
                          -ds['Zl'],
                          ds['psir'].sel(time=5 * 7 * 24 * 60 * 60, method='nearest'),
                          cmap=cmo.balance, shading='nearest',
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

    ax0.set_title('1 weeks')
    ax1.set_title('3 weeks')
    ax2.set_title('5 weeks')
    
    ax0.set_title('(a)', loc='left')
    ax1.set_title('(b)', loc='left')
    ax2.set_title('(c)', loc='left')

    fig.suptitle('Overturning streamfunction in a 2D model')

    ax0.set_ylabel('Depth (m)')
    ax1.set_xlabel('Longitude (km)')

    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbax = fig.add_subplot(gs[1, :])
    cb = fig.colorbar(cax0, cax=cbax, orientation='horizontal',
                      label='$\psi$ (m$^2\\,$s$^{-1}$)', format=fmt)

    fig.tight_layout()

    fig.savefig(figure_path / '2d_overturning.pdf', dpi=dpi)