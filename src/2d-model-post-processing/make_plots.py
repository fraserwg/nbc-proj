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
from matplotlib.ticker import EngFormatter, ScalarFormatter
import xarray as xr
import cmocean.cm as cmo
import xmitgcm
import f90nml
import zarr

logging.info('Importing custom python libraries')
import pvcalc
from importlib import reload
reload(pvcalc)

logging.info('Setting paths')
base_path = Path('../../').absolute().resolve()
figure_path = base_path / 'figures'
figure_path.mkdir(exist_ok=True)
data_path = base_path / 'data'
raw_path = data_path / 'raw'
processed_path = data_path / 'processed'
interim_path = data_path / 'interim'
run_folder = raw_path / 'mitgcm-models-2d'
run_names = ['StandardNoSlip', 'ViscousNoSlip']


logging.info('Setting plotting defaults')
# fonts
if Path('/System/Library/Fonts/Supplemental/PTSans.ttc').exists():
    fpath = Path('/System/Library/Fonts/Supplemental/PTSans.ttc')
elif Path('/home/n01/n01/fwg/.local/share/fonts/PTSans-Regular.ttf').exists():
    fpath = Path('/home/n01/n01/fwg/.local/share/fonts/PTSans-Regular.ttf')
else:
    fpath = None
if fpath != None:
    print(fpath)
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


def open_dataset(run_name):
    run_path = run_folder / run_name
    
    logging.info('Reading in model parameters from the namelist')
    with open(run_path / 'data') as data:
        data_nml = f90nml.read(data)

    delta_t = data_nml['parm03']['deltat']
    f0 = data_nml['parm01']['f0']
    try:
        beta = data_nml['parm01']['beta']
    except KeyError:
        beta = 0
    no_slip_bottom = data_nml['parm01']['no_slip_bottom']
    no_slip_sides = data_nml['parm01']['no_slip_sides']


    logging.info('Reading in the model dataset')
    ds = xmitgcm.open_mdsdataset(run_path,
                                prefix=['ZLevelVars', 'IntLevelVars'],
                                delta_t=delta_t,
                                geometry='cartesian',
                                chunks=-1
                                )


    logging.info('Calculating the potential vorticity')
    grid = pvcalc.create_xgcm_grid(ds)
    ds['drL'] = pvcalc.create_drL_from_dataset(ds)
    ds['rho'] = pvcalc.calculate_density(ds['RHOAnoma'], ds['rhoRef'])
    ds['b'] = pvcalc.calculate_buoyancy(ds['rho'])

    ds['db_dx'], ds['db_dy'], ds['db_dz'] = pvcalc.calculate_grad_buoyancy(ds['b'], ds, grid)

    ds['zeta_x'], ds['zeta_y'], ds['zeta_z'] = pvcalc.calculate_curl_velocity(ds['UVEL'],
                                                                            ds['VVEL'],
                                                                            ds['WVEL'],
                                                                            ds,
                                                                            grid,no_slip_bottom,
                                                                            no_slip_sides, diff_y=False
                                                                            )


    ds['Q'] = pvcalc.calculate_C_potential_vorticity(ds['zeta_x'],
                                                    ds['zeta_y'],
                                                    ds['zeta_z'],
                                                    ds['b'],
                                                    ds,
                                                    grid,
                                                    beta,
                                                    f0, diff_y=False
                                                    )

    return ds

potential_vorticity = True

if potential_vorticity:
    ds = open_dataset('StandardNoSlip')
    logging.info('Plotting the PV')
    clim = 2e-9
    
    fig = plt.figure(figsize=[6, 4])
    gs = gridspec.GridSpec(2, 3,
                           height_ratios=[14, 1])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds['XG'] * 1e-3,
                          -ds['Zl'],
                          ds['Q'].isel(time=0, YG=0),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds['XG'] * 1e-3,
                          -ds['Zl'],
                          ds['Q'].isel(YG=0).sel(time=np.timedelta64(14, 'D'), method='nearest'),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = ax2.pcolormesh(ds['XG'] * 1e-3,
                          -ds['Zl'],
                          ds['Q'].isel(YG=0).sel(time=np.timedelta64(28, 'D'), method='nearest'),
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

    fig.suptitle('Potential vorticity in a 2D model')

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

    fig.savefig(figure_path / '2d_pv.pdf', dpi=dpi)


def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc


wvel_subsetting = True
if wvel_subsetting:
    for run in run_names:
        ds = open_dataset(run)
        t = np.arange(0, np.timedelta64(50, 'D'), np.timedelta64(7, 'D'))
        ds_subset = ds['WVEL'].sel(time=t, Zl=-50, method='nearest').to_dataset(name='WVEL')
        out_name = processed_path / ('WVEL50m2D' + run + '.zarr')
        print(out_name)
        enc = create_encoding_for_ds(ds_subset, 5)
        ds_subset.to_zarr(out_name, mode='w', encoding=enc)