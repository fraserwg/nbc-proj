import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                    )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import font_manager as fm
from matplotlib.ticker import ScalarFormatter
import xarray as xr
import cmocean.cm as cmo
import xrft
from cycler import cycler

logging.info('Importing custom python libraries')


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/nbc-proj')
figure_path = base_path / 'figures'
figure_path.mkdir(exist_ok=True)
data_path = base_path / 'data'
raw_path = data_path / 'raw'
processed_path = data_path / 'processed'
interim_path = data_path / 'interim'
run_path = interim_path / '2DStandardNoSlip'
log_path = base_path / 'src/3d-model-post-processing/.tmp'
dask_worker_path = log_path / 'dask-worker-space'
env_path = base_path / 'nbc-proj/bin/activate'

logging.info('Setting plotting defaults')
# fonts
fpath = Path('/home/n01/n01/fwg/.local/share/fonts/PTSans-Regular.ttf')
assert fpath.exists()
font_prop = fm.FontProperties(fname=fpath)
plt.rcParams['font.family'] = font_prop.get_family()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

# font size
mpl.use("pgf")
plt.rc('xtick', labelsize='8')
plt.rc('ytick', labelsize='8')
plt.rcParams['axes.titlesize'] = 10
plt.rcParams["text.latex.preamble"] = "\\usepackage{euler} \\usepackage{paratype}  \\usepackage{mathfont} \\mathfont[digits]{PT Sans}"
plt.rcParams["pgf.preamble"] = plt.rcParams["text.latex.preamble"]
plt.rc('text', usetex=False)


# output
dpi = 600

# Set plots to make
pv_50m = False
vorticity_comparison = False
pv_slice = False
vorticity_correlations = False
power_spectra = True
plot_formation = False
v_init = False

run_names = ['StandardNoSlip',
             'StandardFreeSlip',
             'ViscousNoSlip',
             'ViscousFreeSlip']

plot_titles = {'StandardNoSlip': 'Standard no-slip',
               'StandardFreeSlip': 'Standard free-slip',
               'ViscousNoSlip': 'Viscous no-slip',
               'ViscousFreeSlip': 'Viscous free-slip'}


def plot_pv_50m(run_name):
    clim = 2e-8
    run_path = processed_path / f'PV3D{run_name}.zarr'
    assert run_path.exists()
    ds = xr.open_dataset(run_path, engine='zarr')

    height = 8
    fig = plt.figure(figsize=[6, height])
    gs = gridspec.GridSpec(2, 3,
                           height_ratios=[height, 2 / 14])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds['XG'] * 1e-3,
                          ds['YG'] * 1e-3,
                          ds['Q'].isel(time=0),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds['XG'] * 1e-3,
                          ds['YG'] * 1e-3,
                          ds['Q'].isel(time=1),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = ax2.pcolormesh(ds['XG'] * 1e-3,
                          ds['YG'] * 1e-3,
                          ds['Q'].isel(time=2),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax0.set_ylim(-512 + 40, 2176 - 350 - 250)
    ax1.set_ylim(-512 + 40, 2176 - 350 - 250)
    ax2.set_ylim(-512 + 40, 2176 - 350 - 250)

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
                      format=fmt)
    
    cb.set_label("$Q$ (s$^{-3}$)", usetex=True)

    fig.tight_layout()

    out_figure = figure_path / f'PV3DSnapshot{run_name}.pdf'
    
    logging.info(f"Saving to {out_figure}")
    fig.savefig(out_figure, dpi=dpi)

if pv_50m:
    for run in run_names: plot_pv_50m(run)


def plot_pv_slice(run_name):
    clim = 2e-9
    run_path = processed_path / f'PV3DSlice{run_name}.zarr'
    assert run_path.exists()
    ds = xr.open_dataset(run_path, engine='zarr')

    fig = plt.figure(figsize=[6, 4])
    gs = gridspec.GridSpec(2, 3,
                           height_ratios=[14, 1])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds['XG'] * 1e-3,
                          -ds['Zl'],
                          ds['Q'].isel(time=0),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds['XG'] * 1e-3,
                          -ds['Zl'],
                          ds['Q'].isel(time=1),
                          cmap=cmo.curl, shading='nearest',
                          vmin=-clim, vmax=clim, rasterized=True)

    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = ax2.pcolormesh(ds['XG'] * 1e-3,
                          -ds['Zl'],
                          ds['Q'].isel(time=2),
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

    ax0.set_title('1 week')
    ax1.set_title('2 weeks')
    ax2.set_title('3 weeks')
    
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
                      format=fmt)
    
    cb.set_label("$Q$ (s$^{-3}$)", usetex=True)

    fig.tight_layout()
    
    out_figure = figure_path / f'PV3DSliceSnapshot{run_name}.pdf'
    logging.info(f"Saving to {out_figure}")
    fig.savefig(out_figure, dpi=dpi)


if pv_slice:
    for run in run_names: plot_pv_slice(run)


if vorticity_comparison:
    rclim, qclim = 2.2e-5, 2e-8

    file_path = processed_path / f'VorticityComparisonStandardNoSlip.zarr'

    ds = xr.open_dataset(file_path,
                         engine='zarr')

    height = 8
    fig = plt.figure(figsize=[6, 7.5])
    gs = gridspec.GridSpec(2, 2,
                           height_ratios=[height, 2 / 14])
    
    ax0 = fig.add_subplot(gs[0, 0])
    
    cax0 = ax0.pcolormesh(ds['XG'] * 1e-3,
                          ds['YG'] * 1e-3,
                          ds['zeta_z'].squeeze(),
                          cmap=cmo.balance, shading='nearest',
                          vmin=-rclim, vmax=rclim, rasterized=True)
    
    ax1 = fig.add_subplot(gs[0, 1])
    cax1 = ax1.pcolormesh(ds['XG'] * 1e-3,
                          ds['YG'] * 1e-3,
                          ds['Q'].squeeze(),
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
                       format=fmt)
    
    qcb.set_label("$Q$ (s$^{-3}$)", usetex=True)

    rcbax = fig.add_subplot(gs[1, 0])
    rcb = fig.colorbar(cax0, cax=rcbax, orientation='horizontal',
                       format=fmt)
    rcb.set_label("$\\xi$ (s$^{-3}$)", usetex=True)

    fig.tight_layout()

    out_figure = figure_path / 'VorticityComparison.pdf'
    logging.info(f"Saving to {out_figure}")
    fig.savefig(out_figure, dpi=dpi)

if power_spectra:
    n = 8
    color = cmo.tempo(np.linspace(0.1, 0.9, n))
    plt.rcParams['axes.prop_cycle'] = cycler(color=color)
    wvels = ['3DStandard', '3DViscous', '2DStandard', '2DViscous']
    i = 0
    
    
    fig, axs = plt.subplots(2, 2, figsize=(6, 4.5), sharex=True, sharey=True)

    axs = axs.flatten()
    for wvel in wvels:
        run_path = processed_path / f'WVEL50m{wvel}NoSlip.zarr'

        print(run_path)
        assert run_path.exists()
        ds = xr.open_zarr(run_path)
        if wvel.startswith('3D'):
            ds = ds.drop(['Depth', 'dxF', 'dyF', 'maskInC', 'rA'])
            ds = ds.sel(XC=slice(0, 200e3), YC=slice(250e3, 750e3))
            wvel_spec = xrft.power_spectrum(ds['WVEL'],
                                            dim=['XC'],
                                            detrend='constant') * 0.5
            wvel_spec = wvel_spec.mean(dim='YC').squeeze()
        
        elif wvel.startswith('2D'):
            ds = ds.drop(['Depth', 'dxF', 'dyF', 'rA'])
            ds = ds.sel(XC=slice(0, 200e3))
            wvel_spec = xrft.power_spectrum(ds['WVEL'],
                                            dim=['XC'],
                                            detrend='constant') * 0.5
            wvel_spec = wvel_spec.squeeze()
    
        lambda_x = 1 / wvel_spec['freq_XC'] * 1e-3
        for nt in range(wvel_spec.sizes['time']):
            axs[i].loglog(lambda_x, wvel_spec.isel(time=nt))
            
        axs[i].set_title(f'{wvel[2:]} {wvel[:2]}')

        i += 1
        axs[0].set_xlim(4, 2e2)
        times = np.arange(0, 50, 7)
        fig.legend(times, title='time (days)')
    
    axs[0].set_title('(a)', loc='left')
    axs[1].set_title('(b)', loc='left')
    axs[2].set_title('(c)', loc='left')
    axs[3].set_title('(d)', loc='left')
    
    fig.supxlabel("Lengthscale (km)", fontsize=10)
    fig.supylabel("Power density (m$^2$\,s$^{-2}$)", fontsize=10)
    
    
    fig.tight_layout()
    
    out_figure = figure_path / "WVELSpectra.pdf"
    logging.info(f"Saving to {out_figure}")
    fig.savefig(out_figure, dpi=dpi)
    
    
if vorticity_correlations:
    logging.info("Plotting PV RV Correlations")
    
    run_names = ["StandardNoSlip",
                 "ViscousNoSlip",
                 "StandardFreeSlip",
                 "ViscousFreeSlip"]
    
    line_colours = ["k", "k", "grey", "grey"]
    line_styles = ["-", ":", "-", ":"]
    labels = ["No-slip", "No-slip viscous", "Free-slip", "Free-slip viscous"]
    
    fig, ax = plt.subplots(figsize=(5, 2.75))
    for run, col, sty, label in zip(run_names, line_colours, line_styles, labels):
        corr_path = processed_path / f"Correlation{run}.zarr"
        ds = xr.open_zarr(corr_path)
        ax.plot(ds['YG'] * 1e-3, ds["r"], color=col, ls=sty, label=label)
    
    ax.set_xlim(-512 + 40, 2176 - 350)
    ax.set_xlabel("Latitude (km)")
    ax.set_ylabel("$r$", usetex=True)
    ax.axvline(0, c="k", ls="-.")
    fig.suptitle("Correlations between relative and potential vorticity")
    ax.legend(loc="lower left")
    fig.tight_layout()
    
    figure_out = figure_path / "VorticityCorrelation.pdf"
    logging.info(f"Saving to {figure_out}")
    fig.savefig(figure_out)
    
    
if v_init:
    logging.info("Plotting v_init")
    
    ds = xr.open_zarr(interim_path / "3d/StandardNoSlip.zarr")
    
    fig, ax = plt.subplots(figsize=(5, 2.75))
    cax = ax.contourf(ds["XC"] * 1e-3,
                      -ds["Z"],
                      ds["VVEL"].isel(time=0, YG=0),
                      cmap=cmo.tempo,
                      vmin=0,
                      vmax=1, 
                      levels=np.linspace(0, 1, endpoint=True, num=11))
    
    ax.set_xlim(0, 400)
    ax.set_facecolor("grey")
    ax.invert_yaxis()
    
    ax.set_xlabel("Longitude (km)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Surface intensified Bickley Jet")
    
    cb = fig.colorbar(cax)
    cb.set_label("$V$ (m$\\,$s$^{-1}$)", usetex=True)
    
    fig.tight_layout()
    
    figure_out = figure_path / "BickleyJet.pdf"
    logging.info(f"Saving to {figure_out}")
    fig.savefig(figure_out)
        

if plot_formation:
    ds_standard = xr.open_zarr(processed_path / "FormationStandardNoSlip.zarr")
    ds_viscous = xr.open_zarr(processed_path / "FormationViscousNoSlip.zarr")

    fig, axs = plt.subplots(5, 3, figsize=(6, 6.5), sharex=True, sharey=True)

    for lat_class in ds_standard["lat_class"]:
        for rho_class in ds_standard["rho_class"]:
            ax = axs[-lat_class - 1,rho_class]
            ax.plot(ds_standard["time"] * 1e-9 / 24 / 60 / 60,
                    ds_standard["integrated_formation"].sel(rho_class=rho_class, 
                                                lat_class=lat_class) * 1e-10,
                    c="k",
                    ls="-",
                    label="standard")
            
            ax.plot(ds_viscous["time"] * 1e-9 / 24 / 60 / 60,
                    ds_viscous["integrated_formation"].sel(rho_class=rho_class, 
                                                lat_class=lat_class) * 1e-10,
                    c="k",
                    ls=":",
                    label="viscous")
            
    axs[-1, 1].set_xlabel("Time (days)")

    axs[2, 0].set_ylabel("Cumulative water mass formation ($\\times 10^{10}$ m$^3\,$km$^{-1}$)", usetex=True)

    axs[0, 0].set_xlim(0, ds_standard["time"].isel(time=-1).astype("float32").values * 1e-9 / 24 / 60 / 60)

    axs[0, 0].set_title("(a)", loc="left")
    axs[0, 1].set_title("(b)", loc="left")
    axs[0, 2].set_title("(c)", loc="left")
    axs[1, 0].set_title("(d)", loc="left")
    axs[1, 1].set_title("(e)", loc="left")
    axs[1, 2].set_title("(f)", loc="left")
    axs[2, 0].set_title("(g)", loc="left")
    axs[2, 1].set_title("(h)", loc="left")
    axs[2, 2].set_title("(i)", loc="left")
    axs[3, 0].set_title("(j)", loc="left")
    axs[3, 1].set_title("(k)", loc="left")
    axs[3, 2].set_title("(l)", loc="left")
    axs[4, 0].set_title("(m)", loc="left")
    axs[4, 1].set_title("(n)", loc="left")
    axs[4, 2].set_title("(o)", loc="left")

    ax2 = []
    for i in range(5):
        ax2 += [axs[i, 2].twinx()]
        ax2[i].set_ylim(-5.1, 5.1)
    axs[0, 0].set_ylim(-5.1, 5.1)

    ax2[0].set_ylabel("750 km to 1500 km", fontsize=8)
    ax2[1].set_ylabel("250 km to 750 km", fontsize=8)
    ax2[2].set_ylabel("0 km to 250 km", fontsize=8)
    ax2[3].set_ylabel("-250 km to 0 km", fontsize=8)
    ax2[4].set_ylabel("-500 km to -250 km", fontsize=8)

    axs[0, 0].set_title("$\\gamma^n \leq 23.45$", fontsize=9, usetex=True)
    axs[0, 1].set_title("$23.45 < \\gamma^n \leq 26.50$", fontsize=9,
                        usetex=True)
    axs[0, 2].set_title("$\\gamma^n > 26.50$", fontsize=9, usetex=True)

    axs[-1, 0].legend(loc="lower left")

    fig.suptitle("Integrated water mass formation")

    fig.tight_layout()

    fig.subplots_adjust(wspace=8e-2, hspace=9/32)

    fig.savefig(figure_path / "CumulativeWaterMassFormation.pdf")



    ds_standard["filtered_formation"] = ds_standard["formation"].rolling(time=24, center=True).mean().dropna("time")

    ds_viscous["filtered_formation"] = ds_viscous["formation"].rolling(time=24, center=True).mean().dropna("time")

    fig, axs = plt.subplots(5, 3, figsize=(6, 6.5), sharex=True, sharey=True)

    for lat_class in ds_standard["lat_class"]:
        for rho_class in ds_standard["rho_class"]:
            ax = axs[-lat_class - 1,rho_class]
            ax.plot(ds_standard["time"] * 1e-9 / 24 / 60 / 60,
                    ds_standard["filtered_formation"].sel(rho_class=rho_class, 
                                                lat_class=lat_class) * 1e-6 * 1e2,
                    c="k",
                    ls="-",
                    label="standard")
            
            ax.plot(ds_viscous["time"] * 1e-9 / 24 / 60 / 60,
                    ds_viscous["filtered_formation"].sel(rho_class=rho_class, 
                                                lat_class=lat_class) * 1e-6 * 1e2,
                    c="k",
                    ls=":",
                    label="viscous")
            
    axs[-1, 1].set_xlabel("Time (days)")

    axs[2, 0].set_ylabel("Water mass formation ($\\times 10^{-2}$ Sv$\\,$km$^{-1}$)", usetex=True)

    axs[0, 0].set_xlim(0, ds_standard["time"].isel(time=-1).astype("float32").values * 1e-9 / 24 / 60 / 60 - 3)

    axs[0, 0].set_title("(a)", loc="left")
    axs[0, 1].set_title("(b)", loc="left")
    axs[0, 2].set_title("(c)", loc="left")
    axs[1, 0].set_title("(d)", loc="left")
    axs[1, 1].set_title("(e)", loc="left")
    axs[1, 2].set_title("(f)", loc="left")
    axs[2, 0].set_title("(g)", loc="left")
    axs[2, 1].set_title("(h)", loc="left")
    axs[2, 2].set_title("(i)", loc="left")
    axs[3, 0].set_title("(j)", loc="left")
    axs[3, 1].set_title("(k)", loc="left")
    axs[3, 2].set_title("(l)", loc="left")
    axs[4, 0].set_title("(m)", loc="left")
    axs[4, 1].set_title("(n)", loc="left")
    axs[4, 2].set_title("(o)", loc="left")

    ax2 = []
    for i in range(5):
        ax2 += [axs[i, 2].twinx()]
        ax2[i].set_ylim(-2.8, 2.8)
    axs[0, 0].set_ylim(-2.8, 2.8)

    ax2[0].set_ylabel("750 km to 1500 km", fontsize=8)
    ax2[1].set_ylabel("250 km to 750 km", fontsize=8)
    ax2[2].set_ylabel("0 km to 250 km", fontsize=8)
    ax2[3].set_ylabel("-250 km to 0 km", fontsize=8)
    ax2[4].set_ylabel("-500 km to -250 km", fontsize=8)

    axs[0, 0].set_title("$\\gamma^n \leq 23.45$", fontsize=9, usetex=True)
    axs[0, 1].set_title("$23.45 < \\gamma^n \leq 26.50$", fontsize=9,
                        usetex=True)
    axs[0, 2].set_title("$\\gamma^n > 26.50$", fontsize=9, usetex=True)

    axs[-1, 0].legend(loc="lower left")

    fig.suptitle("Filtered water mass formation")

    fig.tight_layout()

    fig.subplots_adjust(wspace=8e-2, hspace=9/32)
    fig.savefig(figure_path / "WaterMassFormation.pdf")