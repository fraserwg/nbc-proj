import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                    )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
import xarray as xr
import zarr
import f90nml
from dask_jobqueue import SLURMCluster
from distributed import Client

logging.info('Importing custom python libraries')
import pvcalc

logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/nbc-proj')
data_path = base_path / 'data'
raw_path = data_path / 'raw'
processed_path = data_path / 'processed'
interim_path = data_path / 'interim'
log_path = base_path / 'src/3d-model-post-processing/.tmp'
dask_worker_path = log_path / 'dask-worker-space'
env_path = base_path / 'nbc-proj/bin/activate'

run_names = [#'StandardNoSlip',
             #'StandardFreeSlip',
             #'ViscousNoSlip',
             'ViscousFreeSlip'
             ]

def open_dataset(run_name):
    raw_run_path = raw_path / 'mitgcm-models-3d' / run_name
    assert raw_run_path.exists()
    interim_run_path = interim_path / f'3d/{run_name}.zarr'
    assert interim_run_path.exists()
    
    logging.info('Reading in model parameters from the namelist')
    data_nml = f90nml.read(raw_run_path / 'data')
    f0 = data_nml['parm01']['f0']
    try:
        beta = data_nml['parm01']['beta']
    except KeyError:
        beta = 0
    
    no_slip_bottom = data_nml['parm01']['no_slip_bottom']
    no_slip_sides = data_nml['parm01']['no_slip_sides']


    logging.info('Reading in the model dataset')
    ds = xr.open_zarr(interim_run_path)

    logging.info('Calculating the potential vorticity')
    grid = pvcalc.create_xgcm_grid(ds)
    ds['drL'] = pvcalc.create_drL_from_dataset(ds)
    ds['rho'] = pvcalc.calculate_density(ds['RHOAnoma'], ds['rhoRef'])
    ds['b'] = pvcalc.calculate_buoyancy(ds['rho'])

    grad_b = pvcalc.calculate_grad_buoyancy(ds['b'], ds, grid)

    ds['db_dx'], ds['db_dy'], ds['db_dz'] = grad_b
    
    zeta_all = pvcalc.calculate_curl_velocity(ds['UVEL'],
                                              ds['VVEL'],
                                              ds['WVEL'],
                                              ds,
                                              grid,no_slip_bottom,
                                              no_slip_sides,
                                              diff_y=False)

    ds['zeta_x'], ds['zeta_y'], ds['zeta_z'] = zeta_all
    ds['Q'] = pvcalc.calculate_C_potential_vorticity(ds['zeta_x'],
                                                    ds['zeta_y'],
                                                    ds['zeta_z'],
                                                    ds['b'],
                                                    ds,
                                                    grid,
                                                    beta,
                                                    f0,
                                                    diff_y=False)

    return ds

def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc

for run in run_names:
    ds = open_dataset(run)
    
    # PV map
    times = [np.timedelta64(26, 'D'),
             np.timedelta64(34, 'D'),
             np.timedelta64(42, 'D')]
    
    da_Q = ds['Q'].sel(time=times, Zl=-50)
    
    flat_out_path = processed_path / f'PV3D{run}.zarr'
    logging.info('Creating compression encoding')
    
    ds_Q = da_Q.to_dataset()
    ds_Q = ds_Q.chunk(YG=ds_Q.dims['YG'],
                      XG=ds_Q.dims['XG'])
    enc = create_encoding_for_ds(ds_Q, 9)
    
    logging.info(f"Saving {flat_out_path}")
    # ds_Q.to_zarr(flat_out_path, encoding=enc, mode='w-')
    
    
    # PV slice
    slimes = [np.timedelta64(7, 'D'),
              np.timedelta64(14, 'D'),
              np.timedelta64(21, 'D')]
    
    da_Q_slice = ds['Q'].sel(time=slimes, YG=750e3)
    ds_Q_slice = da_Q_slice.to_dataset()
    ds_Q_slice = ds_Q_slice.chunk(Zl=ds_Q_slice.dims['Zl'],
                                  XG=ds_Q_slice.dims['XG'])
    enc = create_encoding_for_ds(ds_Q_slice, 9)
    
    slice_out_path = processed_path / f'PV3DSlice{run}.zarr'
    logging.info(f"Saving {slice_out_path}")
    # ds_Q_slice.to_zarr(slice_out_path, encoding=enc, mode='w-')
    
    
    # WVEL on 50 m
    time_slice = np.arange(np.timedelta64(0, 'D'),
                           np.timedelta64(50, 'D'),
                           np.timedelta64(7, 'D'))
    
    da_WVEL = ds['WVEL'].sel(Zl=-50,
                             time=time_slice)
    
    ds_WVEL = da_WVEL.to_dataset()
    ds_WVEL = ds_WVEL.chunk(YC=ds_WVEL.dims['YC'],
                            XC=ds_WVEL.dims['XC'])
    
    enc = create_encoding_for_ds(ds_WVEL, 9)
    wvel_out_path = processed_path / f'WVEL50m3D{run}.zarr'
    logging.info(f"Saving {wvel_out_path}")
    # ds_WVEL.to_zarr(wvel_out_path, encoding=enc, mode='w-')
    

    def interp_zeta_z(da_zeta_z, target=-50):
        zeta_z_p = da_zeta_z.sel(Z=target, method='ffill')
        zeta_z_m = da_zeta_z.sel(Z=target, method='bfill')
        
        zeta_z_grad = (zeta_z_p - zeta_z_m) / (zeta_z_p['Z'] - zeta_z_m['Z'])
        
        zeta_z_target = zeta_z_grad * (target - zeta_z_m['Z']) + zeta_z_m
        return zeta_z_target
    
    # The 35 day vorticity comparison plot.
    if run == 'StandardNoSlip':
        

        
        ds_35d_comparison = ds.sel(time=np.timedelta64(35, 'D'))
        da_Q = ds_35d_comparison['Q'].sel(Zl=-50)
        da_zeta_z = interp_zeta_z(ds_35d_comparison['zeta_z'], target=-50)
        ds_35d_comparison = xr.Dataset({'Q': da_Q,
                                        'zeta_z': da_zeta_z})
        
        ds_35d_comparison = ds_35d_comparison.chunk(YG=ds_35d_comparison.dims['YG'],
                                                    XG=ds_35d_comparison.dims['XG'])
        
        enc = create_encoding_for_ds(ds_35d_comparison, 9)
        comparison_out_path = processed_path / f'VorticityComparison{run}.zarr'
        logging.info(f"Saving {comparison_out_path}")
        #ds_35d_comparison.to_zarr(comparison_out_path, encoding=enc, mode='w-')
        
        
    # The correlations between PV and RV
    times = slice(np.timedelta64(21, 'D'), np.timedelta64(49, 'D'))
    ds_corr = ds.sel(time=times, Zl=-50)
    
    da_zeta_z50 = interp_zeta_z(ds_corr['zeta_z'], target=-50)
    
    ds_corr = xr.Dataset({'Q': ds_corr['Q'],
                          'zeta_z_interp': da_zeta_z50})
    
    da_r = xr.corr(ds_corr['Q'], ds_corr['zeta_z_interp'], dim=['XG', 'time'])
    ds_r = da_r.to_dataset(name='r')
    ds_r = ds_r.chunk(YG=ds_r.dims['YG'])
    enc = create_encoding_for_ds(ds_r, 9)
    
    logging.info('Initialising the dask cluster')
    # Set up the dask cluster
    scluster = SLURMCluster(queue='standard',
                            project="n01-SiAMOC",
                            job_cpu=128,
                            log_directory=log_path,
                            local_directory=dask_worker_path,
                            cores=64,
                            processes=16,  # Can change this
                            memory="256 GiB",
                            header_skip= ['#SBATCH --mem='],  
                            walltime="00:20:00",
                            death_timeout=60,
                            interface='hsn0',
                            job_extra=['--qos=short', '--reservation=shortqos'],
                            env_extra=['module load cray-python',
                                    'source {}'.format(str(env_path.absolute()))]
                        )

    client = Client(scluster)
    scluster.scale(jobs=4)
    print(scluster)
    
    r_out_path = processed_path / f'Correlation{run}.zarr'
    logging.info(f"Saving {r_out_path}")
    ds_r.to_zarr(r_out_path, encoding=enc, mode='w-')
    
    scluster.close()
    
formation1 = False
if formation1:
    run_names = ["StandardNoSlip",
                "ViscousNoSlip"]
    
    density_classes = [0, 1023.45, 1026.50, np.inf]
    upper_classes = density_classes[:-1]
    lower_classes = density_classes[1:]

    latitude_bands = [-500e3, -250e3, 0, 250e3, 750e3, 1500e3]
    southern_bounds = latitude_bands[:-1]
    northern_bounds = latitude_bands[1:]



    short_directive = ["--qos=short",
                        "--reservation=shortqos"]
    

    scluster = SLURMCluster(queue="highmem",
                            project="n01-SiAMOC",
                            job_cpu=128,
                            log_directory=log_path,
                            local_directory=dask_worker_path,
                            cores=12,
                            processes=12,  # Can change this
                            memory="512 GiB",
                            header_skip= ['#SBATCH --mem='],  
                            walltime="00:20:00",
                            death_timeout=60,
                            interface='hsn0',
                            job_extra_directives=['--qos=highmem',
                                                    #'--reservation=highmem'
                                                    ],
                            job_script_prologue=['module load cray-python',
                                                    f'source {env_path.absolute()}'
                                        ]
                            )
    dask.config.set({"distributed.workers.memory.spill": 0.85})
    dask.config.set({"distributed.workers.memory.target": 0.75})
    dask.config.set({"distributed.workers.memory.terminate": 0.98})

    
    for run in run_names:
        ds = xr.open_zarr(interim_path / f"3d/{run}.zarr")
        ds["rho"] = ds["RHOAnoma"] + ds["rhoRef"]
        dVol = ds["dxF"] * ds["dyF"] * ds["drF"]
        
        ds_class = xr.Dataset(coords={"rho_class": [0, 1, 2],
                                    "lat_class": [0, 1, 2, 3, 4]})

        ds_class = ds_class.assign_coords({"lower_rho": ("rho_class",
                                                        lower_classes),
                                        "upper_rho": ("rho_class",
                                                      upper_classes),
                                        "southern_bound": ("lat_class",
                                                        southern_bounds),
                                        "northern_bound": ("lat_class",
                                                        northern_bounds)})

        ds_mask = xr.where(ds["rho"] > ds_class["upper_rho"], 1, 0) \
                * xr.where(ds["rho"] <= ds_class["lower_rho"], 1, 0) \
                * xr.where(ds["YC"] >= ds_class["southern_bound"], 1, 0) \
                * xr.where(ds["YC"] < ds_class["northern_bound"], 1, 0) \
                * dVol
                
        da_vol = ds_mask.sum(["Z", "YC", "XC"])
        
        


        client = Client(scluster)
        scluster.scale(jobs=6)
        
        wmt_out = processed_path / f"Volume{run}.zarr"
        logging.info(f"Saving to {wmt_out}")
        ds_vol = da_vol.to_dataset(name="Volume")
        ds_vol.to_zarr(wmt_out, mode="w")


formation2 = False
if formation2:
    run_names = ["StandardNoSlip",
                "ViscousNoSlip"]

    density_classes = [0, 1023.45, 1026.50, np.inf]
    upper_classes = density_classes[:-1]
    lower_classes = density_classes[1:]

    latitude_bands = [-500e3, -250e3, 0, 250e3, 750e3, 1500e3]
    southern_bounds = latitude_bands[:-1]
    northern_bounds = latitude_bands[1:]

    ds_class = xr.Dataset(coords={"rho_class": [0, 1, 2],
                                "lat_class": [0, 1, 2, 3, 4]})

    ds_class = ds_class.assign_coords({"lower_rho": ("rho_class",
                                                    lower_classes),
                                    "upper_rho": ("rho_class",
                                                    upper_classes),
                                    "southern_bound": ("lat_class",
                                                    southern_bounds),
                                    "northern_bound": ("lat_class",
                                                    northern_bounds)})

    scluster = SLURMCluster(queue="standard",
                            project="n01-SiAMOC",
                            job_cpu=128,
                            log_directory=log_path,
                            local_directory=dask_worker_path,
                            cores=64,
                            processes=16,  # Can change this
                            memory="256 GiB",
                            header_skip= ['#SBATCH --mem='],  
                            walltime="00:20:00",
                            death_timeout=60,
                            interface='hsn0',
                            job_extra_directives=['--qos=short',
                                                '--reservation=shortqos'
                                                    ],
                            job_script_prologue=['module load cray-python',
                                                    f'source {env_path.absolute()}'
                                        ]
                            )
    dask.config.set({"distributed.workers.memory.spill": 0.85})
    dask.config.set({"distributed.workers.memory.target": 0.75})
    dask.config.set({"distributed.workers.memory.terminate": 0.98})

    client = Client(scluster)
    scluster.scale(jobs=2)


    for run in run_names:
        ds = xr.open_zarr(interim_path / f"3d/{run}.zarr")

        ds["rho"] = ds["RHOAnoma"] + ds["rhoRef"]
        dVol = ds["dxF"] * ds["dyF"] * ds["drF"]

        da_vol = xr.open_zarr(processed_path / f"Volume{run}.zarr")["Volume"]
        ds_psi = xr.Dataset()
        ds_psi["VVEL_n"] = ds["VVEL"].sel(YG=da_vol["northern_bound"])
        ds_psi["VVEL_s"] = ds["VVEL"].sel(YG=da_vol["southern_bound"])
        ds_psi["mask_n"] = xr.where(ds["rho"].sel(YC=da_vol["northern_bound"], method="bfill") > ds_class["upper_rho"], 1, 0) \
                * xr.where(ds["rho"].sel(YC=da_vol["northern_bound"], method="bfill") <= ds_class["lower_rho"], 1, 0)

        ds_psi["mask_s"] = xr.where(ds["rho"].sel(YC=da_vol["southern_bound"], method="ffill") > ds_class["upper_rho"], 1, 0) \
                * xr.where(ds["rho"].sel(YC=da_vol["southern_bound"], method="ffill") <= ds_class["lower_rho"], 1, 0)
        dA_n = ds["drF"] * ds["dxG"].sel(YG=da_vol["northern_bound"])
        dA_s = ds["drF"] * ds["dxG"].sel(YG=da_vol["southern_bound"])
        ds_psi["psi_n"] = (dA_n * ds_psi["mask_n"]).sum(["Z", "XC"])
        ds_psi["psi_s"] = (dA_s * ds_psi["mask_s"]).sum(["Z", "XC"])
        ds_psi["delta_psi"] = ds_psi["psi_n"] - ds_psi["psi_s"]
        dt = ds_psi["time"].diff(dim="time", label="upper")
        # Concatenate the differences with the areas at the first depth
        dt = xr.concat([ds_psi["time"].isel(time=0), dt], dim="time")
        dt = dt.astype("float64") * 1e-9

        ds_psi["int_delta_psi"] = (ds_psi["delta_psi"] * dt).cumsum("time")
        lat_band_width = ds_psi["northern_bound"] \
                        - ds_psi["southern_bound"]
                        
        da_int_form = (da_vol - da_vol.isel(time=0) + ds_psi["int_delta_psi"]) / (lat_band_width * 1e-3)
        da_int_form = da_int_form.assign_attrs({"units": "m3 km-1"})

        ds_formation = xr.Dataset()
        ds_formation["integrated_formation"] = da_int_form
        ds_formation["formation"] = ds_formation["integrated_formation"].chunk(time=-1).differentiate("time")
        file_name = processed_path / f"Formation{run}.zarr"

        ds_formation.to_zarr(file_name, mode="w-")