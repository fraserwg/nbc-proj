import logging
from re import sub
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
import zarr

logging.info('Importing custom python libraries')
#import pvcalc

logging.info('Setting paths')
base_path = Path('../../').absolute().resolve()
figure_path = base_path / 'figures'
figure_path.mkdir(exist_ok=True)
data_path = base_path / 'data'
raw_path = Path('/Volumes/EXT2TB/Paper1Files.nosync')
processed_path = data_path / 'processed'
interim_path = data_path / 'interim'

run_names = ['StandardNoSlip',
             'StandardFreeSlip',
             'ExtraViscousNoSlip',
             'ExtraViscousFreeSlip']


def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc

days = 24 * 60 * 60

def subset_pv_50m(run_name):
    full_run_path = raw_path / run_name / ('PV50m' + run_name + '.nc')
    ds = xr.open_dataset(full_run_path)
    ds_subset = ds.sel(T=[26 * days, 34 * days, 42 * days],
                       method='nearest')
    
    out_name = processed_path / ('3DPV50m' + run_name + '.zarr')
    print(out_name)
    enc = create_encoding_for_ds(ds_subset, 5)
    ds_subset.to_zarr(out_name, mode='w', encoding=enc)

for run in run_names: subset_pv_50m(run)


def subset_pv_slice(run_name):
    full_run_path = raw_path / run_name / ('PV750km' + run_name + '.nc')
    ds = xr.open_dataset(full_run_path)
    ds_subset = ds.sel(T=[0, 14 * days, 28 * days],
                       method='nearest')
    
    out_name = processed_path / ('3DPV750km' + run_name + '.zarr')
    print(out_name)
    enc = create_encoding_for_ds(ds_subset, 5)
    ds_subset.to_zarr(out_name, mode='w', encoding=enc)

for run in run_names: subset_pv_slice(run)


def subset_rv_pv(run_name):
    full_run_path_rv = raw_path / run_name / ('RV50m' + run_name + '.nc')
    full_run_path_pv = raw_path / run_name / ('PV50m' + run_name + '.nc')
    ds = xr.open_mfdataset([full_run_path_pv, full_run_path_rv])
    ds_subset = ds.sel(T=36 * days,
                       method='nearest')
    
    out_name = processed_path / ('3DVorticityComparison50m' + run_name + '.zarr')
    print(out_name)
    enc = create_encoding_for_ds(ds_subset, 5)
    ds_subset.to_zarr(out_name, mode='w', encoding=enc)

subset_rv_pv(run_names[0])


def subset_wvel(run_name):
    full_run_path = raw_path / run_name / ('WVEL50m' + run_name + '.nc')
    ds = xr.open_dataset(full_run_path)
    t = np.arange(0, 50 * days, 7 * days)
    ds_subset = ds.sel(T=t, method='nearest')
    
    out_name = processed_path / ('3DWVEL50m' + run_name + '.zarr')
    print(out_name)
    enc = create_encoding_for_ds(ds_subset, 5)
    ds_subset.to_zarr(out_name, mode='w', encoding=enc)

for run in run_names: subset_wvel(run)

def subset_vorticity_correlations(run_name):
    full_run_path_rv = raw_path / run_name / ('RV50m' + run_name + '.nc')
    full_run_path_pv = raw_path / run_name / ('PV50m' + run_name + '.nc')
    ds = xr.open_mfdataset([full_run_path_pv, full_run_path_rv])
    days = 24 * 60 * 60
    ds_subset = ds.sel(T=slice(21 * days, 49 * days),
                       Xp1=slice(0, 400e3),
                       X=slice(0, 400e3))
    ds_subset['momVort3'] = ds_subset['momVort3'].squeeze().interp(Xp1=ds_subset['X'],
                                                                   Yp1=ds_subset['Y'])

    out_name = interim_path / ('3DVorticityCorrelations50m' + run_name + '.zarr')
    print(out_name)
    enc = create_encoding_for_ds(ds_subset, 5)
    ds_subset.to_zarr(out_name, mode='w', encoding=enc)

vort_corr = True
if vort_corr:
    for run in run_names: subset_vorticity_correlations(run)