
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import xmitgcm
import f90nml
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import zarr


logging.info('Setting file path')
base_path = Path('/work/n01/n01/fwg/nbc-proj')
assert base_path.exists()

env_path = base_path / 'nbc-proj/bin/activate'
assert env_path.exists()

src_path = base_path / 'src/3d-model-post-processing'
assert src_path.exists()

log_path = src_path / '.tmp'
log_path.mkdir(exist_ok=True)

dask_worker_path = log_path / 'dask-worker-space'
dask_worker_path.mkdir(exist_ok=True)

run_name = 'ViscousNoSlip'
run_path = base_path / 'data/raw/mitgcm-models-3d' / run_name
interim_path = base_path / "data/interim/3d"
out_path = interim_path / f'{run_name}.zarr'
assert run_path.exists()

logging.info('Initialising the dask cluster')

# Set up the dask cluster
scluster = SLURMCluster(queue='standard',
                        project="n01-SiAMOC",
                        job_cpu=128,
                        log_directory=log_path,
                        local_directory=dask_worker_path,
                        cores=128,
                        processes=32,  # Can change this
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
scluster.scale(jobs=6)
print(client)

logging.info('Reading in model parameters from the namelist')
with open(run_path / 'data') as data:
    data_nml = f90nml.read(data)

delta_t = data_nml['parm03']['deltat']
f0 = data_nml['parm01']['f0']
beta = data_nml['parm01']['beta']
no_slip_bottom = data_nml['parm01']['no_slip_bottom']
no_slip_sides = data_nml['parm01']['no_slip_sides']


logging.info('Reading in the model dataset')
ds = xmitgcm.open_mdsdataset(str(run_path),
                            prefix=['ZLevelVars', 'IntLevelVars'],
                            delta_t=delta_t,
                            geometry='cartesian')

def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc

logging.info('Creating compression encoding')
enc = create_encoding_for_ds(ds, 9)
logging.info('Saving to compressed zarr dataset')
logging.info(out_path)
ds.to_zarr(out_path, mode='w-', encoding=enc)

client.close()
scluster.close()