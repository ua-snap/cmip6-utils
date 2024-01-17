import xarray as xr
import glob
import warnings
from multiprocessing import Pool
from config import PROJECT_DIR, cmip6_dir, time_dim_error_file

warnings.simplefilter("ignore")

#list all *.nc files in the CMIP6 directory
#files = glob.glob(cmip6_dir.joinpath('**/*.nc'), recursive = True)
files = list(cmip6_dir.glob('**/*.nc'))

#all files passing transfers/tests.slurm should be readable with xarray
#this test assumes they are readable and tests for a time dimension
#files without a time dimension return their filepath as a result
#files with a time dimension will return None
def test_nc_file(file):
    
    try:
        ds = xr.open_dataset(file)
        t = ds.time
    except:
        return file

if __name__ == "__main__":

    with Pool() as pool:
        results = pool.map(test_nc_file, files)
        
    # sort non-None results and write filepaths to csv
    for r in results:
        if r is not None:
            with open(time_dim_error_file, 'a') as f:
                f.write((str(r) + '\n'))
