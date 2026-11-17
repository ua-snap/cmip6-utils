output_dir_name = "bias_adjust"

qc_dir_name = "qc"
doy_summary_dir_name = "doy_summaries"
doy_summary_tmp_fn = "{var_id}_{model}_{scenario}_{kind}.nc"


era5_tmp_fn = "{var_id}_{year}*_era5_{resolution}km_3338.nc"

cmip6_regrid_tmp_fn = "{var_id}_day_{model}_{scenario}_regrid_{year}0101-{year}1231.nc"

cmip6_netcdf_to_zarr_sbatch_tmp_fn = "convert_cmip6_netcdf_to_zarr.slurm"
era5_netcdf_to_zarr_sbatch_tmp_fn = "convert_era5_netcdf_to_zarr.slurm"

cmip6_zarr_tmp_fn = "{var_id}_{model}_{scenario}.zarr"
era5_zarr_tmp_fn = "{var_id}_era5.zarr"

train_qm_sbatch_tmp_fn = "train_qm_{model}_{var_id}.slurm"
bias_adjust_sbatch_tmp_fn = "bias_adjust_{model}_{scenario}_{var_id}.slurm"

trained_qm_tmp_fn = "trained_qdm_{var_id}_{model}.zarr"

cmip6_adjusted_tmp_fn = "{var_id}_{model}_{scenario}_adjusted.zarr"

# ---------------------------------------------------------------------------
# QDM bias-adjustment parameters (shared by train_qm.py and bias_adjust.py).
# See README_patch_31d.md for the history of QDM_RANK_WINDOWED (issue #125).
# ---------------------------------------------------------------------------
QDM_NQUANTILES = 100
QDM_WINDOW = 31  # days; group="time.dayofyear" with a centred +/-15 day window
# True  -> rank `sim` against the same 31-day window used in training (the fix).
# False -> legacy xclim 0.47 behaviour: rank against the exact day-of-year only.
QDM_RANK_WINDOWED = True

# Spatial chunk edge (cells) used for the time=-1 adjustment chunks, per resolution.
# Windowed ranking needs ~0.45 KB per cell-timestep per task (vs ~60x less for the
# legacy exact-day ranking): a 15x15 SSP block peaks at ~2.8 GB, 30x30 at ~13 GB.
# 15 is the measured sweet spot (2026-11-16 tests, PATCH_31D_PLAN.md section 11);
# outputs are bit-identical across chunk sizes and worker layouts.
ADJUST_CHUNK_SIZE = {"12km": 15, "4km": 15}
# (y, x) grid shape -> resolution label, used to infer the resolution from a sim store.
GRID_SHAPE_TO_RESOLUTION = {(211, 282): "12km", (460, 443): "4km"}

# Output zarr layout: dims (time, y, x), float32, zstd-3 + bitshuffle.
OUTPUT_CHUNKS = (365, 100, 100)
OUTPUT_DTYPE = "float32"
OUTPUT_COMPRESSOR = {"cname": "zstd", "clevel": 3, "shuffle": "bitshuffle"}

# Dask LocalCluster sizing for bias_adjust.py. The per-worker memory limit is derived
# from the node/job memory (SLURM_MEM_PER_NODE or psutil) minus a client-side reserve
# that scales with the size of the in-memory result (the whole adjusted array is
# gathered to the client before the squeeze and the synchronous write).
DASK_ADJUST = {
    # per resolution; 12 km: 4x4 on a 120G t2small job (~9 min/SSP, peak 20 GB/worker);
    # 4 km: 6x4 on a 500G analysis job (~43 min/SSP, peak 50 of 68 GB/worker; 7x4 at
    # 400G had worker kills). More, smaller workers fail: each carries ~8 GB (12 km) /
    # ~28 GB (4 km) of queued input on top of the per-block memory.
    "n_workers": {"12km": 4, "4km": 6},
    "threads_per_worker": {"12km": 4, "4km": 4},
    # Worker scratch on node-local storage ($SLURM_TMPDIR or /tmp), never beegfs: dask's
    # spill/worker-space files on beegfs hit FileNotFoundError mid-run (2026-11-16 tests).
    "worker_dir_local": True,
    # Node-local /tmp is tmpfs (RAM) on Chinook, so spilling cannot relieve memory
    # pressure; keep everything in RAM and size the workers so the data fits.
    "spill_to_disk": False,
    "memory_fraction": 0.90,  # share of (node - reserve) handed to the workers
    "client_reserve_min_gb": 16.0,
    "client_reserve_factor": 2.0,  # x size of the float32 result
    "worker_memory_min_gb": 8.0,
    "worker_memory_max_gb": 80.0,
}

# Post-adjustment squeezing (clip) limits, NaN-preserving: var -> (lower, upper).
# None means no limit on that side. dtr uses quantile-based limits (below).
SQUEEZE_LIMITS = {
    "tasmax": (None, 333.15),  # K
    "pr": (0.0, 1650.0),  # mm d-1
    "hurs": (0.0, 100.0),  # %
    "hursmin": (0.0, 100.0),  # %
}
DTR_SQUEEZE_QUANTILES = (0.0000002, 0.9999998)
DTR_HIST_BINS = 10000

# SLURM defaults for run_bias_adjust.py, per resolution / partition.
# "mem" is keyed by (resolution, partition); a missing key / None omits --mem.
# 4 km does not fit on 128 GB t2small nodes (2026-11-16 tests) -> use analysis.
SLURM_BIAS_ADJUST = {
    "time_limit": {"12km": "02:00:00", "4km": "03:00:00"},
    # n_workers x threads; also stops Slurm packing several jobs onto one node's CPUs
    "cpus_per_task": {"12km": 16, "4km": 28},
    "mem": {
        ("12km", "t2small"): "120G",
        ("12km", "analysis"): "120G",
        ("4km", "analysis"): "500G",
    },
}
