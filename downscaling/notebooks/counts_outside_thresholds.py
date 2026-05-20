#!/usr/bin/env python
import xarray as xr
import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description='Count values outside thresholds in Zarr stores.')
parser.add_argument('--config', type=str, default='counts_outside_thresholds.json', 
                    help='Path to the JSON configuration file')
parser.add_argument('--output', type=str, default='threshold_counts.csv',
                    help='Path to the output CSV file')

args = parser.parse_args()

# Load the configuration from JSON
with open(args.config, 'r') as f:
    config = json.load(f)

# Dictionary to store results: {(model, scenario): {var_name: (count_below, count_above)}}
results = defaultdict(dict)

# Process each variable
for var_name, var_config in config['variables'].items():
    base_path = Path(var_config['path'])
    min_threshold, max_threshold = var_config['thresholds']
    
    print(f'\n{"="*80}')
    print(f'Processing variable: {var_name}')
    print(f'Base path: {base_path}')
    print(f'Thresholds: [{min_threshold}, {max_threshold}]')
    print(f'{"="*80}')
    
    # Find all Zarr stores for this variable
    zarr_stores = sorted(base_path.glob(f'{var_name}_*.zarr'))
    
    if not zarr_stores:
        print(f'  Warning: No Zarr stores found for {var_name}')
        continue
    
    print(f'Found {len(zarr_stores)} Zarr store(s)')
    
    # Process each Zarr store
    for zarr_path in zarr_stores:
        print(f'\n  Processing: {zarr_path.name}')
        
        # Parse model and scenario from filename
        # Expected format: {var_name}_{model}_{scenario}_adjusted.zarr
        parts = zarr_path.stem.replace('_adjusted', '').split('_')
        if len(parts) >= 3:
            model = parts[1]
            scenario = parts[2]
        else:
            print(f'    Warning: Could not parse model/scenario from {zarr_path.name}')
            continue
        
        try:
            # Load the Zarr store using xarray
            ds = xr.open_zarr(zarr_path)
            data = ds[var_name]
            
            # Calculate total pixels across entire time series
            total_pixels = data.size
            
            # Count values outside the thresholds
            count_outside_min = int((data < min_threshold).sum().compute())
            count_outside_max = int((data > max_threshold).sum().compute())
            
            # Calculate percentages
            percent_below = (count_outside_min / total_pixels) * 100
            percent_above = (count_outside_max / total_pixels) * 100
            
            print(f'    Model: {model}, Scenario: {scenario}')
            print(f'    Total pixels: {total_pixels:,}')
            print(f'    Count of values below {min_threshold}: {count_outside_min:,} ({percent_below:.2f}%)')
            print(f'    Count of values above {max_threshold}: {count_outside_max:,} ({percent_above:.2f}%)')
            
            # Store results with counts and percentages
            results[(model, scenario)][var_name] = (count_outside_min, percent_below, count_outside_max, percent_above)
            
        except Exception as e:
            print(f'    Error: {e}')

# Write results to CSV
print(f'\n{"="*80}')
print(f'Writing results to {args.output}')
print(f'{"="*80}')

# Get all variables in order from config
variables = list(config['variables'].keys())

# Create header row
header = ['Model', 'Scenario']
for var_name in variables:
    min_threshold, max_threshold = config['variables'][var_name]['thresholds']
    header.extend([f'{var_name} below {min_threshold}', f'{var_name} above {max_threshold}'])

# Write CSV
with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    
    # Sort by model and scenario for consistent output
    for (model, scenario), var_counts in sorted(results.items()):
        row = [model, scenario]
        for var_name in variables:
            if var_name in var_counts:
                count_below, percent_below, count_above, percent_above = var_counts[var_name]
                row.extend([
                    f'{count_below:,} ({percent_below:.2f}%)',
                    f'{count_above:,} ({percent_above:.2f}%)'
                ])
            else:
                row.extend(['N/A', 'N/A'])  # N/A if variable not found for this model/scenario
        writer.writerow(row)

print(f'Successfully wrote {len(results)} rows to {args.output}')

