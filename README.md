# trajectory_calculator

Author: Lianet Hernández pardo

This script calculates the trajectories of parcels in a 4D wind field. It takes initial positions of parcels and integrates their movement over time. The trajectories are computed in either a 2D or 3D mode based on user-defined parameters.

## Usage:

To run this project, follow these steps:

1. Clone the Repository:

   ```bash
   git clone https://github.com/lhpardo90/trajectory_calculator.git
   cd trajectory_calculator
   ```

2. Create a Conda Environment:

   Use the provided YAML file to create a Conda environment with the necessary dependencies:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the Conda Environment:

   ```bash
   conda activate trajectories
   ```

4. Run the Script:

   ```bash
   python trajectories.py [options] <start_datetime> <wind_filename> <in_filename> <out_filename>
   ```
   
### Options:
    -h, --help            Show help message, including the full list of options, and exit.

### Parameters:
    - start_datetime: Start date and time for trajectory calculation (format: <YYYY>-<MM>-<DD>T<HH>).
    - wind_filename: Path to the netcdf file containing wind data.
    - in_filename: Path to the file storing initial parcel positions in CSV format. Each row should contain comma-separated values in the following order: pressure (hPa), latitude (degrees), longitude (degrees).
    - out_filename: Path to the output trajectories file.

### Input data format:
    The wind field needs to be provided as a single netcdf file via the 'wind_filename' argument. This file must contain at least the variables 'u' and 'v' in m/s. For 3D trajectories, the variable 'w' in Pa/s must be provided in the same file as 'u' and 'v'. All time steps in the desired integration interval must be included in the same file. The grid must be regular in the horizontal dimension, using either pressure or hybrid sigma-pressure levels. If using hybrid levels, the wind must be provided at 'full' levels, and a separate netcdf file containing the surface pressure (variable 'sp', in Pa units) and a CSV file containing the hybrid coefficients 'a' and 'b' at interface (or 'half') levels in separate columns must be provided via the '--sp_filename' and '--coeff_filename' arguments (see Options). The netcdf files should have dimensions: 'time', 'level', 'latitude', 'longitude' (excluding 'level' for the surface pressure file).

### Examples:

Calculating trajectories from 3D model-level wind data:

```bash
python trajectories.py -sf /path/to/your/surface/pressure/file/12UTC_sf.nc -cf /path/to/your/hybrid/coefficients/file/L137_hybrid_levels_NOHEADER.csv 2023-01-04T15 /path/to/your/wind/file/12UTC_ml.nc /path/to/your/initial/locations/init_locations_0000_2023-01-04_15Z.txt /path/to/your/output/dir/my_output
```

Calculating trajectories from 3D pressure-level wind data:

```bash
python trajectories.py --levels_type pl 2023-01-04T15 /path/to/your/wind/file/12UTC_pl.nc /path/to/your/initial/locations/init_locations_0000_2023-01-04_15Z.txt /path/to/your/output/dir/my_output
```
## Contact:

If you have any questions or feedback, feel free to reach out to me:
- Email: hernandezpardo at iau.uni-frankfurt.de

## Acknowledgments: 

I am grateful to Prof. Dr. Anna Possner for her insightful discussions and valuable suggestions. The core of this code drew inspiration from Tom Gowa's code, which can be found at https://github.com/tomgowan/trajectories. This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – TRR 301 – Project-ID 4283127
