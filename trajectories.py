#!/usr/bin/env python
# coding: utf-8
"""
Trajectory Calculation Script

This script calculates the trajectories of parcels in a 4D wind field. It takes initial positions of parcels and integrates their movement over time. The trajectories are computed in either a 2D or 3D mode based on user-defined parameters.

Usage:
    python trajectories.py [options] <start_datetime> <wind_filename> <in_filename> <out_filename>

Options:
    -h, --help            Show help message, including the full list of options, and exit.

Parameters:
    - start_datetime: Start date and time for trajectory calculation (format: <YYYY>-<MM>-<DD>T<HH>).
    - wind_filename: Path to the netcdf file containing wind data.
    - in_filename: Path to the file storing initial parcel positions in CSV format. Each row should contain comma-separated values in the following order: pressure (hPa), latitude (degrees), longitude (degrees).
    - out_filename: Path to the output trajectories file.

Input data format:
    The wind field needs to be provided as a single netcdf file via the 'wind_filename' argument. This file must contain at least the variables 'u' and 'v' in m/s. For 3D trajectories, the variable 'w' in Pa/s must be provided in the same file as 'u' and 'v'. All time steps in the desired integration interval must be included in the same file. The grid must be regular in the horizontal dimension, using either pressure or hybrid sigma-pressure levels. If using hybrid levels, the wind must be provided at 'full' levels, and a separate netcdf file containing the surface pressure (variable 'sp', in Pa units) and a CSV file containing the hybrid coefficients 'a' and 'b' at interface (or 'half') levels in separate columns must be provided via the '--sp_filename' and '--coeff_filename' arguments (see Options). The netcdf files should have dimensions: 'time', 'level', 'latitude', 'longitude' (excluding 'level' for the surface pressure file).

Author: Lianet Hernández Pardo (hernandezpardo at iau.uni-frankfurt.de)
Date: October 9, 2023
"""

import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import bisect
import datetime
import json
import math
import numpy as np
import os
import scipy.signal as sps
import sys
import time
import xarray as xr
import csv

from itertools import tee
from multiprocessing import Pool

class Arguments:

    def __init__ (self):
        args = self.get_args()
        for attr in vars(args):
            setattr(self, attr, getattr(args, attr))

        self.validate()

    def __str__(self):
        return f"""

        Request sumary:

        - initial time: {self.start_datetime.strftime('%Y-%m-%d %H')}Z
        - delta_t: {self.delta_t} s
        - direction: {"forward" if self.forw_back == 1 else "backward" if self.forw_back == -1 else None}
        - end time: {(self.start_datetime + datetime.timedelta(seconds = self.forw_back * self.delta_t * self.number_time_steps)).strftime('%Y-%m-%d %H')}Z

        - maximum number of iterations: {self.number_iterations}
        - dimensions: {'2D' if self.twodim else '3D'}

        - input:
            -- model data vertical coordinate: {'model levels' if self.levels_type=='ml' else 'pressure levels'}
            -- model data:
                -- wind: {self.wind_filename}
                -- surface pressure: {self.sp_filename if self.sp_filename is not None else '-'}
                -- vertical levels coefficients: {self.coeff_filename if self.coeff_filename is not None else '-'}
            -- initial parcel locations: {self.in_filename}
        - output: {self.out_filename}

        """

    def get_args(self):
        """
        Read user-defined parameters, provided via command line arguments, or set default values.
        """
        parser = argparse.ArgumentParser(description='Calculates trajectories given initial positions and a 4D wind field')
        # Parse datetime argument
        parser.add_argument('start_datetime', type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%dT%H'), help="trajectories' initial date and time <YYYY>-<MM>-<DD>T<HH>")
        # Parse wind filename argument
        parser.add_argument('wind_filename', type=str, help='path to file storing wind data for the requested trajectory interval')
        # Parse locations input filename argument
        parser.add_argument('in_filename', type=str, help='path to file storing initial locations for the trajectory calculations')
        # Parse output filename argument
        parser.add_argument('out_filename', type=str, help='path to the output trajectories file')
        # Parse sp filename argument
        parser.add_argument('-sf', '--sp_filename', type=str, default=None, help='path to file storing surface pressure data data for the requested trajectory interval')
        # Parse hybrid-levels coefficients input filename argument
        parser.add_argument('-cf', '--coeff_filename', type=str, default=None, help='path to file storing hybi and hyai coefficients of the vertical grid (only if level_type == "ml")')
        # Parse time step argument
        parser.add_argument('-dt', '--delta_t', type=int, default=3600, help='time step in second between input files')
        # Parse direction flag argument
        parser.add_argument('-fb', '--forw_back', type=int, default=1, choices=[-1, 1], help='integration direction: 1 forward, -1 backward')
        # Parse number of time steps argument
        parser.add_argument('-nt', '--number_time_steps', type=int, default=12, help='number of time steps for integration')
        # Parse maximum number of iterations argument
        parser.add_argument('-n', '--number_iterations', type=int, default=20, help='maximum number of iterations of the numerical scheme')
        # Parse three dimensions switch argument
        parser.add_argument('-2d', '--twodim', default=False, action='store_true')
        # Parse verbosity level argument
        parser.add_argument('-v', '--verbosity', type=int, default=1, choices=[0, 1, 2], help='switch for very verbose (2), moderately verbose (1) or non-verbose (0)')
        # Parse levels type argument
        parser.add_argument('-lt', '--levels_type', type=str, default='ml', choices=['ml', 'pl'], help='type of vertical coordinate of the model data: model levels (ml) or pressure levels (pl)')

        return parser.parse_args()

    def validate(self):
        """
        Check parameter consistency
        """
        if self.out_filename[-4:] != '.csv': self.out_filename+='.csv'

        if self.levels_type == 'ml':
            if self.coeff_filename is None:
                raise ValueError("File containing hybrid a and b coefficients at interface levels (hyai and hybi) required")
            if self.sp_filename is None:
                raise ValueError("File containing surface pressure required")

        print(str(self))
        self.confirm()

    def confirm(self):
        """
        Ask the user to confirm the execution parameters
        """
        while True:
            answer = input("Continue? (y or n)\n").strip().lower()
            if answer in {'n', 'no'}:
                print('Aborting...')
                time.sleep(1)
                sys.exit()
            elif answer in {'y', 'yes'}:
                return

class Parcel:

    total_instances = 0  # Class variable to keep track of the total number of instances
    parcels = []  # Class-level list to keep track of parcels

    def __init__(self, initial_time, initial_pres, initial_lat, initial_lon, id=None):

        Parcel.total_instances += 1

        self.id = Parcel.total_instances if id is None else id
        self.time = initial_time
        self.pres = initial_pres
        self.lat = initial_lat
        self.lon = initial_lon

        Parcel.parcels.append(self)

    def __str__(self):
        return f"Parcel {self.id} at {self.pres} hPa, latitude: {self.lat}, longitude: {self.lon}"

    @classmethod
    def save_parcel_data_to_json(cls, json_filename):
        """
        Convert Parcel instances to dictionaries and save to the JSON file
        """
        parcel_data = [parcel.to_dict() for parcel in cls.parcels]
        with open(json_filename, "w") as json_file:
            json.dump(parcel_data, json_file, indent=4, default=str)

    @classmethod
    def save_parcel_data_to_csv(cls, csv_filename):
        """
        Convert Parcel instances to CSV format and save CSV file
        """
        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write CSV file header
            writer.writerow(['identifier', 'time', 'pressure', 'latitude', 'longitude'])
            for parcel in cls.parcels:
                # Write each parcel's data as a row in the CSV file
                writer.writerow([parcel.id, parcel.time, parcel.pres, parcel.lat, parcel.lon])

    def to_dict(self):
        """
        Convert Parcel attributes to a dictionary
        """
        data = {
            "identifier": self.id,
            "time": self.time,
            "pressure": self.pres,
            "latitude": self.lat,
            "longitude": self.lon,
        }
        return data

class FieldsFourD:

    def __init__(self, wind_filename, sp_filename, coeff_filename, levels_type, twodim):

        self.ds = None
        self.ds_sf = None
        self.hyam = None
        self.hybm = None

        self.read_model_data(wind_filename, sp_filename)
        self.check_files(twodim, levels_type, wind_filename, sp_filename)
        self.get_levels_coef(coeff_filename)
        self.change_longitude_range()
        self.get_boundaries(levels_type)

    def __str__(self):
        return 'Time-evolving DataSet(s)/DataArray(s):\n'+("\n".join(list(vars(self))))

    def read_model_data(self, wind_filename, sp_filename):
        """
        Read model data
        """
        self.ds = xr.open_dataset(wind_filename)
        if sp_filename is not None:  self.ds_sf = xr.open_dataset(sp_filename)

    def get_levels_coef(self, coeff_filename):
        """
        Retrieve the hyam and hybm coefficients of the vertical grid from the hyai and hybi coefficients.
        """
        if coeff_filename is not None: 
            # Read hyai and hybi coefficients from CSV file
            levels_info = np.loadtxt(coeff_filename, delimiter=',', usecols=(1, 2))
    
            # Average hyai and hybi coefficients every two consecutive half-levels to obtain hyam and hybm
            hyai = levels_info[:, 0] * 0.01
            hybi = levels_info[:, 1]
            hyam = (hyai[1:] + hyai[:-1]) / 2
            hybm = (hybi[1:] + hybi[:-1]) / 2
    
            # Create xarray DataArrays for hyam and hybm
            self.hyam = xr.DataArray(data=hyam, dims=['level'], coords={'level': self.ds.level},
                                attrs={'description': 'Hybrid sigma-pressure "a" coefficient', 'units': 'hPa'})
            self.hybm = xr.DataArray(data=hybm, dims=['level'], coords={'level': self.ds.level},
                                attrs={'description': 'Hybrid sigma-pressure "b" coefficient', 'units': ' '})
    
            # Future development:
            # Add an option to provide either hyai, hybi or hyam, hybm directly

    def check_files(self, twodim, levels_type, wind_filename, sp_filename):
        """
        Check files structure
        """
        # Check coordinates
        expected_coords = {'time', 'level', 'latitude', 'longitude'}
        if not set(self.ds.coords) == expected_coords:
            raise TypeError(f"coordinates in file {wind_filename} should be {expected_coords}")

        if levels_type == 'ml':
            expected_coords.remove('level')  # Remove 'level' for 'ml' case
            if not set(self.ds_sf.coords) == expected_coords:
                raise TypeError(f"coordinates in file {sp_filename} should be {expected_coords}")

        # Check vertical coordinate long_name
        valid_long_names = {'ml':'model_level_number', 'pl':'pressure_level'}
        if self.ds.level.long_name != valid_long_names[levels_type]:
            raise TypeError(f'Make sure the model data vertical coordinate is {"hybrid levels" if levels_type == "ml" else "pressure levels"}, and set ds.level.long_name as {valid_long_names[levels_type]}')

        # Check variables
        self.check_variable_list(twodim)

        # Future development: Add more checks if needed

    def check_variable_list(self, twodim):
        """
        Check required variables
        """
        FieldsFourD.compare_variables({'u', 'v', 'w'} if not twodim else {'u', 'v'}, self.ds)
        if self.ds_sf is not None: FieldsFourD.compare_variables({'sp'}, self.ds_sf)

    @staticmethod
    def compare_variables(expected_variables, dataset):
        """
        Compare the variables in the given dataset with the expected variables.
        """
        file_variables = set(dataset.variables)
        missing_variables = expected_variables - file_variables
        if missing_variables:
            raise ValueError(f'Missing variables: {", ".join(missing_variables)}')

    def change_longitude_range(self):
        """
        Change longitude range from [0, 360] to [-180, 180]
        """
        self.ds = self.ds.assign_coords(longitude=(((self.ds.longitude + 180) % 360) - 180))
        if self.ds_sf is not None: self.ds_sf = self.ds_sf.assign_coords(longitude=(((self.ds_sf.longitude + 180) % 360) - 180))

    def get_boundaries(self, levels_type):
        """
        Get maximum and minimum of the latitude and longitude in the model data files provided
        """
        min_lon = max(self.ds.longitude.min().item(), self.ds_sf.longitude.min().item() if levels_type == 'ml' else -180 )
        max_lon = min(self.ds.longitude.max().item(), self.ds_sf.longitude.max().item() if levels_type == 'ml' else 180)
        min_lat = max(self.ds.latitude.min().item(), self.ds_sf.latitude.min().item() if levels_type == 'ml' else -90)
        max_lat = min(self.ds.latitude.max().item(), self.ds_sf.latitude.max().item() if levels_type == 'ml' else 90)

        self.horiz_boundaries = {'latitude':{'min':min_lat, 'max':max_lat}, 'longitude':{'min':min_lon,'max':max_lon}}

class FieldsThreeD:

    def __init__(self, time, twodim, ds, ds_sf, hyam, hybm):

        self.get_wind(ds, time, twodim)
        self.get_pres_and_sp(time, ds_sf, hyam, hybm)

    def __str__(self):
        return 'Time-specific DataArrays:\n'+("\n".join(list(vars(self))))

    def get_wind(self, ds, time, twodim):
        """
        Get wind components (u [m/s], v [m/s], w [Pa/s])
        """
        self.u = ds.u.sel(time=time)
        self.v = ds.v.sel(time=time)
        self.w = ds.w.sel(time=time) if not twodim else None

    def get_pres_and_sp(self, time, ds_sf, hyam, hybm):
        """
        Get pressure and surface pressure [hPa]
        """
        if ds_sf is not None and hyam is not None and hybm is not None:
            self.sp = ds_sf.sp.sel(time=time)/100
            self.pres = hyam + hybm*self.sp
        else:
            self.sp = None
            self.pres = None

def euler_iter(id, pres_0, lat_0, lon_0, p0, u0, v0, w0, p1, u1, v1, w1, horiz_bound, delta_t, forw_back, n_max, verbosity):
    """
    Integrate the advection equation (Lagrangian form) one time-step
    """
    if verbosity >= 1: print(f'Integrating parcel {id}: ({pres_0}, {lat_0}, {lon_0})\n')

    # Predict the parcel location at the next step
    pres_tmp, lat_tmp, lon_tmp, dpres_predicted, dlat_predicted, dlon_predicted, bound_prox_flag, __ = iteration(
                        id, pres_0, lat_0, lon_0, None, None, None, None, None, None,
                        p0, u0, v0, w0, 1, horiz_bound, delta_t, forw_back, verbosity)

    # If close to the bottom or top of the vertical domain, stop and return original positions
    if bound_prox_flag: return id, pres_0, lat_0, lon_0, bound_prox_flag, True

    # Correct the parcel location at the next step
    pres, lat, lon, __, __, __, bound_prox_flag, n = iteration(
                        id, pres_0, lat_0, lon_0, pres_tmp, lat_tmp, lon_tmp,
                        dpres_predicted, dlat_predicted, dlon_predicted,
                        p1, u1, v1, w1, n_max, horiz_bound, delta_t, forw_back, verbosity)

    # Check convergence
    if n >= n_max:
        converg_flag = False
        print(f"Parcel {id}, WARNING: Integration did not converge. Stopping time integration. \n")
    else:
        converg_flag = True

    return id, pres, lat, lon, bound_prox_flag, converg_flag

def iteration(id, pres_0, lat_0, lon_0, pres, lat, lon,
              dpres_predicted, dlat_predicted, dlon_predicted,
              p, u, v, w, n_max, horiz_bound, delta_t, forw_back, verbosity):
    """
    Iterate over corrections based on the given parameters.
    """
    # Set locations if prediction was not provided (i.e., called from the predictive step):
    if pres is None and lat is None and lon is None:
        pres, lat, lon = pres_0, lat_0, lon_0

    # Set parameters that control the number of iterations
    n=0
    ddlon=np.inf
    ddlat=np.inf
    ddpres=np.inf
    dpres_old=None
    dlat_old=None
    dlon_old=None

    while (ddlon**2 + ddlat**2 > 0.0001 or abs(ddpres) > 1.0) and n < n_max:

        # Calculate deltas
        dpres, dlat, dlon = calculate_delta_position(delta_t, pres, lat, lon, p, u, v, w, verbosity, id)

        # Set predicted tendencies if prediction was not provided (i.e., called from the predictive step):
        if dpres_predicted is None and dlat_predicted is None and dlon_predicted is None:
            dpres_predicted, dlat_predicted, dlon_predicted = dpres, dlat, dlon

        # Update pres, lat, lon
        lon = lon_0 + 0.5*forw_back*(dlon_predicted+dlon)
        lat = lat_0 + 0.5*forw_back*(dlat_predicted+dlat)
        pres = pres_0 + 0.5*forw_back*(dpres_predicted+dpres)

        # Print debug message
        if id == 1 and verbosity == 2:
            print(f'Parcel {id}: n = {n}, pres = {pres_0}->{pres}, lat = {lat_0}->{lat}, lon = {lon_0}->{lon} \n')

        # Check proximity to the boundaries
        pres_1d = p.interp(latitude=lat, longitude=lon) if isinstance(p, xr.core.dataarray.DataArray) else u.level
        bound_prox_flag = check_proximity_to_boundaries(pres, lat, lon, pres_1d, horiz_bound)

        # If close to the bottom or top of the vertical domain, stop and return original positions
        if bound_prox_flag:
            print(f"Parcel {id}, ERROR: Location ({pres}, {lat}, {lon}) too close to the boundaries of the domain. Stopping time integration (correcting step no. {n}). \n")
            return pres_0, lat_0, lon_0, 0.0, 0.0, 0.0, bound_prox_flag, n

        # Calculate parameters that control the number of iterations
        if n > 0:
            ddlon = dlon - dlon_old
            ddlat = dlat - dlat_old
            ddpres = dpres - dpres_old
        # else, this is the first iteration, continue

        dlon_old = dlon
        dlat_old = dlat
        dpres_old = dpres

        # Update the number of iterations done
        n += 1

    return pres, lat, lon, dpres, dlat, dlon, bound_prox_flag, n

def calculate_delta_position(delta_t, pres, lat, lon, p, u, v, w, verbosity, id):
    """
    Find wind at (pres, lat, lon) and calculate delta-pres, delta-lat, delta-lon
    """
    # Get value of the model data vertical coordinate at pres,lat,lon
    pres_1d = p.interp(latitude=lat, longitude=lon) if isinstance(p, xr.core.dataarray.DataArray) else None
    zpos = set_vertical_coordinate_value(pres, lat, lon, pres_1d)

    # Find u, v and w at pres,lat,lon
    u_pos, v_pos, w_pos = get_wind_at_position(zpos, lat, lon, u, v, w)

    if verbosity == 2 and id == 1: print(f'Wind at location of parcel {id}: u = {u_pos}, v = {v_pos}, w = {w_pos} \n')

    # Return delta position
    return get_dpos(lat, delta_t, u_pos, v_pos, w_pos)

def set_vertical_coordinate_value(pres, lat, lon, pres_1d):
    """
    Find value of the vertical coordinate corresponding to pressure=pres
    """
    if isinstance(pres_1d, xr.core.dataarray.DataArray):
        # Case in which the data vertical coordinate is 'model level'
        return find_lev_idx(pres, pres_1d)
    else:
        # Case in which the data vertical coordinate is pressure
        return pres

def find_lev_idx(pres, pres_1d):
    """
    Interpolate along an array of model levels to the 'subgrid' location with a given pressure value
    """
    lev_p = bisect.bisect_left(pres_1d, pres)
    lev = lev_p - (pres_1d[lev_p] - pres) / (pres_1d[lev_p] - pres_1d[lev_p - 1]) * (pres_1d.level[lev_p] - pres_1d.level[lev_p - 1])

    return lev + 1

def get_wind_at_position(zpos, lat, lon, u, v, w):
    """
    Interpolate the 3D wind fields to (zpos, lat, lon)
    """
    u_pos = u.interp(level=zpos, latitude=lat, longitude=lon).item()
    v_pos = v.interp(level=zpos, latitude=lat, longitude=lon).item()
    if isinstance(w ,xr.core.dataarray.DataArray):
        w_pos = w.interp(level=zpos, latitude=lat, longitude=lon).item()
    else:
        w_pos = 0.

    return u_pos, v_pos, w_pos

def get_dpos(lat, delta_t, u_pos, v_pos, w_pos):
    """
    Calculate 'distance' from velocity and time
    """
    # Define distance [m] corresponding to a latitude change equal to 1°, along a meridian
    delta_y = 1.112E5

    # Calculate distance from time and velocity 
    # -- Divide by the distance between two meridians (deltay*cos(lat_degrees*pi/180.)) to convert to degrees longitude
    dlon = u_pos*delta_t/(delta_y*math.cos(lat*math.pi/180.))
    # -- Divide by deltay to convert to degrees latitude
    dlat = v_pos*delta_t/delta_y 
    # -- Divide by 100 to convert from Pa/s to hPa/s
    dpres = w_pos*delta_t/100

    return dpres, dlat, dlon

def check_proximity_to_boundaries(pres, lat, lon, pres_1d, horiz_bound):
    """
    Check if a given location (pres, lat, lon) is too close to the boundaries of the domain.
    """
    min_pres, max_pres = pres_1d[0].item() + 5, pres_1d[-1].item() - 5
    min_lat, max_lat = horiz_bound['latitude']['min'], horiz_bound['latitude']['max']
    min_lon, max_lon = horiz_bound['longitude']['min'], horiz_bound['longitude']['max']

    out_of_range = (pres > max_pres or pres < min_pres or
                    lat < min_lat or lat > max_lat or
                    lon < min_lon or lon > max_lon)

    if out_of_range:
        print(f"WARNING: Parcel location ({pres}, {lat}, {lon}) is out of range "
              f"(pres=[{min_pres}, {max_pres}], lat=[{min_lat}, {max_lat}], lon=[{min_lon}, {max_lon}]).\n")

    return out_of_range

def increment_time(start_datetime, forw_back, delta_t, t_idx):
    return start_datetime + datetime.timedelta(seconds = forw_back * delta_t * t_idx)

def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)  


if __name__ == "__main__":

    # Init timer
    start_0 = time.time()

    # Get execution parameters
    args = Arguments()

    # Read model datasets
    fields_4d = FieldsFourD(args.wind_filename, args.sp_filename, args.coeff_filename, args.levels_type, args.twodim)

    # Get 3d fields for the initial time
    fields_3d_current_time = FieldsThreeD( args.start_datetime, args.twodim, fields_4d.ds, fields_4d.ds_sf, fields_4d.hyam, fields_4d.hybm )

    # Initialize Parcel class
    Parcel.total_instances = 0
    Parcel.parcels = []

    # Read input file with initial (p, lat, lon) for each parcel
    with open(args.in_filename) as f:
        for line in f:
            pres, lat, lon = map(float, line.split())

            # Check proximity to the boundaries
            pres_1d = fields_3d_current_time.pres.interp(latitude=lat,longitude=lon) if isinstance(fields_3d_current_time.pres, xr.core.dataarray.DataArray) else fields_4d.ds.level
            bound_prox_flag = check_proximity_to_boundaries(pres, lat, lon, pres_1d, fields_4d.horiz_boundaries)

            # Create instance of Parcel
            if not bound_prox_flag:
                Parcel(args.start_datetime, pres, lat, lon)

    # Set number of cores to be used
    nproc_avail = len(os.sched_getaffinity(0))
    nproc = min(min(nproc_avail, Parcel.total_instances), 100)
    print(f"Running with {nproc} cores \n")

    # integrate over time
    times = [ increment_time(args.start_datetime, args.forw_back, args.delta_t, t) for t in range(0, args.number_time_steps+1) ]
    for current_time, next_time in pairwise(times):

        start = time.time()

        # Get 3d fields to be used in the next time step
        fields_3d_next_time = FieldsThreeD( next_time, args.twodim, fields_4d.ds, fields_4d.ds_sf, fields_4d.hyam, fields_4d.hybm )

        print(f"Date: {next_time.strftime('%Y-%m-%d')}, time: {next_time.strftime('%H')}Z \n")

        # Get parcel info at current_time
        parcel_data = [(parcel.id, parcel.pres, parcel.lat, parcel.lon) for parcel in Parcel.parcels if parcel.time == current_time]

        with Pool(processes=nproc) as pool:
            # Calculate parcels' position at next_time using the improved-Euler method and update
            results = pool.starmap(euler_iter, [(id, pres, lat, lon, fields_3d_current_time.pres,
                                         fields_3d_current_time.u, fields_3d_current_time.v,
                                         fields_3d_current_time.w, fields_3d_next_time.pres,
                                         fields_3d_next_time.u, fields_3d_next_time.v,
                                         fields_3d_next_time.w, fields_4d.horiz_boundaries,
                                         args.delta_t, args.forw_back, args.number_iterations, args.verbosity) for id, pres, lat, lon in parcel_data])

        pool.close()
        pool.join()

        # Create new parcel instance with same id
        for id, pres, lat, lon, bound_prox_flag, converg_flag in results:
            if not bound_prox_flag and converg_flag:
                Parcel(next_time, pres, lat, lon, id)

        # Update 3d fields to be used as current_time in the next time step
        fields_3d_current_time = fields_3d_next_time

        print(f"Calculated in {time.time() - start:6.4f} s \n")

    # Save all parcels to JSON and CSV files
    Parcel.save_parcel_data_to_csv(args.out_filename)
    Parcel.save_parcel_data_to_json(args.out_filename.replace('.csv','.json'))

    print(f"Total execution time: {time.time() - start_0:6.4f} s \n")
