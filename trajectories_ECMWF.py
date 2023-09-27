#!/usr/bin/env python
# coding: utf-8

print('Loading ...')

import datetime
import argparse
import time
import os, sys
import xarray as xr
import numpy as np
import scipy.signal as sps
import math
import bisect
import json

from multiprocessing import Pool, current_process
from itertools import repeat

def getArgs():


    # this function reads the user-defined parameters, provided via command line arguments, or sets default values

    parser = argparse.ArgumentParser(description='Calculates trajectories given initial positions and a 4D wind field',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('time', type = lambda s: datetime.datetime.strptime(s, '%H'), help = 'initial time HH (UTC hour)')
    parser.add_argument('-d', '--date', type = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), default = datetime.datetime.utcnow().date(), help = 'initial date YYYY-MM-DD')
    parser.add_argument('-dt', '--delta_t', type = int, default = 3600, help = 'time step in second between input files')
    parser.add_argument('-df', '--direction_flag', type = int, default = 1, choices = [-1,1], help = 'integration direction: 1 forward, -1 backward')
    parser.add_argument('-nt', '--number_time_steps', type = int, default = 12, help = 'number of time steps for integration')
    parser.add_argument('-n', '--number_iterations', type = int, default = 20, help = 'maximum number of iterations of the numerical scheme')
    parser.add_argument('-3d', '--three_dimensions', type = int, default = 1, choices = [0,1], help = 'switch for 3D (1) or 2D (0) trajectories')
    parser.add_argument('-id', type = str, default = '0000', help = 'string identifying the execution')
    parser.add_argument('-v', '--verbose', type = int, default = 0, choices = [0,1,2], help = 'switch for very verbose (2), moderately verbose (1) or non-verbose (0)')
    parser.add_argument('-lt', '--levels_type', type = str, default = 'ml', choices = ['ml','pl'], help = 'type of vertical coordinate of the model data: model levels (ml) or pressure levels (pl)')
    parser.add_argument('-hu', '--height_units', type = str, default = 'ft', choices = ['m','ft'], help = 'units of height outputs: feet (ft) or meters (m)')

    return parser.parse_args()

def findModelDataFile(start_time,end_time,levels_type):

    # this function looks for the model output file that contains all the times necessary for the integration

    shift = 0
    windFileFound = False
    while shift < 5:

        modelInitDay = (start_time-datetime.timedelta(days=shift)).strftime('%Y-%m-%d')
        DIR = f"/work/bb1311/CAFE-Brazil_OPERATIONAL/forecast/{modelInitDay}"
        try:
            dirFiles = os.listdir(f'{DIR}')
        except FileNotFoundError:
            shift += 1
            continue
        availTimes = sorted( set([dirFiles[i][:2] for i in range(len(dirFiles))]), reverse = True)
        for availTime in availTimes:
            modelInitTime = datetime.datetime.fromisoformat(f"{modelInitDay} {availTime}")
            if (start_time > modelInitTime and end_time > modelInitTime):
                windFile = f'{DIR}/{availTime}UTC_{levels_type}.nc'
                windFileFound = True
                break
        if windFileFound: break
        shift += 1

    else:
        print(f"FATAL: could't find model data containing the time interval [{min(start_time,end_time).strftime('%Y-%m-%d %H')}Z, {max(start_time,end_time).strftime('%Y-%m-%d %H')}Z]")
        sys.exit()

    return windFile

def confirmationRequest(start_time,deltat,direction,end_time,n_max,dimensions,windFile,infile,outfile,outfile_alt,logfile):

    logString = f'''

    - initial time: {start_time.strftime('%Y-%m-%d %H')}Z
    - delta_t: {deltat} s
    - direction: {direction}
    - end time: {end_time.strftime('%Y-%m-%d %H')}Z
    
    - maximum number of iterations: {n_max}
    - dimensions: {dimensions}
    
    - input: 
        -- model data: {windFile}
        -- initial parcel locations: {infile}
    - output: 
        -- pressure units: {outfile}
        -- altitude units: {outfile_alt}

    '''

    print(f'Request summary: {logString}')
    answer = input("Continue?\n")
    if answer.lower() in ["y","yes"]:
        with open(logfile,'w') as writer:
            writer.write(f'Running trajectories with the following specifications: {logString}')
    else:
        print('Aborting ...')
        time.sleep(1)
        sys.exit()    

def setupExecution(args):

    if (args.three_dimensions == 1) and (args.levels_type == 'pl'):
        print(f"FATAL: Pressure-levels data only suitable for 2D trajectories. Try again with options: -3d = 0 or -lt 'ml'")
        sys.exit()

    # time for the trajectories initialization
    start_time = datetime.datetime.fromisoformat(f"{args.date.strftime('%Y-%m-%d')} {args.time.strftime('%H')}")
    
    # interval between model outputs
    deltat = args.delta_t 
     
    # forward/backward flag
    forw_back = args.direction_flag
    if forw_back == 1:
        direction = 'forward'
    elif forw_back == -1:
        direction = 'backward'
    
    # number of time steps for integration
    time_steps = args.number_time_steps     
    end_time = start_time+datetime.timedelta(seconds=forw_back*deltat*time_steps)
    
    # maximum number of iterations of the numerical scheme
    n_max = args.number_iterations
    
    # 3d/2d flag
    threedim = args.three_dimensions
    if threedim:
        dimensions = '3D'
    else:
        dimensions = '2D'
    
    # verbose flag
    iverbose = args.verbose

    # paht to model output
    windFile = findModelDataFile(start_time,end_time,args.levels_type)

    # path to file containing initial positions
    infile = f"./init_locations/init_locations_{args.id}_{start_time.strftime('%Y-%m-%d_%H')}Z.txt"
    
    # path to file to store trajectories
    outfile = f"./trajectories/trajectories_{args.id}_{direction}_{dimensions}_{args.levels_type}_{start_time.strftime('%Y-%m-%d_%H')}Z.txt"
    outfile_alt = f"./trajectories/trajectories_alt_{args.id}_{direction}_{dimensions}_{args.levels_type}_{start_time.strftime('%Y-%m-%d_%H')}Z.txt"

    # path to log file
    logfile = f"./log/log_trajectories_{args.id}_{direction}_{dimensions}_{start_time.strftime('%Y-%m-%d_%H')}Z.txt"

    # display setup summary and ask for confirmation
    confirmationRequest(start_time,deltat,direction,end_time,n_max,dimensions,windFile,infile,outfile,outfile_alt,logfile)

    return start_time, deltat, forw_back, time_steps, n_max, threedim, windFile, infile, outfile, outfile_alt, iverbose

def getLevelsCoef(ds_ml):

    # read hyai and hybi (hybrid model levels coefficients a and b at the layer interfaces, i.e., 137+1 values)
    with open('aux/L137_hybrid_levels_NOHEADER.csv', 'rb') as f:
        levels_info = np.loadtxt(f,delimiter = ",",usecols = (1,2))

    # average the hyai and hybi coefficients every two consecutive half-levels to obtain hyam and hybm (full-levels coefficients)
    convolution_kernel=np.ones(2)/2
    hyam = sps.fftconvolve(0.01*levels_info[:,0], convolution_kernel, mode='valid', axes=0)
    hybm = sps.fftconvolve(levels_info[:,1], convolution_kernel, mode='valid', axes=0)

    # hybrid levels coefficient on the layer interfaces
    hyam = xr.DataArray(data=hyam,dims=["level"],coords=dict(level=ds_ml.level),
        attrs=dict(
            description="Hybrid sigma-pressure 'a' coefficient",
            units="hPa",
            ),
    )
    hybm = xr.DataArray(data=hybm,dims=["level"],coords=dict(level=ds_ml.level),
        attrs=dict(
            description="Hybrid sigma-pressure 'b' coefficient",
            units=" ",
            ),
    )    

    return hyam,hybm

def finding_level_pl(p1d,p_point):
    
    # This functions finds the lev location of a given pressure,lat,lon for a model data in pressure levels
   
    #1. Find model level (z_point) corresponding to p_point
    lev_p = bisect.bisect_left(p1d, p_point) #Find the index of the first element >= p_point in the column (from top to bottom)
    delta_lev_p = (p1d[lev_p]-p_point)/(p1d[lev_p]-p1d[lev_p-1]) #Apply a linear interpolation in the vertical dimension...
    z_point = lev_p - delta_lev_p*(p1d.level[lev_p]-p1d.level[lev_p-1]) # ...to find the (subgrid) 'index' of p_point
    z_point = z_point + 1 #Add 1 to match the model-levels coordinate

    return z_point

def finding_level(p3d,p_point,lat_point,lon_point):
    
    # This functions finds the lev location of a given pressure,lat,lon for a model data in hybrid levels
   
    #1. Interpolate to obtain the pressure in the column at lat_point,lon_point
    p1d = p3d.interp(latitude=lat_point, longitude=lon_point)

    #2. Find model level (z_point) corresponding to p_point
    z_point = finding_level_pl(p1d,p_point)

    return z_point

def altitude(t1d,qv1d,p1d,psurf):

    # This function calculates geopotential height* on model levels
    # *height relative to mean sea level
    
    #Input:
    #     1D variables
    #     t1d -- temperature in a column (K)
    #     qv1d -- water vapor mixing ratio (kg/kg)
    #     p1d -- pressure on model levels (hPa) 
    #
    #     0D variables:
    #     psurf -- surface pressure (hPa)
    
    R_d = 287.0          # Dry air constant (J/K-1.kg-1)
    Re = 6371000.0       # Earth's radius (m)
    g = 9.80665          # Gravity acceleration

    pres = np.zeros(138)    # pressure on half levels
    pres[137] = psurf
    for i in range(136,-1,-1):
        pres[i] = 2*p1d[i]-pres[i+1]
    
    tv = t1d*(1+qv1d/0.622)/(1+qv1d)   # virtual temperature [K]

    integrand = np.zeros(137)
    z_ml = np.zeros(137)
    alt = np.zeros(137)  
        
    for i in range(136,-1,-1): 

        deltap = pres[i] - pres[i+1]

        integrand[i] = tv[i]*deltap/p1d[i]
        z_ml[i] = - R_d*integrand[i:137].sum()    # geopotential (m2.s-2)

        alt[i] = z_ml[i]/g                        # geopotential height (m) with respect to MSL (EGM96 Geoid)
                                                  
        #alt[i] = Re*z_ml[i]/(g*(Re-z_ml[i]/g))    # geometric height (m) with respect to MSL 
                                                   # assuming the Earth is spherical
        
        #print("{} {} {}".format(i, p1d[i].data/100, alt[i]/1000.))
        
    alt_xarray = xr.DataArray(data=alt,dims=['level'],coords=p1d.coords,
         attrs=dict(
             description="Geopotential height",
             units="m",
             ),
    )
            
    return alt_xarray


def find_alt_loc(ds,p,sp,parcel_pos,units):
    
    # This function calculates the altitude corresponding to a given pres,lat,lon 
    # (altitude as provided by the 'altitude' function) in the model output
    
    # Input:
    # ds         # reference model output dataset
    # p          # pressure on model full levels [hPa]
    # sp         # surface pressure [hPa]
    # parcel_pos    # parcel coordinates pressure,lat,lon
    
    #rank = f'Finding altitude location on process: {current_process()._identity[0]}'
    #print(rank, parcel_pos)
    
    # Find the altitude for each model level at a given lat,lon location
    t_1d=ds.t.interp(latitude=parcel_pos[1],longitude=parcel_pos[2])
    q_1d=ds.q.interp(latitude=parcel_pos[1],longitude=parcel_pos[2])
    p_1d=p.interp(latitude=parcel_pos[1],longitude=parcel_pos[2])
    PS_0d=sp.interp(latitude=parcel_pos[1],longitude=parcel_pos[2])
    alt=altitude(t_1d,q_1d,p_1d,PS_0d)

    #for i in range(len(p_1d)):
    #    print(i,alt[i].data,p_1d[i].data)
        
    lev_point = finding_level_pl(p_1d,parcel_pos[0])
    
    if units == "ft": 
        factor = 3.280839895
    elif units == "m":
        factor = 1.
    interpolated_altitude=alt.interp(level=lev_point).data.item()*factor

    return [interpolated_altitude,parcel_pos[1],parcel_pos[2]]

def euler_iter(pos0,p0,u0,v0,w0,pos1,p1,u1,v1,w1,deltat,forw_back,n_max,threedim,iverbose,levels_type):

    rank = current_process()._identity[0]
    if iverbose: print(f'Running process {rank}, parcel at {pos0}')

    # check whether the parcels are too close to the boundaries, if so, stop euler_iter and keep the initial position
    if levels_type == 'ml': 
        ps0_pos=p0.interp(level=len(p0.level)-1, latitude=pos0[1], longitude=pos0[2]).data
        ptop0_pos=p0.interp(level=0, latitude=pos0[1], longitude=pos0[2]).data
    elif levels_type == 'pl': 
        ps0_pos=p0[len(p0.level)-1].data
        ptop0_pos=p0[0].data
    if (pos0[0]+5.)>=ps0_pos: 
        print(f"Process {rank}, WARNING: Parcel height ({pos0[0]}) too close to the surface ({ps0_pos}hPa). Clipping to the surface level.")
        pos0[0]=ps0_pos.item()
        pos1=pos0
        return pos1
    if (pos0[0]-5.)<=ptop0_pos: 
        print(f"Process {rank}, WARNING: Parcel height ({pos0[0]}) too close to the top ({ptop0_pos}hPa). Clipping to the surface level.")
        pos0[0]=ps0_pos.item()
        pos1=pos0
        return pos1

    deltay = 1.112E5   #distance in m between 2 lat circles

    # find model level corresponding to pressure pos0[0] at time t
    if levels_type == 'ml':
        zpos = finding_level(p0,pos0[0],pos0[1],pos0[2])
    elif levels_type == 'pl':
        zpos = pos0[0]

    # find u, v and w at pos0[0],pos0[1],pos0[2] (i.e., p,lat,lon at time t)
    u0_pos = u0.interp(level=zpos, latitude=pos0[1], longitude=pos0[2]).data
    v0_pos = v0.interp(level=zpos, latitude=pos0[1], longitude=pos0[2]).data
    if threedim:
        w0_pos = w0.interp(level=zpos, latitude=pos0[1], longitude=pos0[2]).data
    else:
        w0_pos = 0.

    # find dx, dy and dp based on the wind at pos0[0],pos0[1],pos0[2] (i.e., p,lat,lon at time t)
    dx0 = u0_pos*deltat/(deltay*math.cos(pos0[1]*math.pi/180.))     # divide by the distance between two meridians                                                     
                                                                    # [deltay*cos(lat_degrees*pi/180.)] to convert to
                                                                     # degrees longitude
    dy0 = v0_pos*deltat/deltay      # multiply by the distance between two parallels 
                                    # (deltay) to convert to degrees latitude
    dp0 = w0_pos*deltat/100         # divide by 100 to convert from Pa/s to hPa/s   

    # initialize dx1, dy1 and dp1 (i.e., this corresponds to Euler's method if n=0 below)
    dx1 = dx0.copy()
    dy1 = dy0.copy()
    dp1 = dp0.copy()
    
    # set parameters that control the number of iterations
    n = 0  
    toolarge = True
    dx1_old = 0.
    dy1_old = 0.
    dp1_old = 0.
    
    while toolarge and n < n_max:
                
        # obtain p,lat,lon at time t+1    
        pos1[2] = pos0[2] + 0.5*forw_back*(dx0+dx1)
        pos1[1] = pos0[1] + 0.5*forw_back*(dy0+dy1)
        pos1[0] = pos0[0] + 0.5*forw_back*(dp0+dp1)

        # prevent parcels from going into the ground
        if levels_type == 'ml': 
            ps1_pos=p1.interp(level=len(p1.level)-1, latitude=pos1[1], longitude=pos1[2]).data
            ptop1_pos=p1.interp(level=0, latitude=pos1[1], longitude=pos1[2]).data
        elif levels_type == 'pl': 
            ps1_pos=p1[len(p1.level)-1].data
            ptop1_pos=p1[0].data
        if pos1[0]>ps1_pos: 
            print(f"Process {rank}, WARNING: Parcel height ({pos1[0]}) too close to the surface ({ps1_pos}). Clipping to the surface level.")
            pos1[0]=ps1_pos.item()
            return pos1
        if (pos1[0]-5.)<=ptop1_pos: 
            print(f"Process {rank}, WARNING: Parcel height ({pos1[0]}) too close to the top ({ptop1_pos}hPa). Clipping to the surface level.")
            pos1[0]=ps1_pos.item()
            return pos1

        # calculate parameters that control the number of iterations
        ddx = dx1 - dx1_old
        ddy = dy1 - dy1_old
        ddp = dp1 - dp1_old
        dx1_old = dx1.copy()
        dy1_old = dy1.copy()
        dp1_old = dp1.copy()
        dhpos = np.sqrt(ddx**2 + ddy**2)
        if dhpos < 0.01 and abs(ddp) < 1. : toolarge = False

        # find model level corresponding to pressure pos1[0]
        if levels_type == 'ml':
            zpos = finding_level(p1,pos1[0],pos1[1],pos1[2]) #lev
        elif levels_type == 'pl':
            zpos = pos1[0]

        # find u, v and w at pos1[0],pos1[1],pos1[2] (i.e., p,lat,lon at time t+1)
        u1_pos = u1.interp(level=zpos, latitude=pos1[1], longitude=pos1[2]).data
        v1_pos = v1.interp(level=zpos, latitude=pos1[1], longitude=pos1[2]).data
        if threedim:
            w1_pos = w1.interp(level=zpos, latitude=pos1[1], longitude=pos1[2]).data
            if abs(w1_pos)>2.: 
                pos1[0]=ps1_pos.item()
                print(f"Process {rank}, WARNING: Avoinding intense vertical movements ({w1_pos} Pa/s). Clipping to the surface.")
                return pos1
        else:
            w1_pos = 0.

        # calculate dx1,dy1,dz1 to be used in the next iteration        
        dx1 = u1_pos*deltat/(deltay*math.cos(pos1[1]*math.pi/180.))     #divide by the distance between two meridians
                                                                        #[deltay*cos(lat_degrees*pi/180.)] to convert
                                                                        #to degrees longitude
        dy1 = v1_pos*deltat/deltay      #divide by deltay to convert to degrees latitude
        dp1 = w1_pos*deltat/100         #divide by 100 to convert from Pa/s to hPa/s   

        # update the number of iterations corresponding to the next cycle
        n = n + 1

        if iverbose == 2: print(f'\nProcess {rank}: n = {n}, u = {u1_pos}, v = {v1_pos}, w = {w1_pos} , ddx = {ddx}, ddy = {ddy}, ddp = {ddp}, dp = {dp1}') 

    if toolarge and n == n_max: 
        print(f"Process {rank}, WARNING: Integration did not converge")

    return pos1

if __name__ == "__main__":

    args = getArgs()
    start_time, deltat, forw_back, time_steps, n_max, threedim, windFile, infile, outfile, outfile_alt, iverbose = setupExecution(args) 
 
    # read input file with initial (p, lat, lon) for each parcel
    with open(infile) as f:
        dum = [line.split() for line in f]
    init_pos = [[float(column) for column in row] for row in dum]
    n_parcels = len(init_pos)
    dict_pos = {start_time: dict([(i,init_pos[i]) for i in range(n_parcels)])}
    dict_pos_alt = {start_time: dict([(i,init_pos[i]) for i in range(n_parcels)])}

    # init timer
    start_0 = time.time()    

    # get number of cores to be used
    nproc_avail = len(os.sched_getaffinity(0))
    nproc = min(min(nproc_avail,n_parcels),100)
    print("Running with {} cores".format(nproc))

    # read forecast data
    ds = xr.open_dataset(windFile)
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

    if args.levels_type == 'ml': 

        # read surface forecast data
        ds_sf = xr.open_dataset(windFile.replace("_ml.","_sf." ))
        ds_sf = ds_sf.assign_coords(longitude=(((ds_sf.longitude + 180) % 360) - 180))
    
        # get model levels information 
        hyam, hybm = getLevelsCoef(ds)

    #************** Time integration ************** 

    for t_i in range(0, time_steps):

        start = time.time()    # init timer

        current_time = start_time+datetime.timedelta(seconds=forw_back*deltat*t_i)
        next_time = start_time+datetime.timedelta(seconds=forw_back*deltat*(t_i+1))

        print(f"Date: {next_time.strftime('%Y-%m-%d')}, time: {next_time.strftime('%H')}Z")
           
        if args.levels_type == 'ml': 

            # get pressure [hPa] at current_time and next_time
            sp0 = ds_sf.sp.sel(time=current_time)/100
            sp1 = ds_sf.sp.sel(time=next_time)/100
            p0 = hyam + hybm*sp0
            p1 = hyam + hybm*sp1

        elif args.levels_type == 'pl': 

            p0 = ds.level
            p1 = ds.level

        # get horizontal wind components [m/s] at current_time and next_time
        u0 = ds.u.sel(time=current_time)
        u1 = ds.u.sel(time=next_time)

        v0 = ds.v.sel(time=current_time)
        v1 = ds.v.sel(time=next_time)

        # get vertical wind component [Pa/s] at current_time and next_time
        if threedim:
            w0 = ds.w.sel(time=current_time)
            w1 = ds.w.sel(time=next_time)
        else:
            w0 = np.zeros(u0.shape)
            w1 = np.zeros(u1.shape)

        dict_pos[next_time] = dict([(i,[0., 0., 0.]) for i in range(n_parcels)])
        dict_pos_alt[next_time] = dict([(i,[0., 0., 0.]) for i in range(n_parcels)])
        with Pool(processes=nproc) as pool:

            # calculate parcels' position at next_time using the iterative Euler's method (Petterssen's scheme) 
            tmp = pool.starmap(euler_iter, zip(dict_pos[current_time].values(),repeat(p0),repeat(u0),repeat(v0),repeat(w0),
                                                      dict_pos[next_time].values(),repeat(p1),repeat(u1),repeat(v1),repeat(w1),
                                                      repeat(deltat),repeat(forw_back),repeat(n_max),
                                                      repeat(threedim),repeat(iverbose),repeat(args.levels_type)))
            tmp_alt = tmp.copy()

            if args.levels_type == 'ml':
                # Find the height corresponding to the parcels' pressure
                if t_i==0: tmp_alt_0 = pool.starmap(find_alt_loc, zip(repeat(ds.sel(time=current_time)),repeat(p0),repeat(sp0),dict_pos_alt[current_time].values(),repeat(args.height_units)))
                tmp_alt = pool.starmap(find_alt_loc, zip(repeat(ds.sel(time=next_time)),repeat(p1),repeat(sp1),tmp_alt,repeat(args.height_units)))
            
        pool.close()
        pool.join()

        for parcel in range(n_parcels):
            dict_pos[next_time][parcel] = tmp[parcel].copy()
            if args.levels_type == 'ml':
                if t_i==0: dict_pos_alt[current_time][parcel] = tmp_alt_0[parcel].copy()
                dict_pos_alt[next_time][parcel] = tmp_alt[parcel].copy()

        stop = time.time() #end timer
        print("Calculated in {:6.4} s".format(stop-start))   

    # save trajectories to file
    serialized_dict_pos = {k.isoformat(): v for k, v in dict_pos.items()}
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(serialized_dict_pos, f, ensure_ascii=False, indent=4)     
    serialized_dict_pos_alt = {k.isoformat(): v for k, v in dict_pos_alt.items()}
    with open(outfile_alt, 'w', encoding='utf-8') as f:
        json.dump(serialized_dict_pos_alt, f, ensure_ascii=False, indent=4)     

    stop_0 = time.time() #end timer
    print("Total execution time: {:6.4} s".format(stop_0-start_0))









