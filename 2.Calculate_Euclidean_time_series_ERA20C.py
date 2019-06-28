import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import num2date
from numpy import linalg as la
import matplotlib.pyplot as plt
import os  
import cartopy.crs as ccrs
import cartopy.feature
import sompak



###########################################################################################

#### part 1

#Writng related functions



#function1:
# reads ECMWF data about the zonal wind (E-W winds)
def read_u(year_array): # u is the east west velocity(速度)
    # this function reads in 10m zonal (east-west) velocity from ERA-Interim dataset  
    directoryname='D:/Shun_Li/1.2 dataset_ERA20c/'   
    variable_name='u10'
    filename=variable_name+'_resampledu_'+str(year_array[0])+'.nc'
    print('Reading year:'+str(year_array[0]))
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
    
    for year in year_array[1:]:
        print('Reading year:'+str(year))
        filename=variable_name+'_resampledu_'+str(year)+'.nc'
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
        ncid1=xr.merge((ncid1,ncid2))
    return ncid1


#function2：
# reads ECMWF data about the meridional wind (N-S winds)
def read_v(year_array): # v is the north south velocity
    # this function reads in 10m meridional (north-south) velocity from ERA-Interim dataset
    directoryname='D:/Shun_Li/1.2 dataset_ERA20c/' 
    variable_name='v10'
    print('Reading year:'+str(year_array[0]))
    filename=variable_name+'_resampledu_'+str(year_array[0])+'.nc'  
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename)) 
    
    for year in year_array[1:]:
        print('Reading year:'+str(year))
        filename=variable_name+'_resampledu_'+str(year)+'.nc'
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename)) 
        ncid1=xr.merge((ncid1,ncid2))


    
    # output the raw xarray (ncid1) 
    return ncid1


#function3:
# creates an xarray object
def create_resampled_xarray(time,lat,lon,variable_name,variable):
#The dictionnary keys are the variables contained in the Dataset.
#The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    d = {}
    d['time'] = ('time',time)
    d['latitude'] = ('latitude',lat)
    d['longitude'] = ('longitude', lon)
    d[variable_name] = (['time','latitude','longitude'], variable)
    dset = xr.Dataset(d)
    return dset


#function4:
# making another type of xarray object
def create_multi_xarray_time_series(time,variable_name1,variable1,variable_name2,variable2):
#The dictionnary keys are the variables contained in the Dataset.
#The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    d = {}
    d['time'] = ('time',time)
    d[variable_name1] = (['time'], variable1)
    d[variable_name2] = (['time'], variable2)
    dset = xr.Dataset(d)
    return dset


#function5:
# creates a histogram telling you the relative frequency of occurrence of teh different patterns
def create_xarray_histogram_series(time,node_number,variable_name,variable):
#The dictionnary keys are the variables contained in the Dataset.
#The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    d = {}
    d['time'] = ('time',time)
    d['node_number']=('node_number',node_number)
    d[variable_name] = (['time','node_number'], variable)
    dset = xr.Dataset(d)
    return dset


#function6:
# calculaes the best matching unit based on Euclidean distance
def calculate_bmu(test_array, input_array):    
    test_array_tile = np.tile(test_array, (input_array.shape[0], 1))
#    print(input_array)
#    print(test_array)
    return  np.sqrt(np.nansum(np.square(input_array-test_array_tile),axis=1))


#function7:
# this slices the data into individual years and finds the RFO for each year
def create_year_histogram(node_dataset,som_number,year_array):
    histogram2=np.zeros(1)
    som_number= 12
    for year in year_array:
        time_slice1=str(year)+'-01-01'
        time_slice2=str(year)+'-12-31'
        selection=node_dataset.sel(time=slice(time_slice1,time_slice2))
        histogram1=np.histogram(selection.node_number.values,som_number)
        histogram2=np.concatenate((histogram2,histogram1[0]),axis=0)        
       
    histogram3=np.reshape(histogram2[1:434],(year_array.shape[0],som_number))
    time_array= np.arange(1980,2010)  
    variable_name='histogram'
    histogram_xarray=create_xarray_histogram_series(time_array,np.arange(0,12),variable_name,histogram3)
    return histogram_xarray






###########################################################################################

#### part 2: 

#processing 




#step1:
## 1. pre-processing data 


# step1:
# define the period of analysis    
year_array=np.arange(1980,2010)


# step2:
# read in the ECMWF files
xarray_u_0=read_u(year_array)
xarray_v_0=read_v(year_array)



# step4:
# read the topographic map in asociated with the region
directoryname2='D:/Shun_Li/1.0  topography/'   
topography_filename='topography.nc'
topography_data_0=xr.open_dataset(os.path.join(directoryname2,topography_filename))



#read in the specific ranges in latitude & longitude space in grid boxes
#since we need to offset the latitude by 10 degresss each time so that the grid  boxes overlap and the longitudes also overlap.
#Here is the different longitudes ranges to achieve it:

#state1: 20-80    80-140    140-200   200-260      260-320       320-360+0-20
#state2: 30-90    90-150    150-210   210-270      270-330       330-360+0-30
#state3: 40-100   100-160   160-220   220-280      280-340       330-360+0-40
#state4: 50-110   110-170   170-230   230-290      290-350       350-360+0-50
#state5: 60-120   120-180   180-240   240-300      300-360           0-60
#state6: 70-130   130-190   190-250   250-310   310-360+0-10        10-70
#state7: 80-140   140-200   200-260   260-320   320-360+0-20        20-80

#So, since each box size is covered by 60 degress longitude,so one cycle of overlapp just needs 6 times.
# Change some basic parameters can process it. 

area_number = 1
for latitude_1 in [90,60,30,0,-30,-60]:
    for longitude_1 in [20,80,140,200,260,320]:
        longitude_1 = longitude_1 +20
        if longitude_1 ==320+20:
            xarray_u_1 = xarray_u_0.u10.sel(longitude = slice(320 +20,360),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_u_2 = xarray_u_0.u10.sel(longitude = slice(0,320-300 +20),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_u = xr.merge((xarray_u_1,xarray_u_2))
                    
            xarray_v_1 = xarray_v_0.v10.sel(longitude = slice(320+20,360),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_v_2 = xarray_v_0.v10.sel(longitude = slice(0,320-300+20  ),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_v = xr.merge((xarray_v_1,xarray_v_2))
            
                    
                      
        else:
            xarray_u_1 = xarray_u_0.u10.sel(longitude=slice(longitude_1,longitude_1 + 30),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_u_2 = xarray_u_0.u10.sel(longitude=slice(longitude_1 + 30,longitude_1 + 60),latitude=slice(latitude_1,latitude_1-30 )) 
            xarray_u = xr.merge((xarray_u_1,xarray_u_2))
            
            
            xarray_v_1= xarray_v_0.v10.sel(longitude=slice(longitude_1,longitude_1 + 30),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_v_2 = xarray_v_0.v10.sel(longitude=slice(longitude_1+30,longitude_1 + 60),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_v = xr.merge((xarray_v_1,xarray_v_2))
            
            

#calculate day of year climatology 
        climatology_v = xarray_v.v10.groupby('time.dayofyear').mean('time')
            
# calculate the anomaly from the 20 year climatology
        anomalies_v = xarray_v.v10.groupby('time.dayofyear') - climatology_v


# calculate day of year climatology 
        climatology_u = xarray_u.u10.groupby('time.dayofyear').mean('time')
            
# calculate the anomaly from the 20 year climatology
        anomalies_u = xarray_u.u10.groupby('time.dayofyear') - climatology_u
 
 

      
      
# step3:
# calculate the yearly climatology over the map           
        year_climatology_u=np.mean(climatology_u,axis=0)
        year_climatology_v=np.mean(climatology_v,axis=0)
        year_climatology_u=year_climatology_u
        year_climatology_v=year_climatology_v




# step5:
# creating numpy subset

# creating numpy subset for u    
        u_subset=anomalies_u.values[:,:]
# creating numpy subset for v    
        v_subset=anomalies_v.values[:,:]

        u_subset=np.reshape(u_subset,(u_subset.shape[0],u_subset.shape[1]*u_subset.shape[2]))
        v_subset=np.reshape(v_subset,(v_subset.shape[0],v_subset.shape[1]*v_subset.shape[2]))

#step4:
#loading SOM patterns
# this is loading the reference SOM patterns for analysis 
        directoryname='D:/Shun_Li/2.1 SOM_pattern_figure/3.State3/'  # SHun Li to change
        variable_name='variable_u'
        filename=variable_name+'_RISES'+ str(area_number) + '.nc'
        som_u=xr.open_dataset(os.path.join(directoryname,filename)) 

        directoryname='D:/Shun_Li/2.1 SOM_pattern_figure/3.State3/'  # SHun Li to change
        variable_name='variable_v'
        filename=variable_name+'_RISES'+ str(area_number) + '.nc'
        som_v=xr.open_dataset(os.path.join(directoryname,filename)) 



#step5:
#Calculate a time series of BMU 
        column=0
        bmu_u=np.ones((som_u.variable_u.shape[0],u_subset.shape[0]))*np.nan
        bmu_v=np.ones((som_u.variable_u.shape[0],u_subset.shape[0]))*np.nan
        for column in som_u.SOM_node:   
            bmu_u[column,:]=calculate_bmu(som_u.variable_u[column,:], u_subset)
            bmu_v[column,:]=calculate_bmu(som_v.variable_v[column,:], v_subset)


        node_number=np.argmin(bmu_u+bmu_v,axis=0)
        euclidean=np.min(bmu_u+bmu_v,axis=0)
        time1=xarray_u.time.values
        variable_name1='node_number'
        variable_name2='euclidean'

        node_dataset=create_multi_xarray_time_series(time1,variable_name1,np.reshape(node_number,(node_number.shape[0],)),variable_name2,np.reshape(euclidean,(euclidean.shape[0],)))



#step6:
#save data
# writing a large list of the specifications to a file.
        directoryname='D:/Shun_Li/3.1 Node_dataset/'  # SHun Li to change
        filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+ str(area_number) + '.nc'
        node_dataset.to_netcdf(directoryname+filename)



#step7:
##   slicing the data into individual years
## & calculate the RFO for each year

        histogram=create_year_histogram(node_dataset,som_u.SOM_node.shape[0],year_array)
        histogram_RFO = histogram/365 *100

#step8:
#save data
# writing a large list of the specifications to a file.
        directoryname='D:/Shun_Li/3.2 Histogram/'  # SHun Li to change
        filename='histogram_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+ str(area_number) + '.nc'
        histogram.to_netcdf(directoryname+filename)


        filename='histogram_RFO_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+ str(area_number) + '.nc'
        histogram_RFO.to_netcdf(directoryname+filename)
 
        
        area_number+=1
