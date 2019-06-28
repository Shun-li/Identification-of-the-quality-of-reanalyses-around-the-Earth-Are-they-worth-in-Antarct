# -*- coding: utf-8 -*-

"""
This part applys ECMWF dataset to create an optimized self-organizing map and the best classification available for the wind fields.

Created on Fri February 22 12:00:00 2019

@author : Shun Li

"""






##################################################################################################



#Install packages:


import numpy as np
import pandas as pd
import xarray as xr # for analysisng the Multidimensional array
from netCDF4 import num2date # Network Common Data Form; array-oriented scientific data
from numpy import linalg as la # a linear algebra package
import matplotlib.pyplot as plt
import os  # import os commands for making paths
import cartopy.crs as ccrs  # cartopy : a map-making package; crs: coordinate reference system  
import cartopy.feature  # different cartopy.features:
                                      # Borders : country boundaries
                                      # Coastline : coastline, inclinding major islands
                                      # lakes : Natural and artifical lakes
                                      # land : Land polygons , including major isalnds
                                      # Ocean: Ocean polygons
                                      # Rivers: Single-line drainages,including lake centerlines
import sompak  # this is the Self-Organizing MAP package which calculates the SOM classification of the data; SOM:  an Unsupervised-learning clustering method
import scipy
import dask
import toolz




##################################################################################################


#### part 1


#Writing the relevant functions



# function1:
def read_u(year_array): # u is the east west velocity(速度)
    # this function reads in 10m zonal (east-west) velocity from ERA-Interim dataset  
    directoryname='D:/Shun_Li/1.1 dataset_ECMWF/'   
    variable_name='u10'
    filename=variable_name+'_resampledu_'+str(year_array[0])+'.nc'
    print('Reading year:'+str(year_array[0]))
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
    
    for year in year_array[1:]:
        print('Reading year:'+str(year))
        filename=variable_name+'_resampledu_'+str(year)+'.nc'
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
        ncid1=xr.merge((ncid1,ncid2))
        

   
    
###This part is to check the structure of data:
### Reading year 1981
###<xarray.Dataset>
###Dimensions:    (latitude: 91, longitude: 180, time: 1095)
###Coordinates:
### * time       (time) datetime64[ns] 1979-01-01 1979-01-02 ... 1981-12-31
###  * latitude   (latitude) float32 90.0 88.0 86.0 84.0 ... -86.0 -88.0 -90.0
###  * longitude  (longitude) float32 0.0 2.0 4.0 6.0 ... 352.0 354.0 356.0 358.0
###Data variables:
###    u10        (time, latitude, longitude) float64 -4.94 -5.099 ... -0.9837


    
    # output the raw xarray (ncid1) and anomaly patterns and the climatology
    return ncid1





# function2:
def read_v(year_array): # v is the north south velocity
    # this function reads in 10m meridional (north-south) velocity from ERA-Interim dataset
    directoryname='D:/Shun_Li/1.1 dataset_ECMWF/' 
    variable_name='v10'
    print('Reading year:'+str(year_array[0]))
    filename=variable_name+'_resampledu_'+str(year_array[0])+'.nc'  
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename)) 
    
    for year in year_array[1:]:
        print('Reading year:'+str(year))
        filename=variable_name+'_resampledu_'+str(year)+'.nc'
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename)) 
        ncid1=xr.merge((ncid1,ncid2))


    
    # output the raw xarray (ncid1) and anomaly patterns and the climatology
    return ncid1





# function3:
def anomaly_calculation(X):
    # This code reads in a matrix (2-d array) and removes the mean values from the rows.
    value_1d=np.mean(X,axis=1) 
                  # for m*n matrix:
                    #   axis has no number -- calculate the mean of all numbers;
                    #   axis = 0 -- calculate the mean of column and return 1*n matrix;
                    #   axis = 1 -- calculate the mean of row and return m*1 matrix
    sizes=X.shape #return the dimension of each latitude in a array
    values=np.tile(value_1d,[sizes[1],1]) #copy value_1d along the columns with size[1] times
    values=np.transpose(values) # transpose the values
    anomaly=X-values
    return  value_1d,values,anomaly

###An example below:

###Input:
### X = np.array([1,2,3],
###              [4,5,6])

###Output:
### value_1d:
###   array([2., 5.])
### values:
###   array([[2., 2., 2.],
###         [5., 5., 5.]])
### anomaly:
###   array([[-1.,  0.,  1.],
###         [-1.,  0.,  1.]])





# function4:
def Empirical_Orthogonal_Function(X):
    # this is doing Empirical Orthogonal Function - which is the same as Pricipal component analysis basically 
    # this code calculates the eigenvalues and eigenvectors 
    value_1d,values,anomaly=anomaly_calculation(X)
    S=np.dot(anomaly.T,anomaly)# Matrix multiplication
    [eigen_values,eigen_vectors]=la.eigh(S)
    # numpy.linalg.eigh() is sutable for symmetric matrices
    return eigen_values,eigen_vectors,anomaly

###An example below:

###Input:
### X = np.array([1,2,3],
###              [4,5,6])

### output:
### eigen_valus
###   array([0., 0., 4.])
### eigen_vectors
###   array([[-0.70710678,  0., -0.70710678],
###          [ 0. ,         1.,        0.  ],
###          [-0.70710678,  0.,  0.70710678]])
### anomaly:
###   array([[-1.,  0.,  1.],
###         [-1.,  0.,  1.]])





# function5:
def som_perturb(som_shape,circle_data,alpha1,alpha2,radius_factor1,radius_factor2,random1):    
    # this code takes inputs and uses them to create an optimized self-organizing map, best classification available
    
    # set sizes fo som shape
    print(som_shape)
    
    #stage1:
    # initialization into sompak.SOM object
    circle_som = sompak.SOM(data=circle_data, shape=som_shape, topology='rect', neighbourhood='gaussian',random=random1)
    
    #stage2:
    # Map training 
    circle_som.train(rlen=circle_data.shape[0]*1000, alpha=alpha1, radius=radius_factor1*som_shape[0]*som_shape[1])

    #stage3:
    # Quantization error:
    qerr = circle_som.qerror()
    print(qerr)
    
    circle_som.train(rlen=circle_data.shape[0]*500, alpha=alpha2, radius=radius_factor2*som_shape[0]*som_shape[1])
   # Quantization error:
    qerr = circle_som.qerror()
    print(qerr)
    
   #stage4:
   # Extract the code vectors
    cvec = np.squeeze(circle_som.code_vectors())    
   # The SOM mappings; the mappings associate each input data element (municipality in our case) to one of the SOM neurons
    map_code =np.asarray(circle_som.mappings())

    return qerr,cvec,map_code





# function6:     
def reform_grid(input_array,mask_subset,original_shape):
    #this reformats data into a nice simple pattern based on the topography mask 
    final_array=np.ones(original_shape)*np.nan
    final_array[:,mask_subset]=input_array
    return final_array

###An example below:
###Input:
### input_array = np.array([[1,2,3,4],
###                         [2,3,4,5],
###                         [3,4,5,6]])
### original_shape = (3,9)
### mask_subset = np.array([2,3,4,5])

###Output:
###  array([[nan, nan,  1.,  2.,  3.,  4., nan, nan, nan],
###         [nan, nan,  2.,  3.,  4.,  5., nan, nan, nan],
###         [nan, nan,  3.,  4.,  5.,  6., nan, nan, nan]])

    



# function7:
def Antarctic_plot(xarray,subplot_val1,subplot_val2,subplot_val3):
    # basically makes a set of nice plots over the Antarctic given an xarray input and some idea of the position in a grid
    ax1 = plt.subplot(subplot_val1,subplot_val2,subplot_val3, projection=ccrs.SouthPolarStereo(central_longitude=-180))
          # plt,subplot(x1,x2,x3)
                    # x1: the number of rows about subplots
                    # x2: the number of columns about subplots
                    # x3: the index subplot in each row
          # ccrs.SouthPolarStereo : according to "Cartopy projection list"
                    #  link: #  https://scitools.org.uk/cartopy/docs/v0.15/crs/projections.html    
          # central longitude keyword puts New Zealand up rather than down on plot
    kw = dict( central_latitude=-90, central_longitude=180) 
    ax1.set_extent([12,84,60,90],ccrs.PlateCarree())
    # set the extent of plot [145E,215E,55S,80S]
    ax1.add_feature(cartopy.feature.LAND,zorder=0)#add feature--Land

    

    #get size and extent of axes:
    axpos = ax1.get_position()
    pos_x = axpos.x0 + axpos.width + 0.01 # + 0.25 * axpos.width
    pos_y = axpos.y0
    cax_width = 0.04
    cax_height = axpos.height
    cs = ax1.pcolormesh(xarray.longitude.values,xarray.latitude.values,xarray.variable.values,cmap='seismic', transform=ccrs.PlateCarree(),vmin=-8.0,vmax=8.0) 
      # draw a quadrilateral mesh
               #pcolormesh(x,y,Z,camp,vmin,vmax)
                   # camp:colormaps_reference.py； 
                          #link:  https://matplotlib.org/examples/color/colormaps_reference.html
                   #vmin; vmax: normalize luminance data   
                   
    ax1.coastlines()
    ax1.gridlines()
    return cs





# function8:
def region_plot(xarray,subplot_val1,subplot_val2,subplot_val3,longitude,latitude):
    # makes a set of nice plots over the different regions given an xarray input and some idea of the position in a grid around all of the world
    
    ax1 = plt.subplot(subplot_val1,subplot_val2,subplot_val3, projection=ccrs.PlateCarree(central_longitude=longitude+30))
    ax1.set_extent([longitude,longitude+60,latitude-30,latitude],ccrs.PlateCarree())
    ax1.add_feature(cartopy.feature.LAND,zorder=0)
    axpos = ax1.get_position()
    pos_x = axpos.x0 + axpos.width + 0.01 # + 0.25 * axpos.width
    pos_y = axpos.y0
    cax_width = 0.04
    cax_height = axpos.height
    cs = ax1.pcolormesh(xarray.longitude.values,xarray.latitude.values,xarray.variable.values,cmap='seismic',transform=ccrs.PlateCarree(),vmin=-8.0,vmax=8.0)              
    ax1.coastlines()
    ax1.gridlines()
    return cs




#function9:
def create_xarray_time_series(time,variable_name,variable):
    #creates xarray time series
    #The dictionnary keys are the variables contained in the Dataset.
    #The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    
    d = {}
    d['time'] = ('time',time) # keys
    d[variable_name] = (['time'], variable) # value
    dset = xr.Dataset(d)
    return dset





 

# function10:
def create_lat_lon_xarray(lat,lon,variable_name,variable):
    #create a latitude, longitude xarray object that can be written as a netcdf file if required
    #The dictionnary keys are the variables contained in the Dataset.
    #The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself

    d = {}
    d['latitude'] = ('latitude',lat) # keys
    d['longitude'] = ('longitude', lon) # keys
    d[variable_name] = (['latitude','longitude'], variable) # values
    dset = xr.Dataset(d)
    return dset





# function11:
def create_SOM_xarray(SOM_node,pattern,variable_name,variable):
    #The dictionnary keys are the variables contained in the Dataset.
    #The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself

    d = {}
    d['SOM_node'] = ('SOM_node',SOM_node) # keys
    d['pattern_index'] = ('pattern_index', pattern) # keys
    d[variable_name] = (['SOM_node','pattern_index'], variable) # values
    dset = xr.Dataset(d)
    return dset





# function12:
def map_code_to_node(map_code,som_shape):
    # this just reformats data into a nice simple pattern
    map_code=np.floor(map_code) #return the maximim integer that is not greater than the input parameter
    map_int_code=map_code.astype('int')
    node_number=np.ones((map_code.shape[0],1))*np.nan  
    node_value=0
    for i in range(0,som_shape[0]):
        for j in range(0,som_shape[1]):
            print('%3d %3d' %(i,j))            
            tmp_index=np.logical_and((map_int_code[:,0]==i),(map_int_code[:,1]==j))  
            node_number[tmp_index]=node_value 
            node_value=node_value+1
    return node_number

###An example below:
###Input:
### map_code = np.array([[0.2,3.5],
###                     [2.4,-2.4]])
### som_shape = (3,4)

##Output:
###  array([[ 3.],
###         [nan]])






##################################################################################################


#### part 2


## 1. pre-processing data 


# step1:
# define the period of analysis    
year_array=np.arange(1979,2014+1)


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
        longitude_1 = longitude_1 + 40
        if longitude_1 ==320 + 40:
            xarray_u_1 = xarray_u_0.u10.sel(longitude = slice(320 + 40 ,360),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_u_2 = xarray_u_0.u10.sel(longitude = slice(0,320-300 + 40),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_u = xr.merge((xarray_u_1,xarray_u_2))
                    
            xarray_v_1 = xarray_v_0.v10.sel(longitude = slice(320 + 40 ,360),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_v_2 = xarray_v_0.v10.sel(longitude = slice(0,320-300 + 40),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_v = xr.merge((xarray_v_1,xarray_v_2))
            
                    
            
            topography_data_1= topography_data_0.hgt.sel(lon = slice(320 + 40 ,360),lat=slice(latitude_1,latitude_1-30 ))   
            topography_data_2= topography_data_0.hgt.sel(lon = slice(0,320-300 + 40),lat=slice(latitude_1,latitude_1-30 ))
            topography_data = xr.merge((topography_data_1,topography_data_2))
            mask=(topography_data.hgt[0,:,:]<500.0)  #this makes a mask which identifies regions of high topography 
                                                         #for exclusion anything above 500m is excluded            
        else:
            xarray_u_1 = xarray_u_0.u10.sel(longitude=slice(longitude_1,longitude_1 + 30),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_u_2 = xarray_u_0.u10.sel(longitude=slice(longitude_1 + 30,longitude_1 + 60),latitude=slice(latitude_1,latitude_1-30 )) 
            xarray_u = xr.merge((xarray_u_1,xarray_u_2))
            
            
            xarray_v_1= xarray_v_0.v10.sel(longitude=slice(longitude_1,longitude_1 + 30),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_v_2 = xarray_v_0.v10.sel(longitude=slice(longitude_1+30,longitude_1 + 60),latitude=slice(latitude_1,latitude_1-30 ))
            xarray_v = xr.merge((xarray_v_1,xarray_v_2))
            
            
            
            topography_data_1=topography_data_0.hgt.sel(lon=slice(longitude_1,longitude_1 + 30),lat=slice(latitude_1,latitude_1-30 ))
            topography_data_2=topography_data_0.hgt.sel(lon=slice(longitude_1+30,longitude_1 + 60),lat=slice(latitude_1,latitude_1-30 ))
            topography_data = xr.merge((topography_data_1,topography_data_2))
            
            mask=(topography_data.hgt[0,:,:]<500.0)  #this makes a mask which identifies regions of high topography 
                                                         #for exclusion anything above 500m is excluded          
           
           
   
   
   
   
   
   
           
           

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


# step6:
# data masking
#creating numpy subset for mask
                 
        mask_subset=mask.values[:,:]

# creating input vector for EOF analysis  (same as Principal component analysis)

        masked_u=u_subset[:,mask_subset]  
        # just because we have a problem with nans!!!
        
# creating input vector for EOF analysis (same as Principal component analysis)
        masked_v=v_subset[:,mask_subset]  
        # just because we have a problem with nans!!!


# step7:
# EOF analysis
# creating the input for the EOF analysis
        pca_input=np.concatenate((masked_u,masked_v),axis=1)

### An example for numpy.concatenate function:
### cocatenating array:
###           axis = 0: concatenate according to column
###           axis = 1: concatenate according to row
###eg:
### a = np.array([[1, 2], [3, 4]])
### b = np.array([[5, 6]])
### np.concatenate((a, b), axis=0)
###Output:
### array([[1, 2],
###        [3, 4],
###        [5, 6]])
### np.concatenate((a,b),axis = 1)
###Output:
### array([[1, 2, 5],
###        [3, 4, 6]])

# do EOF analysis on the combined zonal and meridional velocities
        [eigen_values,eigen_vectors,anomaly]=Empirical_Orthogonal_Function(pca_input)


# step8:
# PCA analysis

# calculate Principal components
        PC=np.dot(anomaly,eigen_vectors[:,0])
        for i in range(1,anomaly.shape[1]):
            tmp=np.dot(anomaly,eigen_vectors[:,i])
            PC=np.concatenate((PC,tmp))
        PC_reshape=np.reshape(PC,(eigen_vectors.shape[0],anomaly.shape[0]))
        PC_reshape=PC_reshape.T
        PC_reshape=PC_reshape[:,::-1]  # reverse order for later as biggest PC at end
        eigen_vectors=eigen_vectors[:,::-1]  # reverse order for later


# step9:
# Truncating datasets

# the whole point of EOF analysis is to reduce the dimensionality of the dateset and also
# removing noisy less important information
# identify where to truncate datasets
        truncation_test=(np.cumsum(eigen_values[::-1]/np.sum(eigen_values)))
        truncation_threshold=0.9
        truncation_index=np.argmax(truncation_test>truncation_threshold)







## 2. SOM Analysising

# step1:
#define shape
        som_shape=(4,3)  # identify how big the SOM is and the coordinates for the SOM


# step2:
# setting initial value
# leave these factors alone as these are generally the best to make the SOM work
        alpha1=0.5
        radius_factor1=0.2
        number=5
        qerror1=np.zeros((number,1))*np.nan
        i=0
        radius_factor2=radius_factor1/2
        alpha2=0.05
        random_seed = 999 # variable


# step3:
# complete SOM Analysis using SOMPAK C++ code using the python fundtion som_perturb
        qerr,cvec,map_code=som_perturb(som_shape,PC_reshape[:,:truncation_index],alpha1,alpha2,radius_factor1,radius_factor2,random_seed) 


# step4:
# plot out the error to see if things look ok
        print(qerr)    
        print(cvec)
        print(map_code)


#step5:
# reprojects the data into a nice grid after the SOM analysis
        reprojected=np.dot(PC_reshape[:,:truncation_index],eigen_vectors[:,:truncation_index].T)    
        reprojected_u=reprojected[:,:int(reprojected.shape[1]/2)]
        reprojected_v=reprojected[:,int(reprojected.shape[1]/2):]

#
#seting the area number
        area = str(area_number) 


# step6:
# Graphing meridinal velocity
# makes a pretty graph for the meridional velocity and writes it to file
        fig=plt.figure(figsize=(290/25.4,280/25.4))  #smaller than A4 which would be 210 297
        iter=0
        for i in range(0,cvec.shape[0]):
            for j in range(0,cvec.shape[1]):
                cvec_reprojected=np.dot(cvec[i,j,:truncation_index],eigen_vectors[:,:truncation_index].T)        
                cvec_reprojected_v_reform=reform_grid(cvec_reprojected[int(reprojected.shape[1]/2):],mask_subset,v_subset.shape)
                iter=iter+1
                variable_input=cvec_reprojected_v_reform[0,:,:]+year_climatology_v[:,:]
                cs=region_plot(variable_input,cvec.shape[0],cvec.shape[1],iter,longitude_1,latitude_1)
                plt.title('Node'+str(iter),fontsize=24)
        fig.subplots_adjust(right=0.8)
        cs_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar=fig.colorbar(cs, cax=cs_ax)
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=20)
        cbar.set_label('v (m/s)', rotation=90,fontsize=24)
        
        plt.savefig('RISES7_ECMWF_v_figure'+ area + '.png',bbox_extra_artists=(cbar))  
        
#close picture to make sure each pictures will not be overlappe
        plt.pause(1)
        plt.close()

# step7:
# Graphing zonal velocity
# makes a pretty graph for the zonal velocity and writes it to file
        fig=plt.figure(figsize=(290/25.4,280/25.4))  #smaller than A4 which would be 210 297
        iter=0
        for i in range(0,cvec.shape[0]):
            for j in range(0,cvec.shape[1]):
                cvec_reprojected=np.dot(cvec[i,j,:truncation_index],eigen_vectors[:,:truncation_index].T)        
                cvec_reprojected_u_reform=reform_grid(cvec_reprojected[:int(reprojected.shape[1]/2)],mask_subset,u_subset.shape)
                iter=iter+1
                variable_input=cvec_reprojected_u_reform[0,:,:]+year_climatology_u[:,:]
                cs=region_plot(variable_input,cvec.shape[0],cvec.shape[1],iter,longitude_1,latitude_1)
                plt.title('Node'+str(iter),fontsize=24)
        fig.subplots_adjust(right=0.8)
        cs_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar=fig.colorbar(cs, cax=cs_ax) 
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=20)
        cbar.set_label('u (m/s)', rotation=90,fontsize=24)
       
        plt.savefig('RISES7_ECMWF_u_figure'+ area + '.png',bbox_extra_artists=(cbar)) 
        
#close picture to make sure each pictures will not be overlapped
        plt.pause(1)
        plt.close()

# step8:
# create files that are used in RMSD_plot1 and Euclidean_plot_ts2 programs
        variable_u=np.ones((cvec.shape[0]*cvec.shape[1],cvec_reprojected_u_reform.shape[1]*cvec_reprojected_u_reform.shape[2]))
        variable_v=np.ones((cvec.shape[0]*cvec.shape[1],cvec_reprojected_u_reform.shape[1]*cvec_reprojected_u_reform.shape[2]))

        iter=0
        for i in range(0,cvec.shape[0]):
            for j in range(0,cvec.shape[1]):
                cvec_reprojected=np.dot(cvec[i,j,:truncation_index],eigen_vectors[:,:truncation_index].T)    
                cvec_reprojected_u_reform=reform_grid(cvec_reprojected[:int(reprojected.shape[1]/2)],mask_subset,u_subset.shape)
                cvec_reprojected_v_reform=reform_grid(cvec_reprojected[int(reprojected.shape[1]/2):],mask_subset,v_subset.shape)        
                variable_u[iter,:]=np.ndarray.flatten(cvec_reprojected_u_reform[0,:,:]+year_climatology_u.values[:,:])
                variable_v[iter,:]=np.ndarray.flatten(cvec_reprojected_v_reform[0,:,:]+year_climatology_v.values[:,:])
                                                      
                iter=iter+1

        directoryname='D:/Shun_Li/2.1 SOM_pattern_figure/'  # Shun LI will need to change
        variable_name='variable_u'
        SOM_pattern_u=create_SOM_xarray(np.arange(0,cvec.shape[0]*cvec.shape[1]),np.arange(0,variable_u.shape[1]),variable_name,variable_u)
        filename=variable_name+'_RISES' + area + '.nc'
        SOM_pattern_u.to_netcdf(directoryname+filename)

        variable_name='variable_v'
        SOM_pattern_v=create_SOM_xarray(np.arange(0,cvec.shape[0]*cvec.shape[1]),np.arange(0,variable_v.shape[1]),variable_name,variable_v)
        filename=variable_name+'_RISES' + area + '.nc'
        SOM_pattern_v.to_netcdf(directoryname+filename)
        
        area_number+=1


