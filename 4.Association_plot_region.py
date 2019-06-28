# -*- coding: utf-8 -*-
"""

Read in all the euclidean distance based time series of Best MAtcing Unit and EUclidean ditance from 
ERA-INterim and ECMWF. Then make some nice time series diagrams anda histgram plot.



Created on Thu April 4 12:00:00 2019

@author: Shun li


"""


##################################################################################################



#Install packages:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os  # import os commands for making paths
import datetime as datetime
import seaborn as sns
import time





##################################################################################################


#### part 1


## step 1:

# Regrouping ERA20C dataset according to different regions
for state in (1,2,3,4,5,6):
    for region in (1,  2, 3, 4, 5, 6,
                   7,  8, 9,10,11,12,
                   13,14,15,16,17,18,
                   19,20,21,22,23,24,
                   25,26,27,28,29,30,
                   31,32,33,34,35,36):
        directoryname1='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/time/1900-1920/'
        filename='Node_dataset_ECMWF-1900-1919' + '_RISES'+ str(region) + '.nc'
        ncid1 =xr.open_dataset(os.path.join(directoryname1,filename))
        directoryname_1='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/region/region'+ str(region) +'/'
        ncid1.to_netcdf(directoryname_1+filename)
  
        directoryname2='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/time/1920-1940/'
        filename='Node_dataset_ECMWF-1920-1939' + '_RISES'+ str(region) + '.nc'
        ncid2 =xr.open_dataset(os.path.join(directoryname2,filename))
        directoryname_2='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/region/region'+ str(region) +'/'
        ncid2.to_netcdf(directoryname_2+filename)
    
        directoryname3='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/time/1940-1960/'
        filename='Node_dataset_ECMWF-1940-1959' + '_RISES'+ str(region) + '.nc'
        ncid3 =xr.open_dataset(os.path.join(directoryname3,filename))
        directoryname_3='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/region/region'+ str(region) +'/'
        ncid3.to_netcdf(directoryname_3+filename)
        

        directoryname4='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/time/1960-1980/'
        filename='Node_dataset_ECMWF-1960-1979' + '_RISES'+ str(region) + '.nc'
        ncid4 =xr.open_dataset(os.path.join(directoryname4,filename))
        directoryname_4='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/region/region'+ str(region) +'/'
        ncid4.to_netcdf(directoryname_4+filename)
        
   
        directoryname5='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/time/1980-2010/'
        filename='Node_dataset_ECMWF-1980-2009' + '_RISES'+ str(region) + '.nc'
        ncid5 =xr.open_dataset(os.path.join(directoryname5,filename))
        directoryname_5='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/region/region'+ str(region) +'/'
        ncid5.to_netcdf(directoryname_5+filename)
     
 


        
##################################################################################################   


## step 2:
    
# Writing some relevant functions


# function1:
def read_ERA20C_data(state,region):
#reading ERA20C dataset    
    year_array=np.arange(1900,1919+1)
    
    directoryname='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/region/region'+str(region)+'/'
    filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+str(region)+'.nc'
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    for year_span in np.arange(0,4):
        if(year_span==0):
            year_array=np.arange(1920,1939+1)
        elif(year_span==1):
            year_array=np.arange(1940,1959+1)
        elif(year_span==2):
            year_array=np.arange(1960,1979+1)            
        else:
            year_array=np.arange(1980,2009+1)
        # writing xarray to a netcdf file.
        filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+str(region)+'.nc'
        
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
        ncid1=xr.merge((ncid1,ncid2))
        
    return ncid1









# function2:

def read_ECMWF_data(state,region):
#reading ECMWF dataset    
    year_array=np.arange(1979,2015)
    
    directoryname='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ECMWF/1979-2015/' 
    filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+ str(region)+ '.nc'
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    

    return ncid1














# function3:
def contigency_entropy(x,y):  
#Given a two-dimensional contingency table in the form of an integer array nn[i][j], where i labels the x variable and ranges from 1 to ni, j labels the y variable and ranges from 1 to nj,
#this routine returns the entropy h of the whole table, the entropy hx of the x distribution, the entropy hy of the y distribution, the entropy hygx of y given x, the entropy hxgy of x given y,
#the dependency uygx of y on x (eq. 14.4.15), the dependency uxgy of x on y (eq. 14.4.16),and the symmetrical dependency uxy (eq. 14.4.17).

#  See details from "14.4 Contingency Table Analysis of Two Distributions", 
#  from Chapter 14 "Statistical Description of Data",
#  Book Title: Numerical Recipes in C. The Art of Scientific Computing, 2nd Edition, 1992, ISBN 0-521-43108-5.
#  Also some konwledge can be found from Lecture 21 in course STAT101-Statistics from university of canterbury.


# part 1:  Measure of Association based on Chi-Square:
    
    table=np.asarray(pd.crosstab(x,y))
    
    ni=table.shape[0]
    nj=table.shape[1]

# get the row totals
    sumi=np.sum(table,1)
    
# get the column totals
    sumj=np.sum(table,0)

# get the whole totals
    sum_total=np.sum(table)

# expectation  
    expectation=np.zeros((ni,nj))

    for i in np.arange(0,sumi.shape[0]):   # rows
        for j in np.arange(0,sumj.shape[0]):   # columns
            expectation[i,j]=(sumi[i]*sumj[j])/sum_total

#  chi^2 calculation           
    chi2=np.sum( ( (table-expectation)**2.0)/expectation) # Equation (14.4.3). 
    
# cramer's V
    cramer=np.sqrt(chi2/(sum_total*min(ni-1,nj-1))) # Equation (14.4.4).

# the property of cramer's V: 
  # equal to zero, no association
  # equal to one, perfect association                              





# part 2:  Measure of Association based on Entropy:
# see section 14.4 in Numerical Recipes

    TINY=1.0E-06
    table=np.asarray(pd.crosstab(x,y))
    
    ni=table.shape[0]
    nj=table.shape[1]

# get the row totals
    sumi=np.sum(table,1)
        
# get the column totals
    sumj=np.sum(table,0)
    
# get the whole totals
    sum_total=np.sum(table)

    
# Entropy of the x distribution
    hx=0.0  
    for i in np.arange(0,ni):
        if (sumi[i]):
            p=sumi[i]/sum_total # Equation (14.4.8).
            hx =hx -p*np.log(p) # Equation (14.4.9).
            
# Entropy of the y distribution        
    hy=0.0  
    for j in np.arange(0,nj):
        if (sumj[j]):
            p=sumj[j]/sum_total # Equation (14.4.8).
            hy =hy- p*np.log(p) # Equation (14.4.9).
            
# Total entropy: loop over both x and y    
    h=0.0
    for i in np.arange(0,ni): 
        for j in np.arange(0,nj):
            if (table[i,j]):
                p=table[i,j]/sum_total 
                h =h- p*np.log(p) # Equation (14.4.10).
                
   
# entropy of y given x            
    hygx=(h)-(hx) # Equation (14.4.18)
    
# entropy of x given y
    hxgy=(h)-(hy) # Equation (14.4.18)

#uncertainty coefficient of y 
# or called dependency of y on x
    uygx=(hy-hygx)/(hy+TINY) # Equation (14.4.15).
    
#uncertainty coefficient of x
# or called dependency of x on y
    uxgy=(hx-hxgy)/(hx+TINY) # Equation (14.4.16).
    
#combination of uncertainty coefficient 
# or called combination of dependency
    uxy=2.0*(hx+hy-h)/(hx+hy+TINY) # Equation (14.4.17).

# dependency    
    dependency=uxy
    
    return table,cramer,dependency














# function4:
def create_xarray_association_series(time,variable_name1,variable1,variable_name2,variable2):
#The dictionnary keys are the variables contained in the Dataset.
#The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    d = {}
    d['time'] = ('time',time)
    d[variable_name1] = ('time', variable1)
    d[variable_name2] = ('time', variable2)
    dset = xr.Dataset(d) 
    return dset













#  function5：
def create_year_association(node_number1,node_number2,year_array):
    
    # create histogram for yearly slices based on som number
    cramer=np.zeros(year_array.shape[0])
    dependency=np.zeros(year_array.shape[0])
    i=0
    for year in year_array:
        time_slice1=str(year)+'-01-01'
        time_slice2=str(year)+'-12-31'
        selection1=node_number1.sel(time=slice(time_slice1,time_slice2))
        selection2=node_number2.sel(time=slice(time_slice1,time_slice2))
        table,cramer[i],dependency[i]=contigency_entropy(selection1,selection2)
        i=i+1
     
        
     # create time array
    start_year=str(year_array[0])+'-01-01'
    end_year =str(year_array[-1]+1)+'-01-01'
    time_array=pd.date_range(start_year,end_year,freq='BAS-JUL')#creatine a datetimeindex


    # create output xarray for graphing
    variable_name1='cramer'
    variable_name2='dependency'
    association_xarray=create_xarray_association_series(time_array,variable_name1,cramer,variable_name2,dependency)
    return association_xarray

















# function6:
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)













##################################################################################################


#### part2

# The process of analysis

for state in (1,2,3,4,5,6):
    for region in (1,  2, 3, 4, 5, 6,
                   7,  8, 9,10,11,12,
                   13,14,15,16,17,18,
                   19,20,21,22,23,24,
                   25,26,27,28,29,30,
                   31,32,33,34,35,36):


#step1:
#reading ECMWF dataset
        year_array_ECMWF=np.arange(1979,2015)
        ncid_ECMWF=read_ECMWF_data(state,region)


#step2:
#reading ERA20C dataset
        year_array_ERA20C=np.arange(1900,2010)
        ncid_ERA20C=read_ERA20C_data(state,region)


#step3:
#set the color index 
        colors = cm.rainbow(np.linspace(0, 1, 10))


#step4:
#set the font and pass the font dict as kwargs
        font = {'family' : 'normal',
               'weight' : 'medium',
                'size'   : 14}
        plt.rc('font', **font)
        plt.figure(figsize=(297/25.4,160.0/25.4))  #smaller than A4 which would be 210 297


#step5:
#plot the results
        output=create_year_association(ncid_ERA20C.node_number,ncid_ECMWF.node_number,year_array_ECMWF[:-5])  # hardwiring here.
        plt.plot(output.time,output.dependency.values,color=colors[6,:],marker="^", linestyle="none")
        output.dependency[-19]=np.nan
        plt.plot(output.time,output.dependency.values,color=colors[6,:],marker="^",label="ERA20C vs. ERA-Interim",linewidth=3,markersize=7)


#step6:
#seting the parameters of results
        plt.xlabel('Year')
        plt.ylabel('Dependency')
        plt.legend(loc='upper left')
        plt.grid()


# step7:
#saving results
        save_result_to ='D:/Shun_Li/5.1 Association_plot_region/'+str(state)+'.State'+str(state)+ '/'
        filename =  'Association_plot_region'+str(region)+'.png'
        plt.savefig(save_result_to + filename,dpi=100) 
        
#close picture to make sure each pictures will not be overlapped
        plt.pause(1)
        plt.close()
    