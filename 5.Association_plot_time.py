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







#### part 1



# function:
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









##################################################################################################


#### part2

# The process of analysis

#step1:
#select the year 

for state in (1,2,3,4,5,6):
    for year in range(1980,2010):
        list1 =[]
        list2 =[]    
        for region in ( 1, 2, 3, 4, 5, 6,
                        7, 8, 9,10,11,12,
                       13,14,15,16,17,18,
                       19,20,21,22,23,24,
                       25,26,27,28,29,30,
                       31,32,33,34,35,36):
        
#step2:        
# selcet the specific year from dataset ERA20C
            year_array=np.arange(1980,2010)
            directoryname='D:/Shun_Li/3.1 Node_dataset/'+ str(state)+'.State'+str(state)+'/ERA20c/time/1980-2010/' 
            filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+ str(region)+ ".nc"
            node_dataset =xr.open_dataset(os.path.join(directoryname,filename)) 
            selection1 = node_dataset.sel(time =slice( str(year) +"-01-01", str(year)+"-12-31"))
        

#step3:
# selcet the specific year  from dataset ECMWF
            year_array=np.arange(1979,2015)
            directoryname='D:/Shun_Li/3.1 Node_dataset/'+ str(state)+'.State'+str(state)+'/ECMWF/1979-2015/' 
            filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+ str(region)+ ".nc"
            node_dataset =xr.open_dataset(os.path.join(directoryname,filename)) 
            selection2 = node_dataset.sel(time =slice( str(year) +"-01-01", str(year)+"-12-31"))
 
       
#step4:
#geting related dependency 
            table,cramer,dependency=contigency_entropy(selection1.node_number,selection2.node_number)
            list1.append(dependency)   
            list2.append(cramer)


#step5:
# change result to a 2-D array and change some columns to fit the world-map
        array1= np.array(list1).reshape(6,6)
        array1[:, [0,-1]] = array1[:,[-1,0]]
    
        array2= np.array(list2).reshape(6,6)
        array2[:, [0,-1]] = array2[:,[-1,0]]    
    
#step6:
# plot a heatmap for dependency
        fig,ax = plt.subplots(figsize = (297/25.4,160.0/25.4))
        sns.heatmap(pd.DataFrame(np.round(array1,4)),annot=True, vmax=1,vmin=0,xticklabels =False,yticklabels =False,cmap ='Blues')
    
        name = "Global Heatmap for dependency in" + str(year) +", ECMWF VS ERA20C" 
        ax.set_title(name)  
        ax.set_ylabel('Latitude',fontsize = 20)
        ax.set_xlabel('Longitude',fontsize = 20)
        plt.show()
    
    
    
#step7:
#saving pictures
        filename = "ECMWF_ERA20C_dependency_time_" + str(year) +".png"
        save_result_to ='D:/Shun_Li/5.2 Association_plot_time/'+str(state)+'.State'+str(state)+ '/'
        plt.savefig(save_result_to + filename,dpi=100) 
            
            
            
           
                       
#step8:
#close picture to make sure each pictures will not be overlapped
        plt.pause(1)
        plt.close()



#step9:
# plot a heatmap for cramer'V
        fig,ax = plt.subplots(figsize = (297/25.4,160.0/25.4))
        sns.heatmap(pd.DataFrame(np.round(array2,4)),annot=True, vmax=1,vmin=0,xticklabels =False,yticklabels =False,cmap ='Greens')
    
        name = "Global Heatmap for cramer'V in" + str(year) +", ECMWF VS ERA20C" 
        ax.set_title(name)     
        ax.set_ylabel('Latitude',fontsize = 20)
        ax.set_xlabel('Longitude',fontsize = 20)
        plt.show()
        
        
#step10:
#saving pictures
        filename = "ECMWF_ERA20C_cramer_time_" + str(year) +".png"
        save_result_to ='D:/Shun_Li/5.2 Association_plot_time/'+str(state)+'.State'+str(state)+ '/'
        plt.savefig(save_result_to + filename,dpi=100)             
                        
         #close picture to make sure each pictures will not be overlapped
        plt.pause(1)
        plt.close()
    
    
    







