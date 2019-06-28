import numpy as np
import pandas as pd
import xarray as xr
#from netCDF4 import num2date
from numpy import linalg as la
import matplotlib.pyplot as plt
import os  # import os commands for making paths
import cartopy.crs as ccrs
import cartopy.feature
import sompak
import patsy
import statsmodels.api as sm
import seaborn as sn




#select the year and region
state =6
for year in range(1980,2010):
    for region in (1, 2, 3, 4 , 5, 6,
                   7, 8, 9, 10,11,12,
                   13,14,15,16,17,18,
                   19,20,21,22,23,24,
                   25,26,27,28,29,30,
                   31,32,33,34,35,36):
        
        
# selcet the specific year from dataset ERA20C
        year_array=np.arange(1980,2010)
        directoryname='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ERA20c/time/1980-2010/' 
        filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+ str(region)+ ".nc"
        node_dataset =xr.open_dataset(os.path.join(directoryname,filename)) 
        node_dataset1 = node_dataset.sel(time =slice( str(year) +"-01-01", str(year)+"-12-31"))
        dataframe1 = node_dataset1.to_dataframe()



# selcet the specific year from dataset ECMWF
        year_array=np.arange(1979,2015)
        directoryname='D:/Shun_Li/3.1 Node_dataset/'+str(state)+'.State'+str(state)+'/ECMWF/1979-2015/'  
        filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES'+ str(region)+ ".nc"
        node_dataset =xr.open_dataset(os.path.join(directoryname,filename)) 
        node_dataset2 = node_dataset.sel(time =slice( str(year) +"-01-01", str(year)+"-12-31"))
        dataframe2 = node_dataset2.to_dataframe()
        dataframe2 = dataframe2.rename(columns ={"node_number":"node_number2","euclidean":"euclidean2"})



#join the two dataframes into ones
        dataframe1["node_number2"] = dataframe2.node_number2
        dataframe_result = dataframe1.drop("euclidean",axis = 1)
        dataframe_result = dataframe_result.rename(columns ={"node_number":"node_number_ERA20C","node_number2":"node_number_ECMWF"})


#create contingency
        tab = pd.crosstab(dataframe_result["node_number_ERA20C"],dataframe_result["node_number_ECMWF"],margins= True,margins_name = "total")
        table = sm.stats.Table(tab)


#measure of association in contingency
        print(table.table_orig)
        print(table.fittedvalues)
        print(table.resid_pearson)
        print(table.chi2_contribs)


        
#save the contingency table
        table_result = sn.heatmap(pd.crosstab(dataframe_result["node_number_ERA20C"],dataframe_result["node_number_ECMWF"],margins= True,margins_name = "total"),cmap="YlGnBu",annot=True,cbar= False)
        filename = "ECMWF_ERA20C_region" + str(region) + "_year" +str(year) +".png"
        figure = table_result.get_figure()
        save_result_to ='D:/Shun_Li/4.1 contingency_table/'+str(state)+'.State'+str(state)+'/region'+ str(region)+ '/'
        plt.savefig(save_result_to + filename,dpi=100)                          

                       

#close picture to make sure each pictures will not be overlapped
        plt.pause(1)
        plt.close()






