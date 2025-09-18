# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# #### FDNote2W_Ex2025
#
# Prepared by Inmoo Lee for the Financial Databases class at KAIST
#
# inmool@kaist.ac.kr
#
# List of input files
#
#     - note2data.xlsx
#
# List of output files
#
#     - noteex2.csv
#     - noteex2.xlsx

# %%
import os
import numpy as np
import pandas as pd

print(os.getcwd())  #get the current working directory
# path='D\\####'#Choose your directory to work with
# os.chdir(path) # change the working directory
# print(os.getcwd())  #get the current working directory

# %%
###import the excel file
sampletrans=pd.read_excel('./note2data.xlsx', sheet_name="sampletran", header=0)
print(sampletrans.dtypes)
samplesort=sampletrans.sort_values(['id','Date']) # sort by id and date
print(samplesort.head(10))
print(samplesort.id.drop_duplicates()) #print unique ids
print('Min return:  ', samplesort['return'].min())
print('Max return:  ', samplesort['return'].max())

# %%
##Select only those that satisfy a condition
#select the data for MSFT, IBM, and WMT
#You can use the isin() function to filter the data based on a list of values
mdata=samplesort[samplesort.id.isin(['MSFT','IBM','WMT'])]
print(mdata.head())
mdata.reset_index(drop=True,inplace=True) #reset the index of each row
#inplace=True changes the original dataframe
#drop=true eliminate the original index

print(mdata.head())

# %%
#summaryze : descriptive statistics
print(mdata['return'].describe())
#You can get the summary statistics for each group using the groupby function

#Here, you first created a grouped object by grouping the data by 'id' and then selecting the 'return' column.
#Then, you can use the describe() function to get a summary of the statistics for each
grouped=mdata.groupby(['id'])['return']
print(type(grouped))
print(grouped.describe())
print(grouped.count())

print(grouped.agg(['sum','mean','median',\
                   'min','max','std']))
print(grouped.agg(['sum','mean','std']).rename(columns={'sum': 'total','mean':'avg','std':'sd'}))

# %%
##Alternatively, you can use the following without creating a grouped object
print(mdata.groupby('id')['return'].describe())
print(mdata.groupby('id')['return'].mean())

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Classify into two groups based on returns

# %%
### print(mdata.dtypes) #make sure that return is a numeric column

#print(mdata.drop(columns='isign',inplace=True))

# %%
#You can create a new column based on a condition
#For example, you can create a new column 'isign' that indicates whether the return
test=mdata.copy()
test.loc[test['return']>=0.0,'isign']='POS'
test.loc[test['return']<0.0,'isign']='NEG'
print(test.head())

# %%
#Alternatively, you can use np.where to create a new column based on a condition
#This is more efficient for larger datasets
test1=mdata.copy()
test1.loc[:,'isign']=np.where(test1['return']>=0.0,'POS','NEG')
print(test1.head())

# %%
mdata=test.copy()#Copy the modified dataframe back to mdata

# %%
#generate frequency 2&2 frequency table
print(pd.crosstab(index=mdata["id"],columns='id'))
print(pd.crosstab(index=mdata["isign"],columns='isign'))
print(pd.crosstab(index=mdata["id"],columns=mdata["isign"]))

# %%
#Check the data type of the 'Date' column
print(mdata.Date.dtype)

# %%
# Change the format of date using pd.to_datetime (it is recognized as date, time in pd)
print(mdata.Date.head())

# Convert the 'Date' column to datetime format
# Specify the format of the date string if necessary
mdata.loc[:,'Date']=pd.to_datetime(mdata.loc[:,'Date'],format='%Y%m%d')
print(mdata.Date.head())

# %%
##You can add columns in the following way, rather than add them one by one

#First, import the datetime module
from datetime import datetime as dt

# Using datetime functions to extract year, month, and day
# and create new columns in the DataFrame
print(mdata.head())
mdata=mdata.join(pd.DataFrame(
        { 'year':mdata['Date'].dt.year,
          'month':mdata['Date'].dt.month,
          'day':mdata['Date'].dt.day,
         }, index=mdata.index
        ))
print(mdata.head())

# %%
#########################
# You can report isign Ways for each year, for all, or for a specific id

print(pd.crosstab(index=mdata["year"],columns=mdata["isign"]))
print(pd.crosstab(index=mdata[mdata.id=='MSFT']['year'],columns=mdata["isign"]))
print(pd.crosstab(index=mdata[mdata.id=='IBM']['year'],columns=mdata["isign"]))
print(pd.crosstab(index=mdata[mdata.id=='WMT']['year'],columns=mdata["isign"]))

# %%
##Alternatively, you can use the groupby function to count the number of occurrences
print(mdata.groupby(['id','isign','year'])['return'].count())

# %%
#save the data to a csv file#
mdata.to_csv('./noteex2.csv')

#alterantively, you can save it in an excel file 
mdata.to_excel('./noteex2.xlsx', sheet_name='n2ex2')
