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
# ## FDNote1W.ipynb
# Prepared by Inmoo Lee for the Financial Databases class at KAIST
# inmool@kaist.ac.kr
#
# List of input files
#
#     - note1.txt
#     - note1test.xlsx
#     - note1data.xlsx
#     - note1.RData
#
# List of output files
#
#     - note1_out.xlsx
#     - note1.pkl
#     - note1.ft
#     

# %% [markdown]
# ### Package 
#
# To use packages, you need to install them first.
#
# Reference the following for the installation of pakcages 
# https://www.youtube.com/watch?v=Z_Kxg-EYvxM 
#
# When you have problems with installing packages using conda, reference the following
#
# https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html

# %% [markdown]
# ## Set the working environment
#
# If a pakcage is not installed, then you have to install it first using "!pip install ***" or "conda install ***" in Ananconda Powershell Prompt

# %%
import os #import a package called os

print(os.getcwd())  #get the current working directory

# %%
# path="d:\\****\\"# specify the path to change to in your computer
# os.chdir(path) # change the working directory

from pathlib import Path

CWD_PATH = Path.cwd()

# %% [markdown]
# ## Data input/import
#
# ##### Some basic commands
#

# %%
counter=100 #define the value of a variable
miles=1000.0
name="John"

print (counter) #print out to the output screen
print(miles)
print(name)

# %%
# Check the type of the variables
print(type(counter))
print(type(miles))
print(type(name))

# %%
list = [ 'abcd', 786 , 2.23, 'john', 70.2 ] #series
tinylist = [123, 'john']

print(list) # Prints complete list
print(tinylist) #Prints list two times

print(list[1]) #Prints first element of the list
print(list[1:4]) #Prints elements starting from 2nd till 3rd
print(list[2:]) #Prints elements starting from 3rd element
print(tinylist*2) #Prints list two times
print(list + tinylist) #Prints concatenated lists

# %%
# Check the type of the list variables
print(type(tinylist))

# %% [markdown]
# ### Dictionary
#
# Composed of keys and values

# %%
dict = {} # Create an empty dictionary
dict['one'] = "This is one"
dict[2]     = "This is two"

tinydict = {'name': 'john','code':6734, 'dept': 'sales'}


print (dict['one'])       # Prints value for 'one' key
print (dict[2])           # Prints value for 2 key
print (tinydict)          # Prints complete dictionary
print (tinydict.keys())   # Prints all the keys
print (tinydict.values()) # Prints all the values
print(dict.keys())
print(dict.values())

# %%
# Check the type of the dictionary variables
print(type(dict))
print(type(tinydict))

# %% [markdown]
# ## Create and read data
#
# ####  Method 1: Create one directly
#
# First, import packages after installing packages (if not installed yet, you can install inside jupyternote by using the following command
#
# !pip install numpy
#
# )
#
# Combine arrays and create a Pandas dataframe

# %%
import numpy as np
import pandas as pd

# %%
d=np.array((1,2,3,4,5))
print(d)
e=np.array(("M", "F", "F", "M", "F"))
f=np.array((68.5, 50.2, 45.5, 80.5, 55.0))
g=np.array((175, 150, 155, 165, 160))
print(np.column_stack((d,e,f,g)))

test1=pd.DataFrame(data=np.column_stack((d,e,f,g)),columns=["ID", "GENDER", "WEIGHT", "HEIGHT"])
print(test1)

# %%
print(type(g))

# %% [markdown]
# ### Choose a particular column or row in a dataframe
#
# **loc** gets rows (or columns) with particular labels from the index.
# **iloc** gets rows (or columns) at particular positions in the index (so it only takes integers).

# %%
print(test1.columns)
print(test1.loc[0:0,:])
print(test1.iloc[0:1,:])
print(test1.iloc[:,0:2])
print(test1.loc[:,['ID','HEIGHT']])
print(test1.iloc[:,0:2])

# %% [markdown]
# ###  Method 2: Read from a text file

# %%
test2=pd.read_csv("./note1.txt", sep=" ", header=None, names=["ID", "GENDER", "WEIGHT", "HEIGHT"])
print(test2)
type(test2) #print out the type of variable, series or file

#test2.index=["row1", "row2", "row3", "row4","row5"]#for the naming of rows

# %% [markdown]
# ###  Method 3: Read from an Excel file
#

# %%
test3=pd.read_excel('./note1test.xlsx', sheet_name="TEST3", header=0)
print(test3)
print(test3.columns)

# %% [markdown]
# ### Save the data to a DataFrame file using pandas
#
# Compare different formats: e.g., check the class note for comparisons of alternative file formats (CSV, Excel, JSON, HDF5, Feather, Parquet and Pickle)

# %%
# To save the dataframe to an pickle format file
test3.to_pickle("./note1.pkl") #To save in the pickle format

# %%
# To use different file formats such as feather or parquet, you need to install the respective libraries.
# # !pip install pyarrow

# %%
#Alterantivealy, you can use feather
test3.to_feather("./note1.ft")


# %% [markdown]
# ### How to retrieve the saved data?

# %%
## you can call the saved file in the following way
test41=pd.read_pickle("./note1.pkl")
test42=pd.read_feather('./note1.ft')
print(type(test41))
print(type(test42))

print(test3)
print(test41)
print(test42)

# %% [markdown]
# ### How to save to an Excel file?

# %%
#To save to a new excel file, you can use the following command

test3.to_excel('./note1_out.xlsx', sheet_name='TEST4', index=False) #do not include index in the excel

# %% [markdown]
# ### Conduct a simple analysis#
#
# #### Add a category variable
#
# define "STATURE" based on the value of HEIGHT
#

# %%
print(test1.dtypes) # print the type of each columns
print(test2.columns) #print the columns of the dataframe
print(test2) #print the dataframe

# Add a new column 'STATURE' based on the condition of HEIGHT
# If HEIGHT is less than 160, then STATURE is 'Short', otherwise 'Tall
test2.loc[:,'STATURE']=np.where((pd.to_numeric(test1.loc[:,'HEIGHT']) < 160), 'Short', 'Tall')
#######################
print(test2) #Check the updated dataframe with the new column


# %% [markdown]
# ### Alternatively, you can define a function to do the same
#
# Define func() to catogorize and then use **apply** to apply the function to a column of a dataframe

# %%
#Define a function to categorize height
# If HEIGHT is less than 160, then STATURE is 'Short', otherwise 'Tall
def func(x):            #x is an input
    if int(x) < 160:    #int() convert x to integer value
        return 'Short'  #assign the value to the function and return it
    else:
        return 'Tall'

print(test1) #print the original dataframe
# Apply the function to the HEIGHT column and create a new column STATURE
test1.loc[:,'STATURE'] = test1.loc[:,'HEIGHT'].apply(func)
print(test1) #Check the updated dataframe with the new column

# %% [markdown]
# ### How to sort the data by the values of a column or columns?

# %%
#Sort values by 'GENDER' in ascending order.
testout=test1.sort_values(["GENDER",'WEIGHT'],ascending=True)
print(testout)

# %% [markdown]
# ### Calculate group means and standard deviations

# %%
#Convert an object type to a numeric value
print(testout.dtypes)
testout["WEIGHT"]=pd.to_numeric(testout['WEIGHT'])
testout["HEIGHT"]=pd.to_numeric(testout['HEIGHT'])
print(testout.dtypes)


# %%
# After defining a dataframe grouped by variables of your choide, 
# you can use the agg() function to calculate various statistics
# such as sum, mean, median, min, max, and std for the grouped data
grouped=testout.groupby(['GENDER'])[['WEIGHT','HEIGHT']]
print(grouped.agg(['sum','mean','median',\
                   'min','max','std']))

# %%
# Alternatively, you can use the following commands to get the statistics
# for each group
print(testout.groupby(["GENDER"])[['WEIGHT','HEIGHT']].mean())
print(testout.groupby(["GENDER"])[['WEIGHT','HEIGHT']].std())
print(testout.groupby(["GENDER"])[['WEIGHT','HEIGHT']].min())
print(testout.groupby(["GENDER"])[['WEIGHT','HEIGHT']].max())

# %% [markdown]
# ## The following are available in groupby
#
# https://pandas.pydata.org/pandas-docs/stable/groupby.html
#
#     -gb.agg        gb.boxplot    gb.cummin     gb.describe   gb.filter     
#     -gb.get_group  gb.height     gb.last       gb.median     gb.ngroups    
#     -gb.plot       gb.rank       gb.std        gb.transform
#     -gb.aggregate  gb.count      gb.cumprod    gb.dtype      gb.first      
#     -gb.groups     gb.hist       gb.max        gb.min        gb.nth        
#     -gb.prod       gb.resample   gb.sum        gb.var
#     -gb.apply      gb.cummax     gb.cumsum     gb.fillna     gb.gender     
#     -gb.head       gb.indices    gb.mean       gb.name       gb.ohlc       
#     -gb.quantile   gb.size       gb.tail       gb.weight
#

# %% [markdown]
# ### Generate frequency 2&2 frequency table

# %%
# Check how the output changes as you use different parameters
print(pd.crosstab(testout.GENDER, testout.STATURE, margins=False))
print(pd.crosstab(testout.GENDER, testout.STATURE, margins=True))
print(pd.crosstab(testout.GENDER, testout.STATURE, margins=True,normalize=True))
print(pd.crosstab(testout.GENDER, testout.STATURE, margins=True,normalize='all')) #Sum of all will be 1
print(pd.crosstab(testout.GENDER, testout.STATURE, margins=True,normalize='index')) #Sum of row values will be 1
print(pd.crosstab(testout.GENDER, testout.STATURE, margins=True,normalize='columns')) #Sum of column values will be 1

# %% [markdown]
# ## Plot histogram

# %%
# To visualize the data, you can use matplotlib
# Please make sure you have matplotlib installed
# If not, you can install it using the following command
# # !pip install matplotlib

# %%
import matplotlib.pyplot as plt

##plot histogram where you can specify the number of bins or the bin edges
#plt.hist(testout['WEIGHT'],bins=5)
plt.hist(testout['WEIGHT'],bins=[40,50,60,70,80,90,100])
plt.title('Histogram with auto bins')
plt.show()

# %% [markdown]
# ### Transpose the data
#
# Use **melt** to tranpose the data
#
# Date column will become the first column, column name will become 'id' and column values will be called 'return'

# %%
##################Read the data and transpose for easier analysis################
sample=pd.read_excel('./note1data.xlsx', sheet_name="SAMPLE", header=0)
print(sample) #print the sample dataframe
print(sample.columns) # print the columns of the sample dataframe
print(os.getcwd()) #print out the current working directory

print(sample.iloc[0:4,0:3]) #print out first 4 rows and first 3 columns

# %%
#Use pd.melt to transform the dataframe from wide format to long format
sampletran=pd.melt(sample,id_vars=['Date'],var_name='id',value_name='return')
print(sampletran.columns)# Check how the columns are transformed
print(sampletran) #print the transformed dataframe
print(sampletran.head(10)) #print out the first 10 rows
print(sampletran.tail(8)) #print out the last 8 rows

# %%
# To add a new worksheet to an existing excel file
# Create a Pandas Excel writer using Openpyxl as the engine.
import openpyxl
writer = pd.ExcelWriter('./note1_out.xlsx', engine='openpyxl', mode='a')

# Get the list of existing worksheets
print(writer.book.sheetnames)

# Write the DataFrame to the new worksheet
sampletran.to_excel(writer, sheet_name='sampletran', startrow=0, startcol=0, index=False)

# Save the changes to the Excel file
writer.close()

# %% [markdown]
# # SQL
#
# Let's import RData (data saved in R) to import Rdata
#
# Install **pyreadr** using the following command if not installed yet

# %%
# # !pip install pyreadr

# %%
import pyreadr #package used to read RData, data used in R

# %%
data1=pyreadr.read_r('.\\note1.RData') # Read the RData file
# The result is a dictionary with the dataframes as values
print(type(data1))
print(data1)

# %%
# Check the keys of the dictionary to see the names of the dataframes
print(data1.keys())

# %%
# Access the specific dataframe from the dictionary
# Here, 'mdata' is the name of the dataframe in the RData file
data2=data1['mdata']#mdata is the table in RData
print(type(data2)) #Check the type of the diciontary's values
print(data2.head()) # #print the first 5 rows of the dataframe

# %%
print(data2.iloc[1:10,]) #check the first 10 rows of the dataframe
print(data2.columns)

# %% [markdown]
# ### How to run SQL in Python
#
# There are different ways to run SQL.  We will use pandasql that is convenient to run sql using pandas dataframes
#
# Functions available in pandasql
#
#     -AVG() – returns the average value of a group.
#     -COUNT() – returns the number of rows that match a specified condition
#     -MAX() – returns the maximum value in a group.
#     -MIN() – returns the minimum value in a group
#     -SUM() – returns the sum of values
#

# %% [markdown]
# https://community.alteryx.com/t5/Data-Science-Blog/pandasql-Make-python-speak-SQL/ba-p/138435
#
# First install pandasql if not installed yet

# %%
# # !pip install pandasql

# %%
# In the pandasql package, sqldf is a function that allows you to run SQL queries on pandas dataframes
# You can import it as follows
from pandasql import sqldf

# %%
#Define a query statement to be used inside the sqldf function

query1='''select a.id,avg(a.return) as mean, 
                 min(a.return) as minimum, max(a.return) as maximum, 
                 count(a.return) as number
             from data2 as a 
             group by a.id
             order by a.id'''

# %% [markdown]
# #### locals() vs. globals()
#
# When you use sqldf(query statement, locals/globals), you have to specify
# whether the variables are used locally or globally by specifying either locals() or globals() inside sqldf()
#
# When pandasql needs to have access to other variables in your session/environment, you can pass locals() to pandasql when executing a SQL statement.
#
# But if you're running a lot of queries requiring locals(), it might be a pain. 

# %%
# When locals() are specified, the variables are accessed from the local scope
# When globals() are specified, the variables are accessed from the global scope
df1=sqldf(query1,locals())
print(df1)


# %% [markdown]
# ### Alternatively, you can define a function including globals()
#
# To avoid passing locals all the time, you can add this helper function to your script to set globals() like so:
#
# Define your custom function 

# %%
# you can define a function including globals()
# To avoid passing locals all the time, you can add this helper function to your script to
def pysqldf(q):
 return sqldf(q, globals())


# %% [markdown]
# If you use locals() above instead of globals(), you will get error message since it does not recognize data2 that is defined outside function.

# %%
# You can use your defined function to run the SQL query that was defined earlier
meansql=pysqldf(query1)
print(df1) #Query output using sqldf(locals())
print(meansql) #Query output using pysqldf(globals())
# The output should be the same as df1

# %%
#You can defined a query statement inside the pysqldf function
pysqldf('''select a.id,avg(a.return) as mean, 
                 min(a.return) as minimum, max(a.return) as maximum, 
                 count(a.return) as number
             from data2 as a 
             group by a.id
             order by a.id''')

# %%
#You can also store the output of the query directly into a dataframe.
mdata1=pysqldf('''select a.id,a.date,a.return from data2 as a order by a.id,a.date''')
print(data2.columns)
print(mdata1)

# %% [markdown]
# ### Calculate group means

# %%
# Calculate group means using groupby and agg functions
# You can use the groupby function to group the data by 'id' and then calculate
grouped=data2.groupby('id',observed=False)['return']#observed=False means that even the ones with ID caltegory missing will be included
print(grouped.agg(['mean','min','max','std','count']))

# %%
#For a more detailed description of the grouped data, you can use the describe() function
# This will give you a summary of the statistics for each group
print(data2.groupby('id',observed=False)['return'].describe())

# %%
#Alternatively, you can calculate the mean, min, max, std, and count for each group using the following commands

print('mean:  ', data2.groupby('id',observed=False)['return'].mean())
print('min:  ', data2.groupby('id',observed=False)['return'].min())
print('max:  ', data2.groupby('id',observed=False)['return'].max())
print('std:  ', data2.groupby('id',observed=False)['return'].std())
print('count:  ', data2.groupby('id',observed=False)['return'].count())
print('count-alt:  ', data2[(data2.id !=0)].groupby('id',observed=False)['return'].count()) #count only non-zeros

###compare the results with the meansql
print(meansql)
#######################################################################################
