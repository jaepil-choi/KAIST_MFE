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
# # FDNote5W.ipynb
#
# Prepared by Inmoo Lee for the Financial Databases class at KAIST
#
# inmool@kaist.ac.kr
#
# For portfolio return calculations using SQL
#
# Input files used
#
#     - 2020SP500Constituents_2025_Short.xlsx
#     - return_data.ft
#     - note4data.xlsx
#     - fbhrs.ft
#     - Note3w_RHistory2025_Short.xlsx
#     

# %%
import os #import a package called os

# os.getcwd()  #get the current working directory
# path='D:\\####'#Change this to your directory
# os.chdir(path) # change the working directory

# %%
import numpy as np
import pandas as pd
from datetime import datetime as dt

# %% [markdown]
# # Calculate portfolio returns using SQL
#
# Equally-weighted vs. value-weighted returns
#

# %%
#Calculate the equally-weighted average return in each month
from pandasql import sqldf
def pysqldf(q):
 return sqldf(q, globals())


# %%
#Read the data
df=pd.read_feather('./return_data.ft')

# %%


from pandas.tseries.offsets import MonthEnd

## As discussed before, we can find the the month end any # of months before/after
### you have to convert the floating 64 format Date to integer and then to string format
### to use pd_to_datetime

df.loc[:,'date1']=pd.to_datetime(round(df.loc[:,'Date']).astype(int).astype(str),format='%Y%m%d')+MonthEnd(0)
###find the date 12 months after (+12) and before (-12) the current date
df.loc[:,'fmonth12']=df.loc[:,'date1']+MonthEnd(+12)
df.loc[:,'bmonth12']=df.loc[:,'date1']+MonthEnd(-12)
print(df[['date1','fmonth12','bmonth12']].head())

# %%
###find yyyymm to be used as an time indicator (year+month)
### since date1 is a datetime variable, dt.year and dt.month
### can be used to get year and month from a date.
df.loc[:,'yyyymm']=df.loc[:,'date1'].dt.year*100+df.date1.dt.month
print(df[['yyyymm']].head())

# %% [markdown]
# ## Portfolio return calculation

# %%
#Let's import the market cap information 
marketc=pd.read_excel('./note4data.xlsx', sheet_name="marketcap", header=0)


# %%
from pandas.tseries.offsets import MonthEnd

# convert "Date" to a datetime variable recognized in Pandas and call it "rdate"
# make a time indicator variable (yyyymm), which is same the one as used above
marketc['rdate']=pd.to_datetime(marketc['Date'].astype(int).astype(str),format='%Y%m%d')+MonthEnd(0)
marketc['yyyymm']=marketc.rdate.dt.year*100+marketc.rdate.dt.month
print(marketc.head())

# %% [markdown]
# #### Find lagged market cap values and ids
# Here, by using *groupby(['ID'])*, lagged values are *correctly* identified for each id
#
# It is also important to note that the file should be **sorted by (ID and Date)** before lagged values are identified.
#

# %%
print(marketc.columns)
# Calculate the market capitalization by multiplying Price and Numsh
marketc['marketcap']=marketc.Price * marketc.Numsh

# Sort the marketcap data by ID and Date
# This is important to use .sort_values() before using .shift()
# After sorting, reset the index to avoid confusion
marketcap=marketc.sort_values(['ID','Date'],ascending=True).reset_index(drop=True)
# Check whether the sorting is correct
print(marketcap.head())
##Here, it is important to use .shirt(1) with groupby(['ID']) 
### to correctly get the lagged market cap for each ID
### Otherwise, it may use the other ID's market cap as its lagged value

## shift(1) means the previous row in the group
## If you want to use the next row, use shift(-1)
marketcap['lmc']=marketcap.groupby(['ID'])['marketcap'].shift(1)#lagged marketcap
marketcap['lid']=marketcap.groupby(['ID'])['ID'].shift(1)

# %%
# Check whether the lagged market cap is correct
print(marketcap[['ID','Date','marketcap','lmc','lid']].head())
print(marketcap.loc[160:180,['ID','Date','marketcap','lmc','lid']])

# %%
# Read fbhrs.ft, which we created last week.
fbhrs=pd.read_feather('./fbhrs.ft')

# %%
# /*add market cap information to bhr file*/
# Use the SQL to combine two dataframes.
# Note that two dataframes are merged by matching ids and yyyymm values
query='''select a.*,b.marketcap,b.lmc,b.rdate
            from fbhrs as a
            left join marketcap as b
            where a.id = b.ID and a.yyyymm = b.yyyymm
            order by a.id, a.yyyymm'''
#Run the query and store the result in bhrmc
bhrmc=pysqldf(query)

print(bhrmc.head())
print(bhrmc.describe())
print(bhrmc['bhr1y'])
print(bhrmc['bhr1y'].tail(20))

# %% [markdown]
# ## Value-weighted returns
#
# $R_{vw}=\sum_{i=1}^{n} w_i \times R_i$ where $w_i= \frac{MC_i}{\sum_{i=1}^{n} MC_i} $
#
# Therefore, $R_{vw}=\sum_{i=1}^{n} \frac{MC_i}{\sum_{i=1}^{n} MC_i} \times R_i = \frac{\sum_{i=1}^{n} MC_i \times R_i}{\sum_{i=1}^{n} MC_i}$
#
# One thing you have to be aware of is the fact that the market cap used here should be the market cap at the **beginning of the month (lmc)**, not at the end of the month.  

# %%
# /*calculate equally-weighted and value-weighted returns of three stocks */
# Note that sum are calculated in each group specified in "group by"(i.e., in each month (yyyymm)).
# In other words, equally- and value-weighted returns are calculated in each yyyymm

query='''select a.yyyymm, sum(a.bhr1y)/count(a.bhr1y) as ewbhr1y,
                sum(a.bhr1y * a.lmc) / sum(a.lmc) as vwbhr1y, 
                count(a.bhr1y) as numstock
            from bhrmc as a
            group by a.yyyymm
            order by a.yyyymm'''

# Run the query and store the result in portret
portret=pysqldf(query)

# Check the first 20 rows and the last few rows of portret
print(portret.head(20))
print(portret.tail())

# %%
# /*Compare average equally- and value-weighted returns*/
# The following calcualte simple averages of monthly equally- and value-
# weighted returns calculated above across all year/month

query='''select avg(a.ewbhr1y) as avgew, avg(a.vwbhr1y) as avgvw
            from portret as a'''
summary=pysqldf(query)
print('Average using SQL   :\n', summary.head())
print('Average using mean   :\n', portret[['ewbhr1y','vwbhr1y']].mean())
print(portret.describe())

# %% [markdown]
# ## Form portfolios based on market cap
#
# Check which company has larger cap during the sample period

# %%
#Find the average lagged market cap for each id
# and call it avgmarketcap
# Note that avgmarketcap is the average of lagged market cap (lmc) for each ID acorr time

query='''select b.id, avg(b.lmc) as avgmarketcap 
            from marketcap as b group by b.id'''

summc=pysqldf(query)
print('Average calculated using sql: \n',summc)
print('Average calculated using mean()/groupby(): \n',marketcap.groupby('ID').lmc.mean())

# %% [markdown]
# #### Find out the size group using only the market caps in Jan. 2001

# %%
# Assign firms into a size portfolio based on the market cap on 20010131

# First, find out the median of the market cap in Jan. 2001 and call it med
bhrmc['med']=np.median(bhrmc[bhrmc['yyyymm']==200101]['marketcap'])# calcualte the median market cap in Jan 2001

# Check the median value
print(bhrmc.med.describe())
print(bhrmc.head(200))
#check the median market cap in Jan 2001 to see whether med is correct
print(bhrmc.loc[bhrmc.yyyymm==200101,['marketcap']].describe())

# Check the median market cap and market cap of each firm in Jan 2001
print(bhrmc[bhrmc.yyyymm==200101][['med','marketcap','id']])

# %%
#Assign firms into a size portfolio based on the market cap on 20010131

# Use SQL for the portfolio assignment
# find the list of ids available in Jan. 2001 and add a size indicator (isize)
# based on the median market cap in Jan. 2001 found above ("med") (1=large, 2=small)

# Note that "case when" is used to assign a value based on a condition
# If the condition is true, it returns 1, otherwise it returns 2

query='''select a.id, 
                case when a.marketcap >= a.med then 1 else 2 end as isize
            from bhrmc as a
            where a.yyyymm = 200101
            order by a.id'''

# Run the query and store the result in isize
isize=pysqldf(query)

# Check the columns and the first few rows of isize
print(isize.columns)
print(isize)

# %%
# Combine the above result with the bhrmc dataframe by matching ids
# (adding isize to bhrmc file)

query='''select a.*,b.isize
            from bhrmc as a
            left join isize as b
            on a.id = b.id
            order by a.id, a.yyyymm'''
bhrmc2=pysqldf(query)

# Confirm that isize is added to bhrmc2
print(bhrmc2)

# %% [markdown]
# ### Calculate summary statistics for each isize group

# %%
# Use SQL to summarize the average, maximum, minimum, and count of bhr1y
# grouped by isize (size portfolio)

query='''select a.isize,avg(a.bhr1y) as avg, max(a.bhr1y) as max, min(a.bhr1y) as min,
                count(a.bhr1y) as num
            from bhrmc2 as a
            group by a.isize'''
summary=pysqldf(query)
print(summary)

#Alternativley, you can use the following code to summarize the average, maximum, minimum, and count of bhr1y
# grouped by isize (size portfolio)
print(bhrmc2.groupby('isize')['bhr1y'].describe())

# %% [markdown]
# # Calendar-time portfolio formation and return calculation
#
# ### Calendar time portfolio is a portfolio formed in each month
# #### In each month, the portfolio is composed of firms that had an event within the past 24 mont
#
# A firm is included **only once** even if it had multiple events within the past 24 months.

# %%
#Read the data

events=pd.read_excel('./note4data.xlsx', sheet_name="Events", header=0)
print(events.dtypes)
print(events)

# %%
#/*combine the event data with the return data*/
# Match by ids only

query='''select a.*,b.rdate,b.return,b.lmc,b.marketcap
            from events as a
            left join marketcap as b 
            on a.id=b.id
            order by a.id, a.date,b.date'''

eventret=pysqldf(query)

print(eventret.head())
# Check the number of rows and columns for eventret and two input dataframes
print("Number of rows and columns in eventret:", eventret.shape)
print("Number of rows and columns in events:", events.shape)
print("Number of rows and columns in marketcap:", marketcap.shape)

#This is not a Cartesian product, but a join
# Therefore, the number of rows in eventret is NOT the product of
# the number of rows in events and marketcap

print(events.shape[0]*marketcap.shape[0])

# %%
## convert the date format of Date in eventret
## to a datetime variable and call it "edate" (event date)

## In addition, call the return date as "rd" (return date)

## Note that yyyymm indicates year and month of return date (not event date)
eventret['edate']=pd.to_datetime(eventret['Date'].astype(int).astype(str),format='%Y%m%d')

eventret['rd']=pd.to_datetime(eventret['rdate'])

##yyyymm is the year-month of returns
eventret['yyyymm']=eventret.rd.dt.year*100+eventret.rd.dt.month
print(eventret.head())

# %%
from dateutil.relativedelta import *

## You can use the "relativedelta" fundtion to find out
## the date corresponding to a certain number of months 
#  plus (or minus) a certain date.  
# (MonthEnd() is not used here since event date is not necessarily the end of a month)

##Find out the date 24 months after the event date

## "e24" is the date 24 months after the event date
## eyyyymm and e24yyyymm are year and months of event date and 24 months after the corresponding event date.


#appy.(lambda x: f(x)) is used to run f(x) function 
# for each observation of a column (here, eventret.edate)


eventret['e24']=eventret.edate.apply(lambda x: x+relativedelta(months=+24))
#check whether e24 is correct
print(eventret[['edate','e24']])

# eyyyymm and e24yyyymm are year and months of event date and 24 months after the corresponding event date.
eventret['eyyyymm']=eventret.edate.dt.year*100+eventret.edate.dt.month
eventret['e24yyyymm']=eventret.e24.dt.year*100+eventret.e24.dt.month

#Check the output of eventret
print(eventret.head())
print(eventret.tail())
print(eventret[['ID','edate','yyyymm','e24','rd','eyyyymm','e24yyyymm']])

# %%
## Compare the results with those using MonthEnd()
eventret['e24a']=eventret.edate+MonthEnd(+24)
print(eventret[['edate','e24','e24a']])

# %% [markdown]
# ## Calculate calendar time portfolio return
#
# The portfolios are composed of firms with the events that occurred within the past 24 months
#
# Find out the list of the stocks that satisfy the condition in each month
#
# In each month, id will be listed if it satisfy the condition in where i.e., return date is within the 24-month window starting from the month after event month.
#
# #### **distinct** is used in the "select" statement to select only unique observations (prevent same values to be selected multiple times.

# %%
#Save the eventret dataframe to an Excel file
# This will create an Excel file named 'eventret.xlsx' in the current directory
eventret.to_excel('./eventret.xlsx')

# %%
print(eventret.head())

# %%
##Notice the use of "distinct" here
## If not used, it can list the same yyyymm id row mutiple times

## Note that the conditions in the "where" statement dictates which observations to include
## based on yyyymm, event yyyymm (eyyyymm) and 24 months after the event yyyymm (e24yyyymm)

print(eventret.columns)

query='''select distinct a.yyyymm, a.id
                  from eventret as a
                  where  a.yyyymm > a.eyyyymm and a.yyyymm <= a.e24yyyymm
                  order by a.yyyymm,a.id''' # a.yyyymm > a.eyyyymm 해준 것은 forward looking 안하기 위함. 
                  # distinct 는 event가 24m 윈도우 내에서 여러 번 발생하더라도 1 번만 남기기 위함. 

portdat=pysqldf(query)
print(portdat.head())
print(portdat.iloc[20:50,])

# %%
### We are now ready to calculate the calendar time portfolio returns
### We will calculate the equally- and value-weighted returns in each month

## calcualte the equally- and value-weighted calendar time portfolio returns in each month
## after joining market cap information.

query='''select a.yyyymm, sum(b.return)/count(b.return) as ewret,
                  sum(b.return*b.lmc)/sum(b.lmc) as vwret, count(b.return) as numstock
                from portdat as a 
                left join marketcap as b
                on  (a.id=b.id and a.yyyymm=b.yyyymm)
                group by a.yyyymm
                order by a.yyyymm'''

calret=pysqldf(query)
print(calret.head())
print(calret.tail())
print(calret.describe())

# %% [markdown]
# # P/B portfolio formation and return calculation
#
# Input files used
#
#     - 2020SP500Constituents_2025_Short.xlsx
#
# We will make 3 dataframes (return, market cap and MB ratios) out of the input file and then combine them to calculate portfolio returns
#
# 2020SP500Constituents_2025.xlsx file includes the list of firms included in S&P 500 as of 2020 and other information of these firms (returns, market capitalization and market-to-book equity ratio) retrieved from Bloomberg in Excel as will be discussed in Note6W.  You will find that the information can be retrieved directly from Bloomberg using API as will be discussed in Note6W (and shown in FDNote6W2025.ipynb).

# %% [markdown]
# #### First, Make Return Dataframe

# %%
#########################################################
########read the return data
#########################################################
returns=pd.read_excel('./2020SP500Constituents_2025_Short.xlsx', sheet_name="Returns", header=0)
print(returns.columns[:5])
print(returns.iloc[:5,:5])

#discard the first three columns
returns=returns.iloc[:,3:].copy()#The first one is row and the second one is column
print(returns.columns[:5])
print(returns.iloc[:5,:5])


# %%
## select only those with available "Edate" information and rename "Edate" as "date"
ret0=returns.dropna(subset=['Edate']).copy()#drop all rows with Edate=NaN

#Rename
# it is important to use inplace=True to change the original dataframe
ret0.rename(columns={'Edate': 'date'}, inplace=True)

##check
print(ret0.shape)
print(ret0.columns)
print(ret0.columns.values[:5])

# %%
##sort by date
ret0sort=ret0.sort_values(['date'])

print(ret0sort.iloc[:5,:5])

# %% [markdown]
# #### Trasnpose the data to calculate portfolio returns
#
# In the current format, returns are across different columns.
# To calculate portfolio returns, it is easy to have returns in one column, not across different columns

# %%
##transpose the data (column names are called "id" and values are called "ret")
# pd.melt() is used to reshape the DataFrame
# It converts the DataFrame from wide format to long format

#id_vars=['date'] specifies the columns to keep as identifiers
#var_name='id' specifies the name of the new column that will contain the former column names
#value_name='ret' specifies the name of the new column that will contain the former values

returns0=pd.melt(ret0sort,id_vars=['date'],var_name='id',value_name='ret')

#check the first few rows and the rows with ret=-99
print(returns0.head())
print (returns0.loc[returns0['ret']==-99])

# %%
# Print the row at index 352
# This will show the data for the specific date and id at that index
print(returns0.iloc[352,:])

# %%
###Replace the "ret" column values of -99 as missing returns
returns0.loc[returns0['ret']==-99,'ret']=np.nan
### convert the return in % to decimal by dividing the return values by 100
returns0['ret']=returns0['ret']/100.0

#check
print(returns0.iloc[352,:])
print(returns0.head())
print(returns0.ret.describe())

# %%
#use query to check summary statistics of monthly returns

query='''
        select avg(a.ret) as avg, count(a.ret) as num,
            sum(a.ret) as sum, min(a.ret) as min, max(a.ret) as max
        from returns0 as a
        where a.ret not null
    '''
print(pysqldf(query))

# %% [markdown]
# #### Second, Make Market Cap Dataframe

# %%
## Read the market capitalization information 
## and drop the observations without Bdate and rename Bdate as date.
## Bdat is the date of market cap calculation in the "MarketCap" worksheet (Edate is a month after Bdate)

mc=pd.read_excel('./2020SP500Constituents_2025_Short.xlsx', sheet_name="MarketCap", header=0)
print(mc.columns[:5])
mc=mc.iloc[:,2:]#skip the first two columns
print(mc.columns[:5])

# %%
mc=mc.drop(['Edate'],axis=1) #drop the Edate column

##In the data, market cap is the market cap on Bdate
mc.rename(columns={'Bdate': 'date'}, inplace=True)#rename Bdate column as date
mc=mc.dropna(subset=['date'])#if date is missing, drop
mcsort=mc.sort_values(['date'])#sort by date

# %%
print(mcsort.iloc[:5,:5])

# %%
#pd.melt: This function is useful to massage a DataFrame into a format where one or more
# columns are identifier variables (id_vars), while all other columns, 
#considered measured variables (value_vars), are “unpivoted” to the row axis, 
#leaving just two non-identifier columns, ‘variable’ and ‘value’.

mc0=pd.melt(mcsort,id_vars=['date'],var_name='id',value_name='mcap')#change the format
mc0.loc[mc0['mcap']==0.0,'mcap']=np.nan
mc0sort=mc0.sort_values(['id','date']).reset_index(drop=True)


#check
print(mc.iloc[:5,:5])
#print(mc.columns.values[:5])
#print(mc.shape)

print(mc0sort.iloc[:5,])
#print(mc0.loc[mc0['mcap']==0.0])

# %% [markdown]
# #### Third, Make M/B Dataframe

# %%
#########for m/b  ###################################################################################

##In the MB worksheet, Edate is the date of market cap to book value information.

mb=pd.read_excel('./2020SP500Constituents_2025_Short.xlsx', sheet_name="MB", header=0)
print(mb.columns[:5])
mb=mb.iloc[:,2:]#skip the first two columns
print(mb.columns[:5])

# %%
## Process the MB data as we did for other data

mb.rename(columns={'Edate': 'date'}, inplace=True)#rename Edate columns
mb=mb.dropna(subset=['date'])#drop if date is missing
mbsort=mb.sort_values(['date'])# sort by date
mb0=pd.melt(mbsort,id_vars=['date'],var_name='id',value_name='mb')#change the format
mb0['year']=mb0['date'].dt.year#get year out of date and make it "year" column

#check
#print(mb.columns.values[:5])
print(mb.shape)
print(mb0.head())

# %% [markdown]
# #### Drop rows with missing mb values

# %%
print(mb0.shape)
mb1=mb0.dropna(subset=['mb'])#drop those with missing mb
print(mb1.shape)
print(mb1.groupby(['date'])['mb'].describe())

# %% [markdown]
# ##### Find the cutoff points for MB quintiles (5 groups)
#

# %%
## First, find out the percentile values of MB ratios in each month
mb2 = mb1.groupby(['date'])['mb'].describe(percentiles=[.2, .4, .6,.8]).reset_index()
print(mb2.head())

# Second, rename the columns for easier access
mb2 = mb2[['date','20%','40%','60%','80%']]\
.rename(columns={'20%':'quint20','40%':'quint40','60%':'quint60','80%':'quint80'})

print(mb2.columns)
print(mb2.head())

# %% [markdown]
# Add quintile cutoff points and divide firms in to 5 groups based on mb;
#

# %%
# add cutoff points to the original mb data
# Note that columns to be included are selected from mb0 to avoid
# date column being duplicated in the result (both a and b have the date column)
query='''select a.id,a.mb,a.year,b.*
              from mb0 as a 
              left join mb2 as b 
              on a.date = b.date'''
              
mb3=pysqldf(query)
print(mb3.columns)
print(mb3.head())

# %% [markdown]
# ### Use SQL to form portfolios and calculate returns of the portfolio composed of all stocks in each portfolio
#

# %%
#Find out which MB portfolios each stock belongs to in each year

# Assgin firms into 5 groups based on mb using SQL;
# Note that "case when" is used to assign a value based on a condition
# new column "pmb" is the portfolio based on mb

query='''select a.*,
                case when a.mb <= a.quint20 then 1 else 
                    case when a.mb <= a.quint40 then 2 else
                        case when a.mb <= a.quint60 then 3 else
                            case when a.mb <= a.quint80 then 4 else 5
                            end
                        end
                    end
                end as pmb
            from mb3 as a'''
mb3a=pysqldf(query)
print(mb3a.head())

# %% [markdown]
# # Combine 3 dataframes
#
# Add new variables
#
# To combine data for each month, we create year and month columns

# %%
#/*combine the data*/
mb5=mb3a[mb3a['mb'].notnull()].copy()# Get rid of ones with missing mb values

print(mb5.date)
#the following change the format of date to a simmpler datetime format
mb5['date']=pd.to_datetime((mb5['date']).astype(str))
print(mb5.date)

# %%
# Add year and month columns

returns0['year']=returns0['date'].dt.year
mc0sort['year']=mc0sort['date'].dt.year
mb5['year']=mb5['date'].dt.year

returns0['month']=returns0['date'].dt.month
mc0sort['month']=mc0sort['date'].dt.month
mb5['month']=mb5['date'].dt.month

print(returns0.head())
print(mb5.head())
print(mc0.head())

# %% [markdown]
# ### Combine three data sets 
#
# - yyyymm is the year-month of return
# - mcapym is the year-month of market cap to be used for weights in vw
# - mbym is the year-month of mb portfolio
#
# mb portfolio is formed at the end of June and this is used from July of the year till June of the following year in portfolio formation.
#
# Note the conditions used in join "on"
#
# - When you combine the return datafreame and the market cap dataframe, make sure that the market cap is the market cap at one month before the return month.
#     - ((a.year-b.year)*12+(a.month-b.month)) =1: To make sure that lagged market cap is indeed the market cap one month prior to the return month: 
#         - b (mc0sort)' year/month is the year/month for market cap and 
#         - a (returns0)'s year/month is the year/month for returns
# - When you combine the return dataframe and the MB portfolio dataframe, make sure that returns are included from July of the MB portfolio formation year and June of the following year (Portfolios are formed in June of each year)
#     - ((a.year-c.year)*12+(a.month-c.month)) between 1 and 12: To combine June BM of year t with returns from July of year t to June of year t+1: 
#         - c (mb5)'s year/month is the year/month for market-to-book ratio and 
#         - a (returns0)'s year/month is the year/month for returns (returns are included from one month after untill twelve months after the market-to-book calculation month.
#
# Check a simpler way used to retrive M/B ratios and other information using Bloomber API in the next week's notebook, FDNote6W2025.ipynb

# %%
### Combine the returns, market cap, and mb dataframes using SQL
# Note the conditions used in join "on" as explained above
# The conditions make sure that the mcap is the market cap one month prior to the return month
# and that returns are included from July of the MB portfolio formation year and June of the following

query='''select a.date,a.year*100+a.month as yyyymm,a.id
            ,a.ret,b.mcap,c.mb,c.pmb,
            b.year*100+b.month as mcapym,
            c.year*100+c.month as mbym
         from returns0 as a
         left join mc0sort as b on ((a.year-b.year)*12+(a.month-b.month)) =1 and a.id=b.id 
         left join mb5 as c on a.id=c.id and ((a.year-c.year)*12+(a.month-c.month)) between 1 and 12
         order by a.id,a.date'''
              
data=pysqldf(query)
print(data.head())
print(data[data.ret.notnull()].head())#print only those with non-missing ret
print(data.columns)

# %%
###check whether mcapym is the previous month (market cap), 
## and mbym (year/month of MB calculation) changes in July
print(data.loc[data.pmb.notnull(),['date','mcapym','mb','pmb','mbym','id','yyyymm','mcapym','mbym']].reset_index(drop=True).head(50))
#############################################################################             

# %% [markdown]
# #### Calculate the value-weighted and equally-weighted returns 
# using only thoe with available return and portfolio information
#

# %%
#/*calculate returns of the portfolio*/
##Note that in the data, mcap is the market cap at the end of the month prior to the return month
##Therefore, we do not need to use the lagged market cap.

#Include only thoes with non-missing return, pmb and market cap
#An additional condition is that pmb is not 0

data1=data.loc[(data.ret.notnull() & data.pmb.notnull() &\
                data.pmb!=0 & data.mcap.notnull()),:].reset_index(drop=True).copy()
# Check the number of rows and columns before and after filtering
print("Before filtering:", data.shape)
print("After filtering:", data1.shape)
print(data1.head())

# %%
# We are now ready to calculate the portfolio returns
# Note that the portfolio returns are calculated in each month (yyyymm)

# calculate equally- and value-weighted returns 
# of each pmb portfolio in each month

query='''select a.yyyymm, a.pmb, sum(a.ret)/count(a.ret) as ewret,
             sum(a.ret*a.mcap)/sum(a.mcap) as vwret,count(a.ret) as numstock
         from data1 as a
         group by a.yyyymm, pmb
         order by a.yyyymm, pmb'''              
ret=pysqldf(query)
print(ret.head(30))

# %%
##########################################################
###Calculate the average returns of each PMB group
###Note that below is the average of average returns
query='''select a.pmb, avg(ewret),avg(vwret),count(vwret) as num
             from ret as a
             group by pmb'''
avg=sqldf(query,locals())
print(avg)

# %%
print(ret.groupby('pmb')[['ewret','vwret']].mean())
print(ret.groupby('pmb')[['ewret','vwret']].describe())

##########################################################

# %% [markdown]
# # Appendix 
#
# Below are just for reference
#
# - How to calculate returns using price information
# - How to calcualte value-weighted returns using a function
# - How to form MB portfolios using Python
#

# %% [markdown]
# ## Appendix 1: Calculate returns using price information

# %%
########read the company id in the first row#########################################################
LowPEHead=pd.read_excel('./Note3w_RHistory2025_Short.xlsx', sheet_name="RHistory", skiprows=0,nrows=1,header=None)
print(LowPEHead.iloc[:,0:5].head())

# %%
########read the price data                 #########################################################
####Skip the first 3 rows
LowPE=pd.read_excel('./Note3w_RHistory2025_Short.xlsx', sheet_name="RHistory", skiprows=3,header=None)
print(LowPE.iloc[:,0:5].head())

# %%
#change the column headings
##replace column names with the one in LowPEHead
LowPE.columns=LowPEHead.iloc[0,:]
print(LowPE.columns)

# %%
##Rename the first column as date
LowPE.rename(columns={ LowPE.columns[0]: "date"}, inplace=True)
print(LowPE.shape)
print(LowPE.iloc[:,0:5].head())

# %%
print(LowPE.shape)

# %%
###eliminate columns with all NA
LowPE.dropna(axis=1,how='all',inplace=True)
print(LowPE.shape)
print(LowPE.columns)
print(LowPE.head())
print(LowPE.describe().T)# by using .T you can transpose the output


# %%
######### transpose the data #######################################################################
trans=pd.melt(LowPE,id_vars=['date'],var_name='id',value_name='price')
print(trans.head())
trans.sort_values(['id','date'],inplace=True)
trans.reset_index(drop=True,inplace=True)
print(trans.head())

# %% [markdown]
# ### One can create new columns as follows

# %%
#lagged values for each id
trans[['lprice','ldate']]=trans.groupby(['id'])[['price','date']].shift(1)
trans.reset_index(drop=True,inplace=True)# reset the index and drop old index values.  In addition, replace the original
print(trans.head(50))

# %%
#calculate returns
trans['return']=(trans['price']-trans['lprice'])/trans['lprice']
#create aYearMonth columns
trans['yyyymm']=trans['date'].dt.year*100+trans['date'].dt.month

# %% [markdown]
# ### One should be careful not to mistakenly calcualted returns using prices of different companies
# - It can happen when ID changes
# - In addition, returns may not represent the return over the intended time interval (e.g., a month) when there are missing values

# %%
#  Find the differences in the number of months between the lagged observation and the current one
# and eliminate those with more than one month difference

trans['diffm']=(trans['date'].dt.year-trans['ldate'].dt.year)*12+\
                (trans['date'].dt.month-trans['ldate'].dt.month)
print(trans.head())
print(trans.dtypes)#types of each column values

# %% [markdown]
# #### If the gap between the current and the previous month is greater than one month, set the return as missing

# %%
print(trans['return'].describe())

# %%
trans.loc[(trans['diffm']!=1.0),'return']=np.nan
print(trans['return'].describe())

# %% [markdown]
# ## Use "groupby" to correctly calculate returns using prices
#
# Below, you will find that **ret** includes wrong returns when ID changes to a different one while **ret1** and **ret2** correctly calculate returns even when ID changes (missing in those cases) by using **groupby(['id'])**

# %%
#Calculate returns in a simpler way
#Problem with a new ID
trans['ret']=(trans['price']-trans['price'].shift(1))/trans['price'].shift(1)
print(trans.ret.describe())

# %%
#The above code does not work correctly for the first observation of each ID
#because it uses the previous observation of the same ID
#Therefore, we need to use groupby to calculate the returns for each ID separately
#This will ensure that the return is calculated based on the previous observation of the same ID

#Correct one
#############################################
trans['ret1']=(trans['price']-trans.groupby(['id'])['price'].shift(1))\
                /trans.groupby(['id'])['price'].shift(1)
print(trans.ret1.describe())

# %% [markdown]
# #### One can use ".pct_change()" to find the return with "groupby"

# %%

#trans['ret2']=trans.price.pct_change(fill_method=None)
trans['ret2']=trans.groupby(['id']).price.pct_change()
print(trans.ret2.describe())
print((trans.ret2-trans.ret1).describe())

print(trans[['id','date','ret','ret1','ret2']].head(50))


# %% [markdown]
# ## Appendix 2: How to calculate value-weighted returns using a function
#

# %%
####Using pandas##########################################
# One can define a custom function to calculate value weighted return
#df is a dataframe, avg_name is the column name to be used for average calculation
#weight_name is the column name to be used as the weight in value-weighted average

def wavg(df, avg_name, weight_name):
    df = df[df[avg_name].notna() & df[weight_name].notna()].copy()  # Filter out NaN values
    if df.empty:
        return np.nan
    d = df[avg_name]
    w = df[weight_name]
    
    try:
        w_sum = w.sum()
        if w_sum == 0 or np.isnan(w_sum) or np.isinf(w_sum):
            return np.nan
        else:
            return (d * w).sum() / w_sum
    except ZeroDivisionError:
        return np.nan

# Calculate Weighted Average Returns 
# to use the custom function defined above for each group of yyyymm and pmb
# one can use the apply method with a lambda function

data1.groupby(['yyyymm', 'pmb'])[['ret', 'mcap']].apply(lambda x: wavg(x, 'ret', 'mcap')).reset_index()
data1['icount']=np.where(data1['ret'].notnull(),1,0)#icount is set to be 1 if ret is not null.  Otherwise, set to be zero
print(data1.icount.describe())

# %%
ret['vwret1']= data1.groupby(['yyyymm','pmb'])[['ret','mcap']].apply(wavg, 'ret','mcap').reset_index().iloc[:,2]

# Equally-weighted returns are calculated by using the same function
# where the weight is same for all (i.e., 1) in each group

ret['ewret1'] = data1.groupby(['yyyymm','pmb'])[['ret','icount']].apply(wavg, 'ret','icount').reset_index().iloc[:,2]
print(ret.head(20))

###Compare the results calculated using SQL above
print((ret.ewret-ret.ewret1).describe())
print((ret.vwret-ret.vwret1).describe())


# %% [markdown]
# ## Appendix 3: The following are alternative ways to find out the MB portfolios in Python

# %%
#The following are alternative ways to find the portfolios in Python
#assign MB portfolio value - define a function

# Define a function to assign quintiles based on the mb value
def quintile(x):#note that the input is a dataframe
    
    if x['mb'] <= x['quint20']: return 1
    elif x['mb'] <= x['quint40']: return 2
    elif x['mb'] <= x['quint60']: return 3
    elif x['mb'] <= x['quint80']: return 4
    else: return 5
mb3['pmb']=mb3.apply(quintile, axis=1)#apply the function and call the result as pmb
mb3['pmb']=np.where(mb3['mb'].isnull(),0,mb3['pmb'])#If mb is null assign 0

#check
print(mb3.columns)
print(mb3.head())
#print(mb3[mb3['mb'].isnull()].head())
#print(mb3[mb3['mb'].notnull()].head())
#print(mb3.mb.describe())
#print(mb3[mb3.pmb>0].pmb.describe())
print(mb3.groupby(['pmb']).mb.describe())

# %%
######################################################################
# An easier way using pandas' qcut to form deciles in each year
######################################################################
print(mb3.shape)
mb4=mb3[mb3['mb'].notnull()].copy()# Get rid of ones with missing mb values
print(mb4.shape)

#lambda "arguments" : "expression"
#The "expression" with "argument" is executed and the result is returned:


#".transform(lambda x: f(x))" is similar to .apply(lambda x: f(x)) but keep the original shape
# while apply does not
mb4.loc[:,'mbquint']=mb4.groupby(['year'])['mb'].transform(
                     lambda x: pd.qcut(x, 5, labels=range(1,6)))

print(mb3.columns)
print(mb4.columns)

# %%
print(mb4[['pmb','mbquint']])
print(mb4.dtypes)


# %%

# %%
