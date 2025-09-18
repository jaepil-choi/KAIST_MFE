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
# ## FDNote3W
# Prepared by Inmoo Lee for the Financial Databases class at KAIST
#
# inmool@kaist.ac.kr
#
#
# List of output files
#
#     - RHistoryNote3W.xlsx
#     - LowPEHeadKRX.ft
#     - RHistory.ft
#     - LowPEHeadUS.ft
#     - RHistoryESE.ft
#     - RHistory.ft
#

# %%

# %%
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt

# print(os.getcwd())  #get the current working directory
# path='d:\\###'#change this to your directory to work with
# os.chdir(path) # change the working directory
# print(os.getcwd())  #get the current working directory


# %% [markdown]
# ## LSEG Workspace (used to be EIKON) API
# from hhttps://developers.lseg.com/en/api-catalog/eikon/eikon-data-api/quick-start
#
# First, install the eikon package

# %%
# To use the eikon package, you need to install bottleneck first, if you haven't done so already.
# #!pip install --user --upgrade bottleneck
# #!pip install --upgrade bottleneck
# # !pip install bottleneck

# %%
#install eikon package
# # !pip install eikon

# %%
from pathlib import Path

CWD = Path.cwd()
PROJ_PATH = CWD.parent.parent.parent

PROJ_PATH

# %%
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=PROJ_PATH / '.env')

API_KEY = os.getenv('EIKON_API_KEY')

# %%
#After installing the eikon package, you need to import it to use its functionalities.
#You can also check the attributes and methods of the eikon package using the dir() function
import eikon as ek
dir(ek) # lists the attributes and methods of an object

# %% [markdown]
# ### Appkey
#
# To use LSEG Workspace API, you should get the appkey by following the procedures below
# after loggin into your Workspace account
#
#     - First open your Workspace
#     - Type APPKEY to open App Key Generator
#     - Choose Eikon Data API and then click Register New App
#     - Copy the App Key to the following, inside the parenthesis
#

# %%
help(ek.get_app_key) #get help on get_app_key

# %%
#Get the previously generated app key
ek.set_app_key(API_KEY) #Replace this with your app key
key=ek.get_app_key()
# print(key)

# %% [markdown]
# ### Once you input the key, then you are ready to use EIKON API
#
# You should log in the LSEG Workspace in the same computer that you run your Python
#
# USe Data Item Browser (DIB) App to find data items
#

# %%
###Some exercises: getting news
# Get help regarding the get_news_headlines function
help(ek.get_news_headlines)

# %%
#Get the news for a specific company during a certain time period
#There are time restrictions (only for recent periods)
# Example: Get the latest 5 news headlines for Microsoft in France in English

ek.get_news_headlines("R:MSFT.O IN FRANCE IN ENGLISH", count=5)

# %%
# Get the news for a specific company during a certain time period
# There are time restrictions (only for recent periods)
# Example: Get the news headlines for Microsoft from May 1, 2025, to June 23, 2025
# And store the result in a dataframe called 'news'
news=ek.get_news_headlines('MSFT.O', date_from='2025-07-01T09:00:00',\
                           date_to='2025-07-23T18:00:00')
print(news.columns)
#print(news.info())
print(news.head())
#print(news.storyId)

# %%
# Get the news story for a specific storyId
storyId = news['storyId'].iloc[2]#specify which story to retrieve
ek.get_news_story(storyId)

# %%
###to read it in a web browser, you can store it and then open it in the following way
htmlobj=ek.get_news_story(storyId)

#save the content, htmlobj, in the 'test.htm' file
with open('test.htm','wb') as f:   # Use some reasonable temp name
    f.write(htmlobj.encode("UTF-8"))

# %%
# open an HTML file stored above, test.htm, on my own (Windows) computer
# You can use the webbrowser module to open the HTML file in a web browser.
import webbrowser
url = r'test.htm'
webbrowser.open(url,new=2)
# if new=0, the url is opened in the same browser window if possible
# if new=1, a new browser window is opened if possible
# if new=2, a new browser page("tab") is opened if possible

# %%
# You can also access the news dataframe in various ways
print(news.head()) #get the first 5 rows
print(news.loc[:,'text'])
#print(news.iloc[:,1]) #get the second column of all rows
print(news.iat[0,1]) #get the second column of the first row
print(news.iat[5,1]) #get the second column of the sixth row

# %%
#get the third row's storyid
storyid=news.iat[2,2]#iat[] is used to access a single value for a row/column pair in Pandas

#save the content to 'test.htm' and open it in webbrowser
with open('test.htm','wb') as f:   # Use some reasonable temp name
    f.write(ek.get_news_story(news.iat[3,2]).encode("UTF-8"))
webbrowser.open(r'test.htm',new=2)

# %% [markdown]
# ### Getting timeseries data from LSEG Workspace
#
# You can use get_timeseries to get time series data

# %%
# Get help regarding the get_timeseries function
help(ek.get_timeseries)

# %%
# Getting timeseries data from LSEG Workspace
# Example: Get the price data for Microsoft (MSFT.O) from January 1, 2025, to January 10, 2025

price = ek.get_timeseries(["MSFT.O"], 
                       start_date="2025-08-01",  
                       end_date="2025-08-10")
price

# %% [markdown]
# ### To get multiple firms' latest financial items specifed
#
# Use get_data
#
# You can find the data item from the Data Item Browser
#
# In the Eikon Search Box, type DIB or data item, and select
#
# APP-Data Item Browser from Autosuggest

# %%
###getting multiple firms' latest financial items specifed
# Get help regarding the get_data function
help(ek.get_data)

# %%
# To get the latest financial items for multiple firms, you can use the get_data function.
# Example: Get the latest financial items for Google (GOOG.O) and Microsoft (MSFT.O)    
FS, err = ek.get_data(['GOOG.O','MSFT.O'], 
                      [ 'TR.Revenue','TR.GrossProfit','PERATIO','TR.EBITDAInterestCoverage'])
print(FS)
print(err)# error

# %%
# To get help regarding the TR_Field class
# This class provides access to the Thomson Reuters financial data fields.
help(ek.TR_Field)

# %%
#You can specify fields to be retrieved and then use them in "get_data"
fields = [ek.TR_Field('tr.revenue'),
          ek.TR_Field('tr.open',None,'asc',1),
          ek.TR_Field('TR.GrossProfit',
                      {'Scale': 6, 'Curn': 'EUR'},'asc',0)]
          #Scale is used to set the unit (6=million)
print(fields)

# %%
# To get the latest financial items for multiple firms using specified fields
# Example: Get the latest financial items for Google (GOOG.O) and Microsoft (MSFT.O)
# You store the result in fs1 and any error in err

fs1, err=ek.get_data(['GOOG.O','MSFT.O'], fields)
print(fs1)
print(err)

# %%
#To sort in ascending order, use opening price and then GrossProfit in sorting
fields = [ek.TR_Field('tr.revenue'),
          ek.TR_Field('tr.open',None,'asc',1),
          ek.TR_Field('TR.GrossProfit',
                      {'Scale': 6,'Curn': 'USD'},'asc',2)]
#fields
fs2, err=ek.get_data(['GOOG.O','MSFT.O'], fields)
print(fs2)

# %% [markdown]
# ## You can have a list of instrument codes and then use it in "get_data"

# %%
## List of Reuters Instrument Codes (RICs)
rics=['GE','AAPL.O',
      'EUR=',#EUR/USD exchange rate
      'XAU=',#Gold price
      'DE10YT=RR' #10yr Bund price
      ]

#Unfortunately, we do not have access to the index information
#'.SPX',#s&p 500 STOCK INDEX
#'.VIX',#vix VOLATILITY INDEX


# %%
# Get the timeseries data for the specified RICs
# Example: Get the close prices for the specified RICs from October 10, 2022, to January 31, 2023
data=ek.get_timeseries(rics, #the list of RICs
                       fields='CLOSE', #close field
                       start_date='2025-06-01', #start date
                       end_date='2025-07-30')
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())

# %% [markdown]
# ### There are multiple IDs used in the global financial communities
#
# You can convert one to another using "get_symbology"
#
# Symbology conversions (ISIN, SEDOL etc)
#
#     - ISIN: International Securities Identification Number (International Securities Identification Number, are globally recognized and standardized identifiers. Their structure is defined by ISO 6166 (International Organization for Standardization) and is a 12-character alphanumeric code.)
#     - SEDOL: Stock Exchange Daily Official Lis (Stock Exchange Daily Official List, are primarily used in the United Kingdom and Ireland. They are  seven-character alphanumeric codes assigned by the London Stock Exchange (LSE))

# %%
# To get help regarding the get_symbology function
# This function is used to convert between different symbology formats.
help(ek.get_symbology)

# %%
# Example: Convert RICs to ISINs
# You can convert RICs to ISINs using the get_symbology function.
print(ek.get_symbology(rics[:2],from_symbol_type='RIC',to_symbol_type='ISIN'))

# %%
# Example: Convert RICs to ISINs and tickers
# You can convert RICs to ISINs and tickers using the get_symb
print(ek.get_symbology(rics[:2],from_symbol_type='RIC',to_symbol_type=['ISIN','ticker']))
print(ek.get_symbology(rics,from_symbol_type='RIC',to_symbol_type=['ISIN','ticker']))

# %%
###from SEDOL  to RICs or Others
sedols=['B1YW440','0673123']
print(ek.get_symbology(sedols,from_symbol_type='SEDOL',to_symbol_type=['RIC','ISIN','ticker']))
##from ISINs to
symbols=['US0378331005','US0231351067']
ricex=ek.get_symbology(symbols,from_symbol_type='ISIN',to_symbol_type=['RIC'])
ricex=list(ricex.RIC.values)
print(ricex)


# %%
ek.get_symbology(['FB'], from_symbol_type='ticker', to_symbol_type='ISIN')

# %%
################################################################
##retrieve the data for the list of RICs
################################################################
data=ek.get_timeseries(ricex, #the list of RICs
                       fields='CLOSE', #close field
                       start_date='2025-07-01', #start date
                       end_date='2025-07-30')
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())

# %%
####draw normalized price time series
data = (data-data.min())/(data.max()-data.min()) #Normalize the data
print(data.columns)
print(data.head())
print(data.tail())
print(data.describe())
print(data.index)

# %% [markdown]
# ### You can use pyplot in matplotlib to draw a graph

# %%
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

y1=data['AAPL.O'].tolist() #to make a list
y2=data['AMZN.O'].tolist()

plt.figure() #define a new figure
plt.plot(data.index, y1, linewidth=3,label='AAPL',color='blue') #define the line for AAPL
plt.plot(data.index, y2, linewidth=3,label='AMZN',color='orange') #define the line for AMZN
plt.xlabel('Date') #use label for X axis
plt.xticks(rotation=90) #rotate x axis labels for better visibility
plt.ylabel('Returns') #use label for Y axis
plt.ylim(ymin=0) #specify y axis values to start from 0
plt.legend() # Legend will be included using the labels defined in plt.plot()
plt.show() #Show the figure

# %%
##To fit a regression line between two stock returns
# You can use seaborn's lmplot function to fit a linear regression line between two stock returns.
import seaborn as sns

#Draw a linear regression line between the returns of AMZN and AAPL
sns.lmplot(x="AMZN.O", y="AAPL.O", data=data.astype(float), line_kws={'color':"blue"})

# %%
###################################################################
####get the data for the list of stocks
print(ricex)
###################################################################
#Get multiple items (fields) data for a list of instruments
data, err=ek.get_data(ricex,['TR.PriceClose','TR.Volume',
                                'TR.PriceLow','TR.TotalReturnYTD',
                                'TR.TotalReturn52WK',
                                'TR.TotalReturn'])
print(data)
print(err)

# %%
##find a subset and use Instrument as an index
cols=['YTD Total Return','52 Week Total Return']
# Define a new dataframe with 'Instrument' as the index 
# and only the specified columns
df=data.set_index('Instrument')[cols]
print(df.index)
print(df.iloc[:,0])
print(df)

# %%
print(df.index) # Get the index of the dataframe
print(df.head()) # Get the first 5 rows of the dataframe

# %%
###draw bar graph using the dataframe, df
df.plot(kind="bar")
plt.title("Total Returns")
plt.ylabel("Total Returns")

# %% [markdown]
# ## Use screen to find a list of securities that satisfy the conditions
#
# Create RHistoryNote3W.xlsx data using API
#
# Using DIB of an equity (e.g., Samsung Electronics), find the data item to be
# used as a screening criterion
#

# %%
#Define screen criteria
exp = "SCREEN(U(IN(Equity(active,public,primary))),\
     IN(TR.ExchangeMarketIdCode,""XKRX""),\
     TR.PE <= 10.0,\
     TR.TtlDebtToTtlEquityPct(Period=FY0) <= 100.0,\
     TR.EBITDAInterestCoverage(Period=FY0) >= 10.00,\
     CURN=KRW)"
#Define the fields to be retrieved
fields = ['TR.COmmonName', 'TR.PE',\
          'TR.TtlDebtToTtlEquityPct(Period=FY0)','TR.EBITDAInterestCoverage(Period=FY0)'] 

# Get the data for the specified screen criteria and fields
# The result will be stored in LowPEHead and any error in err
LowPEHead, err = ek.get_data(exp, fields)

print(LowPEHead.dtypes)
print(LowPEHead.head())

# %%
#change the column name from 'Instrument' to 'id'
LowPEHead.rename(columns={'Instrument':'id'},inplace=True)

print(LowPEHead)

LowPEHead.to_feather('LowPEHeadKRX.ft')# save the data

# %%
##store the list of ids in the dataframe, LowPEHead to firms
firms=list(LowPEHead.id)
print(firms)

# %% [markdown]
# ## Retrieve stock returns
#
# There are two ways to get stock returns
#
# One is to use return calculated by LSEG (e.g., TR.TotalReturn1Mo)
#
# The second way is to use return index (which adjusts for stock splits/stock dividends and dividends) to calculate returns.

# %%
## Retrieve the data for the list of RICs
## "get_data" is used with options to specify time period and intervals
## TR.PriceClose adjust prices for stock splits: check below

# Define the fields to be retrieved, spcifying the start and end dates, and frequency
# The frequency is set to 'M' for monthly data
# .calcdate and .date are used to get the calculation date and the date of the data
# respectively, for the TotalReturn1Mo field

fd =['TR.PriceClose(SDate=2020-01-31,EDate=2024-12-31,Frq=M)',
     'TR.TotalReturn1Mo(SDate=2020-01-31,EDate=2024-12-31,Frq=M)',
         'TR.TotalReturn1Mo(SDate=2020-01-31,EDate=2024-12-31,Frq=M).calcdate',
         'TR.TotalReturn1Mo(SDate=2020-01-31,EDate=2024-12-31,Frq=M).date']
dataKR, err=ek.get_data(firms,fields=fd)
print(dataKR)
print(dataKR.columns)
print(dataKR.head())

# %% [markdown]
# ### Let's sort the data by 'Instrument' and 'Date'

# %%
#sort the data and replace the orignal data
dataKR.sort_values(['Instrument','Date'],inplace=True) 
#reset the index and replace the original data
dataKR.reset_index(drop=True,inplace=True)  

# In the above, (drop=True (discard old index) and inplace=True (replace the old data))

print(dataKR.head())
print(dataKR[['Price Close','1 Month Total Return']].head())
print(dataKR.columns)

#save the data
dataKR.to_feather('RHistory.ft')

# %%
# Print out the list of instruments after dropping duplicates    
print(dataKR.Instrument.drop_duplicates())

#print out a certain instrument's information
print(dataKR[dataKR.Instrument=='000240.KS'])

# %% [markdown]
# #### Save the data into an excel file

# %%
import os
os.getcwd() #find out the current working directory

# %%
#Save the data to an Excel file
# The sheet name is set to "LowPEKRX"
dataKR.to_excel('RHistoryNote3W.xlsx',sheet_name="LowPEKRX")

# %% [markdown]
# #### Use adjusted prices to calculated returns
#
# You can get the prices and convert them to returns
#
# Make sure that you get adjusted prices
#
# Here, we are using timeseries

# %%
## Get the timeseries data for the specified RICs
# Example: Get the close prices for the specified RICs from January 31, 2021, to December 31, 2024
# The firms variable contains the list of RICs obtained from LowPEHead
pdata=ek.get_timeseries(firms, #the list of RICs
                       fields='CLOSE', #close field
                       start_date='2020-01-31', #start date
                       end_date='2024-12-31',
                       count = None,
                       interval='monthly',
                       calendar = None, corax = 'adjusted',
                       normalize = False, raw_output = False, debug = False)
print(pdata.head())
#Stack the data to convert it into a long format
# This will create a multi-index dataframe with 'Instrument' and 'Date' as indices
df1=pdata.stack().reset_index()

# %%
print(pdata.head())
print(df1.head())

# %%
#Change the column names for better readability
# Rename the columns to 'id' for Instrument and 'adjprc' for adjusted price
df1.rename(columns={'CLOSE':'id',0:'adjprc'},inplace=True)
print(df1)
#Sort the data by 'id' and 'Date' and replace the original data
df1.sort_values(['id','Date'],inplace=True)
df1.reset_index(drop=True, inplace=True)
print(df1.head())
print(df1.columns)

# %% [markdown]
# #### Let's check whether the returns using adjusted prices are same as returns calculated by EIKON

# %%
print(dataKR.columns)
print(df1.columns)
print(dataKR.dtypes)
print(df1.dtypes)

# %%
#change the format of Calc Date column to datetime format so that
#the merge operation can be performed correctly using the same data format
#of the columns used in the merge operation

dataKR['Calc Date']=pd.to_datetime(dataKR['Calc Date']) 
test=pd.merge(dataKR.drop(columns=['Date']),df1,how='left',left_on=['Instrument','Calc Date'],\
              right_on=['id','Date'])
#In the above, drop(columns=['Date']), drop the specified columns
print(test)

# %%
#Store only those rows that satisfy the conditions
test1=test.loc[test['Price Close'].notnull() & test.adjprc.astype('float').notnull(),:].copy()
#In the above, rows that satisfy the conditions are stored in test1

print((test1['Price Close']-test1.adjprc.astype('float')).describe())

# %% [markdown]
# ### Check adjusted vs. unadjusted prices
#
# Check a case which experienced stock splits
#
# Samsung electronics which splitted stocks on May 11, 2018
#

# %%
# Get the adjusted and unadjusted close prices for Samsung Electronics
# and store them in adp and unadp respectively
# The adjusted prices are adjusted for stock splits and dividends
adp=ek.get_timeseries('081660.KS', #the list of RICs
                       fields='CLOSE', #close field
                       start_date='2018-01-31', #start date
                       end_date='2018-12-31',
                       count = None,
                       interval='monthly',
                       calendar = None, corax = 'adjusted',
                       normalize = False, raw_output = False, debug = False)
unadp=ek.get_timeseries('081660.KS', #the list of RICs
                       fields='CLOSE', #close field
                       start_date='2018-01-31', #start date
                       end_date='2018-12-31',
                       count = None,
                       interval='monthly',
                       calendar = None, corax = 'unadjusted',
                       normalize = False, raw_output = False, debug = False)
# The fields to be retrieved, specifying the start and end dates, and frequency
# The frequency is set to 'M' for monthly data
fd =['TR.PriceClose(SDate=2018-01-31,EDate=2018-12-31,Frq=M)',
         'TR.PriceClose(SDate=2018-01-31,EDate=2018-12-31,Frq=M).date']
closep, err=ek.get_data('081660.KS',fields=fd)

##compare to check whether TR.PriceClose is adjusted price or unadjusted price
print(closep)
print(adp)
print(unadp)

# %% [markdown]
# ### Let's get the US data

# %%
#Specify the screening criteria for US stocks
exp = "SCREEN(U(IN(Equity(active,public,primary))),\
     IN(TR.ExchangeCountryCode,""US""),\
     IN(TR.ExchangeMarketIdCode,""XNYS""),\
     TR.PE <= 10.0,\
     TR.TotalDebtToEV <= 50.0,\
     CURN=USD)"
#Specify the fields to be retrieved
fields = ['TR.COmmonName', 'TR.PE',\
          'TR.TotalDebtToEV'] 
# Get the data for the specified screen criteria and fields
# The result will be stored in LowPEHeadUS and any error in err
LowPEHeadUS, err = ek.get_data(exp, fields)

print(LowPEHeadUS.columns)
LowPEHeadUS.rename(columns={'Instrument':'id'},inplace=True)
print(LowPEHeadUS)
#save the data
LowPEHeadUS.to_feather('LowPEHeadUS.ft')

# %% [markdown]
# ### Get the list of the stocks and retreive stock returns

# %%
#Get the list of ids from LowPEHeadUS
# This will be used to retrieve the timeseries data for these firms
firms=list(LowPEHeadUS.id)
print(firms)

#Specify the fields to be retrieved

fd =['TR.PriceClose(SDate=2020-01-31,EDate=2024-12-31,Frq=M)',
     'TR.TotalReturn1Mo(SDate=2020-01-31,EDate=2024-12-31,Frq=M)',
         'TR.TotalReturn1Mo(SDate=2020-01-31,EDate=2024-12-31,Frq=M).calcdate',
         'TR.TotalReturn1Mo(SDate=2020-01-31,EDate=2024-12-31,Frq=M).date']
dataUS, err=ek.get_data(firms,fields=fd)

print(dataUS.columns)
#Sort the data by 'Instrument' and 'Date' and replace the original data
dataUS.sort_values(['Instrument','Date'],inplace=True)
#Reset the index and replace the original data
dataUS.reset_index(drop=True,inplace=True)
print(dataUS.head())
print(dataUS.describe())

# %% [markdown]
# #### You can convert a string columns to a numeric column

# %%
print(dataUS.dtypes)

# %%
# Convert the '1 Month Total Return' column to numeric type and call the column 'ret'
# This will handle any non-numeric values by converting them to NaN
dataUS['ret']=pd.to_numeric(dataUS['1 Month Total Return'], errors='coerce')
dataUS.ret.describe()

# %%
#Drop the '1 Month Total Return' column as it is no longer needed
# This column has been converted to 'ret' and is no longer needed
dataUS.drop(columns=['1 Month Total Return'],inplace=True)
print(dataUS.dtypes)
print(dataUS[['Price Close','ret']].head(50))

#Save
dataUS.to_feather('RHistoryESE.ft')
print(dataUS.columns)

#Save
dataUS.to_feather('RHistory.ft')

# %%
#List the unique instruments in the dataUS dataframe
dataUS.Instrument.drop_duplicates().tolist()

# %%
#Check an example
print(dataUS.loc[dataUS.Instrument=='BDJ.N',['Date','Price Close']]) 

# %% [markdown]
# ### Save the data in to the excel file used above (different worksheet)

# %%
#To add a worksheet to an existing excel file, use the following method

# #!pip install openpyxl
import openpyxl
#Check the version of openpyxl
print(openpyxl.__version__)

# %%
#To install or upgrade openpyxl, use the following command
# # !pip install --upgrade openpyxl

# %%
with pd.ExcelWriter('RHistoryNote3W.xlsx',engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:  
    dataUS.to_excel(writer, sheet_name='LowPEUS')

# %%
