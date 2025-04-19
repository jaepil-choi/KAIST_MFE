# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% vscode={"languageId": "r"}
rm(list=ls()) # Wipe memory for initialization

# %% vscode={"languageId": "r"}
setwd("./ch1andch2 dataset") # No need to write the full path from now on

# %% vscode={"languageId": "r"}
lightbeer = read.csv("light_beer_preference_survey.csv")
head( lightbeer )  

# %% vscode={"languageId": "r"}
lightbeer[1, 3] # row, col

# %% vscode={"languageId": "r"}
lightbeer[1, ] # row 1, all cols

# %% vscode={"languageId": "r"}
brand = lightbeer$Brand # select column
brand_freq = table(brand)
brand_freq

# %% vscode={"languageId": "r"}

freq_dist = cbind( brand_freq, brand_freq/sum(brand_freq)*100) # relative frequency to percentage
barplot(brand_freq)

# %% vscode={"languageId": "r"}
pie(brand_freq)

# %% vscode={"languageId": "r"}
longdist = read.csv("long_distance_telephone_bills.csv")
head(longdist )

# %% vscode={"languageId": "r"}
bills <- longdist$Bills
hist(bills, breaks=seq(0, 120, 15), col='blue')

# %% vscode={"languageId": "r"}
news_reader <- read.csv('newspaper_readership_survey.csv')
head( news_reader )

# %% vscode={"languageId": "r"}
ctgc_tab <- table( news_reader$Newspaper, news_reader$Occupation )
colnames(ctgc_tab)<- c("Blue Collar", "While Collar", "Professional")
rownames(ctgc_tab)<- c("G&M", "Post", "Star", "Sun")
ctgc_tab

# %% vscode={"languageId": "r"}
barplot( ctgc_tab )

# %% vscode={"languageId": "r"}
barplot( ctgc_tab, beside=TRUE )

# %% vscode={"languageId": "r"}
house <- read.csv('price_and_size_of_houses.csv')
 head( house )
 plot( house$Size, house$Price, xlab="Size", ylab="Price" )

# %% vscode={"languageId": "r"}
gasoline <- read.csv('price_of_gasoline.csv')
  head( gasoline )
plot( gasoline$Price, type="l", xlab="Month", ylab="Price of Gasoline")

# %% vscode={"languageId": "r"}
