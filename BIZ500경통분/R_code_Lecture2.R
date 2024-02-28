rm(list=ls())

setwd("./ch1andch2 dataset")

lightbeer = read.csv("light_beer_preference_survey.csv")
head( lightbeer )  

brand = lightbeer$Brand
brand_freq = table(brand)
freq_dist = cbind( brand_freq, brand_freq/sum(brand_freq)*100)
barplot(brand_freq)

pie(brand_freq)



##
longdist = read.csv("long_distance_telephone_bills.csv")
head(longdist )
bills <- longdist$Bills
hist(bills, breaks=seq(0, 120, 15), col='blue')


##
news_reader <- read.csv('newspaper_readership_survey.csv')
head( news_reader )
ctgc_tab <- table( news_reader$Newspaper, news_reader$Occupation )
colnames(ctgc_tab)<- c("Blue Collar", "While Collar", "Professional")
rownames(ctgc_tab)<- c("G&M", "Post", "Star", "Sun")
ctgc_tab
barplot( ctgc_tab )
barplot( ctgc_tab, beside=TRUE )


##
 house <- read.csv('price_and_size_of_houses.csv')
 head( house )
plot( house$Size, house$Price, xlab="Size", ylab="Price" )


##
gasoline <- read.csv('price_of_gasoline.csv')
  head( gasoline )
plot( gasoline$Price, type="l", xlab="Month", ylab="Price of Gasoline")




