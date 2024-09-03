rm(list=ls())

setwd("./ch3_dataset")

longdist <- read.csv('long_distance_telephone_bills.csv')
head( longdist )
bills <- longdist$Bills

mean( bills )
median( bills )
range( bills )
diff( range( bills ) )
var( bills )
sd( bills )
sd( bills )/mean( bills )


quantile( bills )
quantile( bills , prob=c(0.3, 0.8))



bills <- longdist$Bills
times <- read.csv('serving_times_of_drive_throughs.csv')
head( times )
boxplot( times, horizontal=TRUE, col="lightblue" )



scores <- read.csv('GMAT_and_GPA_scores_for_MBA_students.csv')
head( scores )
gmat <- scores$GMAT
gpa <- scores$GPA
plot( gmat, gpa )

cov( gmat, gpa )
cor( gmat, gpa )
