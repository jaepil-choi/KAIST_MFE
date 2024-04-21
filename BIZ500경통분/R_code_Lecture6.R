rm(list=ls())

setwd("C:/Users/Administrator/Dropbox/KAIST Course/2023 Spring/BIZ500/DataExamples")
timedt <- read.csv('amount_of_television_watched.csv')
time <- timedt$Time
time


xbar <- mean( time ) 
sx <- sd( time )
n <- length( time )
CI_m <- c( xbar - qnorm(0.975)*sx/sqrt(n), xbar + qnorm(0.975)*sx/sqrt(n))
CI_m



demand <- c(235, 374, 309, 499, 253, 421, 361, 
             514, 462, 369, 394, 439, 348, 344, 
             330, 261, 374, 302, 466, 535, 386, 
             316, 296, 332, 334)
xbar <- mean( demand ) 
sx <- 75
n <- length( demand )
CI_m <- c( xbar - qnorm(0.975)*sx/sqrt(n), xbar + qnorm(0.975)*sx/sqrt(n))
CI_m



phat <- 35/50
CI_p <- c( phat + qnorm(0.99)*sqrt( phat*(1-phat)/50 ), phat + qnorm(0.01)*sqrt( phat*(1-phat)/50 ))
CI_p




S2 <- 50^2
n <- 21
CI_sigma2 <- c( (n-1)*S2/qchisq(0.975, df=n-1), (n-1)*S2/qchisq(0.025, df=n-1))
CI_sigma2
