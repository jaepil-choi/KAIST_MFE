rm(list=ls())

#
sammean <- rep( 0, times=200 )
for ( i in 1:200 ){
  sam10 <- rnorm(10, mean=3, sd=2)
  sammean[i] <- mean( sam10 )
}
hist( sammean )


mean( sammean )
sd( sammean )
sqrt( 2^2/10 )


#
curve( df(x, df1=5, df2=8), from=0, to=5, ylab="f(x)", 
main="density of F[5, 8]")

sammean <- rep( 0, times=200 )
for ( i in 1:200 ){
  sam10 <- rf(10, df1=5, df2=8)
  sammean[i] <- mean( sam10 )
}
hist( sammean, breaks=10 )


 
sammean <- rep( 0, times=200 )
for ( i in 1:200 ){
  sam10 <- rf(100, df1=5, df2=8)
  sammean[i] <- mean( sam10 )
}
hist( sammean, breaks=10 )



#
pnorm( 32, mean=32.2, sd=0.3, lower.tail=FALSE)
pnorm( 32, mean=32.2, sd=0.3/sqrt(4), lower.tail=FALSE)
pnorm( 0.05, mean=0.03, sd=sqrt( 0.03*0.97/400 ), lower.tail=FALSE )
