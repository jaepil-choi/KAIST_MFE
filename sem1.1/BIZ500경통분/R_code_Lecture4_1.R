rm(list=ls())


x <- 0:5
y <- c( 1218, 32379, 37961, 19387, 7714, 2842 )
pmf_table <- cbind( x, y/sum(y))
colnames( pmf_table ) <- c("X","f(X)")
pmf_table



Ex <- sum( pmf_table[,"X"] * pmf_table[,"f(X)"] )
Ex
Vx <- sum( pmf_table[,"X"]^2 * pmf_table[,"f(X)"] ) - Ex^2
Vx
Sx <- sqrt(Vx)
Sx


##
mu <- 25000
sigma <- 4000
0.3*mu-6000
(0.3^2)*(sigma^2)


##
Pxy <- matrix( c( 0.12, 0.21, 0.07, 0.42, 0.06, 0.02, 0.06, 0.03, 0.01), nrow=3, ncol=3 )
rownames( Pxy ) <- c(0, 1, 2)
colnames( Pxy ) <- c(0, 1, 2)
Pxy
Px <- apply( Pxy, 2, sum )
Px
Py <- apply( Pxy, 1, sum )
Py

 
Ex <- sum( (0:2) * Px )
Sx <- sqrt( sum( (0:2)^2 * Px ) - Ex^2 )
Ex
Sx
Ey <- sum( (0:2) * Py )
Sy <- sqrt( sum( (0:2)^2 * Py ) - Ey^2 )
Ey
Sy


COVxy <- sum( outer( 0:2, 0:2 ) * Pxy ) - Ex*Ey
CORxy <- COVxy / ( Sx * Sy )
COVxy 
CORxy
Ex + Ey
Sx^2+Sy^2+2COVxy
