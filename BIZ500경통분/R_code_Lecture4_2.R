rm(list=ls())


1- pbinom( 2, size = 10, prob=0.2 )

p_ref <- 1 - dbinom( 0, size=10, prob=0.01 )
p_ref
dbinom( 1, size=3, prob=p_ref )


##
pnorm(1100, mean=1000, sd=100)
pnorm( 0, mean=10, sd=5 )
pnorm( 0, mean=10, sd=10 )
qnorm( 0.99, mean=490, sd=61 )


##
qchisq(0.95, df=8)
qchisq(0.05, df=8)
1-pchisq(33, df=16)
pchisq(33, df=16, lower.tail=FALSE)
pchisq(28, df=16) - pchisq(23, df=16)
qchisq(0.10, df=16)
qchisq(0.95, df=16)



##
qt(0.95, df=25)
qt(0.95, df=74)
qt(0.05, df=10)


##
qf(0.95, df1=5, df2=7)
qf(0.05, df1=4, df2=8)

