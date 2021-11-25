mu <-2
Z <-rnorm(20000, mu)
MSE <-function(estimate, mu) {
        return(sum((estimate -mu)^2) /length(estimate))
}

n <-10
shrink <-seq(0,0.5, length=n)
mse <-numeric(n)
bias <-numeric(n)
variance <-numeric(n)

for (i in 1:n) {
        mse[i] <- MSE((1 -shrink[i]) *Z, mu)
        bias[i] <- mu *shrink[i]
        variance[i] <- (1 -shrink[i])^2
}

# Bias-Variance tradeoff plot

plot(shrink, mse, xlab='Shrinkage', ylab='MSE', type='l', col='pink', 
     lwd=3, lty=1, ylim=c(0,1.2))

lines(shrink, bias^2, col='green', lwd=3, lty=2)

lines(shrink, variance, col='red', lwd=3, lty=2)

legend(0.02,0.6, c('Bias^2', 'Variance', 'MSE'), col=c('green', 'red', 
                                                       'pink'), lwd=rep(3,3), lty=c(2,2,1))


