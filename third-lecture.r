# These R-files are scripts produced in the lab of Supervised Statistical Learning 
# course held by professor Laura Anderlucci, in the University of Bologna.
#-----------------------------------------------------

# Load the data

names(prostate)
x<-prostate[,-ncol(prostate)] 
p <- ncol(x) - 1

install.packages('glmnet')
library(glmnet)

## Ridge Regression ####

x<-model.matrix(lpsa~.,prostate[,-ncol(prostate)])[,-1]
y <- prostate$lpsa

grid <- 10^(seq(10,-2,length=100)) #grid of values for lambda
ridge.mod<-glmnet(x, y, alpha=0, lambda=grid) #alpha=0 is ridge. alpha=1 is lasso
coef(ridge.mod) #access the coefficients, for every value of lambda
ridge.mod$lambda[50] # value of lambda for the 50th element
sqrt(sum(coef(ridge.mod)[-1,50]^2)) #l2 norm

predict(ridge.mod,s=50, type ="coefficients")[1:9,] #ridge regression coefficient for lambda=50

# Test error

set.seed(1234)
train <- sample(1:nrow(x),ceiling(nrow(x)/2))
test <- (-train)
y.test <- y[test]

ridge.mod<-glmnet(x[train,], y[train], alpha=0, lambda=grid,thresh=1e-12)
ridge.pred<-predict(ridge.mod, s=4, newx=x[test,])#evaluate its MSE on the test set, using lambda = 4.
mean((ridge.pred-y.test)^2)

ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])#evaluate its MSE on the test set, using lambda = 10^10.
mean((ridge.pred-y.test)^2)

# ols
ridge.pred=predict(ridge.mod, s=0, newx=x[test,], exact=T,x=x[train,],y=y[train])
mean((ridge.pred-y.test)^2)

# Choice of lambda: cross validation
set.seed(1234)
cv.out <- cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)

best.lambda<-cv.out$lambda.min
best.lambda

#test mse with this value of lambda
ridge.pred=predict(ridge.mod,s=best.lambda,newx=x[test,],x=x[train,],y=y[train])
mean((ridge.pred-y.test)^2)

#refit our model on the full dataset with the lambda chosen by cross-validation
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=best.lambda)[1:9,]
#-----------------------------------------------------

# Lasso ####
# we use the same function glmnet, but in the lasso alpha = 1.

lasso.mod = glmnet(x[train,], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)

# choice of lambda via cross validation
set.seed(1234)
cv.out <- cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)

best.lambda<-cv.out$lambda.min
best.lambda

lasso.pred=predict(lasso.mod,s=best.lambda,newx=x[test,],x=x[train,],y=y[train])
mean((lasso.pred-y.test)^2)

out=glmnet(x,y,alpha=1, lamda=grid)
lasso.coef = predict(out,type="coefficients",s=best.lambda)[1:9,]
lasso.coef #some of them are exactly 0! 
#-----------------------------------------------------

# Dimension Reduction: Principal Components Regression ####
library(pls)

set.seed(1234)

pcr.fit <- pcr(lpsa~., data = prostate[,-ncol(prostate)], scale=TRUE, validation ="CV") #pcr without the last column of the df. 10 fold cv.
summary(pcr.fit) #outputs root mean squared error for each possible number of components

validationplot(pcr.fit, val.type="MSEP", legendpos = "top")

#to choose the optimal number of components we have two ways
ncomp.onesigma <- selectNcomp(pcr.fit, method = "onesigma", plot = TRUE)
ncomp.permut <- selectNcomp(pcr.fit, method = "randomization", plot = TRUE)

#test error estimate of PCR
set.seed(1234)
train <- sample(1:nrow(x),nrow(x)/2)
test <- (-train)
y.test <- prostate[test,9]

set.seed(1234)
pcr.fit<-pcr(lpsa~., data=prostate[,-ncol(prostate)], subset= train, scale =TRUE, validation ="CV")
validationplot(pcr.fit, val.type="MSEP",legendpos = "top") # M = 5 is the best

pcr.pred<-predict(pcr.fit, prostate[test,1:8], ncomp=5)
mean((pcr.pred-y.test)^2)

## It is possible to compute PCR also via eigen() and via svd()

# fit PCR on the full data frame with M = 5
pcr.fit<-pcr(y~x, scale=TRUE, ncomp=5)
summary(pcr.fit)
