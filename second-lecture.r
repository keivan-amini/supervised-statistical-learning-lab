# These R-files are scripts produced in the lab of Supervised Statistical Learning 
# course held by professor Laura Anderlucci, in the University of Bologna.
#-----------------------------------------------------

## Load the dataframe

library(ElemStatLearn)
load("SAheart.RData")
source("misc.R")
summary(SAheart)


## Divide the dataset into training and validation sets.

n <-nrow(SAheart) #number of observations
set.seed(1234)
index <-sample(1:n,ceiling(n/2),replace=F) #vector with n/2 entries that randomly contains number between 1 and n
training_set <- SAheart[index,] 
validation_set <- SAheart[-index,] 
y_validation_set <- SAheart$chd[-index]

## Perform Linear Discriminant Analysis ####

library(MASS) 
out.lda<-lda(chd~.,data=training_set)
out.lda
y.hat <- predict(out.lda, newdata= validation_set)$class #class will categorize the variables
table(y.hat, y_validation_set)
misc(y.hat, y_validation_set) #estimate test error


## Comparison with logistic regression
out.log<-glm(chd~.,data=training_set, family = "binomial")
p.hat.lr <- predict(out.log, newdata= validation_set, type ="response") #class will categorize the variables
y.hat.lr <- ifelse(p.hat.lr>0.5,1,0)
table(y.hat, y.hat.lr)

misc(y.hat.lr, y_validation_set)

#-----------------------------------------------------

## Model Selection 
load("prostate.RData")
head(prostate)

x<- prostate[,-ncol(prostate)] #we do not want the last column
p<- ncol(x)-1 #number of features

summary(x) #we got rid of the train variable.

#there is a function for the subset selection. we must load a library
install.packages('leaps')
library(leaps)

## Best subset selection ####
regfit.full <- regsubsets(lpsa~.,x) #same syntax we have seen
reg.summary <- summary(regfit.full) #how many variables do we include?
names(reg.summary)

reg.summary$rsq #but in fact, we should compute which number of parameters is the best considering rss, adjr2, bic and cp.

which.max(reg.summary$adjr2) #output 7 -> we choose the model with 7 predictors
which.min(reg.summary$bic) #outputs 3
which.min(reg.summary$cp) #outputs 5

#let's make a plot, in order to understand
par(mfrow=c(2,2)) #divides the plot windows into 4 regions

plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l") 
plot(reg.summary$adjr2,xlab ="Number of Variables",ylab="Adjusted RSq",type="l") #high
plot(reg.summary$bic,xlab ="Number of Variables",ylab="BIC",type="l")
plot(reg.summary$cp,xlab ="Number of Variables",ylab="C_p",type="l") #low


coef(regfit.full, 7) #outputs the vector of regression coefficient with the model with 7 predictors.


## Forward selection (the syntax is basically the same) ####
regfit.fwd <- regsubsets(lpsa~.,x, method = "forward")
summary(regfit.fwd)


## Backward selection ####
regfit.bkw <- regsubsets(lpsa~.,x, method = "backward")
summary(regfit.bkw)


## Hybrid selection ####
regfit.seqrep <- regsubsets(lpsa~.,x, method = "seqrep")
summary(regfit.seqrep)


## Choose the best model
# we can select the best model via validation set approach and cross-validation, in order to estimate the test error.
# unfortunately, the regsubset fuction does not include a predict function.

# Validation set Approach ####
x<-prostate[,-ncol(prostate)]
set.seed(1234)
train<-sample(c(TRUE,FALSE), nrow(x),replace=TRUE)
test<-(!train)

regfit.best <- regsubsets(lpsa~.,data = x[train,])


test.mat <- model.matrix(lpsa~.,data = x[test,]) #syntax of the regression

mse.valid <- rep(NA,8) #prepare a void vector: this will contain the MSE for each model

for (i in 1:8) {
  coefi <- coef(regfit.best,id = i) #id = model size
  yhat <- test.mat[,names(coefi)]%*%coefi #matrix product
  mse.valid[i] <- mean((x$lpsa[test]-yhat)^2)
}

which.min(mse.valid) #model with 5 predictors

## Cross Validation ####

#create a function. input: output of the regsubsets function, the new data onto which we want to do prediction, and the model size. output: prediction
predict.regsubsets <- function(object, newdata, id,...) {
  form <- as.formula(object$call[[2]]) # we specify the formula of the model
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id= id)
  xvars <- names(coefi)
  yhat <- c(mat[,xvars]%*%coefi)
  return(yhat)
}

k<- 5
set.seed(1234)
folds <- sample(1:k,nrow(x),replace=TRUE)

cv.error <- matrix(NA,k,p, dimnames= list(NULL,paste(1:p)))

for (i in 1:k) {
  train.x <- x[folds!=i,]
  test.x <- x[folds==i,]
  best.fit <- regsubsets(lpsa~.,train.x)
  for (j in 1:8) {
    pred <- predict.regsubsets(best.fit, test.x, j)
    cv.error[i,j] <- mean((test.x$lpsa-pred)^2)
  }
}
 
cv.error # 5 × 8 matrix, of which the (i, j)th element corresponds to the test MSE for the ith cross-validation fold for the best j-variable model.
mean.cv.errors <- apply(cv.error,2,mean)
par(mfrow=c(1,1))
plot(mean.cv.errors,type='b')

which.min(apply(cv.error,2,mean)) #3

best.model <- regsubsets(lpsa~.,x)
best.bhat3 <- coef(best.model,id = 3)
best.bhat3

#-----------------------------------------------------
