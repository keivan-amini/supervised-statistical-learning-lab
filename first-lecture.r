# These R-files are scripts produced in the lab of Supervised Statistical Learning 
# course held by professor Laura Anderlucci, in the University of Bologna.
#-----------------------------------------------------

# Package installation
install.packages("ElemStatLearn")
library(ElemStatLearn)

# Loading the data
load("SAheart.RData")
summary(SAheart)

# Data visualization
pairs(SAheart[,-ncol(SAheart)], col = SAheart[,ncol(SAheart)]+1, lwd = 1.5) #all columns except the last one

#-----------------------------------------------------

# Multiple Logistic regression (out.log = output of the logistic model)
out.log <- glm (chd ~ ., data = SAheart, family = "binomial") #since Y is categorical
summary(out.log)

# now we should perform a logistic regression just considering the relevant predictors:
out.log.vs <-  glm (chd ~ tobacco+ldl+famhist+typea+age, data = SAheart, family = "binomial")
summary(out.log.vs) # we have reduced the AIC, which is good.

# estimate the test error via 5-fold cross validation. we will use a for cycle:
x <- subset(SAheart, select = c("chd","tobacco","ldl","famhist","typea","age"))
K <- 5
n <- nrow(SAheart)

set.seed(1234)
folds <- sample(1:K, n, replace = TRUE) # i want to sample n integers from 1 to K
table(folds)

err.cv <- NULL #empty vector
yhat <- rep(NA, n) #empty vector of n entries

for (i in 1:K){
  x.train<-x[folds!=i,] #all the groups different from i will constitute my training set.
  x.test<-x[folds==i,] # the i group will be the validation test
  y.test<-x$chd[folds==i]
  
  out.cv<-glm(chd~.,data=x.train,family="binomial")
  p.hat<-predict(out.cv,newdata=x.test,type="response") #posterior probability 
  y.hat<-ifelse(p.hat>0.5,1,0) #if p.hat is larger than 0.5, assign 1. otherwise, assign 0.
  
  err.cv[i]<-misc(y.hat,y.test)
  yhat[folds==i]<-y.hat
}

err.cv

# test error estimate.

mean(err.cv)
table(yhat,x$chd) #confusion matrix

#-----------------------------------------------------

# Naive Bayes Classifier: we need to load the library klaR with the function NaiveBayes
library(klaR)

K <- 5
set.seed(1234)

# two vector that will store the error with naive gaussian and naive with non parametric:
err.nb.g <- NULL #error naive bayes gaussian
err.nb.k <- NULL #error naive bayes kernel
yhat.nb.g <- rep(NA,n)
yhat.nb.k <- rep(NA,n)

for (i in 1:K){
  x.train<-x[folds!=i,-1]
  x.test<-x[folds==i,-1]
  y.test<-x$chd[folds==i]
  y.train<-x$chd[folds!=i]
  
  # Naive Bayes with Gaussian density estimate
  out.nb.g<-NaiveBayes(x=x.train,grouping=as.factor(y.train), usekernel = FALSE)
  y.hat.g<-predict(out.nb.g,newdata=x.test)$class #we want the element class
  yhat.nb.g[folds==i]<-y.hat.g
  
  # Naive Bayes with kernel density estimate
  out.nb.k<-NaiveBayes(x=x.train,grouping=as.factor(y.train), usekernel = TRUE)
  y.hat.k<-predict(out.nb.k,newdata=x.test)$class
  yhat.nb.k[folds==i]<-y.hat.k
  
  
  err.nb.g[i]<-misc(y.hat.g,y.test)
  err.nb.k[i]<-misc(y.hat.k,y.test)
}

err.nb.g
err.nb.k

# Test error estimate of Naive Bayes
mean(err.nb.g)
mean(err.nb.k)

#-----------------------------------------------------

# k-Nearest Neighbors
library(class)

#we get rid of two columns: famhist and chd since they are categorical variables and response variable rispectively.
x<- SAheart[,-c(5,10)] 
y<- SAheart[,10]

#divide the dataset into training and validation sets
set.seed(1234)
index<-sample(1:n,ceiling(n/2),replace=F)

train<-x[index,]
test<-x[-index,]
train.y<-y[index]
test.y<-y[-index]

# standardization of the test set according to the parameters of the mean and st dev of the training set.
train.std <- scale(train, T, T) # we are subtracting the mean and we are dividing for the std dev.
ntr<-nrow(train.std)

# k-fold cross validation to choose the best number of neighbors
set.seed(1234)
folds<-sample(1:K,ntr,replace=T)
table(folds)
k<-c(1,3,5,11,25,45,105)
err.cv<-matrix(NA,K,length(k),dimnames=list(NULL,paste0("k=",k)))

for(i in 1:K){
  x.train<-train.std[folds!=i,]
  y.train<-train.y[folds!=i]
  x.test<-train.std[folds==i,]
  y.test<-train.y[folds==i]
  
  for (j in 1:length(k)){
    yhat<-knn(train=x.train,test=x.test,cl=y.train,k=k[j])
    err.cv[i,j]<-misc(yhat,y.test)
  }
}

err.cv

# to select the best k, we can compute the cross validation error:
apply(err.cv,2,mean) #apply mean on every column of the matrix
which.min(apply(err.cv,2,mean))
best_k<-k[which.min(apply(err_cv,2,mean))]
best_k

# use the validation test to estimate the test error
#standardize the data
mean_x <- colMeans(train)
sd_x <- apply(train,2,sd)

test.std <- test

for (j in 1:ncol(test)){
  test.std[,j] <- (test[,j]-mean_x[j])/sd_x[j]
}

# prediction
y.hat <- knn(train = train.std, test = test.std, cl = train.y, k= best_k) #could work with k = 45
table(y.hat,test.y)
misc(y.hat,test.y)
