# These R-files are scripts produced in the lab of Supervised Statistical Learning 
# course held by professor Laura Anderlucci, in the University of Bologna.
#-----------------------------------------------------

# Load data set

load("prostate.RData")

x <- prostate[,-ncol(prostate)] #remove last column
p <- ncol(x)-1
n <- nrow(x)


set.seed(1234)
train <- sample(1:n,ceiling(n/2))
x_test <- x[-train,]


# Regression Trees ####

install.packages("tree")
library(tree)

tree_prostate <- tree(lpsa~.,x,subset = train)
summary(tree_prostate)
tree_prostate

plot(tree_prostate)
text(tree_prostate,digits=3)

# compute the mse
tree_pred <- predict(tree_prostate, x_test)
mse <- mean((tree_pred-x_test$lpsa)^2)
mse

# perform cross validation to choose the number of terminal nodes
set.seed(1234)
cvtree_prostate <- cv.tree(tree_prostate, K=5, FUN=prune.tree)

best_terminal <- cvtree_prostate$size[which.min(cvtree_prostate$dev)]
best_terminal

prune_prostate <- prune.tree(tree_prostate, best=best_terminal)
plot(prune_prostate)
text(prune_prostate)

pruned_tree_pred <- predict(prune_prostate, x_test)
mse <- mean((pruned_tree_pred-x_test$lpsa)^2)
mse #smaller than before: pruned_tree is better than tree_prostate as a regression tree.
#-----------------------------------------------------

# Classification Trees ####

load("SAheart.RData")
data("SAheart")

summary(SAheart)

p <- ncol(SAheart)-1
n <- nrow(SAheart)

y <- SAheart$chd
x <- SAheart[,-10]

heart<-data.frame(chd=as.factor(y),x) #we admit only two levels: NECESSARY for classification trees

set.seed(1234)
train <- sample(1:n,ceiling(n/2))
heart_test <- heart[-train,]

tree_heart<-tree(chd~.,heart,subset=train)
summary(tree_heart)
plot(tree_heart)
text(tree_heart, pretty=0) #pretty=0 makes the write "absent" or "present for the famhist predictor

tree_pred <- predict(tree_heart, heart_test, type = "class")
table(tree_pred, heart_test$chd)

misc<-function(yhat,y){
  if (length(table(yhat))!=length(table(y)))
    stop("The levels of the two vectors do not match")
  1-sum(diag(table(yhat,y)))/length(y)
}

misc(tree_pred, heart_test$chd)

# Prune with CV

cv_heart <- cv.tree(tree_heart, FUN = prune.misclass)
prune_heart <- prune.misclass(tree_heart, best = 5) #5 is chosen since it is the size of the entries with less deviance
plot(prune_heart)
text(prune_heart,pretty=0)


tree_pred <- predict(prune_heart, type="class")
table(tree_pred,heart_test$chd)

misc(tree_pred,heart_test$chd)
#-----------------------------------------------------


# Bagging ####
# we need to install the package

install.packages("randomForest")
library(randomForest)

set.seed(1234)
bag.heart <- randomForest(chd~.,heart,subset=train, mtry = ncol(heart)-1, importance=TRUE) #same arguments of tree + mtry and importance.
bag.heart

importance(bag.heart) # the higher the decreases, the most importants are the variables.
varImpPlot(bag.heart) #ldl and tobacco are the most important variables.

#estimate test error
yhat.bag <- predict(bag.heart,newdata= heart[-train,])
table(yhat.bag,heart_test$chd)
misc(yhat.bag,heart_test$chd)
#-----------------------------------------------------


# Random Forest ####
# basically we copy and paste. here, we will use as the number of variables sqrt(m)

set.seed(1234)
rf.heart <- randomForest(chd~.,heart,subset=train, importance=TRUE) #removed mtry
rf.heart

importance(rf.heart)
varImpPlot(rf.heart)

yhat.rf <- predict(rf.heart,newdata= heart[-train,])
table(yhat.rf,heart_test$chd)
misc(yhat.rf,heart_test$chd)
#-----------------------------------------------------


# Boosting ####
install.packages("gbm")
library(gbm)

heart<-data.frame(chd=y,x)
heart.test <- SAheart[-train,]
heart.train <- SAheart[train,]

out.boost <- gbm(chd~.,data = heart[train,], distribution = "bernoulli", n.trees=100, interaction.depth = 4, bag.fraction = 1)
out.boost

summary(out.boost)

# now tune the parameters with cv
K = 5
set.seed(1234)
folds <- sample(1:K, length(train), replace=TRUE)
table(folds)

# vector with the best number of trees to choose on
B <- c(25,50,100,150)
err.cv <- matrix(NA,K,length(B)) 
for (k in 1:K){
  x.test <- heart.train[folds==k,]
  x.train <-heart.train[folds!=k,]
  
  for ( b in 1:length(B)) {
    boost.out <- gbm(chd~.,x.train, distribution= "bernoulli", n.trees = B[b], interaction.depth=4, bag.fraction=1)
    p.hat <- predict(boost.out, newdata=x.test, n.trees=B[b],type="response")
    yhat<-ifelse(p.hat >0.5,1,0)
    err.cv[k,b] <- 1-mean(yhat==x.test$chd) #it is the misc function: 1- units correctly classified.
     
    }
}

err.cv
colMeans(err.cv) #the first b has the lowest error ->  b = 25

b_best<-B[which.min(colMeans(err.cv))]
b_best

# let use boosting with b=best_min

boost.heart <- gbm(chd~.,heart.train, distribution= "bernoulli", n.trees = b_best, interaction.depth=4, bag.fraction=1)

p.hat.test <- predict(boost.heart,newdata=heart.test,n.trees=b_best, type="response")
y.hat.test<- ifelse(p.hat.test>0.5,1,0)

table(y.hat.test,heart.test$chd)
misc(y.hat.test,heart.test$chd)
#-----------------------------------------------------

# Support Vector Machines ####


# Support Vector Classifier (linear)
install.packages("e1071")
library(e1071)

heart <- data.frame(chd=as.factor(y),x) #since we want classification, not regression

svmfit <- svm(chd~.,data=heart, kernel="linear",cost=10) #increase cost -> more expensive to make mistake -> less support vectors. In R, cost = "make a mistake is not a big deal"
svmfit$index #index of support vectors

summary(svmfit)

#the package e1071 does not need the for cycle to prove test error with different costs!
set.seed(1234)
tune_out <- tune(svm,chd~.,data=heart[train,],kernel="linear",ranges=list(cost=c( 0.001,0.01,0.1,1,5,10,100)))
tune_out #uses 10 fold cross validation. best cost is 0.1

best_model <- tune_out$best.model

yhat <- predict(best_model,heart[-train,])
table(yhat,heart$chd[-train])
misc(yhat,heart$chd[-train])
#-----------------------------------------------------

# Radial Kernel
svmfit <- svm(chd~.,data=heart[train,], kernel="radial",gamma=1, cost=1) 
summary(svmfit)

#tune parameters
set.seed(1234)
svm_out <- tune(svm,chd~.,data=heart[train,],kernel="radial",ranges=list(cost=c(0.1,1,10,100,1000), gamma= c(1,2,3,4,5)))
summary(svm_out)

best_model <- svm_out$best.model
yhat.rad <- predict(best_model,newdata= heart[-train,])
table(yhat.rad,heart$chd[-train])
misc(yhat.rad,heart$chd[-train])
#-----------------------------------------------------

# Polynomial kernel
svmfit <- svm(chd~.,data=heart[train,], kernel="polynomial",gamma=1, cost=1) 
summary(svmfit)

#tune parameters
set.seed(1234)
svm_out <- tune(svm,chd~.,data=heart[train,],kernel="polynomial",ranges=list(cost=c(0.1,1,10,100,1000), d= c(1,2,3,4,5)))
summary(svm_out)

best_model <- svm_out$best.model
yhat.pol <- predict(best_model,newdata= heart[-train,])
table(yhat.pol,heart$chd[-train])
misc(yhat.pol,heart$chd[-train])
#-----------------------------------------------------

# Overall best Support Vector Machines algorithm!

set.seed(1234)
svm_out <- tune(svm,chd~.,data=heart[train,],ranges=list(cost=c(0.1,1,10), kernel=c("linear","radial","polynomial")))
summary(svm_out) #linear kernel is the best

best_model <- svm_out$best.model
yhat.overall <- predict(best_model,newdata= heart[-train,])
table(yhat.overall,heart$chd[-train])
misc(yhat.overall,heart$chd[-train])


