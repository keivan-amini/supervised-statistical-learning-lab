# These R-files are scripts produced in the lab of Supervised Statistical Learning 
# course held by professor Laura Anderlucci, in the University of Bologna.
#-----------------------------------------------------

# Neural Networks ####

summary(zip.train)

install.packages("nnet")
library(nnet)

zip.df <- data.frame(y=as.factor(zip.train[,1]),zip.train[,-1])
zip.test.df <- data.frame(y=as.factor(zip.test[,1]),zip.test[,-1])

set.seed(1234)
out.nnet<- nnet(y~.,data=zip.df,size = 16,MaxNWts=4500,maxit = 1000)

y.hat.tr <- predict(out.nnet,zip.df[,-1],type = "class")
y.hat.test <- predict(out.nnet,zip.test.df[,-1],type = "class")

table(y.hat.tr,zip.df$y)
misc(y.hat.tr,zip.df$y)

misc(y.hat.test,zip.test.df$y)

# Comparison with Linear Discriminant Analysis ####

library(MASS)
out.lda <- lda(y~.,zip.df)
y.hat.lda <- predict(out.lda,zip.test.df,type="class")

misc(y.hat.lda$class,zip.test.df$y) # the error is bigger, but not that much bigger.




