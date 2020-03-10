library(MASS)
library(leaps)

set.seed(13254675)
data(Boston)
index <- sample(nrow(Boston),nrow(Boston)*0.70)
boston.train <- Boston[index,]
boston.test <- Boston[-index,]

#Fit a linear regression model
boston.lm <- regsubsets(medv~.,data=boston.train, nbest=1, nvmax = 14)
summary(boston.lm)
plot(boston.lm, scale="bic")

full.model.lm <- lm(medv~., data=boston.train)
model_step_b <- step(full.model.lm,direction='backward')
summary(model_step_b)

model.lm.final <- lm(medv~rm+ptratio+dis+lstat+black, data=boston.train)

lm.model.summary <- summary(model.lm.final)

boston.lm.pred <- predict(object = model.lm.final, newdata = boston.test)
##MSE and MSPE_LM
boston.lm.MSE <- (lm.model.summary$sigma)^2
boston.lm.MSPE <- (mean((boston.lm.pred-boston.test$medv)^2))

library(rpart)
library(rpart.plot)
boston.rpart <- rpart(formula = medv ~ ., data = boston.train)
summary(boston.rpart)
prp(boston.rpart,digits = 4, extra = 1)
boston.train.pred.tree = predict(boston.rpart)
boston.test.pred.tree = predict(boston.rpart,boston.test)

##MSE and MSPE_Tree
boston.tree.train.MSE <- mean((boston.train.pred.tree - boston.train$medv)^2)
boston.tree.test.MSPE <- mean((boston.test.pred.tree - boston.test$medv)^2)


#Bagging
library(randomForest)

boston.bag<- randomForest(medv~., data = boston.train, mtry=13,ntree=100)
boston.bag
varImpPlot(boston.bag)

##OOB Prediction
boston.bag.oob<- randomForest(medv~., data = boston.train,mtry=13, nbagg=100)
boston.bag.oob$err.rate[,1]

##Prediction in training sample
boston.bag.pred.train <- predict(boston.bag)
boston.bag.train.MSE <- mean((boston.train$medv-boston.bag.pred.train)^2)

##Prediction in the testing sample
boston.bag.pred.test <- predict(boston.bag,newdata = boston.test)
boston.bag.test.MSPE <- mean((boston.test$medv-boston.bag.pred.test)^2)


##------------------------2. Random forests-------------------------
boston.rf1 <- randomForest(medv~.,data=boston.train,mtry=3,importance=TRUE)
boston.rf
#Higher importance IncNodePurity is better for a variables
boston.rf$importance
varImpPlot(boston.rf)

#OOB error for every number of trees from 1-500
plot(boston.rf$mse,type='l',col=2,lwd=2,xlab="ntree",ylab="OOB Error")

##Prediction on the training set
boston.rf.train.pred <- predict(boston.rf)
boston.rf.train.MSE <- mean((boston.train$medv-boston.rf.train.pred)^2)
##Prediction of the testing set
boston.rf1.pred <- predict(boston.rf1,boston.test)
boston.rf1.test.MSPE <- mean((boston.test$medv-boston.rf1.pred)^2)

#evaluate performance based on mtry arguements
oob.err <- rep(0,13)
test.err <- rep(0,13)

for(i in 1:13){
  fit <- randomForest(medv~., data=boston.train,mtry=i)
  oob.err[i] <- fit$mse[500]
  test.err[i] <- mean((boston.test$medv-predict(fit, newdata = boston.test))^2)
  cat(i," ")
}

matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))

##----3. Boosting------------------------#####
library(gbm) #also check out xgboost()
?gbm

boston.boost<- gbm(medv~., data = boston.train, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)

plot(boston.boost,i="lstat")
plot(boston.boost, i="rm")

boston.boost.pred.train<- predict(boston.boost,n.trees = 10000)
boston.boost.train.MSE <- mean((boston.train$medv-boston.boost.pred.train)^2)

boston.boost.pred.test<- predict(boston.boost, boston.test, n.trees = 10000)
boston.boost.test.MSPE <- mean((boston.test$medv-boston.boost.pred.test)^2)

##change in testing error based on number of trees

ntree <- seq(100, 10000, 100)
predmat <- predict(boston.boost,newdata=boston.test,n.trees = ntree)
err<- apply((predmat-boston.test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)
min(err)
##Final Table fo MSPE

stats.models <- data.frame("Model Name" = c("Linear Regression","Regression Tree", "Bagging", "Random Forest", "Boosting", "xgBoosting"),
                             "MSE" = c(boston.lm.MSE,boston.tree.train.MSE,boston.bag.train.MSE,boston.rf.train.MSE,boston.boost.train.MSE,boston.xgboost.train.MSE), 
                           "MSPE" = c(boston.lm.MSPE,boston.tree.test.MSPE,boston.bag.test.MSPE,boston.rf.test.MSPE,boston.boost.test.MSPE,boston.xgboost.test.MSPE))

rownames(stats.models) <- c("Linear Regression","Regression Tree", "Bagging", "Random Forest", "Boosting", "xgBoosting")
colnames(stats.models) <- c("MSE","MSPE")
model.table <- as.table(stats.models)


##Appendix
library(xgboost)
xgbst <- xgboost(data = as.matrix(boston.train[, -c(which(colnames(boston.train)=='medv'))]), label = boston.train$medv,
               max_depth = 6, eta = 0.79099,
               objective = "reg:squarederror",nrounds = 20)
pred <- predict(xgbst, as.matrix(boston.test[, -c(which(colnames(boston.test)=='medv'))]))

boston.xgboost.test.MSPE <- mean((as.matrix(boston.test[, c(which(colnames(boston.test)=='medv'))])-pred)^2)

xgb.pred.train <- predict(xgbst, as.matrix(boston.train[, -c(which(colnames(boston.train)=='medv'))]))
boston.xgboost.train.MSE <- mean((as.matrix(boston.train[, c(which(colnames(boston.train)=='medv'))])-xgb.pred.train)^2)





