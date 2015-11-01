# This script creates a sample submission using the excellent Xgboost algorithm in R.
# Download xgboost for R : https://github.com/dmlc/xgboost/tree/master/R-package
# Xgboost parameters taken from author Devin's script (thanks Devin !). 
#
# To submit the sample, download xgbboost_r_benchmark.csv.gz
# from the Output tab and submit it as normal to the competition
#
# @ author Darragh

# load packages
require(xgboost)
library(caret)
library(readr)

# load raw data
train = read_csv('train.csv')
test = read_csv('test.csv')

########
# EDA
########
attach(train)
oldpar <- par(mfrow = c(3, 1))
var1 =(train$`Months since Last Donation`)
var1 =(train$`Number of Donations`)
hist(var1, col = "grey", main = "R default", ylab = "Frequency",
     freq = FALSE)
lines(density(var1), lwd = 2)
lines(density(var1, adjust = 0.5), lwd = 1)
rug(var1)
box()
qqnorm(var1)
qqline(var1)
train$`Made Donation in March 2007`

#boxplots
par(mfrow = c(2, 2))
boxplot(var1~`Made Donation in March 2007`, notch = FALSE, col = "grey", ylab = "var",main = "Boxplot of ...", boxwex = 0.5)
table(var1)
table(var1,`Made Donation in March 2007`)




y = train$`Made Donation in March 2007`

# We'll convert all the characters to factors so we can train a randomForest model on them
mtrain = train[,-c(1,6)]
mtest = test[,-c(1)]
dummies <- dummyVars(~ ., data = mtrain)
mtrain = predict(dummies, newdata = mtrain)
mtest = predict(dummies, newdata = mtest)


# Set necessary parameters and use parallel threads
param <- list("objective" = "binary:logistic", "verbose"=0)

clf <- xgboost(param       =  param,
               data        =  mtrain,
               label       = y,
               nrounds     = 1700,
               subsample   = 0.8,
               max.depth   = 6, 
               eta         = 0.1,
               verbose     = 1,
               early.stop.round    = 20,
               objective   = "binary:logistic",
               eval_metric = "auc")


submission <- data.frame(ID=test$`[EMPTY]`)
submission$`Made Donation in March 2007` <- predict(clf, mtest)

cat("saving the submission file\n")
write_csv(submission, "blood_xgb1.csv")

# Predict Hazard for the test set
submission <- data.frame(Id=test$`[EMPTY]`)
submission$Hazard <- predict(xgb.fit1, mtest)+predict(xgb.fit2, mtest)+predict(xgb.fit3, mtest)+predict(xgb.fit4, mtest)+predict(xgb.fit5, mtest)+predict(xgb.fit6, mtest)
write_csv(submission, "xgbboost_r_benchmark.csv")
