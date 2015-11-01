# DRIVEN DATA 
#
# Predict Blood Donations
############################
require(xgboost)
library(caret)
library(readr)

# load the data
blood <- read.csv(file.choose()) # load the "charity.csv" file

#EDA
attach(blood)

# check column names and structure of dataset
length(names(blood))
names(blood)
str(blood)

#check for missing vlaues
sum(is.na(blood[2:6])) #0

summary(blood)
head(blood,10)
cor(blood[2:6])

# Months.since.Last.Donation,
require(MASS)
par <- par(mfrow = c(3, 1))
var1 = Months.since.Last.Donation
hist(var1, col = "grey", main = "R default", ylab = "Frequency",freq = FALSE)
qqnorm(var1)
qqline(var1)
par(mfrow = c(1, 1))
boxplot(var1 ~ Made.Donation.in.March.2007, notch = TRUE, col = "grey",ylab = "inca", main = "Boxplot",boxwex = 0.5, varwidth = TRUE)
plot(Months.since.Last.Donation)

# Months.since.Last.Donation,
par <- par(mfrow = c(3, 1))
var1 = sqrt(Months.since.Last.Donation)
hist(var1, col = "grey", main = "R default", ylab = "Frequency",freq = FALSE)
qqnorm(var1)
qqline(var1)
par(mfrow = c(1, 1))
boxplot(var1 ~ Made.Donation.in.March.2007, notch = TRUE, col = "grey",ylab = "inca", main = "Boxplot",boxwex = 0.5, varwidth = TRUE)

table(Months.since.Last.Donation)
blood$Months.since.Last.Donation <- ifelse(blood$Months.since.Last.Donation >40,39,blood$Months.since.Last.Donation)

# Number.of.Donations
par <- par(mfrow = c(3, 1))
var1 = Number.of.Donations
hist(var1, col = "grey", main = "R default", ylab = "Frequency",freq = FALSE)
qqnorm(var1)
qqline(var1)
par(mfrow = c(1, 1))
boxplot(var1 ~ Made.Donation.in.March.2007, notch = TRUE, col = "grey",ylab = "inca", main = "Boxplot",boxwex = 0.5, varwidth = TRUE)

# Total.Volume.Donated..c.c..
par <- par(mfrow = c(3, 1))
var1 = (Total.Volume.Donated..c.c..)
hist(var1, col = "grey", main = "R default", ylab = "Frequency",freq = FALSE)
qqnorm(var1)
qqline(var1)
par(mfrow = c(1, 1))
boxplot(var1 ~ Made.Donation.in.March.2007, notch = TRUE, col = "grey",ylab = "Total.Volume.Donated..c.c..", main = "Boxplot",boxwex = 0.5, varwidth = TRUE)

# Months.since.First.Donation
par <- par(mfrow = c(3, 1))
var1 = (Months.since.First.Donation)
hist(var1, col = "grey", main = "R default", ylab = "Frequency",freq = FALSE)
qqnorm(var1)
qqline(var1)
par(mfrow = c(1, 1))
boxplot(var1 ~ Made.Donation.in.March.2007, notch = TRUE, col = "grey",ylab = "Total.Volume.Donated..c.c..", main = "Boxplot",boxwex = 0.5, varwidth = TRUE)
table(Months.since.First.Donation,Made.Donation.in.March.2007)

#scatterplot
pairs(cbind(Months.since.Last.Donation,Months.since.First.Donation,Total.Volume.Donated..c.c..,Number.of.Donations), gap = 0, panel = panel.smooth)

#boxcox
summary(blood)
boxcox(Number.of.Donations ~ Made.Donation.in.March.2007, data = data.boxcox,lambda = seq(-4,2, length = 10))
boxcox(Total.Volume.Donated..c.c.. ~ Made.Donation.in.March.2007, data = data.boxcox,lambda = seq(-4,2, length = 10))
boxcox(Months.since.First.Donation ~ Made.Donation.in.March.2007, data = data.boxcox,lambda = seq(-4,2, length = 10))


# implement transformations based on 
# what we see in EDA
blood$Months.since.Last.Donation <- sqrt(blood$Months.since.Last.Donation)
#blood.t$Number.of.Donations = log(Number.of.Donations)
#blood.t$Total.Volume.Donated..c.c.. = log(Total.Volume.Donated..c.c..)

# let's look at training data
#data.train <- blood.t[blood$Part=="Train",]
# split training into two holdouts
bound <- floor((nrow(blood)/5)*2)         #define % of training and test set
blood.t <- blood[sample(nrow(blood)), ]           #sample rows 
data.train <- blood.t[1:bound, ]              #get training set
data.valid <- blood.t[(bound+1):nrow(blood.t), ]    #get validation set


# set up data for analysis
x.train <- data.train[,2:5]
c.train <- data.train[,6] # 2007.donations
n.train.c <- length(c.train) # 288

#data.valid <- blood.t[blood$Part=="Valid",]
x.valid <- data.valid[,2:5]
c.valid <- data.valid[,6] # 2007 donations
n.valid.c <- length(c.valid) # 288

data.test <- read.csv("test.csv")
#data.test$Total.Volume.Donated..c.c.. <- NULL
n.test <- dim(data.test)[1] # 200
x.test <- data.test[,2:5]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, Made.Donation.in.March.2007=c.train) # to classify donors

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, Made.Donation.in.March.2007=c.valid) # to classify 2007 donors

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

####################################
#     CLASSIFICATION MODELING      #
####################################
#----------------------
# linear discriminant analysis
#----------------------
library(MASS)

#-------------------
# LDA model 1
#-------------------
model.lda1 <- lda(Made.Donation.in.March.2007 ~ Months.since.Last.Donation + Number.of.Donations, 
                  data.train.std.c) # include additMonths.since.First.Donationional terms on the fly using I()

model.lda1
plot(model.lda1)

pred.lda1 <- predict(model.lda1, data.valid.std.c,type="response")
lda.class=pred.lda1$class
#table(lda.class ,spring.valid$target)
mean(lda.class==data.valid.std.c$Made.Donation.in.March.2007)

model.lda2 <- lda(Made.Donation.in.March.2007 ~ Months.since.First.Donation + Months.since.Last.Donation + Total.Volume.Donated..c.c.., 
                  data.train.std.c) # include additMonths.since.First.Donationional terms on the fly using I()

model.lda2
plot(model.lda2)

pred.lda2 <- predict(model.lda2, data.valid.std.c,type="response")
lda.class=pred.lda2$class
#table(lda.class ,spring.valid$target)
mean(lda.class==data.valid.std.c$Made.Donation.in.March.2007)

submission <- data.frame(ID=data.test$X)
post.pred <- predict(model.lda1, data.test.std,type="response")
submission$Donate <-post.pred$posterior[,2]

cat("saving the submission file\n")
write_csv(submission, "blood_lda3.csv")

############################
#  LOG
############################
model.log <- glm(Made.Donation.in.March.2007 ~ Months.since.Last.Donation + Number.of.Donations, 
                 data.train.std.c,family=binomial("logit"))

# review the results and select best 
# p values that have the best AIC score
summary(model.log)

#  model predictions
post.valid.log <- predict(model.log, data.valid.std.c, type="response") 

# get prediction accuracy
glm.pred=rep (0 ,346)
glm.pred[post.valid.log >.5]=1
#table(glm.pred,data.valid.std.c$target)
#(22900 + 95)/30000
# [1] 0.7683044
mean(glm.pred==data.valid.std.c$Made.Donation.in.March.2007)

##################
#     TREES
##################
library(tree)

#---------------
#   model 1
#---------------
donr.class=ifelse (c.train ==0,"No","Yes")
data.train.std.c.tree =data.frame(data.train.std.c ,donr.class)
model.tree1 =tree(donr.class ~ Number.of.Donations + Total.Volume.Donated..c.c.. + Months.since.First.Donation +
                    Months.since.Last.Donation 
                  -Made.Donation.in.March.2007, data.train.std.c.tree )
summary (model.tree1)
par(mfrow = c(1, 1))
plot(model.tree1)
text(model.tree1,pretty =0)

donr.class=ifelse (c.valid ==0,"No","Yes")
data.valid.std.c.tree =data.frame(data.valid.std.c ,donr.class)
post.valid.tree1 = predict(model.tree1 ,data.valid.std.c.tree ,type ="class")
table(post.valid.tree1 ,donr.class)
#donr.class
#post.valid.tree1  No Yes
              #No  222  25
              #Yes  17  24
#pred_accuracy = (222+ 24) /288  = .8541667 correct predictions of donors

# prune the tree to see if reuslts improve
set.seed(2)
cv.blood = cv.tree(model.tree1 ,FUN=prune.misclass)
cv.blood

# the tree with 4 terminal nodes results
# in lowest cross-validation error rate w/ 86 errors
# plot the error rate
par(mfrow =c(1,2))
plot(cv.blood$size ,cv.blood$dev ,type="b")
plot(cv.charity$k ,cv.blood$dev ,type="b")

# prune the tree to obtain a 4 node tree
prune.blood =prune.misclass (model.tree1 ,best=4)
plot(prune.blood)
text(prune.blood ,pretty =0)

model.tree2=predict(prune.blood, data.valid.std.c.tree,type="class")
table(model.tree2,donr.class)
#donr.class
#model.tree2  No Yes
        #No  223  28
        #Yes  16  21
#pred_accuracy = (223 + 28) /288  = .8715278 correct predictions of donors


###################
# REgression Trees
###################

#To Do

########################
#     BAGGING
########################
library (randomForest)
set.seed (1)
model.bag1 =randomForest(Made.Donation.in.March.2007 ~.,data=data.train.std.c , importance =TRUE)
model.bag1
# Mean of squared residuals: 0.2115465
# % Var explained: 0.93

post.valid.bag1 = predict(model.bag1,newdata = data.valid.std.c )
plot(post.valid.bag1,c.valid)
abline (0,1)
mean((post.valid.bag1-c.valid)^2)
# [1] 0.1345902

set.seed (1)
model.bag2 =randomForest(Made.Donation.in.March.2007 ~.,data=data.train.std.c , 
                         mtry=1, ntree=25)
model.bag2
#Mean of squared residuals: 0.2183862
#% Var explained: -2.27

post.valid.bag2 = predict(model.bag2,newdata = data.valid.std.c )
mean((post.valid.bag2-c.valid)^2)
# [1] 0.1396108

set.seed (1)
rf.blood = randomForest(Made.Donation.in.March.2007 ~.,data=data.train.std.c, 
                        mtry=1, importance=TRUE)
importance (rf.blood )
varImpPlot (rf.blood )

########################
#         BOOSTING
########################
library (gbm)
set.seed (1)
model.boo1 =gbm(Made.Donation.in.March.2007 ~ Months.since.First.Donation +
                Months.since.Last.Donation + Number.of.Donations,data=data.train.std.c, 
                distribution="bernoulli",
                n.trees =5000 , interaction.depth =4)
summary(model.boo1)
par(mfrow =c(1,2))
plot(model.boo1,i="Months.since.First.Donation")
plot(model.boo1,i="Number.of.Donations")

post.valid.boo1 = predict (model.boo1,newdata=data.valid.std.c,
                           n.trees =5000)
mean((post.valid.boo1 - c.valid)^2)
# [1] 2.70933

# model 2
set.seed (1)
model.boo2 = gbm(Made.Donation.in.March.2007 ~ Months.since.First.Donation +
                  Months.since.Last.Donation + Number.of.Donations,data=data.train.std.c, 
                distribution="bernoulli",
                n.trees =5000 , interaction.depth = 3,shrinkage =0.01,
                verbose =F)

post.valid.boo2 = predict (model.boo2,newdata=data.valid.std.c,
                           n.trees =5000)
mean((post.valid.boo2 - c.valid)^2)
# [1] 4.279558

# select model.boo1 since it has maximum profit in the validation sample
profit.boo2 <- cumsum(1*c.valid[order(post.valid.boo2, decreasing=T)])
post.test.boo <- predict(model.boo2, data.test.std,n.trees =5000) # n.valid.c post probs

# Oversampling adjustment for calculating number of mailings for test set
n.donors.valid <- which.max(profit.boo2)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.donors.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.donors.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.donors.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test.boo, decreasing=T)[n.donors.test+1] # set cutoff based on n.mail.valid
chat.test <- ifelse(post.test.boo>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test) # classification table
#   0    1 
#   43 157
# based on this model we'll mail to the 157 highest posterior probabilities

# See below for saving chat.test into a file for submission
ip <- data.frame(chat=chat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="blood_boo1.csv", row.names=FALSE) # use your initials for the file name
