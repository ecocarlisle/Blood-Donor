############################
# DRIVEN DATA 
#
# Compettion: Predict Blood Donations
# TEAM: Jon Carlisle & Joshus Meek
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

# Months.since.Last.Donation transformed
# note, not sold on this transformation
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
boxplot(var1 ~ Made.Donation.in.March.2007, notch = TRUE, col = "grey",ylab = "Number.of.Donations", main = "Boxplot",boxwex = 0.5, varwidth = TRUE)

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
boxplot(var1 ~ Made.Donation.in.March.2007, notch = TRUE, col = "grey",ylab = "First Donation", main = "Boxplot",boxwex = 0.5, varwidth = TRUE)
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
#   MODEL 1
#-------------------
model.lda1 <- lda(Made.Donation.in.March.2007 ~ Months.since.Last.Donation + Number.of.Donations, 
                  data.train.std.c) # include additMonths.since.First.Donationional terms on the fly using I()

model.lda1
plot(model.lda1)

pred.lda1 <- predict(model.lda1, data.valid.std.c,type="response")
lda.class=pred.lda1$class
#table(lda.class ,spring.valid$target)
mean(lda.class==data.valid.std.c$Made.Donation.in.March.2007)

#-------------------
#   MODEL 2
#-------------------
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

