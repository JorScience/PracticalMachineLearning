# Library
library(ggplot2)
library(scales)
library(dplyr)
library(magrittr)
library(caret)
library(MASS)
library(nnet)
library(randomForest)
library(parallel)
library(doParallel)
library(e1071)

# Working directory
setwd("D:/R Folder/Coursera")

# Load test- and train datasets
test <- read.csv("pml-testing.csv")
train <- read.csv("pml-training.csv")

# Remove response and unnecessary variables
response <- train[,160]
train <- train[,-c(1:7, 160)]
test <- test[,-c(1:7, 160)]

# Missing values
dfm <- rbind(train,test) %>%
      apply(2, as.numeric) %>%
      apply(2, function(x) sum(is.na(x))) %>%
      {data.frame(Names=names(.), MV=.)}

table(dfm$MV)

dfm <- dfm %>%
      filter(MV >0) %>%
      .$Names
      
train <- train[,!(names(train) %in% dfm)]
test <- test[,!(names(test) %in% dfm)]
rm(dfm)

# Near zero variance
nearZeroVar(train, saveMetrics=F)

# Split training set
inTrain <- createDataPartition(response, p=3/4, list=F)
valid <- train[-inTrain,]
train <- train[inTrain,]
responseT <- response[inTrain]
responseV <- response[-inTrain]
rm(inTrain, response)

# Training parameters
ctrl <- trainControl(method="cv", number=10, allowParallel=T)
preP <- preProcess(train, method=c("center", "scale"))
train_stand <- predict(preP, train)

# Enable parallelized computing
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

# Multinomial logistic regression
set.seed(2017)
logfit <- train(x=train_stand, y=responseT, method="multinom", 
                trControl=ctrl, trace=FALSE)

# Linear discriminant analysis
set.seed(2017)
ldafit <- train(x=train_stand, y=responseT, method="lda", 
                trControl=ctrl)

# Random forest
set.seed(2017)
rffit <- train(x=train_stand, y=responseT, method="rf", 
               trControl=ctrl)

# Support vector machine with linear kernel
set.seed(2017)
lsvmfit <- train(x=train_stand, y=responseT, method="svmLinear", tunelength=14, 
                trControl=ctrl)

# XGBOOST
set.seed(2017)
xgbfit <- train(x=train_stand, y=responseT, method="xgbLinear",  
                trControl=ctrl, nthread=1)

# Stop parallel
stopCluster(cluster)
rm(cluster)

#
allResamples <- resamples(list("Multinom log" = logfit,
                               "LDA" = ldafit,
                               "RandomForest" = rffit,
                               "Linear SVM" = lsvmfit,
                               "XGBoost" = xgbfit)
)

summary(allResamples)
parallelplot(allResamples)

confusionMatrix(xgbfit)
#parallelplot(allResamples, metric = "Rsquared")



