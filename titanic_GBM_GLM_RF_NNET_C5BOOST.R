
library(caret)
library(randomForest)
library(rpart.plot)
library(rattle)
library(lars)
library(leaps)
library(DMwR)
library(gbm)
library(relaxo)
library(kernlab)
library(stringr)
library(C50)
library(nnet)

# Read/write test and train sets
ttrain.uc.df <- read.csv("ttrain.csv")
ttest.uc.df <- read.csv( "ttest.csv")

ttrain.uc.df$Embarked[270] <- "S"
ttrain.uc.df$Embarked[271] <- "S"
ttrain.uc.df$Embarked <- factor(ttrain.uc.df$Embarked, levels = c("C", "Q", "S"))

ttrain.c.df <- ttrain.uc.df
ttest.c.df <- ttest.uc.df

# Creating Status Variable
ttrain.c.df$Status <- ifelse(grepl('Mr\\.|Col\\. |Major\\.|Rev\\.|Dr\\.|Sir\\.|Don\\.|Capt\\.|Jonkheer\\.',ttrain.c.df$Name),'Mr',ifelse(grepl('Mrs\\.|Lady\\. |Countess\\.|Mme\\.|Dona\\.',ttrain.c.df$Name),'Mrs',ifelse(grepl('Miss\\.|Mlle\\.|Ms\\.',ttrain.c.df$Name),'Miss',ifelse(grepl('Master', ttrain.c.df$Name),'Boy', 'None')))) 
ttrain.c.df$Status <- factor(ttrain.c.df$Status)

ttest.c.df$Status <- ifelse(grepl('Mr\\.|Col\\. |Major\\.|Rev\\.|Dr\\.|Sir\\.|Don\\.|Capt\\.|Jonkheer\\.',ttest.c.df$Name),'Mr',ifelse(grepl('Mrs\\.|Lady\\. |Countess\\.|Mme\\.|Dona\\.',ttest.c.df$Name),'Mrs',ifelse(grepl('Miss\\.|Mlle\\.|Ms\\.',ttest.c.df$Name),'Miss', ifelse(grepl('Master', ttest.c.df$Name),'Boy', 'None'))))
ttest.c.df$Status <- factor(ttest.c.df$Status)

### Function for imputing meadian, mean or mode to X predictor grouped by Y predictor
impute_by_group <- function(x,y, FUN = median,...) {
  y_levels <- levels(y)
  for ( i in 1: length(y_unique)) {
  x[as.logical(is.na(x)*(y == y_levels[i])) ] <-FUN(x[which(y == y_levels[i])], na.rm = TRUE)  
  }
  return (x)
}

### Imputing median age by status groups??? or
ttrain.c.df$Age <- impute_by_group(ttrain.c.df$Age, ttrain.c.df$Status)

### Or imputing mean age by status groups !!!!!
ttrain.c.df$Age <- impute_by_group(ttrain.c.df$Age, ttrain.c.df$Status, FUN = mean)


#### Imputing mdeian fare to replace NA in the 3rd class
ttest.c.df$Fare[153] <-median(ttrain.c.df$Fare[which(ttrain.c.df$Pclass == 3)])


#### Other option for Imputation: Imputing missing values in training set with KNN  ??????
ttrain.c.df <- knnImputation(ttrain.c.df, k = 9 )
ttest.c.df <- knnImputation(ttest.c.df, k = 9)


##### Creating Child variable !!!!!
ttrain.c.df$Child <- ifelse(ttrain.c.df$Age < 16, 1, 0)
ttest.c.df$Child <- ifelse(ttest.c.df$Age < 16, 1, 0)

##### Creating FamSize variable !!!!!!
ttrain.c.df$FamSize <- ttrain.c.df$SibSp + ttrain.c.df$Parch + 1

ttest.c.df$FamSize <- ttest.c.df$SibSp + ttest.c.df$Parch + 1

# Logging Fare !!!!!
ttrain.c.df$Fare <- log(ttrain.c.df$Fare+1)
ttest.c.df$Fare <- as.numeric(ttest.c.df$Fare)
ttest.c.df$Fare <- log(ttest.c.df$Fare+1)

# Adding outcome to training set
ttrain.c.df$Survived <- ifelse(ttrain.uc.df$Survived == 0, "No", "Yes")
ttrain.c.df$Survived <- as.factor(ttrain.c.df$Survived)


### Random Forest: Survived Pclass    Sex     Fare Embarked Status FamSize
set.seed(1234)
rf.fit <- randomForest(Survived ~., ttrain.c.df[,-c(1,4,6,7,8,9,11,14)], ntree = 400)
rf.fit
                  # Best Fit so far

# Call:
#  randomForest(formula = Survived ~ ., data = ttrain.c.df[, -c(1,      4, 6, 7, 8, 9, 11, 14)], ntree = 500) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 2

# OOB estimate of  error rate: 17.28%
# Confusion matrix:
#       No Yes class.error
# No  505  44  0.08014572
# Yes 110 232  0.32163743


varImpPlot(rf.fit, main = "RF_MODEL")

rf.predict <- predict(rf.fit, newdata = ttest.c.df[,-c(1,3,5,8,10)])
rf.predict2 <- ifelse(rf.predict == "Yes", 1, 0)
rf.solution <- data.frame(PassengerId = ttest.uc.df$PassengerId, Survived = rf.predict2)
write.csv(rf.solution, "titanic_41_rf_t500.csv", row.names = FALSE)

# RF CARET

ctrl <- trainControl(method = "oob", number = 10, classProbs = TRUE,returnResamp='none', summaryFunction = twoClassSummary )
rf.grid <- expand.grid(.mtry = c(1,2,3,4,5,6) )
set.seed(1234)
rf.caret.Fit <- train(ttrain.c.df[,-c(1,2,4,6,7,8,9,11,14)],ttrain.c.df[,2], trControl = ctrl, method = "rf",
                      metric = "Accuracy", preProc = c("center", "scale"), tuneGrid = rf.grid, verbose = FALSE)

rf.predict <- predict(rf.caret.Fit, newdata = ttest.c.df[,-c(1,3,5,8,10)])
rf.predict2 <- ifelse(rf.predict == "Yes", 1, 0)
rf.solution <- data.frame(PassengerId = ttest.uc.df$PassengerId, Survived = rf.predict2)
write.csv(rf.solution, "titanic_47_rf_m3_caret.csv", row.names = FALSE)


### Neural Net - NNET
set.seed(1234)
nnet.fit <- nnet(Survived ~., ttrain.c.df[,-c(1,4,6,7,8,9,11,14)],entropy = TRUE, MaxNWts= 10000, size = 500, decay = 0.001, maxit = 5000)
nnet.fit

nnet.predict <- predict(nnet.fit, newdata = ttest.c.df[,-c(1,3,5,8,10)])
nnet.predict2 <- ifelse(nnet.predict >0.5, 1, 0)
nnet.solution <- data.frame(PassengerId = ttest.uc.df$PassengerId, Survived = nnet.predict2)
write.csv(nnet.solution, "titanic_43_nnet.csv", row.names = FALSE)


### Neural Net - CARET/NNET


ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE,returnResamp='none', summaryFunction = twoClassSummary )
nnet.grid <- expand.grid(.decay = c(0.8,0.6, 0.5, 0.4, 0.3), .size = c(3, 5, 6, 7,8,9, 10))
set.seed(1234)
nnet.caret.Fit <- train(ttrain.c.df[,-c(1,2,4,6,7,8,9,11,14)], ttrain.c.df[,2], trControl = ctrl, method = "nnet",
                       metric = "Accuracy", preProc = c("center", "scale"), tuneGrid = nnet.grid, trace = FALSE)

nnet.predict <- predict(nnet.caret.Fit, newdata = ttest.c.df[,-c(1,3,5,6,7,8,10,13)])
nnet.predict2 <- ifelse(nnet.predict == "Yes", 1, 0)
nnet.solution <- data.frame(PassengerId = ttest.uc.df$PassengerId, Survived = nnet.predict2)
write.csv(nnet.solution, "titanic_44_nnet_caret.csv", row.names = FALSE)


#### Fitting GBM  
ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE,returnResamp='none', summaryFunction = twoClassSummary )
gbm.grid <- expand.grid(.n.trees = c(500, 800, 1000,1200), .shrinkage = c(.1,.05,.01), .interaction.depth = c(1,2,3,4), .n.minobsinnode = c(2,3,4,5))
set.seed(1234)
gbm.caret.Fit <- train(ttrain.c.df[,-c(1,2,4,5,6,7,8,9,11,14)],ttrain.c.df[,2], trControl = ctrl, method = "gbm",
                    metric = "Accuracy", preProc = c("center", "scale"), tuneGrid = gbm.grid, verbose = FALSE)

gbm.predict <- predict(gbm.caret.Fit, newdata = ttest.c.df[,-c(1,3,5,6,7,8,10,13)])
gbm.predict2 <- ifelse(gbm.predict == "Yes", 1, 0)
gbm.solution <- data.frame(PassengerId = ttest.uc.df$PassengerId, Survived = gbm.predict2)
write.csv(gbm.solution, "titanic_46_gbm.csv", row.names = FALSE)


############   Fitting GLMNET CARET

glm.train.control <- trainControl(method='cv', number=10, returnResamp='none')
set.seed(1234)
glm.grid <- expand.grid(.alpha = (0:10) * 0.01, .lambda = (0:10) * 0.1)

glm.caret.Fit <- train(model.matrix(~.-1,ttrain.c.df[,-c(1,2,4,6,7,8,9,11,14)]),ttrain.c.df[,2], method='glmnet',
                 tuneGrid = glm.grid, metric = "Accuracy", trControl=glm.train.control)

glm.predict <- predict(glm.caret.Fit, newdata = model.matrix(~.-1,ttest.c.df[,-c(1,3,5,6,7,8,10,13)]))
glm.predict2 <- ifelse(glm.predict == "Yes", 1, 0)
glm.solution <- data.frame(PassengerId = ttest.uc.df$PassengerId, Survived = glm.predict2)
write.csv(glm.solution, "titanic_48_glm_p01_p.csv", row.names = FALSE)

############   Fitting C5Boost

C5.train.control <- trainControl(method="repeatedcv", number=10, repeats=3)
set.seed(1234)
C5boost.Fit <- train(ttrain.c.df[,-c(1,2,4,7,8,9,11,14)], ttrain.c.df[,2], method='C5.0', trControl=C5.train.control)

C5.predict <- predict(C5boost.Fit, ttest.c.df[,-c(1,3,5,6,7,8,10,13)])
C5.predict2 <- ifelse(C5.predict == "Yes", 1, 0)
C5.solution <- data.frame(PassengerId = ttest.uc.df$PassengerId, Survived = C5.predict2)
write.csv(C5.solution, "titanic_49_C5.csv", row.names = FALSE)

