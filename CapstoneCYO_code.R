# CapstoneCYO_code.R

# This R file is derived from the Capstone.Rmd R markdown file
# It contains all the code needed to build the models and reproduce the predictions.

# Download base dataset
DataURL <- "https://www.kaggle.com/jsphyg/weather-dataset-rattle-package/downloads/weather-dataset-rattle-package.zip/2"
DataFile <- "weather-dataset-rattle-package.zip"
if (!file.exists(DataFile)) {
  download.file(DataURL, destfile = DataFile)
  unzip(DataFile)
}
rain <- read.csv("weatherAUS.csv")
str(rain)

# Cleaning data

rain$Date <- as.Date(rain$Date)
library(lubridate)
rain$Month <- month(ymd(rain$Date))
rain$RISK_MM <- NULL
rain$RainToday <- NULL
rain$Evaporation <- NULL
rain$Sunshine <- NULL
rain$Cloud9am <- NULL
rain$Cloud3pm <- NULL

# Imputation for missing values

library(dplyr)
library(tidyr)
rain <- rain %>%
   arrange(Date) %>%
   group_by(Location) %>%
   fill(WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Rainfall, Pressure9am, Pressure3pm, MinTemp, MaxTemp, Temp9am, Temp3pm, WindGustDir, WindDir9am, WindDir3pm) %>%
   fill(WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Rainfall, Pressure9am, Pressure3pm, MinTemp, MaxTemp, Temp9am, Temp3pm, WindGustDir, WindDir9am, WindDir3pm, .direction = "up")
rain <- rain %>% ungroup()
rain$Date <- NULL
rain$WindGustDir[is.na(rain$WindGustDir)] <- rain$WindDir9am[is.na(rain$WindGustDir)] 
rain$WindGustSpeed[is.na(rain$WindGustSpeed)] <- max(rain$WindSpeed9am[is.na(rain$WindGustSpeed)], rain$WindSpeed3pm[is.na(rain$WindGustSpeed)]) 
rain$Pressure9am[is.na(rain$Pressure9am)] <- mean(rain$Pressure9am[!is.na(rain$Pressure9am)]) 
rain$Pressure3pm[is.na(rain$Pressure3pm)] <- mean(rain$Pressure3pm[!is.na(rain$Pressure3pm)]) 

## Training and testing set
library(caret)
set.seed(1971)
test_index <- createDataPartition(rain$RainTomorrow, times = 1, p = 0.2, list = FALSE)
test_set <- rain[test_index, ]
train_set <- rain[-test_index, ]

## Logistic Regression

mod_glm <- glm(RainTomorrow ~ WindGustSpeed + WindSpeed9am + Humidity3pm + Pressure3pm + MinTemp + MaxTemp + Month + Location, data = train_set, family = binomial)
pred_glm <- predict(mod_glm, type = 'response', newdata = test_set)
pred_glm2 <- as.factor(ifelse (pred_glm < 0.5, "No", "Yes"))
confusionMatrix(data = pred_glm2, reference = test_set$RainTomorrow, positive = "Yes")
library(ROCR)
pr <- prediction(pred_glm, test_set$RainTomorrow)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

## Tree

library(rpart)
mod_tree <- rpart(RainTomorrow ~ WindGustSpeed + WindSpeed9am + Humidity3pm + Pressure3pm + MinTemp + MaxTemp + Month + Location, data = train_set, method = 'class')
library(rpart.plot)
prp(mod_tree, type = 2, extra = 4)
pred_tree <- predict(mod_tree, newdata = test_set, type = "class")
confusionMatrix(pred_tree, test_set$RainTomorrow, positive = "Yes")
pred_tree_prob <- predict(mod_tree, newdata = test_set)
pr <- prediction(pred_tree_prob[,2], test_set$RainTomorrow)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

## Random Forest

library(randomForest)
mod_rf <- randomForest(RainTomorrow ~ ., data = train_set, family = binomial)
pred_rf <- predict(mod_rf, type = 'response', newdata = test_set)
confusionMatrix(data = pred_rf, reference = test_set$RainTomorrow, positive = "Yes")

## Enable parallel computation
library(caret)
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv", number = 2, allowParallel = TRUE)

## Caret Random Forest

Gridrf <-  expand.grid(mtry = c(3, 5))
before <- proc.time()
fit_caret_rf <- train(RainTomorrow ~ WindGustSpeed + WindSpeed9am + Humidity3pm + Pressure3pm + MinTemp + MaxTemp + Month + Location, data = train_set, method = "rf", trControl = fitControl, tuneGrid = Gridrf)
predict_caret_rf <- predict(fit_caret_rf, test_set)
proc.time() - before
confusionMatrix.train(fit_caret_rf)

## Caret Bayesian Generalized Linear Model

before <- proc.time()
fit_caret_bayesglm <- train(RainTomorrow ~ WindGustSpeed + WindSpeed9am + Humidity3pm + Pressure3pm + MinTemp + MaxTemp + Month + Location, data = train_set, method = "bayesglm", trControl = fitControl)
predict_caret_bayesglm <- predict(fit_caret_bayesglm, test_set)
proc.time() - before
confusionMatrix.train(fit_caret_bayesglm)

## Stop parallel computation
stopCluster(cluster)
registerDoSEQ()

## Ensemble model
predictions <- data.frame(pred_rf, predi_tree, pred_rf, predict_caret_rf, predict_caret_bayesglm)
library(prettyR)
final_prediction <- apply(predictions, 1, Mode)
confusionMatrix(data = final_prediction, reference = test_set$RainTomorrow, positive = "Yes")

# Results

# Conclusion

