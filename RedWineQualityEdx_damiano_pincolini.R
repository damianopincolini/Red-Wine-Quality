# RED WINE QUALITY MACHINE LEARING PROJECT
# Damiano Pincolini



# Loading libraries

library(readr)
library(tidyverse)
library(moments)
library(GGally)
library(caret)
library(MASS)
library(stats) 
library(rpart.plot) # for decision tree plots
library("xgboost") # for X Gradient Boosting model


# Loading Data

# IMPORTANT NOTE: since it is not possible to download any dataset from Kaggle without
# providing account credentials to other graders,
# the following code simply load the dataset in R from my laptop where it has previuosly
# been saved.
# This link (https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/code?datasetId=4458&language=R)
# allows to land on the kaggle's page where it is possibile to download the dataset I have used
# (provided you log in). 

wines <- read_csv("wineQualityReds.csv")
colnames(wines)[1] <- "wineId"


# 1. EXPLORATORY DATA ANALYSIS

head(wines)
colnames(wines)
str(wines)
summary(wines)


# Univariate analysis of the output (quality): the main goal here is to understand how
# the outcome I want to investigate (quality) behaves.


# Five numbers by Tukey

summary(wines$quality)


# Distribution graphic analysis

ggplot(wines, aes(quality))+
  geom_bar(width=0.75)+
  scale_x_continuous(n.breaks=10,limits=c(0,10))+
  coord_cartesian(xlim=c(0,10))+
  labs(x = "quality (grades)",
       y = "number of grades",
       title ="Grade frequency")+
  theme_bw()

ggplot(wines, aes(quality))+
  geom_boxplot(outlier.colour="red", orientation="y")+
  labs(x = "quality (grades)",
       title ="Grade frequency")+
  theme_bw()


# Skweness and kurtosis 

skewness(wines$quality) 

# It is visible a slightly positive skewness (0.217)


kurtosis(wines$quality)

# With a 3.29 value, the distribution is slightly leptokurtic, but it does not seems a 
# major issue.


# Normality tests: let's take some test on the distribution of quality variable from the
# moments package.

jarque.test(wines$quality)

# In the Jarque-Bera normality test the null hypothesis is that the variable has a skewness and
# kurtosis that matches a normal distribution, while the alternative hypothesis is that the
# dataset has a skewness and kurtosis that does not match a normal distribution.
# Since the p-value is 0.0001062 which is less than 0.05, we can not reject the
# alternative hypothesis.


agostino.test(wines$quality)

# The D'Agostino skewness test returns a p-value of 0.0004182 is lower than 0.05 so we can
# accept the alternative hypotesis (skewness).


anscombe.test(wines$quality)

# The anscombe kurtosis test returns a p-value of 0.028, lower than 0.05, that leads me to
# accept the alternative hypotesis (kurtosis).

# It seems we are facing a non-normal distribution of red wine quality grades.

# Is there some issue with outliers?
# As we have already seen, the first quantile corresponds to grade 5 and the third quantile 
# corresponds to grade 6.
# So IQR is simply 1.

# The limit for lower outliers is

as.vector(summary(wines$quality)[2] - 1.5*IQR(wines$quality))

# The limit for higher outliers is

as.vector(summary(wines$quality)[5] + 1.5*IQR(wines$quality))


# It appears that the minimun and the maximum grade (respectively 3 and 5) are supposed to be
# considered as outliers.
# In the dataset documentation each wine rate is based on sensory evaluation  which is
# the median of at least 3 evaluations.
# Due to this information, I don't want to lose the information of 3 and 8 grades because
# the evaluation process seems pretty solid since it takes into account more than 
# a single grade (a least 3); in my opinion this can be considered as a good tip to exclude
# the presence of misleading information.


# Correlation: I want to detect if there are specific correlations either between quality and
# other features and between features themselves.
# First of all, I need to cancel the wineId column which is a simple progressive counter that
# does not represent a feature for quality wine prediction. 

wines2 <- wines %>%
  dplyr::select(2:13)

ggpairs(wines2, title="Correlation analysis", proportions = "auto")+
  theme_bw()

# 1. The features most correlated (respectively in a negative and a positive way)
# with quality are volatile acidity (-0.391) and alcohol (0.476).
# 2. Volatile acidity appears to be well positively correlated to citric acid (-0.552).
# 3. Alcohol is well negatively correlated with density (-0.496).
# According to what's above, I will focus on volatile acidity and alcohol (the two most
# important feature since I have spotted a direct correlation with quality).

 
# Dataset reduction and analysis of the selected features: the aim is here to focus on the
# features that seems to have a fairly good correlation with quality, thus managing the
# "curse of dimensionality".

# I rescale the dataset values via square root transformation

wines2sqrt <- wines2 %>%
  transmute(volatile.aciditysqrt = sqrt(volatile.acidity),
            alcoholsqrt = sqrt(alcohol),
            qualitysqrt = sqrt(quality)) %>%
  dplyr::select(volatile.aciditysqrt, alcoholsqrt, qualitysqrt)


# I check how variables behaves after transformation:

# Volatile acidity

wines2sqrt %>% ggplot(aes(volatile.aciditysqrt))+
  geom_boxplot()

skewness(wines2sqrt$volatile.aciditysqrt) 

kurtosis(wines2sqrt$volatile.aciditysqrt)


# Alcohol

wines2sqrt %>% ggplot(aes(alcoholsqrt))+
  geom_boxplot()

skewness(wines2sqrt$alcoholsqrt) 

kurtosis(wines2sqrt$alcoholsqrt)


#  Quality

wines2sqrt %>% ggplot(aes(qualitysqrt))+
  geom_boxplot()

skewness(wines2sqrt$qualitysqrt) 

kurtosis(wines2sqrt$qualitysqrt)

cor.test(x=wines2$volatile.acidity, y=wines2$quality, method="pearson")

cor.test(x=wines2$alcohol, y=wines2$quality, method="pearson")

# After squared root transformation I can detect what follows:
# 1. quality skeness has significantly reduced (from 0.217 to -0.05),
# 2. quality kurtosis has slightly increased (from 3.29 to 3.58),
# 3. correlation between quality and both alcohol and volatile acidity has not changed (0.476
# and -0.391 respectively).


# 2. MODELING

# Pre-processing: data have already been transformed, the number of features have already
# been reduced. There is no need for other pre-processing activities.


# Data partitioning

# The starting point of modeling activity is the dataset called wines2sqrt,
# which contains both outcomes (quality) and two features (alcohol and volatile acidity).
# I want to split the dataset in the two following items:
# 1. wineQuality, a vector containing quality marks.
# 2. wineFeatures, a matrix of two numeric features.

wineQuality <- wines2sqrt$qualitysqrt

wineFeatures <- wines2sqrt %>%
  dplyr::select(c(1:2)) %>%
  as.matrix()

# After setting the seed to 1, I proceed with data partition splitting both wineQuality vector
# and wineFeatures matrix into a 20% test set and 80% train.

set.seed(1, sample.kind = "Rounding")    
test_index <- createDataPartition(wineQuality, times = 1, p = 0.2, list = FALSE)
wineFeatureTest <- wineFeatures[test_index,]
wineQualityTest <- wineQuality[test_index]
wineFeatureTrain <- wineFeatures[-test_index,]
wineQualityTrain <- wineQuality[-test_index]


# I check that the training and test wineQuality datasets have similar proportions of marks.

mean(wineQuality)

mean(wineQualityTest)

mean(wineQualityTrain)

# Everything seems fine: the average mark in the original, train and test tests is 2.37 in 
# all of the three cases.


densityplot(wineQualityTest)

densityplot(wineQualityTrain)

wineQualityTest %>%
  as_tibble() %>%
  group_by(value) %>%
  summarize(numMarks=n()) %>%
  mutate(percMark=numMarks/sum(numMarks)*100)

wineQualityTrain %>%
  as_tibble() %>%
  group_by(value) %>%
  summarize(numMarks=n()) %>%
  mutate(percMark=numMarks/sum(numMarks)*100)

# Distribution of test and train set is similar with highs and lows at the same points as 
# it is visibile from the two plot above.
# The 2.24 value appears for the 42/44 per cent of the times; the 2.45 value for the 39.9
# per cent of the times, the 2.65 value has 13.4/12.2 per cent of frequency.
# The marginal (transformed) marks (1.73, 2 and 2.83) show slightly meaningful differences 
# but we can deduce that data partitioning has not produced unbalanced train and test dataset.


# Training

# I want to try the following algorithms: linear regression, decision tree, random forest,
# XG boost and k-nearest neighbors.


# Linear regression

train_lm <- train(x = wineFeatureTrain,
                  y = wineQualityTrain,
                  method = "lm")

train_lm        

summary(train_lm)


# Decision Tree

train_d.tree <- train(x = wineFeatureTrain,
                y = wineQualityTrain, 
                method = "rpart")

train_d.tree


# Random forest model

train_rf <- train(x = wineFeatureTrain,
                  y = wineQualityTrain,
                  method = "ranger")

train_rf


# XG Boost

train_xg.boost <- train(x = wineFeatureTrain,
                        y = wineQualityTrain, 
                        method = "xgbTree")

train_xg.boost


# K-Nearest Neighbor

train_knn <- train(x = wineFeatureTrain,
                   y = wineQualityTrain,
                   method = "knn")

train_knn



# Testing: prediction and performance metrics

# For each model I will predict outcomes based on the features of test set and then,
# with postSample() command, I will get the most used KPI: RMSE, R squared and MAE which
# will be discussed in the following sections of this report.

# Linear regression

hat_lm <- predict(train_lm, wineFeatureTest)

Perf_lm <- postResample(pred=hat_lm, obs=wineQualityTest)

Perf_lm

ggplot(varImp(train_lm)) 


# Decision Tree

hat_d.tree <- predict(train_d.tree, wineFeatureTest)

Perf_d.tree <- postResample(pred=hat_d.tree, obs=wineQualityTest)

Perf_d.tree

ggplot(varImp(train_d.tree))   


# Random forest

hat_rf <- predict(train_rf, wineFeatureTest)

Perf_rf <- postResample(pred=hat_rf, obs=wineQualityTest)

Perf_rf


# XG Boost

hat_xg.boost <- predict(train_xg.boost, wineFeatureTest)

Perf_xg.boost <- postResample(pred=hat_xg.boost, obs=wineQualityTest)

Perf_xg.boost

ggplot(varImp(train_xg.boost))  


# K-Nearest Neighbor

hat_knn <- predict(train_knn, wineFeatureTest)

Perf_knn <- postResample(pred=hat_knn, obs=wineQualityTest)

Perf_knn

ggplot(varImp(train_knn)) 


# 3. MODEL PERFORMANCES EVALUATION

# The goal is to gather the performances of the models I have tried in order to find out
# which is more suitable to predict wine quality.

Perf_overview <- rbind(Perf_lm, Perf_d.tree, Perf_rf, Perf_xg.boost, Perf_knn)

Perf_overview2 <- Perf_overview %>%
  as_tibble %>%
  cbind(model = row.names(Perf_overview), .)

Perf_overview2 %>% ggplot(aes(x=RMSE, y=Rsquared, color=model))+
  geom_point(size=4)+
  labs(x = "RMSE", y = "Rsquared",
       title ="MODEL EVALUATION")+
  theme_bw()

# As said in the introduction section, I am interested in understanding models performance
# using RMSE and R squared index.
# As far as our models are concerned, we can visualize an inverse relation between RMSE and
# the R squared.
