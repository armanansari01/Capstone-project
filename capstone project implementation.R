#reading data
capdata <- read.csv(file.choose())
#viewing data
View(capdata)
#details and summary of the data
str(capdata)

summary(capdata)


#load all the required packages
library(e1071)
library(caTools)
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(plyr)
library(class)
library(party)
library(randomForest)

#First we will create subset for depression and sarcasm data
depdata <- capdata[,c(1,2,3,4,5,7)]
View(depdata)

#splitting the depdata into train and test with the ratio of 70 and 30 respectively
split <- sample.split(depdata, SplitRatio = 0.7)
traindep_cl <- subset(depdata, split == "TRUE")
testdep_cl <- subset(depdata, split == "FALSE")

#building Naive Bayes model
set.seed(120)  # Setting Seed
depclassifier_cl <- naiveBayes(Diagnosis ~ ., data = traindep_cl)
depclassifier_cl

#prediction of the model
depy_pred <- predict(depclassifier_cl,newdata = testdep_cl)
depy_pred

#confusion matrix
depcm <- table(testdep_cl$Diagnosis, depy_pred)
depcm

#model evaluation
confusionMatrix(depcm)

naivedepaccuracy <- (sum(diag(depcm))/sum(depcm))*100
naivedepaccuracy


#now we perform the same operation on sarcasm data
sardata<-capdata[,c(1,2,3,11,15,19)]
View(sardata)

 
#splitting the sardata into train and test into the ratio of 70 and 30 respectively
splitsar <- sample.split(sardata, SplitRatio = 0.7)

trainsar_cl <- subset(sardata, splitsar == "TRUE")
testsar_cl <- subset(sardata, splitsar == "FALSE")

#building model for sarcasm data
set.seed(120)  # Setting Seed
sarclassifier_cl <- naiveBayes(Recognize_sarcasm ~ ., data = trainsar_cl)
sarclassifier_cl

sary_pred <- predict(sarclassifier_cl,newdata = testsar_cl)
sary_pred


#confusion matrix
sarcm <- table(testsar_cl$Recognize_sarcasm, sary_pred)
sarcm

#model evaluation
confusionMatrix(sarcm)

naivesaraccuracy <- (sum(diag(sarcm))/sum(sarcm))*100
naivesaraccuracy

#------------------------------------------------------------------------------#
#Now we build the Knn model for depression and sarcasm

#subset for knn of depression data
depdata <- capdata[,c(2,4,5,7)]
View(depdata)

#firstly we need to revalue the columns
depdata$Diagnosis = revalue(depdata$Diagnosis, c("Yes"=1))
depdata$Diagnosis = revalue(depdata$Diagnosis, c("No"=0))
depdata$Affect = revalue(depdata$Affect, c("Positively"=1))
depdata$Affect = revalue(depdata$Affect, c("Negatively"=-1))
depdata$Affect = revalue(depdata$Affect, c("No Impact"=0))
View(depdata)

#subset for knn of sarcasm data
sardata <- capdata[,c(2,11,15,19)]
View(sardata)

sardata$Recognize_sarcasm = revalue(sardata$Recognize_sarcasm, c("Yes"=1))
sardata$Recognize_sarcasm = revalue(sardata$Recognize_sarcasm, c("No"=0))
sardata$sarcasm_ambiguity = revalue(sardata$sarcasm_ambiguity, c("Yes"=1))
sardata$sarcasm_ambiguity = revalue(sardata$sarcasm_ambiguity, c("Can't say"=0))
sardata$sarcasm_ambiguity = revalue(sardata$sarcasm_ambiguity, c("No"=-1))
View(sardata)

# Convert the column to a factor
depdata$Affect <- as.factor(depdata$Affect)
sardata$sarcasm_ambiguity <- as.factor(sardata$sarcasm_ambiguity)


#splitting again as we have updated the subset data
#splitting the depdata into train and test with the ratio of 70 and 30 respectively
split <- sample.split(depdata, SplitRatio = 0.7)
traindep_cl <- subset(depdata, split == "TRUE")
testdep_cl <- subset(depdata, split == "FALSE")

xdeptrain <- traindep_cl[,-4]
ydeptrain <- traindep_cl[,4]
xdeptest <- testdep_cl[,-4]
ydeptest <- testdep_cl[,4]

# Train the KNN model with k=5
depknn_model <- knn(xdeptrain,xdeptest,ydeptrain,k=5)

depcm <- table(testdep_cl$Affect,depknn_model)
depcm

# Check the accuracy of the model
knndepaccuracy <- (sum(diag(depcm))/sum(depcm))*100
knndepaccuracy

#now we apply it on sarcasm data
#splitting the sardata into train and test into the ratio of 70 and 30 respectively
splitsar <- sample.split(sardata, SplitRatio = 0.7)
trainsar_cl <- subset(sardata, splitsar == "TRUE")
testsar_cl <- subset(sardata, splitsar == "FALSE")


xsartrain <- trainsar_cl[,-4]
ysartrain <- trainsar_cl[,4]
xsartest <- testsar_cl[,-4]
ysartest <- testsar_cl[,4]

# Train the KNN model with k=5
sarknn_model <- knn(xsartrain,xsartest,ysartrain,k=5)
sarknn_model

sarcm <- table(testsar_cl$sarcasm_ambiguity,sarknn_model)
sarcm


# Check the accuracy of the model
knnsaraccuracy <- (sum(diag(sarcm))/sum(sarcm))*100
knnsaraccuracy

#------------------------------------------------------------------------------#

#SVM model
#depression
# Split the data into training and testing datasets
set.seed(123)
trainIndex <- sample(1:nrow(capdata), 0.7*nrow(capdata))
train_in <- capdata[trainIndex,]
test_in <- capdata[-trainIndex,]

# Fit the SVM model
depsvm_model <- svm(Depression_scale ~ Age, data = train_in, kernel = "linear")

# Make predictions on the testing data-set
depsvm_pred <- predict(depsvm_model, test_in)
depsvm_pred
# Evaluate the performance of the model
depcm <- table(depsvm_pred, test_in$Affect[1:length(depsvm_pred)])
depcm

svmdepaccuracy <- (sum(diag(depcm))/sum(depcm))*100
svmdepaccuracy

#svm on sarcasm
sarsvm_model <- svm(sarcasm_scale ~ Age, data = train_in, kernel = "linear")

svmsar_pred <- predict(sarsvm_model,test_in)
svmsar_pred

sarcm <- table(svmsar_pred,test_in$sarcasm_ambiguity[1:length(svmsar_pred)])
sarcm

svmsaraccuracy <- (sum(diag(sarcm))/sum(sarcm))*100
svmsaraccuracy

#------------------------------------------------------------------------------#
#random forest model
#depression

deprfdata <- capdata[,c(1,2,3,4,5,7)]
View(deprfdata)

#converting non-numeric dataset into factors
deprfdata$Diagnosis <- as.factor(deprfdata$Diagnosis)

#splitting dataset
set.seed(123)

depsplit <- sample.split(deprfdata, SplitRatio = 0.7)
deprftrain_cl <- subset(deprfdata, split == "TRUE")
deprftest_cl <- subset(deprfdata, split == "FALSE")

#applying model
deprfmodel <- randomForest(Diagnosis ~.,data = deprfdata, ntree=100)
print(deprfmodel)

deprfpred <- predict(deprfmodel, newdata = deprftest_cl)
deprfpred

depcm <- table(deprfpred, deprftest_cl$Diagnosis)
depcm

rfdepaccuracy <- (sum(diag(depcm))/sum(depcm))*100
rfdepaccuracy

#sarcasm
rfsardata<-capdata[,c(1,2,3,11,15)]
View(rfsardata)


#converting non-numeric dataset into factors
rfsardata$Recognize_sarcasm <- as.factor(rfsardata$Recognize_sarcasm)

#splitting dataset
set.seed(123)

sarsplit <- sample.split(rfsardata, SplitRatio = 0.7)
sarrftrain_cl <- subset(rfsardata, split == "TRUE")
sarrftest_cl <- subset(rfsardata, split == "FALSE")

rfsarmodel <- randomForest(Recognize_sarcasm ~.,data = rfsardata,ntree = 100)
print(rfsarmodel)

rfsarpred <- predict(rfsarmodel, newdata = sarrftest_cl)
rfsarpred

sarcm <- table(rfsarpred, sarrftest_cl$Recognize_sarcasm)
sarcm

rfsaraccuracy <- (sum(diag(sarcm))/sum(sarcm))*100
rfsaraccuracy

#------------------------------------------------------------------------------#
#Decision tree
#depression

#we split the capdata again for the decision tree into the ratio of 80 and 20
ind <- sample.split(Y=capdata,SplitRatio = 0.8)
trainData <- capdata[ind,]
testData <- capdata[!ind,]


set.seed(1234)

# build the decision tree
mydeptree <- rpart(Diagnosis ~ Gender + Depression_scale, data=capdata, 
                   method="class",
                   control =rpart.control(minsplit =1,minbucket=1, cp=0))

# print the tree
print(mydeptree)

#plot the tree
plot(mydeptree)
text(mydeptree,pretty=0)
summary(mydeptree)
fancyRpartPlot(mydeptree)
rpart.plot(mydeptree,extra = 106)

#Test the model
depprediction_model <- predict(mydeptree,testData)
depprediction_model

depcm<-table(depprediction_model,
             testData$Diagnosis[1:length(depprediction_model)])

#Evaluating the performance of Regression trees
MAE <- function(actual,pred) {mean(abs(actual-pred))}

depmae <- MAE(testData$Depression_scale,depprediction_model)
depmae

dep_MSE <- mean((depprediction_model-testData$Depression_scale)*(depprediction_model-testData$Depression_scale))
dep_MSE

#Calculate the Complexity Parameter
printcp(mydeptree)
plotcp(mydeptree)

#Prune the tree
deppruned_model <- prune.rpart(mydeptree,cp=0.01)
plot(deppruned_model)
text(deppruned_model)
fancyRpartPlot(mydeptree)

#Test the pruned model
depy1 <- predict(deppruned_model, testData)
depy1

pcm <- table(depy1[1:length(testData$Diagnosis)],testData$Diagnosis)
pcm

decdepaccuracy <- (sum(diag(pcm))/sum(pcm))*100
decdepaccuracy


#Plot Result
plot(depy1, testData$Depression_scale[1:length(depy1)],ylab = "Depression scale")
abline(0,1)

#Evaluate the performance of Pruned Trees
depMSE2 <- mean((depy1-testData$Depression_scale)*(depy1-testData$Depression_scale))
depMSE2

#decision tree on sarcasm
mysartree <- rpart(Recognize_sarcasm ~ Gender + sarcasm_scale, data=capdata, 
                   method="class",control =rpart.control(minsplit =1,minbucket=1, cp=0))
print(mysartree)

#plotting the sartree
plot(mysartree)
text(mysartree,pretty=0)
summary(mysartree)
fancyRpartPlot(mysartree)
rpart.plot(mysartree,extra = 106)

#Test the model
predictionsar_model <- predict(mysartree,testData)
predictionsar_model

sarcm <- table(predictionsar_model[1:length(testData$Recognize_sarcasm)],
               testData$Recognize_sarcasm)
sarcm

decsaraccuracy <- (sum(diag(sarcm))/sum(sarcm))*100
decsaraccuracy 

#Evaluating the performance of Regression trees
sarmae <- MAE(testData$sarcasm_scale,predictionsar_model)
sarmae

MSE1sar <- mean((predictionsar_model-testData$sarcasm_scale)*(predictionsar_model-testData$sarcasm_scale))

#Calculate the Complexity Parameter
printcp(mysartree)
plotcp(mysartree)

#Prune the tree
sarpruned_model <- prune.rpart(mysartree,cp=0.01)
plot(sarpruned_model)
text(sarpruned_model)
fancyRpartPlot(mysartree)

#Test the pruned model
sary1 <- predict(sarpruned_model, testData)
sary1

scm <- table(sary1[1:length(testData$Recognize_sarcasm)],testData$Recognize_sarcasm)
scm

decsaraccuracy <- (sum(diag(scm))/sum(scm))*100
decsaraccuracy 

#Plot Result
plot(sary1, testData$sarcasm_scale[1:length(sary1)],ylab = "Sarcasm scale")
abline(0,1)

#Evaluate the performance of Pruned Trees
MSE2sar <- mean((sary1-testData$sarcasm_scale)*(sary1-testData$sarcasm_scale))
MSE2sar

#comparing the MSE of the pruned trees of depression and sarcasm
algorithm <- c("Depression MSE", "Sarcasm MSE")
MSE_of_both <- c(depMSE2, MSE2sar)
df <- data.frame(algorithm, MSE_of_both)

barplot(df$MSE_of_both, names.arg = df$algorithm, ylim = c(0, 100))

barplot(df$MSE_of_both, names.arg = df$algorithm, ylim = c(0, 100), 
        col = c("blue", "red"), 
        border = NA, main = "MSE of Depression and sarcasm")

#comparing the MAE of the pruned trees of depression and sarcasm
algorithm <- c("Depression MAE", "Sarcasm MAE")
MAE_of_both <- c(depmae, sarmae)
df <- data.frame(algorithm, MAE_of_both)

barplot(df$MAE_of_both, names.arg = df$algorithm, ylim = c(0, 50))

barplot(df$MAE_of_both, names.arg = df$algorithm, ylim = c(0, 50), 
        col = c("blue", "red"),
        border = NA, main = "MAE of Depression and sarcasm")

#------------------------------------------------------------------------------#
#compare the accuracies of all the models

Model <- c(c("NB Depression","NB Sarcasm"),c("KNN Depression","KNN Sarcasm"),
             c("SVM Depression","SVM Sarcasm"),c("RF Depression","RF Sarcasm"),
             c("DT Depression","DT Sarcasm"))

Percentage <- c(c(naivedepaccuracy,naivesaraccuracy),c(knndepaccuracy,knnsaraccuracy),
             c(svmdepaccuracy,svmsaraccuracy),c(rfdepaccuracy,rfsaraccuracy),
             c(decdepaccuracy,decsaraccuracy))

df <- data.frame(Model,Percentage)
df

library(tidyverse)
# Bar chart side by side
ggplot(df, aes(x = Model, y = Percentage, group = Percentage, fill = Model)) +
  geom_bar(stat = "identity",position = "dodge")


################################################################################
 
