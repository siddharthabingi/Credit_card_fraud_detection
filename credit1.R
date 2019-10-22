############################################################################
####-----------------------Credit Card Fruad Detection-----------------#####
############################################################################

#I've used only RandomForest ML algorithm for prediction.

##-------Imbalance classification is handled using following methods in this code:

#1. Undersampling: Randomly seleting observations from majority class to make equal to minority
#Limitations: We might loose valuable data as we're doing random selection

#2. Oversampling : Replicating the minority class (fraud) to certain observations 
#Limitations: No data loss but we might overfit the minority class

#3. Both over and undersampling

#4. Synthetic Minority Oversampling technique (SMOTE): 
#We produce synthetic minority data by using k-nearest neighbours. 
#We select one observation as reference from minority and select k neighbours 
#We calculate the distance between reference and 1st neighbour
#We multiply that distance by some random number between 0 and 1. 
#Result produces a point between reference and 1st selected neighbour.
#We iterate the process to produce desired sythetic samples. 

#Difference between ROSE and SMOTE

#ROSE uses smoothed bootstrapping to draw artificial samples from the 
#feature space neighbourhood around the minority class.

#SMOTE draws artificial samples by choosing points that lie on the line connecting 
#the rare observation to one of its nearest neighbors in the feature space.


########-------------------Code starts here-----------------------------------


#Libraries to be loaded
library(DMwR)
library(caret)
library(randomForest)
library(unbalanced)
library(ROSE)
library(ROCR)
library(ggplot2)
setwd("C:/Users/Siddharth Bingi/Desktop/Gyan Data/Case Study/Credit Card")

#Reading the dataset
credit.card <- read.csv("creditcard.csv")

#Checking the summary and structure of dataset
summary(credit.card)
str(credit.card)

#Feature "class" should be categorical/factor variable 
credit.card$Class <- factor(credit.card$Class)
str(credit.card)

#Missing values in the data = 0
cat("\n We could see no missing values in the data")

#Checking the no of observations for fraud and genuine
table(credit.card$Class)

prop.table(table(credit.card$Class))*100 #0.17% of fraud cases. Highly imbalanced

fraud <- nrow(credit.card[credit.card$Class == 1,])
genuine <- nrow(credit.card[credit.card$Class == 0,])

#Base accuracy is 99.8% ie; if we classify all transactions as genuine
base.accuracy <- genuine/ nrow(credit.card)
cat("\n Base Accuracy = ",base.accuracy*100)

rm(base.accuracy,fraud,genuine)

#Plotting feature 'Class'
plot(credit.card$Class, main = "No of fraud and genuine transactions", 
     xlab = '0 : Genuine , 1 : Fraud', ylab = 'Frequency')

cat("\n Highly imbalanced data")


#Scree plot
variance <- c()

for(i in 1:28)
{
  variance[i] <- var(credit.card[,i+1])
}
prop_varex <- variance/sum(variance)

plot(prop_varex*100, type = 'b',xaxt = 'n',yaxt = 'n', main = 'Scree Plot',
     xlab = 'Principal components 1 to 28', 
     ylab = "Proportion of Variance Explained")
axis(1, at=seq(1,28, by=1), labels = 1:28)
axis(side = 2, at = seq(0,15, by=0.5))


#Cummulative Scree plot
plot(cumsum(prop_varex), main = 'Scree Plot - 2', 
     xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
axis(1, at=seq(1,28, by=1), labels = 1:28)


#Plotting corrplot for Principal Components
corrplot(cor(credit.card[,2:29]))

#Missing values

df <- data.frame(Variable = rep(NA,31), MissingValues = rep(NA,31) )



for(i in 1:ncol(credit.card))
{
  df$Variable[i] <- names(credit.card[i])
  df$MissingValues[i] <- sum(is.na(credit.card[i]))
  
}

#write.table(df,file = 'MissingValue.csv',sep = '/t')


#Plotting Feature 'Amount' for fraud transactions
plot(credit.card$Amount[credit.card$Class == 1], 
     xlab = 'Observations', ylab = 'Amount', 
     main = "Plot of Feature 'Amount' for fraud transactions")

plot(credit.card$Amount[credit.card$Class == 0], 
     xlab = 'Observations', ylab = 'Amount', 
     main = "Plot of Feature 'Amount' for non-fraud transactions")


credit.card$Time_hour = floor(credit.card[,1]/3600)
fraud = subset(credit.card,credit.card$Class==1)

#------Fraud Transactions
hist(fraud$Time_hour, breaks = 48, main = 'No of Fraud Transaction per hour',
     xlab = 'Time in Hours', ylab = 'No of observations', col = 'lightgreen')

genuine = subset(credit.card, credit.card$Class==0)

#-------Non Fraud
hist(genuine$Time_hour, breaks = 48, main = 'No of Non- Fraud Transaction per hour',
     xlab = 'Time in Hours', ylab = 'No of observations',col = 'lightgreen')

#-------All-Transactions
hist(credit.card$Time_hour, breaks = 48, main = 'No of Transaction per hour',
     xlab = 'Time in Hours', ylab = 'No of observations',col = 'lightgreen')

#Feature 'Time' doesn't indicate actual time of transaction
#'Time' has less/no significance to the data. 
#Fraud transaction trend doesn't follow any pattern.
#So removing that variable

rm(genuine, fraud)
credit.card$Time_hour <- NULL

credit.card <- credit.card[,-1]
credit.card <- credit.card[,c(30, 1:29)]

#Data Partition
set.seed(1234)

index <- createDataPartition(y = credit.card$Class, p = 0.70,list = F) #Using caret package

train.credit <- credit.card[index,]
test.credit <- credit.card[-index,]

prop.table(table(train.credit$Class))*100 #Train Data Proportion is same as original data
prop.table(table(test.credit$Class))*100 #Test Data Proportion is same as original data


#Four types of methods to solve imbalanced datasets

#1. Undersampling - Using ROSE package
#2. Oversampling - Using ROSE package
#3. Synthetic Data Generation - using DmWr package
#4. Cost Sensitive Learning (Will be doing later)


#Accuracy shouldn't considered as a performance metric for imbalanced class problem. 
#For example: As in this case study, we have 99.98 percent of data is genuine. If we blindly 
# predict the outcome as genuine, we get 99.98% accuracy of the model. 

#There are metrics like Sensitivity, Specificity, Receiver Operating characteristic (ROC curve)
#         and Area under the curve (AUC)

#Accuracy = (TP + TN)/(TP + TN + FP + FN)

#No information Rate = largest proportion of the observed class = (TP + FN) / (TP + TN + FP + FN)

#Specificity : TN / TN + FP
#Sensitivity : TP / TP + FN


############################################################################
#########---------------------END OF CASE STUDY------------------###########
############################################################################