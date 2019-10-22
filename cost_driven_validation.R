
cred <- credit.card1

cred <- credit.card1[!(credit.card1$Amount==0),]

nrow(cred[(cred$Amount>2825),])

cred <- (cred[!(cred$Amount>2825),])

set.seed(2222)

index <- createDataPartition(y = cred$Class, p = 0.70,list = F) #Using caret package

train.credit <- cred[index,]
test.credit <- cred[-index,]

prop.table(table(train.credit$Class))*100 #Train Data Proportion is same as original data
prop.table(table(test.credit$Class))*100 #Test Data Proportion is same as original data




smote.data <- SMOTE(form = Class~. , data = train.credit, perc.over = 200, k = 5, 
                    perc.under = 200)

table(train.credit$Class)

prop.table(table(smote.data$Class))
table(smote.data$Class)


plot(smote.data$Amount[smote.data$Class == 1], 
     xlab = 'Observations', ylab = 'Amount', 
     main = "Plot of Feature 'Amount' for fraud transactions")

plot(smote.data$Amount[smote.data$Class == 0], 
     xlab = 'Observations', ylab = 'Amount', 
     main = "Plot of Feature 'Amount' for non-fraud transactions")
mean(smote.data$Amount[smote.data$Class == 1])
mean(smote.data$Amount[smote.data$Class == 0])


library(rpart)

fit_samp <- rpart(formula = Class~.,data = smote.data,method = 'class')

pred_test <- predict(object = fit_samp, newdata = test.credit[,-1],type = 'class')

pred_train <- predict(object = fit_samp, newdata = train.credit[,-1], type = 'class')

confusionMatrix(data = pred_train,reference = train.credit$Class, positive = '0')
confusionMatrix(data = pred_test,reference = test.credit$Class, positive = '0')

library(rpart.plot)

rpart.plot(fit)

#####################################################


fit2 <- rpart(formula = Class~.,data = smote.data,method = 'class',
              parms = list(split = 'information'))

pred2 <- predict(object = fit2, newdata = test.credit[,-1],type = 'class')


library(pROC)
roc(test.credit$Class,fit2)


confusionMatrix(data = pred2,reference = test.credit$Class, positive = '1')

library(rpart.plot)

rpart.plot(fit)

##############################################################
cost.driven.model <- rpart(Class~.,data = smote.data,parms=list(loss=matrix(c(0,1,5,0),
                                         byrow=TRUE,
                                         nrow=2)))

pred_cost_test <- predict(object = cost.driven.model, newdata = test.credit[,-1],type = 'class')

pred_cost_train <- predict(object = cost.driven.model, newdata = train.credit[,-1],type = 'class')

confusionMatrix(data = pred_cost_test,reference = test.credit$Class, positive = '0')

confusionMatrix(data = pred_cost_train,reference = train.credit$Class, positive = '0')

roc.curve(response = test.credit$Class,predicted = pred_cost_test)


#ROC Cruves - Test Data

roc.curve(response = test.credit$Class, predicted = pred_test,
          col = 'blue', lwd = 2, main = 'ROC Curve : Test Data')

roc.curve(response = test.credit$Class, predicted = pred_cost_test,
          col = 'red', lwd = 2, add.roc = T)

legend("bottomright", c("SMOTE", "COST Sensitive"), 
       col=c('blue','red'),lwd=2)

#Train Data

roc.curve(response = train.credit$Class, predicted = pred_train,
          col = 'blue', lwd = 2, main = 'ROC Curve : Train Data')

roc.curve(response = train.credit$Class, predicted = pred_cost_train,
          col = 'red', lwd = 2, add.roc = T)

legend("bottomright", c("SMOTE", "COST Sensitive"), 
       col=c('blue','red'),lwd=2)

