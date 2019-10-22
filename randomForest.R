#######-------------Creating a vector to save metrics for plotting---------######
sensitivity.rf <- c(rep(NA,3))
names(sensitivity.rf) <- c('Undersampling','Oversampling','SMOTE')
specificity.rf <- c(rep(NA,3))
names(specificity.rf) <- c('Undersampling','Oversampling','SMOTE')




########---------------Undersampling---------------########
train.fraud <- nrow(train.credit[train.credit$Class == 1,])

#undersampling using ROSE package
undersample <- ovun.sample(formula = Class ~ ., data = train.credit, 
                           method = "under", N =  train.fraud*2)

#Predicting model using Random forest
rfTrain.under <- randomForest(formula = Class~.,data = undersample$data)
rfPred.under <- predict(object = rfTrain.under, newdata = test.credit)

conf.under.rf <- confusionMatrix(reference = test.credit$Class, data = rfPred.under)

sensitivity.rf['Undersampling'] <- conf.under.rf$byClass[1]
specificity.rf['Undersampling'] <- conf.under.rf$byClass[2]

roc.curve(response = test.credit$Class, predicted = rfPred.under, 
          col = 'red', lwd = 2, main = 'ROC Curve for Sampling Technique')


#####--------------Oversampling------------######

#Taking randomly 2000 samples from class == 1 (genuine) 
genuine.sample.2000 <- train.credit[sample(x = nrow(train.credit[train.credit$Class == 0,]),
                                           size = 2000),]  

#Combining data of random genuine samples and fraud samples
oversample.data <- rbind(genuine.sample.2000, train.credit[train.credit$Class == 1,])


#Oversampling the data to majority class(genuine)
oversample <- ovun.sample(formula = Class~., data = oversample.data,method = 'over', 
                          N = 4000)

#Predicting using RandomForest
rfTrain.over <- randomForest(formula = Class~.,data = oversample$data)
rfPred.over <- predict(object = rfTrain.over, newdata = test.credit)

conf.over.rf <- confusionMatrix(reference = test.credit$Class, data = rfPred.over)

sensitivity.rf['Oversampling'] <- conf.over.rf$byClass[1]
specificity.rf['Oversampling'] <- conf.over.rf$byClass[2]

roc.curve(response = test.credit$Class,predicted = rfPred.over,
          add.roc =TRUE, col = 'green', lwd = 2)


#####----------- SMOTE----------#######

#Making balanced data using 'S'ynthetic 'M'inority 'O'versampling 'TE'chnique
smote.data <- SMOTE(form = Class~. , data = train.credit, perc.over = 200, k = 5, 
                    perc.under = 200)
#58% data belongs to majority and 42% belong to minority 
prop.table(table(smote.data$Class))

#RandomForest algorithm on SMOTEd data
rfTrain.smote <- randomForest(formula = Class~.,data = smote.data)
rfPred.smote <- predict(object = rfTrain.smote, newdata = test.credit[,-1])


conf.smote.rf <- confusionMatrix(reference = test.credit$Class, data = rfPred.smote)

sensitivity.rf['SMOTE'] <- conf.smote.rf$byClass[1]
specificity.rf['SMOTE'] <- conf.smote.rf$byClass[2]

roc.curve(response = test.credit$Class,predicted = rfPred.smote,
          add.roc =TRUE, col = 'blue', lwd = 2)

legend("bottomright", c("Undersampling", "Oversampling", 'SMOTE'), 
       col=c('red','green','blue'),lwd=2)

#Plotting histogram for sensitivity and specificity
barchart(sensitivity.rf, main = "Senstivity plot for Random Forest Technique",
         xlab = "Sensitivity (Class = 'Non-Fraud')", col = c('red','green','blue'),
         xlim = c(0.85,1))

barchart(specificity.rf, main = "Specificity plot for Random Forest Technique",
         xlab = "Specificity (Class = 'Fraud')",col = c('red','green','blue'),
         xlim = c(0.85,1))


# We're only getting 95% AUC, will be using more methods to boost AUC
#1. Boosting
#2. Bagging
#3. XG Boost
#4. GBM - Gradient Boosting Algorithm


#We'll be using more Classification algorithms to performance metrics
#1. Logistic Regression
#2. Decision Tree..................etc

########---------------Both : Under and over ---------------########

#Sampling using ROSE package
both <- ovun.sample(formula = Class ~ ., data = train.credit, 
                    method = "both", N =  2000)
#Proportion of data 0: 48%, 1: 52%
prop.table(table(both$data$Class))

#Predicting model using Random forest
rfTrain.both <- randomForest(formula = Class~.,data = both$data)
rfPred.both <- predict(object = rfTrain.both, newdata = test.credit)

confusionMatrix(reference = test.credit$Class,data = rfPred.both)

#ROC curve plots and gives AUC value
roc.curve(response = test.credit$Class, predicted = rfPred.both)



########---------------ROSE ---------------########

#Sampling using ROSE function
rose.credit <- ROSE(formula = Class ~ ., data = train.credit, N = 5000)
#Proportion of data 0: 49.56%, 1: 50.44%
prop.table(table(rose.credit$data$Class))

#Predicting model using Random forest
rfTrain.rose <- randomForest(formula = Class~.,data = rose.credit$data)
rfPred.rose <- predict(object = rfTrain.rose, newdata = test.credit)

confusionMatrix(reference = test.credit$Class,data = rfPred.rose)

#ROC curve plots and gives AUC value
roc.curve(response = test.credit$Class, predicted = rfPred.rose)