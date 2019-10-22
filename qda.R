#######-------------Creating a vector to save metrics for plotting---------######
sensitivity.qda <- c(rep(NA,3))
names(sensitivity.qda) <- c('Undersampling','Oversampling','SMOTE')
specificity.qda <- c(rep(NA,3))
names(specificity.qda) <- c('Undersampling','Oversampling','SMOTE')

library(MASS)

########---------------Undersampling---------------########
train.fraud <- nrow(train.credit[train.credit$Class == 1,])

#undersampling using ROSE package
undersample <- ovun.sample(formula = Class ~ ., data = train.credit, 
                           method = "under", N =  train.fraud*2)

qda.under <- qda(Class~.,data = undersample$data)

qda.under.pred <- predict(qda.under, newdata = test.credit[,-1],type = 'class')

conf.under.qda <- confusionMatrix(data = qda.under.pred$class, reference = test.credit$Class,
                                  positive = '1')

sensitivity.qda['Undersampling'] <- conf.under.qda$byClass[1]
specificity.qda['Undersampling'] <- conf.under.qda$byClass[2]


roc.curve(response = test.credit$Class, predicted = qda.under.pred$class,
          col = 'red', lwd = 2, main = 'QDA : ROC Curve for Sampling Technique')



#####--------------Oversampling------------######

#Taking randomly 2000 samples from class == 1 (genuine) 
genuine.sample.2000 <- train.credit[sample(x = nrow(train.credit[train.credit$Class == 0,]),
                                           size = 2000),]  

#Combining data of random genuine samples and fraud samples
oversample.data <- rbind(genuine.sample.2000, train.credit[train.credit$Class == 1,])


#Oversampling the data to majority class(genuine)
oversample <- ovun.sample(formula = Class~., data = oversample.data,method = 'over', 
                          N = 4000)

qda.over <- qda(Class~.,data = oversample$data)

qda.over.pred <- predict(qda.over, newdata = test.credit[,-1],type = 'class')

conf.over.qda <- confusionMatrix(data = qda.over.pred$class, reference = test.credit$Class,
                                  positive = '1')

sensitivity.qda['Oversampling'] <- conf.over.qda$byClass[1]
specificity.qda['Oversampling'] <- conf.over.qda$byClass[2]


roc.curve(response = test.credit$Class, predicted = qda.over.pred$class,
          col = 'green', lwd = 2, add.roc = T)


#####----------- SMOTE----------#######

#Making balanced data using 'S'ynthetic 'M'inority 'O'versampling 'TE'chnique
smote.data <- SMOTE(form = Class~. , data = train.credit, perc.over = 200, k = 5, 
                    perc.under = 200)
#58% data belongs to majority and 42% belong to minority 
prop.table(table(smote.data$Class))



qda.smote <- qda(Class~.,data = smote.data)

qda.smote.pred <- predict(qda.smote, newdata = test.credit[,-1],type = 'class')

conf.smote.qda <- confusionMatrix(data = qda.smote.pred$class, reference = test.credit$Class,
                                 positive = '1')

sensitivity.qda['SMOTE'] <- conf.smote.qda$byClass[1]
specificity.qda['SMOTE'] <- conf.smote.qda$byClass[2]


roc.curve(response = test.credit$Class, predicted = qda.smote.pred$class,
          col = 'blue', lwd = 2, add.roc = T)


barchart(sensitivity.qda, main = "Senstivity plot for QDA Technique",
         xlab = "Sensitivity (Class = 'Fraud')", col = c('red','green','blue'))
barchart(specificity.qda, main = "Specificity plot for QDA Technique",
         xlab = "Specificity (Class = 'Non-Fraud')",col = c('red','green','blue'))


