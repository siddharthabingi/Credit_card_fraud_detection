#######-------------Creating a vector to save metrics for plotting---------######
sensitivity.svm <- c(rep(NA,3))
names(sensitivity.svm) <- c('Undersampling','Oversampling','SMOTE')
specificity.svm <- c(rep(NA,3))
names(specificity.svm) <- c('Undersampling','Oversampling','SMOTE')


library(e1071)

########---------------Undersampling---------------########
train.fraud <- nrow(train.credit[train.credit$Class == 1,])

#undersampling using ROSE package
undersample <- ovun.sample(formula = Class ~ ., data = train.credit, 
                           method = "under", N =  train.fraud*2)

svm.under <- svm(formula = Class~.,data = undersample$data, 
                 kernel = "linear",type = 'C-classification',
                 cost = 100,gamma = 1,scale = FALSE)

svm.under.pred <- predict(svm.under, newdata = test.credit[,-1],type = 'class')

conf.under.svm <- confusionMatrix(data = svm.under.pred, reference = test.credit$Class)

sensitivity.svm['Undersampling'] <- conf.under.svm$byClass[1]
specificity.svm['Undersampling'] <- conf.under.svm$byClass[2]


roc.curve(response = test.credit$Class, predicted = svm.under.pred,
          col = 'red', lwd = 2, main = 'SVM : ROC Curve for Sampling Technique')


#####--------------Oversampling------------######

#Taking randomly 2000 samples from class == 1 (genuine) 
genuine.sample.2000 <- train.credit[sample(x = nrow(train.credit[train.credit$Class == 0,]),
                                           size = 2000),]  

#Combining data of random genuine samples and fraud samples
oversample.data <- rbind(genuine.sample.2000, train.credit[train.credit$Class == 1,])


#Oversampling the data to majority class(genuine)
oversample <- ovun.sample(formula = Class~., data = oversample.data,method = 'over', 
                          N = 4000)


svm.over <- svm(formula = Class~.,data = oversample$data, 
                 kernel = "linear",type = 'C-classification',
                 cost = 100,gamma = 1,scale = FALSE)

svm.over.pred <- predict(svm.over, newdata = test.credit[,-1],type = 'class')

conf.over.svm <- confusionMatrix(data = svm.over.pred, reference = test.credit$Class)

sensitivity.svm['Oversampling'] <- conf.over.svm$byClass[1]
specificity.svm['Oversampling'] <- conf.over.svm$byClass[2]


roc.curve(response = test.credit$Class, predicted = svm.over.pred,
          col = 'green', lwd = 2, add.roc = TRUE)


#####----------- SMOTE----------#######

#Making balanced data using 'S'ynthetic 'M'inority 'O'versampling 'TE'chnique
smote.data <- SMOTE(form = Class~. , data = train.credit, perc.over = 200, k = 5, 
                    perc.under = 200)
#58% data belongs to majority and 42% belong to minority 
prop.table(table(smote.data$Class))


svm.smote <- svm(formula = Class~.,data = smote.data, 
                kernel = "linear",type = 'C-classification',
                cost = 100,gamma = 1,scale = FALSE)

svm.smote.pred <- predict(svm.smote, newdata = test.credit[,-1],type = 'class')

conf.smote.svm <- confusionMatrix(data = svm.smote.pred, reference = test.credit$Class)

sensitivity.svm['SMOTE'] <- conf.smote.svm$byClass[1]
specificity.svm['SMOTE'] <- conf.smote.svm$byClass[2]


roc.curve(response = test.credit$Class, predicted = svm.smote.pred,
          col = 'blue', lwd = 2, add.roc = T)

legend("bottomright", c("Undersampling", "Oversampling", 'SMOTE'), 
       col=c('red','green','blue'),lwd=2)



barchart(sensitivity.svm, main = "Senstivity plot for SVM Technique",
         xlab = "Sensitivity (Class = 'Fraud')", col = c('red','green','blue'),
         xlim = c(0.85,1))

barchart(specificity.svm, main = "Specificity plot for SVM Technique",
         xlab = "Specificity (Class = 'Non-Fraud')",col = c('red','green','blue'),
         xlim = c(0.85,1))

