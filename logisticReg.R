############################################################################
####-----------------------Credit Card : Logistic Reg------------------#####
############################################################################



#######-------------Creating a vector to save metrics for plotting---------######
sensitivity.logis <- c(rep(NA,3))
names(sensitivity.logis) <- c('Undersampling','Oversampling','SMOTE')
specificity.logis <- c(rep(NA,3))
names(specificity.logis) <- c('Undersampling','Oversampling','SMOTE')




########---------------Undersampling---------------########
train.fraud <- nrow(train.credit[train.credit$Class == 1,])

#undersampling using ROSE package
undersample <- ovun.sample(formula = Class ~ ., data = train.credit, 
                           method = "under", N =  train.fraud*2)

logis.under <- glm(Class~ ., family=binomial(link="logit"), data = undersample$data, maxit = 50)

logis.under.pred <- predict(logis.under,newdata = test.credit[,-1], type="response")

summary(logis.under)

logis.under.pred <- ifelse(logis.under.pred <=0.5, 0, 1)

conf.under.logis <- confusionMatrix(data = logis.under.pred, 
                                    reference = test.credit$Class)

ROCRpred <- prediction(logis.under.pred, test.credit$Class)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))


sensitivity.logis['Undersampling'] <- conf.under.logis$byClass[1]
specificity.logis['Undersampling'] <- conf.under.logis$byClass[2]

theme_update(plot.title = element_text(hjust = 0.5))
ggplot(test.credit, aes(x =  logis.under.pred, 
                       fill = (Class), colour = (Class))) +
  geom_density() + xlab("Predicted values on Undersampled data") + ylab("density") + 
  ggtitle("Density of fail/success on Validation set")

roc.curve(response = test.credit$Class, predicted = logis.under.pred,
          col = 'red', lwd = 2, main = 'Logistic Regression: ROC Curve for Sampling Technique')


#####--------------Oversampling------------######

#Taking randomly 2000 samples from class == 1 (genuine) 
genuine.sample.2000 <- train.credit[sample(x = nrow(train.credit[train.credit$Class == 0,]),
                                           size = 2000),]  

#Combining data of random genuine samples and fraud samples
oversample.data <- rbind(genuine.sample.2000, train.credit[train.credit$Class == 1,])


#Oversampling the data to majority class(genuine)
oversample <- ovun.sample(formula = Class~., data = oversample.data,method = 'over', 
                          N = 4000)

logis.over <- glm(Class~ ., family=binomial(link="logit"), data = oversample$data, maxit = 50)

logis.over.pred <- predict(logis.over,newdata = test.credit[,-1], type="response")


logis.over.pred <- ifelse(logis.over.pred <=0.5, 0, 1)

conf.over.logis <-confusionMatrix(data = logis.over.pred, reference = test.credit$Class)

sensitivity.logis['Oversampling'] <- conf.over.logis$byClass[1]
specificity.logis['Oversampling'] <- conf.over.logis$byClass[2]

theme_update(plot.title = element_text(hjust = 0.5))
ggplot(test.credit, aes(x =  logis.over.pred, 
                        fill = (Class), colour = (Class))) +
  geom_density() + xlab("Predicted values on Oversampled data") + ylab("density") + 
  ggtitle("Density of fail/success on Validation set")

roc.curve(response = test.credit$Class, predicted = logis.over.pred,
          add.roc =TRUE, col = 'green', lwd = 2)



#####----------- SMOTE----------#######

#Making balanced data using 'S'ynthetic 'M'inority 'O'versampling 'TE'chnique
smote.data <- SMOTE(form = Class~. , data = train.credit, perc.over = 200, k = 5, 
                    perc.under = 200)
#58% data belongs to majority and 42% belong to minority 
prop.table(table(smote.data$Class))

logis.smote <- glm(Class~ ., family=binomial(link="logit"), data = smote.data, maxit = 50)

logis.smote.pred <- predict(logis.smote,newdata = test.credit[,-1], type="response")



logis.smote.pred <- ifelse(logis.smote.pred <=0.5, 0, 1)

conf.smote.logis <- confusionMatrix(data = logis.smote.pred, reference = test.credit$Class)

sensitivity.logis['SMOTE'] <- conf.smote.logis$byClass[1]
specificity.logis['SMOTE'] <- conf.smote.logis$byClass[2]


theme_update(plot.title = element_text(hjust = 0.5))
ggplot(test.credit, aes(x =  logis.smote.pred, 
                      fill = (Class), colour = (Class))) +
  geom_density() + xlab("Predicted values on Oversampled data") + ylab("density") + 
  ggtitle("Density of fail/success on Validation set")

roc.curve(response = test.credit$Class, predicted = logis.smote.pred,
          add.roc =TRUE, col = 'blue', lwd = 2)

legend("bottomright", c("Undersampling", "Oversampling", 'SMOTE'), 
       col=c('red','green','blue'),lwd=2)


#Plotting histogram for sensitivity and specificity
par(mfrow = c(1,2))

barchart(sensitivity.logis, main = "Senstivity plot for Logistic Regression Technique",
         xlab = "Sensitivity (Class = 'Non-Fraud')", col = c('red','green','blue'), 
         xlim = c(0.85,1))
barchart(specificity.logis, main = "Specificity plot for Logistic Regression Technique",
         xlab = "Specificity (Class = 'Fraud')",col = c('red','green','blue'),
         xlim = c(0.85,1))
