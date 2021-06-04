setwd("")
library(readxl)
library(nFactors)
library(corrplot)
library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
library(ggord)
library(ggplot2)
library(Hmisc)
library(klaR)
library(klaR)
library(MASS)
library(nnet)
library(plyr)
library(pROC)
library(psych)
library(scatterplot3d)
library(SDMTools)
library(dplyr)
library(ElemStatLearn)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neuralnet)
miniproject4.dataset <- read_xlsx("Cellphone.xlsx")

#Exploratory data analysis
attach(miniproject4.dataset)
boxplot(miniproject4.dataset$AccountWeeks, horizontal=TRUE, main="Boxplot for AccountWeeks")
boxplot(miniproject4.dataset$DataUsage, horizontal=TRUE, main="Boxplot for DataUsage")
boxplot(miniproject4.dataset$CustServCalls, horizontal=TRUE, main="Boxplot for CustServCalls")
boxplot(miniproject4.dataset$DayMins, horizontal=TRUE, main="Boxplot for DayMins")
boxplot(miniproject4.dataset$DayCalls, horizontal=TRUE, main="Boxplot for DayCalls")
boxplot(miniproject4.dataset$MonthlyCharge, horizontal=TRUE, main="Boxplot for MonthlyCharge")
boxplot(miniproject4.dataset$OverageFee, horizontal=TRUE, main="Boxplot for OverageFee")
boxplot(miniproject4.dataset$RoamMins, horizontal=TRUE, main="Boxplot for RoamMins")
pairs(miniproject4.dataset[,-c(1,3,4)])

#Principal Component Analysis
mynewdata=miniproject4.dataset[,-c(1,3,4)]
myfulldata=miniproject4.dataset[,-1]
cormatbig=cor(myfulldata)
cormatbig
corrplot(cormatbig,method="number")
myeigenvalues=eigen(cormatbig)
myeigenvalues$values
myeigenvalues$vectors
factorids=c(1:10)
screeplot=plot(data.frame(factorids,myeigenvalues$values),col="Blue")
lines(data.frame(factorids,myeigenvalues$values),col="Blue")
rotatematrix=principal(myfulldata,nfactors = 8,rotate = "varimax")
rotatematrix
moredata=rotatematrix$scores
moredata
moredata=data.frame(moredata,miniproject4.dataset$Churn)
attach(moredata)


#Model Building
set.seed(12)
pd2<-sample(2,nrow(moredata),replace=TRUE, prob=c(0.7,0.3))



train2<-moredata[pd2==1,]
val2<-moredata[pd2==2,]


sum(moredata$miniproject4.dataset.Churn)
sum(val2$miniproject4.dataset.Churn)
sum(train2$miniproject4.dataset.Churn)

#logistic regression for full model
logreg=miniproject4.dataset.Churn~RC1+RC2+RC3+RC4+RC5+RC6+RC7+RC8
logit.plot<-glm(logreg, data=moredata, family=binomial())
summary(logit.plot)

pred.logit.final <- predict.glm(logit.plot, newdata=moredata, type="response")


tab.logit<-confusion.matrix(miniproject4.dataset.Churn,pred.logit.final,threshold = 0.5)
tab.logit
accuracy.logit<-roc.logit<-roc(miniproject4.dataset.Churn,pred.logit.final )
roc.logit
plot(roc.logit)


#logistic regression for training model
logit.plot<-glm(miniproject4.dataset.Churn~RC1+RC2+RC3+RC4+RC5+RC8, data=train2, family=binomial())
summary(logit.plot)
pred.logit.final <- predict.glm(logit.plot, newdata=train2, type="response")


tab.logit<-confusion.matrix(train2$miniproject4.dataset.Churn,pred.logit.final,threshold = 0.5)
tab.logit
accuracy.logit<-roc.logit<-roc(train2$miniproject4.dataset.Churn,pred.logit.final )
roc.logit
plot(roc.logit)


#logistic regression for testing model
logit.plot<-glm(miniproject4.dataset.Churn~RC1+RC2+RC3+RC4+RC5+RC8, data=train2, family=binomial())
summary(logit.plot)
pred.logit.final <- predict.glm(logit.plot, newdata=val2, type="response")


tab.logit<-confusion.matrix(val2$miniproject4.dataset.Churn,pred.logit.final,threshold = 0.5)
tab.logit
accuracy.logit<-roc.logit<-roc(val2$miniproject4.dataset.Churn,pred.logit.final )
roc.logit
plot(roc.logit)


#knn for training and testing model
y_pred<-knn(train=train2[,-9],test=val2[,-9], cl=train2[,9],k=19)
tab.knn<-table(val2[,9],y_pred)
tab.knn


accuracy.knn<-sum(diag(tab.knn))/sum(tab.knn)
accuracy.knn

loss.knn<-tab.knn[2,1]/(tab.knn[2,1]+tab.knn[1,1])
loss.knn
opp.loss.knn<-tab.knn[1,2]/(tab.knn[1,2]+tab.knn[2,2])
opp.loss.knn
tot.loss.knn<-0.95*loss.knn+0.05*opp.loss.knn
tot.loss.knn


#naive-bayes for the training and testing dataset
train2$faChurn<-as.factor(train2$miniproject4.dataset.Churn)
val2$faChurn<-as.factor(val2$miniproject4.dataset.Churn)

NB<-naiveBayes(x=train2[-9], y=train2$faChurn)
#predict
y_pred.NB<-predict(NB,newdata=val2[-9])
y_pred.NB


#Confusion matrix

tab.NB=table(val2[,9],y_pred.NB)
tab.NB



accuracy.NB<-sum(diag(tab.NB))/sum(tab.NB)
accuracy.NB
loss.NB<-tab.NB[2,1]/(tab.NB[2,1]+tab.NB[1,1])
loss.NB
opp.loss.NB<-tab.NB[1,2]/(tab.NB[1,2]+tab.NB[2,2])
opp.loss.NB
tot.loss.NB<-0.95*loss.NB+0.05*opp.loss.NB
tot.loss.NB

