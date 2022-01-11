portdata=read.csv('PortData.csv')
head(portdata)
str(portdata)
hist(portdata$Length,col='lightblue')
mydata=portdata[,c(3,6)]
head(mydata)
liner=mydata[which(mydata$VoyageType=='Liner'),]
feeder=mydata[which(mydata$VoyageType=='Feeder'),]
x1=liner[,c(1)]
x2=feeder[,c(1)]
x=list('liner vessels'=x1,'feeder vessels'=x2)
stripchart(x,xlab='points',ylab = 'vessel length',
           method = 'jitter',log='',col = c('orange','red')
           ,pch = 16)
boxplot(x,horizontal = TRUE,names = c('liner','feeder'))
mydata2=portdata[,c(2:5,7:9,11)]
cormat=cor(mydata2)
round(cormat,2)
source("http://www.sthda.com/upload/rquery_cormat.r")
library(corrplot)
rquery.cormat(mydata2,graphType = 'heatmap')
rquery.cormat(mydata2,type = 'flatten',graph = FALSE)
corrplot(cormat,method = 'number')













