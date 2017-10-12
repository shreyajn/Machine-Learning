########################Question 5########################
#read in the data
path <- system.file("mat-files", package="R.matlab")
pathname <- file.path("/Users/hellofutrue/Google Drive/Schoolwork_CU/Schoolwork Fall 2017/COMS4771 Machine Learning/HW1", "hw1data.mat")
data <- readMat(pathname)
#print(data)
X<-data$X
Y<-data$Y

#Select the features that has the top 200 variability
VAR=NULL
for (i in 1:ncol(X)){
 VAR[i]<-var(X[,i])}
VAR_ID<-cbind(c(1:ncol(X)),VAR) #ID and var of each feature
number<-VAR_ID[order(VAR_ID[,2],decreasing=T),][1:200,1]#select the features have the top 200 variability
X<-X[,number]

#normalized each features (transfer into Z score)
mu <- colMeans(X) 

ZX=matrix(,ncol=ncol(X),nrow=nrow(X))
for (i in 1:ncol(X)){
  ZX[,i]<-as.matrix((X-mu)[,i]/sqrt(var(X[,i])))
}

#create the whole dataset
dat<-cbind(ZX,Y)

#split the data into training and test
splitratio<-0.8
sample <- sample.int(n = nrow(dat), size = floor(splitratio*nrow(dat)), replace = F)
train <- dat[sample, ] #8000 datapoints for training
test  <- dat[-sample, ] #2000 datapoints for testing
testx<-test[,-201]
trainx<-train[,-201]


#(i) Gaussian classfier 
#splititng the training data into 10 subgroups
y0<-subset(train[which(train[,201]==0),])
y1<-subset(train[which(train[,201]==1),])
y2<-subset(train[which(train[,201]==2),])
y3<-subset(train[which(train[,201]==3),])
y4<-subset(train[which(train[,201]==4),])
y5<-subset(train[which(train[,201]==5),])
y6<-subset(train[which(train[,201]==6),])
y7<-subset(train[which(train[,201]==7),])
y8<-subset(train[which(train[,201]==8),])
y9<-subset(train[which(train[,201]==9),])

#calculate MLEs
xy0<-y0[,-201]#only the features when y=0
xy1<-y1[,-201]
xy2<-y2[,-201]
xy3<-y3[,-201]
xy4<-y4[,-201]
xy5<-y5[,-201]
xy6<-y6[,-201]
xy7<-y7[,-201]
xy8<-y8[,-201]
xy9<-y9[,-201]

#mu
mu0<-colMeans(xy0)
mu1<-colMeans(xy1)
mu2<-colMeans(xy2)
mu3<-colMeans(xy3)
mu4<-colMeans(xy4)
mu5<-colMeans(xy5)
mu6<-colMeans(xy6)
mu7<-colMeans(xy7)
mu8<-colMeans(xy8)
mu9<-colMeans(xy9)

#sigma matrices 
matrices0<-list()
for (i in 1:nrow(xy0)){
  matrices0[[i]]<-t(t(xy0[i,]-mu0))%*%t(xy0[i,]-mu0)
}
sigma0<-apply(simplify2array(matrices0), 1:2, mean)

matrices1<-list()
for (i in 1:nrow(xy1)){
  matrices1[[i]]<-t(t(xy1[i,]-mu1))%*%t(xy1[i,]-mu1)
}
sigma1<-apply(simplify2array(matrices1), 1:2, mean)

matrices2<-list()
for (i in 1:nrow(xy2)){
  matrices2[[i]]<-t(t(xy2[i,]-mu2))%*%t(xy2[i,]-mu2)
}
sigma2<-apply(simplify2array(matrices2), 1:2, mean)

matrices3<-list()
for (i in 1:nrow(xy3)){
  matrices3[[i]]<-t(t(xy3[i,]-mu3))%*%t(xy3[i,]-mu3)
}
sigma3<-apply(simplify2array(matrices3), 1:2, mean)

matrices4<-list()
for (i in 1:nrow(xy4)){
  matrices4[[i]]<-t(t(xy4[i,]-mu4))%*%t(xy4[i,]-mu4)
}
sigma4<-apply(simplify2array(matrices4), 1:2, mean)

matrices5<-list()
for (i in 1:nrow(xy5)){
  matrices5[[i]]<-t(t(xy5[i,]-mu5))%*%t(xy5[i,]-mu5)
}
sigma5<-apply(simplify2array(matrices5), 1:2, mean)

matrices6<-list()
for (i in 1:nrow(xy6)){
  matrices6[[i]]<-t(t(xy6[i,]-mu6))%*%t(xy6[i,]-mu6)
}
sigma6<-apply(simplify2array(matrices6), 1:2, mean)

matrices7<-list()
for (i in 1:nrow(xy7)){
  matrices7[[i]]<-t(t(xy7[i,]-mu0))%*%t(xy7[i,]-mu7)
}
sigma7<-apply(simplify2array(matrices7), 1:2, mean)

matrices8<-list()
for (i in 1:nrow(xy8)){
  matrices8[[i]]<-t(t(xy8[i,]-mu8))%*%t(xy8[i,]-mu8)
}
sigma8<-apply(simplify2array(matrices8), 1:2, mean)

matrices9<-list()
for (i in 1:nrow(xy9)){
  matrices9[[i]]<-t(t(xy9[i,]-mu9))%*%t(xy9[i,]-mu9)
}
sigma9<-apply(simplify2array(matrices9), 1:2, mean)

#det(sigma0)

# h<-eigen(sigma0)$values[201]
# add<-h*diag(ncol(sigma0))
# sigma0new<-sigma0+add
# det(sigma0new)

#learning about class priors (y=0....9)
freq<-table(train[,201])/10000
py0<-as.numeric(freq[1]) 
py1<-as.numeric(freq[2]) 
py2<-as.numeric(freq[3]) 
py3<-as.numeric(freq[4]) 
py4<-as.numeric(freq[5]) 
py5<-as.numeric(freq[6]) 
py6<-as.numeric(freq[7]) 
py7<-as.numeric(freq[8]) 
py8<-as.numeric(freq[9]) 
py9<-as.numeric(freq[10]) 


#calculte the probablity with test data
mu0<-matrix(mu0,ncol=ncol(testx),nrow=1,byrow = T)
mu1<-matrix(mu1,ncol=ncol(testx),nrow=1,byrow = T)
mu2<-matrix(mu2,ncol=ncol(testx),nrow=1,byrow = T)
mu3<-matrix(mu3,ncol=ncol(testx),nrow=1,byrow = T)
mu4<-matrix(mu4,ncol=ncol(testx),nrow=1,byrow = T)
mu5<-matrix(mu5,ncol=ncol(testx),nrow=1,byrow = T)
mu6<-matrix(mu6,ncol=ncol(testx),nrow=1,byrow = T)
mu7<-matrix(mu7,ncol=ncol(testx),nrow=1,byrow = T)
mu8<-matrix(mu8,ncol=ncol(testx),nrow=1,byrow = T)
mu9<-matrix(mu9,ncol=ncol(testx),nrow=1,byrow = T)

p<-matrix(,ncol=10,nrow=nrow(testx)) #prepare the form of probability
for (n in 1:nrow(testx)){
  p[n,1]<-(((2*pi)^ncol(testx)*det(sigma0))^(-0.5)*(exp(-0.5*(testx[n,]-mu0)%*%solve(sigma0)%*%t(testx[n,]-mu0))))*py0
  p[n,2]<-(((2*pi)^ncol(testx)*det(sigma1))^(-0.5)*(exp(-0.5*(testx[n,]-mu1)%*%solve(sigma1)%*%t(testx[n,]-mu1))))*py1
  p[n,3]<-(((2*pi)^ncol(testx)*det(sigma2))^(-0.5)*(exp(-0.5*(testx[n,]-mu2)%*%solve(sigma2)%*%t(testx[n,]-mu2))))*py2
  p[n,4]<-(((2*pi)^ncol(testx)*det(sigma3))^(-0.5)*(exp(-0.5*(testx[n,]-mu3)%*%solve(sigma3)%*%t(testx[n,]-mu3))))*py3
  p[n,5]<-(((2*pi)^ncol(testx)*det(sigma4))^(-0.5)*(exp(-0.5*(testx[n,]-mu4)%*%solve(sigma4)%*%t(testx[n,]-mu4))))*py4
  p[n,6]<-(((2*pi)^ncol(testx)*det(sigma5))^(-0.5)*(exp(-0.5*(testx[n,]-mu5)%*%solve(sigma5)%*%t(testx[n,]-mu5))))*py5
  p[n,7]<-(((2*pi)^ncol(testx)*det(sigma6))^(-0.5)*(exp(-0.5*(testx[n,]-mu6)%*%solve(sigma6)%*%t(testx[n,]-mu6))))*py6
  p[n,8]<-(((2*pi)^ncol(testx)*det(sigma7))^(-0.5)*(exp(-0.5*(testx[n,]-mu7)%*%solve(sigma7)%*%t(testx[n,]-mu7))))*py7
  p[n,9]<-(((2*pi)^ncol(testx)*det(sigma8))^(-0.5)*(exp(-0.5*(testx[n,]-mu8)%*%solve(sigma8)%*%t(testx[n,]-mu8))))*py8
  p[n,10]<-(((2*pi)^ncol(testx)*det(sigma9))^(-0.5)*(exp(-0.5*(testx[n,]-mu9)%*%solve(sigma9)%*%t(testx[n,]-mu9))))*py9
}

#Give out prediction
colnames(p)<-c(0:9)
predi<-colnames(p)[apply(p,1,which.max)]
predi<-as.matrix(predi,ncol=1,nrow=nrow(testx))

#Calculate accuracy
accu<-cbind(predi,test[,201])
accuracy<-accu[,1]==accu[,2]
accuracyrate<-length(accuracy[accuracy==TRUE])/nrow(accu)


#K-NN classifier
#for a single test point, calculate the euclidian distance of it to all the training data
#select the nearest k points, capture the label with highest frequency, assign it to the test point

#calculate the euclidian distance from every test data to all the training data
euc<-matrix(,nrow =nrow(testx),ncol=nrow(trainx))
for (n in 1:nrow(testx)){
  x<-matrix(testx[n,],nrow=nrow(trainx),ncol=ncol(testx),byrow=T)
  d<-sqrt(rowSums((x-trainx)^2))
  euc[n,]<-d
}
colnames(euc)<-c(1:nrow(trainx))

k=3
labelpredi<-matrix(,nrow=nrow(euc),ncol=1)
  for (n in 1:nrow(euc)){
    topk<-as.numeric(names(sort(euc[n,])[1:k]))
    label<-train[topk,201]
    labelpredi[n,]<-names(sort(table(label))[length(table(label))])
  }
  
#Calculate accuracy
  knnaccu<-cbind(labelpredi,test[,201])
  knnaccuracy<-knnaccu[,1]==knnaccu[,2]
  knnaccuracyrate<-length(knnaccuracy[knnaccuracy==TRUE])/nrow(knnaccu)
  table<-rbind(k,knnaccuracyrate)
  

# (iii) compare two classifiers
compare<-function(splitratio){
  sample <- sample.int(n = nrow(dat), size = floor(splitratio*nrow(dat)), replace = F)
  train <- dat[sample, ] #8000 datapoints for training
  test  <- dat[-sample, ] #2000 datapoints for testing
  testx<-test[,-201]
  trainx<-train[,-201]
  
  
  #(i) Gaussian classfier 
  #splititng the training data into 10 subgroups
  y0<-subset(train[which(train[,201]==0),])
  y1<-subset(train[which(train[,201]==1),])
  y2<-subset(train[which(train[,201]==2),])
  y3<-subset(train[which(train[,201]==3),])
  y4<-subset(train[which(train[,201]==4),])
  y5<-subset(train[which(train[,201]==5),])
  y6<-subset(train[which(train[,201]==6),])
  y7<-subset(train[which(train[,201]==7),])
  y8<-subset(train[which(train[,201]==8),])
  y9<-subset(train[which(train[,201]==9),])
  
  #calculate MLEs
  xy0<-y0[,-201]#only the features when y=0
  xy1<-y1[,-201]
  xy2<-y2[,-201]
  xy3<-y3[,-201]
  xy4<-y4[,-201]
  xy5<-y5[,-201]
  xy6<-y6[,-201]
  xy7<-y7[,-201]
  xy8<-y8[,-201]
  xy9<-y9[,-201]
  
  #mu
  mu0<-colMeans(xy0)
  mu1<-colMeans(xy1)
  mu2<-colMeans(xy2)
  mu3<-colMeans(xy3)
  mu4<-colMeans(xy4)
  mu5<-colMeans(xy5)
  mu6<-colMeans(xy6)
  mu7<-colMeans(xy7)
  mu8<-colMeans(xy8)
  mu9<-colMeans(xy9)
  
  #sigma matrices 
  matrices0<-list()
  for (i in 1:nrow(xy0)){
    matrices0[[i]]<-t(t(xy0[i,]-mu0))%*%t(xy0[i,]-mu0)
  }
  sigma0<-apply(simplify2array(matrices0), 1:2, mean)
  
  matrices1<-list()
  for (i in 1:nrow(xy1)){
    matrices1[[i]]<-t(t(xy1[i,]-mu1))%*%t(xy1[i,]-mu1)
  }
  sigma1<-apply(simplify2array(matrices1), 1:2, mean)
  
  matrices2<-list()
  for (i in 1:nrow(xy2)){
    matrices2[[i]]<-t(t(xy2[i,]-mu2))%*%t(xy2[i,]-mu2)
  }
  sigma2<-apply(simplify2array(matrices2), 1:2, mean)
  
  matrices3<-list()
  for (i in 1:nrow(xy3)){
    matrices3[[i]]<-t(t(xy3[i,]-mu3))%*%t(xy3[i,]-mu3)
  }
  sigma3<-apply(simplify2array(matrices3), 1:2, mean)
  
  matrices4<-list()
  for (i in 1:nrow(xy4)){
    matrices4[[i]]<-t(t(xy4[i,]-mu4))%*%t(xy4[i,]-mu4)
  }
  sigma4<-apply(simplify2array(matrices4), 1:2, mean)
  
  matrices5<-list()
  for (i in 1:nrow(xy5)){
    matrices5[[i]]<-t(t(xy5[i,]-mu5))%*%t(xy5[i,]-mu5)
  }
  sigma5<-apply(simplify2array(matrices5), 1:2, mean)
  
  matrices6<-list()
  for (i in 1:nrow(xy6)){
    matrices6[[i]]<-t(t(xy6[i,]-mu6))%*%t(xy6[i,]-mu6)
  }
  sigma6<-apply(simplify2array(matrices6), 1:2, mean)
  
  matrices7<-list()
  for (i in 1:nrow(xy7)){
    matrices7[[i]]<-t(t(xy7[i,]-mu0))%*%t(xy7[i,]-mu7)
  }
  sigma7<-apply(simplify2array(matrices7), 1:2, mean)
  
  matrices8<-list()
  for (i in 1:nrow(xy8)){
    matrices8[[i]]<-t(t(xy8[i,]-mu8))%*%t(xy8[i,]-mu8)
  }
  sigma8<-apply(simplify2array(matrices8), 1:2, mean)
  
  matrices9<-list()
  for (i in 1:nrow(xy9)){
    matrices9[[i]]<-t(t(xy9[i,]-mu9))%*%t(xy9[i,]-mu9)
  }
  sigma9<-apply(simplify2array(matrices9), 1:2, mean)
  
  #det(sigma0)
  
  # h<-eigen(sigma0)$values[201]
  # add<-h*diag(ncol(sigma0))
  # sigma0new<-sigma0+add
  # det(sigma0new)
  
  #learning about class priors (y=0....9)
  freq<-table(train[,201])/10000
  py0<-as.numeric(freq[1]) 
  py1<-as.numeric(freq[2]) 
  py2<-as.numeric(freq[3]) 
  py3<-as.numeric(freq[4]) 
  py4<-as.numeric(freq[5]) 
  py5<-as.numeric(freq[6]) 
  py6<-as.numeric(freq[7]) 
  py7<-as.numeric(freq[8]) 
  py8<-as.numeric(freq[9]) 
  py9<-as.numeric(freq[10]) 
  
  
  #calculte the probablity with test data
  mu0<-matrix(mu0,ncol=ncol(testx),nrow=1,byrow = T)
  mu1<-matrix(mu1,ncol=ncol(testx),nrow=1,byrow = T)
  mu2<-matrix(mu2,ncol=ncol(testx),nrow=1,byrow = T)
  mu3<-matrix(mu3,ncol=ncol(testx),nrow=1,byrow = T)
  mu4<-matrix(mu4,ncol=ncol(testx),nrow=1,byrow = T)
  mu5<-matrix(mu5,ncol=ncol(testx),nrow=1,byrow = T)
  mu6<-matrix(mu6,ncol=ncol(testx),nrow=1,byrow = T)
  mu7<-matrix(mu7,ncol=ncol(testx),nrow=1,byrow = T)
  mu8<-matrix(mu8,ncol=ncol(testx),nrow=1,byrow = T)
  mu9<-matrix(mu9,ncol=ncol(testx),nrow=1,byrow = T)
  
  p<-matrix(,ncol=10,nrow=nrow(testx)) #prepare the form of probability
  for (n in 1:nrow(testx)){
    p[n,1]<-(((2*pi)^ncol(testx)*det(sigma0))^(-0.5)*(exp(-0.5*(testx[n,]-mu0)%*%solve(sigma0)%*%t(testx[n,]-mu0))))*py0
    p[n,2]<-(((2*pi)^ncol(testx)*det(sigma1))^(-0.5)*(exp(-0.5*(testx[n,]-mu1)%*%solve(sigma1)%*%t(testx[n,]-mu1))))*py1
    p[n,3]<-(((2*pi)^ncol(testx)*det(sigma2))^(-0.5)*(exp(-0.5*(testx[n,]-mu2)%*%solve(sigma2)%*%t(testx[n,]-mu2))))*py2
    p[n,4]<-(((2*pi)^ncol(testx)*det(sigma3))^(-0.5)*(exp(-0.5*(testx[n,]-mu3)%*%solve(sigma3)%*%t(testx[n,]-mu3))))*py3
    p[n,5]<-(((2*pi)^ncol(testx)*det(sigma4))^(-0.5)*(exp(-0.5*(testx[n,]-mu4)%*%solve(sigma4)%*%t(testx[n,]-mu4))))*py4
    p[n,6]<-(((2*pi)^ncol(testx)*det(sigma5))^(-0.5)*(exp(-0.5*(testx[n,]-mu5)%*%solve(sigma5)%*%t(testx[n,]-mu5))))*py5
    p[n,7]<-(((2*pi)^ncol(testx)*det(sigma6))^(-0.5)*(exp(-0.5*(testx[n,]-mu6)%*%solve(sigma6)%*%t(testx[n,]-mu6))))*py6
    p[n,8]<-(((2*pi)^ncol(testx)*det(sigma7))^(-0.5)*(exp(-0.5*(testx[n,]-mu7)%*%solve(sigma7)%*%t(testx[n,]-mu7))))*py7
    p[n,9]<-(((2*pi)^ncol(testx)*det(sigma8))^(-0.5)*(exp(-0.5*(testx[n,]-mu8)%*%solve(sigma8)%*%t(testx[n,]-mu8))))*py8
    p[n,10]<-(((2*pi)^ncol(testx)*det(sigma9))^(-0.5)*(exp(-0.5*(testx[n,]-mu9)%*%solve(sigma9)%*%t(testx[n,]-mu9))))*py9
  }
  
  #Give out prediction
  colnames(p)<-c(0:9)
  predi<-colnames(p)[apply(p,1,which.max)]
  predi<-as.matrix(predi,ncol=1,nrow=nrow(testx))
  
  #Calculate accuracy
  accu<-cbind(predi,test[,201])
  accuracy<-accu[,1]==accu[,2]
  accuracyrate<-length(accuracy[accuracy==TRUE])/nrow(accu)

  
  #K-NN classifier
  #calculate the euclidian distance from every test data to all the training data
  euc<-matrix(,nrow =nrow(testx),ncol=nrow(trainx))
  for (n in 1:nrow(testx)){
    x<-matrix(testx[n,],nrow=nrow(trainx),ncol=ncol(testx),byrow=T)
    d<-sqrt(rowSums((x-trainx)^2))
    euc[n,]<-d
  }
  colnames(euc)<-c(1:nrow(trainx))
  
  k=3
  labelpredi<-matrix(,nrow=nrow(euc),ncol=1)
  for (n in 1:nrow(euc)){
    topk<-as.numeric(names(sort(euc[n,])[1:k]))
    label<-train[topk,201]
    labelpredi[n,]<-names(sort(table(label))[length(table(label))])
  }
  
  #Calculate accuracy
  knnaccu<-cbind(labelpredi,test[,201])
  knnaccuracy<-knnaccu[,1]==knnaccu[,2]
  knnaccuracyrate<-length(knnaccuracy[knnaccuracy==TRUE])/nrow(knnaccu)
  table<-rbind(k,knnaccuracyrate)
  
  out<-c(splitratio,accuracyrate,knnaccuracyrate)
  
  print(splitratio)
  return(out)
}

RESULT<-NULL
for (s in (c(0.8,0.6,0.4,0.3))){
  result<-compare(s)
  RESULT<-rbind(result,RESULT)
}

ggplot(RESULT, aes(x=RESULT$splitratio)) + 
  geom_line(aes(y=RESULT$Gaussian.accuracy.rate, color="Gaussian classifier")) + xlab("Percentage of training data in the whole data")+
  geom_line(aes(y=RESULT$k.nn.accuracy.rate, color="k-NN classifier, k=3"))+ ylab("Accuracy rate")

#(iv)Different distance definitions L1, L2 and L-infinite

distance<-function(splitratio){
  sample <- sample.int(n = nrow(dat), size = floor(splitratio*nrow(dat)), replace = F)
  train <- dat[sample, ] #8000 datapoints for training
  test  <- dat[-sample, ] #2000 datapoints for testing
  testx<-test[,-201]
  trainx<-train[,-201]

L1<-matrix(,nrow =nrow(testx),ncol=nrow(trainx))
L2<-matrix(,nrow =nrow(testx),ncol=nrow(trainx))
Linf<-matrix(,nrow =nrow(testx),ncol=nrow(trainx))
  for (n in 1:nrow(testx)){
    x<-matrix(testx[n,],nrow=nrow(trainx),ncol=ncol(testx),byrow=T)
    d1<-rowSums(abs(x-trainx))
    d2<-sqrt(rowSums((x-trainx)^2))
    dinf<-apply(abs(x-trainx),1,max)
    L1[n,]<-d1
    L2[n,]<-d2
    Linf[n,]<-dinf
  }
colnames(L1)<-c(1:nrow(trainx))
colnames(L2)<-c(1:nrow(trainx))
colnames(Linf)<-c(1:nrow(trainx))

k=3
labelpredi1<-matrix(,nrow=nrow(L1),ncol=1)
labelpredi2<-matrix(,nrow=nrow(L2),ncol=1)
labelpredi3<-matrix(,nrow=nrow(Linf),ncol=1)
for (n in 1:nrow(L1)){
  #predict based on L1
  topk1<-as.numeric(names(sort(L1[n,])[1:k]))
  label1<-train[topk1,201]
  labelpredi1[n,]<-names(sort(table(label1))[length(table(label1))])
  #predict based on L2
  topk2<-as.numeric(names(sort(L2[n,])[1:k]))
  label2<-train[topk2,201]
  labelpredi2[n,]<-names(sort(table(label2))[length(table(label2))])
  #predict based on Linf
  topk3<-as.numeric(names(sort(Linf[n,])[1:k]))
  label3<-train[topk3,201]
  labelpredi3[n,]<-names(sort(table(label3))[length(table(label3))])
}

#Calculate accuracy
knnaccu<-cbind(labelpredi1,labelpredi2,labelpredi3,test[,201])
L1knnaccuracy<-knnaccu[,1]==knnaccu[,4]
L1knnaccuracyrate<-length(L1knnaccuracy[L1knnaccuracy==TRUE])/nrow(knnaccu)

L2knnaccuracy<-knnaccu[,2]==knnaccu[,4]
L2knnaccuracyrate<-length(L2knnaccuracy[L2knnaccuracy==TRUE])/nrow(knnaccu)

Linfknnaccuracy<-knnaccu[,3]==knnaccu[,4]
Linfknnaccuracyrate<-length(Linfknnaccuracy[Linfknnaccuracy==TRUE])/nrow(knnaccu)

out<-rbind(splitratio,L1knnaccuracyrate,L2knnaccuracyrate,Linfknnaccuracyrate)
}

D<-NULL
for (s in (c(0.8,0.6,0.4))){
  result<-distance(s)
  D<-cbind(D,result)
}

ggplot(D, aes(x=D$splitratio)) + 
  geom_line(aes(y=D$L1knnaccuracyrate, color="L1 distance")) + xlab("Percentage of training data in the whole data")+
  geom_line(aes(y=D$L2knnaccuracyrate, color="L2 distance"))+ ylab("Accuracy rate,k=3")+
  geom_line(aes(y=D$Linfknnaccuracyrate, color="L-infinite"))


