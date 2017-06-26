# -*- coding: utf-8 -*-
from scipy import sparse
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import linear_model

#threshold=0.5,0.6,0.7,0.75,0.8,0.9
def nBayesClassifier(traindata,trainlabel,testdata,testlabel,threshold):
    clf = MultinomialNB()
    clf.fit(traindata,trainlabel)
    ypred= clf.predict_proba(testdata)
    ypred=np.where(ypred[:,1]>threshold,1,-1)
    count=0
    for i in range(len(testlabel)):
        if testlabel[i]==ypred[i]:
            count=count+1
    accuracy=float(count)/len(testlabel)
    return ypred,accuracy

#lambda=1e-4,0.01,0.1,1,0.5,1,10,100,1000,5000,10000
def lsClassifier(traindata,trainlabel,testdata,testlabel,Lambda):
    reg = linear_model.Ridge(alpha=Lambda)
    reg.fit(traindata,trainlabel)
    ypred=reg.predict(testdata)
    ypred = np.where(ypred > 0, 1, -1)
    count = 0
    for i in range(len(testlabel)):
        if testlabel[i] == ypred[i]:
            count = count + 1
    accuracy = float(count) / len(testlabel)
    return ypred, accuracy

#singma=0.01d,0.1d,d,10d,100d
#c=1,10,100,1000
def softsvm(traindata,trainlabel,testdata,testlabel,sigma,c):
    if sigma==0:
        clf = svm.SVC(kernel='linear',C=c)
    else:
        clf=svm.SVC(kernel='rbf',gamma=1.0/(sigma*sigma),C=c)
    clf.fit(traindata, trainlabel)
    ypred=clf.predict(testdata)
    count = 0
    for i in range(len(testlabel)):
        if testlabel[i] == ypred[i]:
            count = count + 1
    accuracy = float(count) / len(testlabel)
    return ypred,accuracy

def getFeature(comment): # read a comment
    row=[]
    col=[]
    data=[]
    comment=comment[1:]
    idf = np.loadtxt(open("idf.txt", "r"), delimiter=",")
    for i in range(len(comment)):
        index_value=comment[i].split(":")
        index=int(index_value[0])
        value = int(index_value[1])
        col.append(index)
        data.append(value)
        row.append(0)
    vcom = sparse.coo_matrix((data, (row, col)), shape=(1, 89527))
    v = np.array(vcom.toarray())
    v= preprocessing.normalize(v, norm='l1')
    v=v[0]*idf
    return v

trainlabel=[]
count=0
row = []
col = []
data = []
idf= np.loadtxt(open("idf.txt","r"),delimiter=",")
with open("data") as file:
    for line in file.readlines():
        line=line.strip().split(' ')
        label = int(line[0])
        if label >= 7:
            label = 1
        elif label <= 4:
            label = -1
        else:
            label = 0
        trainlabel.append(label)
        comment = line[1:]
        for i in range(len(comment)):
            index_value = comment[i].split(":")
            index = int(index_value[0])
            value = int(index_value[1])
            col.append(index)
            data.append(value*idf[index])
            row.append(count)
        count=count+1
traindata= sparse.coo_matrix((data, (row, col)), shape=(50000, 89527))
traindata= preprocessing.normalize(traindata, norm='l1')
trainlabel=np.array(trainlabel)
#print("finish")
kf=KFold(n_splits=5,shuffle=True)

threshold=[0.5,0.6,0.7,0.75,0.8,0.9]
Lambda=[0.0001,0.01,0.1,0.5,1,10,100,1000,5000,10000]
nbasy_accuracy=np.zeros((len(threshold),5))
lsq_accuracy=np.zeros((len(Lambda),5))

for i in range(len(threshold)):
    j=0
    for train_index, test_index in kf.split(traindata, trainlabel):
        ypred, nbasy_accuracy[i,j] = nBayesClassifier(traindata[train_index], trainlabel[train_index], traindata[test_index],
                                trainlabel[test_index], threshold[i])
        j=j+1
print nbasy_accuracy
np.savetxt("nbasy_accuracy.txt",nbasy_accuracy)

for i in range(len(Lambda)):
    j = 0
    for train_index, test_index in kf.split(traindata, trainlabel):
        ypred, lsq_accuracy[i,j] = lsClassifier(traindata[train_index], trainlabel[train_index], traindata[test_index],
                            trainlabel[test_index], Lambda[i])
        print(lsq_accuracy[i,j])
        j=j+1
print lsq_accuracy
np.savetxt("lsq_accuracy.txt",lsq_accuracy)

d=0.033
sigma=[0.01*d,0.1*d,d,10*d,100*d]
c=[1,10,100,1000]
svm_accuracy=np.zeros((len(sigma),len(c),5))
for i in range(len(sigma)):
    for j in range(len(c)):
        k = 0
        for train_index, test_index in kf.split(traindata, trainlabel):
            ypred, svm_accuracy[i, j,k] = softsvm(traindata[train_index], trainlabel[train_index], traindata[test_index],
                        trainlabel[test_index], sigma[i],c[j])
            print(svm_accuracy[i, j,k])
        k = k + 1
print svm_accuracy
np.savetxt("svm_accuracy.txt",svm_accuracy)
