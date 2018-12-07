#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:24:03 2018

@author: wajih
"""


import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import numpy as np


inputData = scipy.io.loadmat('/media/wajih/Disk1 500 GB/Onus/RnD/DataSet/handwritten/mnist/mnist_uint8')
if inputData is not None:
    keys = inputData.keys()
    for key in keys:
        if key == "train_x":
            train_X = inputData[key].copy()
            train_X = train_X/255.0
        elif key == "train_y":
            train_Y = inputData[key].copy()
            train_Y = train_Y
        elif key == "test_x":
            test_X = inputData[key].copy()
            test_X = test_X/255.0
        elif key == "test_y":
            test_Y = inputData[key].copy()
            test_Y = test_Y            
        
# Reduce the one hot encoding
train_Y = np.argmax(train_Y, axis=1)
test_Y = np.argmax(test_Y,axis=1)
# Lets view an entry 
index = 77
v = test_X[index,:].reshape(28,28)
fig = plt.figure(figsize = (10,10))
ax = fig.gca()
plt.title("Original Image - 1D plot",fontsize=5)
plt.plot(test_X[index,:])
plt.show()

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
plt.title("Original Image",fontsize=10)
plt.imshow(v)
plt.show()

nSamples = 0.25 #Percentage
nSamplesTrain = np.int(np.round(train_X.shape[0]*nSamples))
nSamplesTest = np.int(np.round(test_X.shape[0]*nSamples))
# We already have the train and test data, so lets throw them into XGBOOST and see if we can
# do some classification


from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


classifier =  XGBClassifier(
learning_rate =0.3,
 n_estimators=18,
 max_depth=7,
 min_child_weight=0.23,
 gamma=0.015,
 reg_alpha=0.005,
 subsample=0.5,
 colsample_bytree=0.7,
 objective='multi:softmax',
 eval_metric = "mlogloss",
 nthread=4,
 seed=27,
 num_class = 10
 )


# we need to use values.ravel to ensure the labels are sent to classifier correctly
classifier.fit(train_X[0:nSamplesTrain,:],train_Y[0:nSamplesTrain].ravel())
print(classifier)

predictions = classifier.predict(test_X[0:nSamplesTest,:])
print ("\naccuracy_score :",accuracy_score(test_Y[0:nSamplesTest],predictions))
print ("\nclassification report :\n",(classification_report(test_Y[0:nSamplesTest],predictions)))
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(test_Y[0:nSamplesTest],predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.show()


test_x_gist = scipy.io.loadmat('//media/wajih/Disk1 500 GB/Onus/RnD/Codes/gistdescriptor/mnist_gists/test_x_gist')
train_x_gist = scipy.io.loadmat('//media/wajih/Disk1 500 GB/Onus/RnD/Codes/gistdescriptor/mnist_gists/train_x_gist')


keys = train_x_gist.keys()
for key in keys:
    if key == "train_x_gist":
        train_X = train_x_gist[key].copy()
keys = test_x_gist.keys()
for key in keys:
    if key == "test_x_gist":
        test_X = test_x_gist[key].copy()


# Lets view an entry 
fig = plt.figure(figsize = (10,10))
ax = fig.gca()
plt.title("Gist plot",fontsize=10)
plt.plot(test_X[index,:])
plt.show()

from xgboost import XGBClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

del classifier

classifier =  XGBClassifier(
learning_rate =0.3,
 n_estimators=18,
 max_depth=7,
 min_child_weight=0.23,
 gamma=0.015,
 reg_alpha=0.005,
 subsample=0.5,
 colsample_bytree=0.7,
 objective='multi:softmax',
 eval_metric = "mlogloss",
 nthread=4,
 seed=27,
 num_class = 10
 )

classifier.fit(train_X[0:nSamplesTrain,:],train_Y[0:nSamplesTrain].ravel())
print(classifier)

predictions = classifier.predict(test_X[0:nSamplesTest,:])
print ("\naccuracy_score :",accuracy_score(test_Y[0:nSamplesTest],predictions))
print ("\nclassification report :\n",(classification_report(test_Y[0:nSamplesTest],predictions)))
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(test_Y[0:nSamplesTest],predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.show()
