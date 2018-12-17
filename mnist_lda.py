#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:56:39 2018

@author: wajih
"""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import numpy as np
from timeit import default_timer as timer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
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
Model=LinearDiscriminantAnalysis()

nSamples = 1.0 #Percentage
nSamplesTrain = np.int(np.round(train_X.shape[0]*nSamples))
nSamplesTest = np.int(np.round(test_X.shape[0]*nSamples))
start = timer()
# we need to use values.ravel to ensure the labels are sent to classifier correctly
Model.fit(train_X[0:nSamplesTrain,:],train_Y[0:nSamplesTrain].ravel())
end = timer()
print(Model)
print("Time taken to fit:",end-start)

start = timer()
predictions = Model.predict(test_X[0:nSamplesTest,:])
end = timer()
print("Time taken to predict:",end-start)

print ("\naccuracy_score :",accuracy_score(test_Y[0:nSamplesTest],predictions))
print ("\nclassification report :\n",(classification_report(test_Y[0:nSamplesTest],predictions)))
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(test_Y[0:nSamplesTest],predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.show()

test_x_gist = scipy.io.loadmat('//media/wajih/Disk1 500 GB/Onus/RnD/Codes/gistdescriptor/mnist_gists/test_x_gist')
train_x_gist = scipy.io.loadmat('//media/wajih/Disk1 500 GB/Onus/RnD/Codes/gistdescriptor/mnist_gists/train_x_gist')

del Model
Model = LinearDiscriminantAnalysis()
keys = train_x_gist.keys()
for key in keys:
    if key == "train_x_gist":
        train_X = train_x_gist[key].copy()
keys = test_x_gist.keys()
for key in keys:
    if key == "test_x_gist":
        test_X = test_x_gist[key].copy()


start = timer()
# we need to use values.ravel to ensure the labels are sent to classifier correctly
Model.fit(train_X[0:nSamplesTrain,:],train_Y[0:nSamplesTrain].ravel())
end = timer()
print(Model)
print("Time taken to fit:",end-start)

start = timer()
predictions = Model.predict(test_X[0:nSamplesTest,:])
end = timer()
print("Time taken to predict:",end-start)

print ("\naccuracy_score :",accuracy_score(test_Y[0:nSamplesTest],predictions))
print ("\nclassification report :\n",(classification_report(test_Y[0:nSamplesTest],predictions)))
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(test_Y[0:nSamplesTest],predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.show()