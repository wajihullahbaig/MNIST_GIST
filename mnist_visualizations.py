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

nSamples = 0.025#Percentage
nSamplesTrain = np.int(np.round(train_X.shape[0]*nSamples))
nSamplesTest = np.int(np.round(test_X.shape[0]*nSamples))

from sklearn.manifold import Isomap
model = Isomap(n_components=2)
proj = model.fit_transform(train_X[0:nSamplesTrain,:])

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
plt.title("ISOMAP MNIST RAW",fontsize=10)
plt.scatter(proj[:, 0], proj[:, 1], c=train_Y[0:nSamplesTrain], cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
plt.show()

from sklearn.manifold import TSNE
model = TSNE(n_components=2,init='pca',random_state = 0)
proj = model.fit_transform(train_X[0:nSamplesTrain,:])

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
plt.title("TSNE MNIST RAW",fontsize=10)
plt.scatter(proj[:, 0], proj[:, 1], c=train_Y[0:nSamplesTrain], cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
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

del model
model = Isomap(n_components=2)
proj = model.fit_transform(train_X[0:nSamplesTrain,:])

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
plt.title("ISOMAP MNIST GIST",fontsize=10)
plt.scatter(proj[:, 0], proj[:, 1], c=train_Y[0:nSamplesTrain], cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
plt.show()

del model
model = TSNE(n_components=2,init='pca',random_state = 0)
proj = model.fit_transform(train_X[0:nSamplesTrain,:])

fig = plt.figure(figsize = (10,10))
ax = fig.gca()
plt.title("TSNE MNIST GIST",fontsize=10)
plt.scatter(proj[:, 0], proj[:, 1], c=train_Y[0:nSamplesTrain], cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5);
plt.show()