# MNIST_GIST
## Experinmenting with GIST feartures of MNIST dataset using XGBOOST

This is a python code repo that experiments with the image descriptor GIST on MNIST dataset for classification.
This descriptors has the ability to wholisttically capture image information that can be useful for classification. 
Originally written by Torallaba in 2001, the source code for MATLAB Gist can be found [here](http://people.csail.mit.edu/torralba/code/spatialenvelope/) a python implementation [here](https://pypi.org/project/pyleargist/)

Please note that I did not try the python version.

The important aspect of Gist is the features extracted at mutliple scales on N channels of an image. In this test I have used 
three scales. In Matlab you can set the scales and image size as I did as follows

## GIST parameters
param.imageSize = [28 28]; % it works also with non-square images
param.orientationsPerScale = [8 4 2];
param.numberBlocks = 4;
param.fc_prefilt = 4;

With some prefiltering parameters as show in the last line.

## Dataset
You will need the GIST source code to extract the image features on your own. I have already done the lengthy work and the Gist 
features can be downloaed at the following link
https://drive.google.com/drive/folders/1BGHB7GRmNMTtvSvgyNQso4uFb2VeO5sY

You can download the original MNIST dataset from http://yann.lecun.com/exdb/mnist/

Once you have the dataset correctly downloaded, ensure you have the correct paths to the dataset.

The code runs on original MNIST dataset and then follows on gist features of the mnist dataet.
## Results
You can see that accuracy of classifcation using XGBOOST, both on MNIST (0.9612) and MNIST_GISTS (0.9748) with same parameters.
Both are highly accurate, and that is because of XGBOOST. The gist featuers being wholistic are providing more accurate results. 

## Conclusion
On comparing the two features, its interesting to note that MNSIT images have 784 features while gist features stand at 256 features
per image. This very useful in terms of machine learning where reduction of features and increase in accuracy is a highly lucrative 
aspect.
