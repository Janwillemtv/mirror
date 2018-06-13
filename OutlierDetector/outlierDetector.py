print(__doc__)


# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import os.path
import pickle
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
# fetch original mnist dataset
from sklearn.datasets import fetch_mldata

# import custom module
from mnist_helpers import *

mnist = fetch_mldata('MNIST original', data_home='./')

# minist object contains: data, COL_NAMES, DESCR, target fields
# you can check it by running
mnist.keys()

# data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target

# Let's have a look at the random 16 images,
# We have to reshape each data row, from flat array of 784 int to 28x28 2D array

# pick  random indexes from 0 to size of our dataset
#show_some_digits(images, targets)

# ---------------- classification begins -----------------
# scale data for [0,255] -> [0,1]
# sample smaller size for testing
# rand_idx = np.random.choice(images.shape[0],10000)
# X_data =images[rand_idx]/255.0
# Y      = targets[rand_idx]

# full dataset classification
X_data = images / 255.0
Y = targets

digitData = {}

for digit in range(10):
    tempArray = []
    for idx,label in enumerate(Y):
        if digit == label:
            tempArray.append(X_data[idx])
    digitData[digit] = tempArray



# split data to train and test
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)




filename = "finalized_model.sav"
if os.path.isfile(filename):
    classifier = pickle.load(open(filename, 'rb'))
    print("Loaded model")
else:
    ################ Classifier with good params ###########
    #Create a classifier: a support vector classifier

    param_C = 5
    param_gamma = 0.05
    classifier = svm.SVC(C=param_C,gamma=param_gamma)

    #We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    classifier.fit(X_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))
    filename = 'finalized_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))


for digitToPredict in range(10):
    print("Predicting", digitToPredict)
    tempOutliers = []
    tempLabels = []
    X_test = digitData[digitToPredict]

    predicted = classifier.predict(X_test)
    for idx, outcome in enumerate(predicted):
        if outcome != digitToPredict: #outlier
            tempOutliers.append(X_test[idx])
            tempLabels.append(predicted[idx])
    print("Nr of outliers", len(tempOutliers))
    outliers = np.array(tempOutliers)
    labels = np.array(tempLabels)

    print("Writing files")
    outliername = "/outliers/" + str(digitToPredict)
    np.save(os.path.abspath(outliername),outliers)
    labelname = "/outliers/" + str(digitToPredict) + "_labels"
    np.save(os.path.abspath(outliername),labels)
