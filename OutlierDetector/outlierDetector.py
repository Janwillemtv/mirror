print(__doc__)


# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import os.path
from os.path import dirname, abspath

import pickle
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
# fetch original mnist dataset
from sklearn.datasets import fetch_mldata

# import custom module
from mnist_helpers import *
from sklearn.model_selection import train_test_split



#load data
mnist = fetch_mldata('MNIST original', data_home='./')

# minist object contains: data, COL_NAMES, DESCR, target fields
# you can check it by running
mnist.keys()

# data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target


# full dataset classification
X_data = images / 255.0
Y = targets


# split data to train and test
# from sklearn.cross_validation import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.5, random_state=42)


digitData = {}

for digit in range(10):
    tempArray = []
    for idx,label in enumerate(y_test):
        if digit == label:
            tempArray.append(X_test[idx])
    digitData[digit] = tempArray

def load_model(model):
    filepath = os.path.join("models", model)
    if os.path.isfile(filepath):
        classifier = pickle.load(open(filepath, 'rb'))
        print("Loaded model")
    else:
        ################ Classifier with good params ###########
        #Create a classifier: a support vector classifier
        print("Model not found, training new one")
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

        pickle.dump(classifier, open(os.path.join("models",model), 'wb'))
    return classifier


def predict(classifier, digitToPredict):

    print("Predicting", digitToPredict)
    X_test = digitData[digitToPredict]

    predicted = classifier.predict(X_test)
    return predicted


def predict_and_save(classifier):
    for digitToPredict in range(10):
        print("Predicting", digitToPredict)
        tempOutliers = []
        tempGoodDigits = []
        tempLabels = []
        X_test = digitData[digitToPredict]

        predicted = classifier.predict(X_test)
        for idx, outcome in enumerate(predicted):
            if outcome != digitToPredict: #outlier
                tempOutliers.append(X_test[idx])
                tempLabels.append(predicted[idx])
            else:
                tempGoodDigits.append(X_test[idx])
        print("Nr of outliers", len(tempOutliers))
        outliers = np.array(tempOutliers)
        labels = np.array(tempLabels)
        goodDigits = np.array(tempGoodDigits)

        print("Writing files")

        np.save(os.path.join('outliers', str(digitToPredict)),outliers)
        labelname = str(digitToPredict) + "_labels"
        np.save(os.path.join('outliers', labelname),labels)
        np.save(os.path.join('goodDigits',str(digitToPredict)),goodDigits)



############SELECT MODEL TO USE HERE##########
## finalized_model.sav    this works best (doesnt create so much outliers)
## finalized_model_bad    this creates more outliers (worse model)
model = "finalized_model_bad.sav"

classifier = load_model(model)
predict_and_save(classifier)










