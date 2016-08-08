from StackedMarginalizedDenoisingAutoencoder import StackedMarginalizedDenoisingAutoencoder
import numpy as np
from load_amazon import load_data
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from scipy import io
import os

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 15
BATCH_SIZE = 20
DATASET = '../Datasets/office images/'

from random import shuffle

import timeit

import sys

sys.path.append("../GrassmanianDomainAdaptation/")

from GrassmanianSampling import flow

grassmannian_sampling = True
from sklearn.decomposition import PCA
from sys import stdout

if __name__ == '__main__':

    domains = os.listdir(DATASET+"amazon/interest_points/")
    original_train_x, original_test_x = [], []
    original_train_y, original_test_y = [], []
    for i, domain in enumerate(domains):
        print "Loading domain ", domain
        directory = DATASET+"amazon/interest_points/"+domain
        for f in os.listdir(directory):
            matfile = scipy.io.loadmat(directory+"/"+f)
            histogram = matfile['histogram']
            original_train_x.append(histogram[0])
            original_train_y.append(i)

        directory = DATASET+"webcam/interest_points/"+domain
        for f in os.listdir(directory):
            matfile = scipy.io.loadmat(directory+"/"+f)
            histogram = matfile['histogram']
            original_test_x.append(histogram[0])
            original_test_y.append(i)

    original_train_x = np.asarray(original_train_x)
    original_train_y = np.asarray(original_train_y)
    original_test_x = np.asarray(original_test_x)
    original_test_y = np.asarray(original_test_y)

    if grassmannian_sampling:
        dimensions = 600
        grassmanian_subspaces = flow(original_train_x, original_test_x, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")
        pre_train, train_x, test_x = None, None, None
        for i in range(grassmanian_subspaces.shape[0]):
           A = grassmanian_subspaces[i]
           if pre_train == None:
                train_x = original_train_x.dot( A.dot(A.T) )
                test_x = original_test_x.dot( A.dot(A.T) )
                pre_train = np.vstack( (train_x, test_x) )
                train_y = original_train_y
                test_y = original_test_y
           else:
                train = original_train_x.dot( A.dot(A.T) )
                test = original_test_x.dot( A.dot(A.T) )

                ###Extend training and testing with projected data
                train_x = np.vstack( (train_x, train) )
                train_y = np.hstack( (train_y, original_train_y) )

                test_x = np.vstack( (test_x, test) )
                test_y = np.hstack( (test_y, original_test_y) )

                ###Extend pre-train with projected training and testing data
                pre_train = np.vstack( (pre_train, train) )
                pre_train = np.vstack( (pre_train, test) )
    else:
        train_x = original_train_x
        train_y = original_train_y
        test_x = original_test_x
        test_y = original_test_y

        dimensions = train_x.shape[1]
        pre_train = np.vstack( (train_x, test_x) )

    
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################
    start_time = timeit.default_timer()
    smda = StackedMarginalizedDenoisingAutoencoder(num_layers=2, corruption_level=CORRUPTION_LEVEL, inputs=pre_train)

    smda.train(train_x, train_y)    
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    
    print "Running time ", training_time
    

    predictions = smda.get_output(test_x)
    correct = 0.0
    total = 0.0
    print "Shape pred ", predictions.shape
    print "Test y ", test_y.shape
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        if pred == test_y[i]:
            correct += 1
        total += 1
    stdout.write("\rPredicting %d/%d " %(i, predictions.shape[0]))
    stdout.flush()

    print "Correct: ", correct
    print "Total: ", total
    print "Accuracy: ", (correct / total)

    with open("Results_Office_G.txt", "a") as myfile:
    	accuracy = correct / total
        myfile.write("%s -> %s\nAccuracy: %f \n" %("amazon", "webcam", accuracy))
