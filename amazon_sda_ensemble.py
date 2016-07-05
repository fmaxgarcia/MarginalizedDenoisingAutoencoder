from StackedMarginalizedDenoisingAutoencoder import StackedMarginalizedDenoisingAutoencoder
import numpy as np
from load_amazon import load_data
from sklearn.feature_extraction.text import CountVectorizer

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 15
BATCH_SIZE = 20
DATASET = '../Datasets/amazon reviews/'

from random import shuffle

import timeit

import sys

sys.path.append("../GrassmanianDomainAdaptation/")

from GrassmanianSampling import flow

from sklearn.decomposition import PCA
from sys import stdout

if __name__ == '__main__':

    vectorizer = CountVectorizer(min_df=2)
    domains = [str(sys.argv[1]), str(sys.argv[2])] #"books", "dvd", "kitchen_&_housewares", "electronics", "dvd"] 
    all_ratings, all_text = [], []
    num_samples = 0
    for domain in domains:
        print "loading domain: ", domain
        corpus, ratings = load_data(DATASET, domain)
        indices = range(len(corpus))
        shuffle(indices)
        corpus = np.asarray(corpus)[indices]
        ratings = np.asarray(ratings)[indices]
        all_ratings.extend( list(ratings) )
        all_text.extend( list(corpus) )
        if domain != str(sys.argv[2]):
            num_samples += len(indices) 


    X = vectorizer.fit_transform(all_text)
    X = X.toarray()

    train_x = np.asarray(X[:num_samples,:])
    train_y = np.asarray(all_ratings)[:num_samples]

    test_x = np.asarray(X[num_samples:,:])
    test_y = np.asarray(all_ratings)[num_samples:]


    dimensions = 1000
    grassmanian_subspaces = flow(train_x, test_x, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")
    projected_data, smdas = [], []
    for i in range(grassmanian_subspaces.shape[0]):
       A = grassmanian_subspaces[i]
       projected_train = train_x.dot( A.dot(A.T) )
       projected_test = test_x.dot( A.dot(A.T) ) 

       pre_train = np.vstack( (projected_train, projected_test) )
 
       projected_data.append( projected_test )

       smda = StackedMarginalizedDenoisingAutoencoder(num_layers=2, corruption_level=CORRUPTION_LEVEL, inputs=pre_train)
       smda.train(train_x, train_y)
       
       smdas.append( smda )    
           

    all_predictions = []
    for i, smda in enumerate(smdas):    
       print "Getting predictions for subspace ", i
       projected_test = projected_data[i] 
       predictions = smda.get_output(test_x)
       all_predictions.append( predictions )

     
    correct = 0.0
    total = 0.0
    for i in range(all_predictions[0].shape[0]):
	vote_dictionary = dict()
	for predictions in all_predictions: 
	    pred = predictions[i]
	    if pred not in vote_dictionary:
	        vote_dictionary[pred] = 1
	    else:
		vote_dictionary[pred] += 1

	final_prediction, count = 0, 0
	for pred in vote_dictionary:
	    if vote_dictionary[pred] > count:
		count = vote_dictionary[pred]
		final_prediction = pred

        if final_prediction == test_y[i]:
            correct += 1
        total += 1
	stdout.write("\rPredicting %d/%d " %(i, predictions.shape[0]))
	stdout.flush()

    print "Correct: ", correct
    print "Total: ", total
    print "Accuracy: ", (correct / total)

    with open("Results_Vote.txt", "a") as myfile:
	accuracy = correct / total
        myfile.write("%s -> %s\nAccuracy: %f \n" %(str(sys.argv[1]), str(sys.argv[2]), accuracy))
