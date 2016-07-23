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

grassmannian_sampling = True
from sklearn.decomposition import PCA
from sys import stdout


def create_projected_data(proj_x, msda):   

    return msda.get_reconstruction(proj_x)



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

    original_train_x = np.asarray(X[:num_samples,:])
    original_train_y = np.asarray(all_ratings)[:num_samples]

    original_test_x = np.asarray(X[num_samples:,:])
    original_test_y = np.asarray(all_ratings)[num_samples:]

    dimensions = 1000
    grassmanian_subspaces = flow(original_train_x, original_test_x, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")
    
    proj_train, proj_test = [], []
    for i, subspace in enumerate(grassmanian_subspaces):
        print "Denoising subspace #%d" %(i)
        proj_s = original_train_x.dot( subspace.dot(subspace.T) )
        proj_t = original_test_x.dot( subspace.dot(subspace.T) )
        pre_train = np.vstack( (proj_s, proj_t) )
        
        smda = StackedMarginalizedDenoisingAutoencoder(num_layers=2, corruption_level=CORRUPTION_LEVEL, inputs=pre_train)
        
        reconstruction_train = create_projected_data(proj_s, msda)
        reconstruction_test = create_projected_data(proj_t, msda)

        proj_train.append( reconstruction_train )
        proj_test.append( reconstruction_test )

    
    print "Creating mixture of subspaces..."
    mixture_of_subspaces = MixtureOfSubspaces(num_subspaces=len(proj_train), proj_dimension=proj_train[0].shape[1], original_dimensions=original_train_x.shape[1])

    mixture_of_subspaces.train_mixture(X=original_train_x, Y=original_train_y, X_proj=proj_train)

    predictions = mixture_of_subspaces.make_prediction(original_test_x, proj_test)
    correct = 0.0
    total = 0.0
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

    with open("Results_M_G.txt", "a") as myfile:
        accuracy = correct / total
        myfile.write("%s -> %s\nAccuracy: %f \n" %(str(sys.argv[1]), str(sys.argv[2]), accuracy))
