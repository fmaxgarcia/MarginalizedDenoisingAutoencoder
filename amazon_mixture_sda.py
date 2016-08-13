from StackedMarginalizedDenoisingAutoencoder import *
import numpy as np
from load_amazon import load_data
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append("../GrassmanianDomainAdaptation/")
sys.path.append("../MixtureOfSubspaces/")
sys.path.append("../Visualization/")
sys.path.append("../SubspaceTransformation/")
from sklearn.decomposition import PCA

from GrassmanianSampling import flow
from MixtureOfSubspaces import MixtureOfSubspaces
from visualize_isomap import visualize_data
from SubspaceTransformation import LinearTransformation

CORRUPTION_LEVEL = 0.5
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 10
BATCH_SIZE = 20
DATASET = '../Datasets/amazon reviews/'

from sys import stdout
from random import shuffle


def create_projected_data(proj_x, msda):   

    return msda.get_reconstruction(proj_x)



if __name__ == '__main__':

    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
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
    X = X.toarray() / 10.0

    original_train_x = np.asarray(X[:num_samples,:])
    original_train_y = np.asarray(all_ratings)[:num_samples]

    original_test_x = np.asarray(X[num_samples:,:])
    original_test_y = np.asarray(all_ratings)[num_samples:]

    # visualize_data( np.vstack((original_train_x, original_test_x)), np.vstack((original_train_y.reshape( (original_train_y.shape[0], 1)), original_test_y.reshape( (original_test_y.shape[0], 1)))), original_train_x.shape[0], original_test_x.shape[0] )

    tranformation = LinearTransformation(original_train_x, original_test_x)
    X = tranformation.transformed_features()
    reconstruction_train = X.T[:original_train_x.shape[0]]
    reconstruction_test = X.T[original_train_x.shape[0]:]

    # pre_train = np.vstack( (original_train_x, original_test_x) )
    # smda = StackedMarginalizedDenoisingAutoencoder(num_layers=2, corruption_level=CORRUPTION_LEVEL, inputs=pre_train)
    #
    # reconstruction_train = create_projected_data(original_train_x, smda)
    # reconstruction_test = create_projected_data(original_test_x, smda)

    visualize_data( np.vstack((reconstruction_train, reconstruction_test)), np.vstack((original_train_y.reshape( (original_train_y.shape[0], 1)), original_test_y.reshape( (original_test_y.shape[0], 1)))), original_train_x.shape[0], original_test_x.shape[0] )

    dimensions = 1000
    num_outputs = np.unique(original_train_y).shape[0]
    grassmanian_subspaces = flow(reconstruction_train, reconstruction_test, t=np.linspace(0, 1, num=num_outputs*2), k=dimensions, dim_reduction="PCA")

    num_outputs = np.unique(original_train_y).shape[0]
    proj_train, proj_test = [], []
    for i, subspace in enumerate(grassmanian_subspaces):
        print "Denoising subspace #%d" %(i)
        proj_s = reconstruction_train.dot( subspace.dot(subspace.T) )
        proj_t = reconstruction_test.dot( subspace.dot(subspace.T) )

        proj_train.append( proj_s )
        proj_test.append( proj_t )
    
    print "Creating mixture of subspaces..."
    mixture_of_subspaces = MixtureOfSubspaces(num_subspaces=len(proj_train), proj_dimension=proj_train[0].shape[1],
                                              original_training=original_train_x, num_outputs=num_outputs, task_type=CLASSIFICATION)

    mixture_of_subspaces.train_mixture(X=reconstruction_train, Y=original_train_y, X_proj=proj_train)

    mixture_of_subspaces.set_best_parameters()
    predictions = mixture_of_subspaces.make_prediction(reconstruction_test, proj_test)
    correct = 0.0
    total = 0.0
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        if np.argmax(pred) == original_test_y[i]:
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
