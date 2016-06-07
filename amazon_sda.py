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

if __name__ == '__main__':

    vectorizer = CountVectorizer(min_df=2)
    domains = ["books", "kitchen_&_housewares", "electronics", "dvd"] 
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
        if domain != "dvd":
            num_samples += len(indices) 


    X = vectorizer.fit_transform(all_text)
    X = X.toarray()

    train_x = np.asarray(X[:num_samples,:])
    train_y = np.asarray(all_ratings)[:num_samples]

    test_x = np.asarray(X[num_samples:,:])
    test_y = np.asarray(all_ratings)[num_samples:]
    
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################
    start_time = timeit.default_timer()
    train_x = train_x[:1000,:]
    train_y = train_y[:1000]
    smda = StackedMarginalizedDenoisingAutoencoder(num_layers=2, corruption_level=CORRUPTION_LEVEL, inputs=train_x, outputs=train_y)

    smda.train(train_x, train_y)    
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    
    print "Running time ", training_time
    

    predictions = smda.get_output(test_x)
    correct = 0.0
    total = 0.0
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        if pred == test_y[i]:
            correct += 1
        total += 1

    print "Correct: ", correct
    print "Total: ", total
    print "Accuracy: ", (correct / total)
