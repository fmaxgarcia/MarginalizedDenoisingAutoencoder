from StackedMarginalizedDenoisingAutoencoder import StackedMarginalizedDenoisingAutoencoder
import numpy as np
from load_mnist import load_data

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 15
BATCH_SIZE = 20
DATASET = '../Datasets/mnist.pkl.gz'

import timeit

if __name__ == '__main__':

    datasets = load_data(DATASET)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE

    train_x = train_set_x.get_value()
    train_y = train_set_y.get_value()

    test_x = test_set_x.get_value()
    test_y = test_set_y.get_value()
    
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################
    start_time = timeit.default_timer()
    train_x = train_x[:10000,:]
    train_y = train_y[:10000]
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
