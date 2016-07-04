from StackedMarginalizedDenoisingAutoencoder import StackedMarginalizedDenoisingAutoencoder
import numpy as np
from load_mnist import load_data

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 15
BATCH_SIZE = 20
DATASET = '../Datasets/mnist.pkl.gz'

import sys
sys.path.append("../GrassmanianDomainAdaptation/")
from sklearn.decomposition import PCA

from GrassmanianSampling import flow

import timeit
from sys import stdout
grassmannian_sampling = True

if __name__ == '__main__':

    datasets = load_data(DATASET)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE

    original_train_x = train_set_x.get_value()
    original_train_y = train_set_y.get_value()

    original_test_x = test_set_x.get_value()
    original_test_y = test_set_y.get_value()
    
    if grassmannian_sampling:
        dimensions = 20*20
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
		print train_x.shape
		print train_y.shape
		train_x = np.vstack( (train_x, train) )
		train_y = np.hstack( (train_y, original_train_y) )
		print train_x.shape
		print train_y.shape		

		test_x = np.vstack( (test_x, test) )
		test_y = np.hstack( (test_y, original_test_y) ) 	   	

		###Extend pre-train with projected training and testing data
		pre_train = np.vstack( (pre_train, train) )
		pre_train = np.vstack( (pre_train, test) )
		
    else:
	train_x = original_train_x
	test_x = original_test_x
	train_y = original_train_y
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
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        if pred == test_y[i]:
            correct += 1
        total += 1
	stdout.write("%d/%d" %(i, predictions.shape[0]))
	stdout.flush()

    print "Correct: ", correct
    print "Total: ", total
    print "Accuracy: ", (correct / total)
