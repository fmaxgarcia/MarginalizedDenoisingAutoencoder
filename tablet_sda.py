import numpy as np
from StackedDenoisingAutoencoder import StackedDenoisingAutoencoder
import sys
sys.path.append("../GrassmanianDomainAdaptation/")

from GrassmanianSampling import flow

grassmannian_sampling = False
from sklearn.decomposition import PCA


DATASET = "../Datasets/Mars/tablet/"
CORRUPTION_LEVEL = 0.5

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    input1 = np.load(DATASET+"instrument1.npy")
    input2 = np.load(DATASET+"instrument2.npy")
    outputs = np.load(DATASET+"labels.npy")

    if grassmannian_sampling:
            dimensions = 400
            grassmanian_subspaces = flow(input1, input2, t=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), k=dimensions, dim_reduction="PCA")
            pre_train, train_x, test_x = None, None, None
            for i in range(grassmanian_subspaces.shape[0]):
               A = grassmanian_subspaces[i]
               if pre_train == None:
                    train_x = input1.dot( A.dot(A.T) )
                    test_x = input2.dot( A.dot(A.T) )
                    pre_train = np.vstack( (train_x, test_x) )
                    train_y = outputs
                    test_y = outputs
               else:
                    train = input1.dot( A.dot(A.T) )
                    test = input2.dot( A.dot(A.T) )

                    ###Extend training and testing with projected data
                    train_x = np.vstack( (train_x, train) )
                    train_y = np.vstack( (train_y, outputs) )

                    test_x = np.vstack( (test_x, test) )
                    test_y = np.vstack( (test_y, outputs) )

                    ###Extend pre-train with projected training and testing data
                    pre_train = np.vstack( (pre_train, train) )
                    pre_train = np.vstack( (pre_train, test) )
    else:
        pca = PCA(n_components=600)
        combined_inputs = np.vstack( (input1, input2) )
        pca.fit( combined_inputs )
        combined_inputs = pca.transform( combined_inputs )
        train_x = combined_inputs[:input1.shape[0]]
        train_y = outputs
        test_x = combined_inputs[input1.shape[0]:]
        test_y = outputs

        # train_x = input1
        # train_y = outputs
        # test_x = input2
        # test_y = outputs

        dimensions = train_x.shape[1]
        pre_train = np.vstack( (train_x, test_x) )

    out_max = np.max( np.vstack((train_y, test_y) ), axis=0 )
    out_mins = np.min( np.vstack((train_y, test_y) ), axis=0 )

    ############## Normalize outputs for comparison ################
    # train_y =  (train_y - out_mins) / (out_max - out_mins)
    # test_y =  (test_y - out_mins) / (out_max - out_mins)
    # print "Normalized ", normalized_outputs
    # print "Recovered ", (normalized_outputs * (out_max - out_mins)) + out_mins
    

    print "Training autoencoder..."
    smda = StackedMarginalizedDenoisingAutoencoder(num_layers=2, corruption_level=CORRUPTION_LEVEL, inputs=pre_train, prediction_task=1)
    smda.train(train_x, train_y)    

    print "Making predictions..."
    print "Test x shape ", test_x.shape
    print "Test y shape ", test_y.shape
    predictions = smda.get_output(test_x)
    test_y = test_y[:predictions.shape[0]] #Predicting in batches has to match batch size

    errors = (test_y - predictions)**2
    mean_error = np.sum(errors, axis=0) / errors.shape[0]
    print "Mean Error is ", mean_error
    print "Sum of mean errors ", np.sum(mean_error)

    xs = np.linspace(0, test_y.shape[0]-1, num=test_y.shape[0])
    for i in range(test_y.shape[1]):
        plt.plot(xs, test_y[:,i], "r", xs, predictions[:,i], "b")
        plt.show()