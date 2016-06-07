import numpy as np
from MarginalizedDenoisingAutoencoder import MarginalizedDenoisingAutoencoder

from sklearn import svm

class StackedMarginalizedDenoisingAutoencoder():

    def __init__(self, num_layers, corruption_level, inputs, outputs):

        for i in range(num_layers):
            print "Creating mDA ", i
            mda = MarginalizedDenoisingAutoencoder(inputs, corruption_level)
            if i < num_layers-1:
                reconstruction = mda.get_output(inputs, nonlinearity=np.tanh)
            else:
                reconstruction = mda.get_output(inputs)

            sq_error = mda.evaluate_square_error(inputs, reconstruction)
            print "MSE %d: %f " %(i, sq_error)
            inputs = reconstruction
            self.reconstruction = reconstruction


    def train(self, inputs, outputs):
        ones = np.ones( (inputs.shape[0], 1) ) #[n_samples, n_features]
        inputs = np.hstack( (inputs, ones) )
        self.classifier = svm.SVC()
        self.classifier.fit(inputs, outputs)


    def get_output(self, inputs):
        ones = np.ones( (inputs.shape[0], 1) ) #[n_samples, n_features]
        inputs = np.hstack( (inputs, ones) )
        return self.classifier.predict( inputs )
