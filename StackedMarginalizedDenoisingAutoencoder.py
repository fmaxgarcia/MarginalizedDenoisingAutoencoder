import numpy as np
from MarginalizedDenoisingAutoencoder import MarginalizedDenoisingAutoencoder

class StackedMarginalizedDenoisingAutoencoder():

    def __init__(self, num_layers, corruption_level, inputs):

        for i in range(num_layers):
            mda = MarginalizedDenoisingAutoencoder(inputs, corruption_level)
            if i < num_layers-1:
                reconstruction = mda.get_output(inputs, nonlinearity=np.tanh)
            else:
                reconstruction = mda.get_output(inputs)

            sq_error = mda.evaluate_square_error(inputs, reconstruction)
            print "MSE %d: %f " %(i, sq_error)
            inputs = reconstruction
            self.reconstruction = reconstruction


