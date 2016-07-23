import numpy as np
from MarginalizedDenoisingAutoencoder import MarginalizedDenoisingAutoencoder

import theano
import theano.tensor as T
import lasagne

from sklearn import svm

import sys
sys.path.append("../mixture_of_experts/")

CLASSIFICATION = 0
REGRESSION = 1


class StackedMarginalizedDenoisingAutoencoder():

    def __init__(self, num_layers, corruption_level, inputs, prediction_task=0):
        if prediction_task not in [CLASSIFICATION, REGRESSION]:
            raise ValueError("Invalid Argument. prediction_task must be 0 or 1")

        self.prediction_task = CLASSIFICATION if prediction_task == CLASSIFICATION else REGRESSION

        self.layers = []
        for i in range(num_layers):
            print "Creating mDA ", i
            mda = MarginalizedDenoisingAutoencoder(inputs, corruption_level)
            self.layers.append( mda )
            if i < num_layers-1:
                reconstruction = mda.get_output(inputs, nonlinearity=np.tanh)
            else:
                reconstruction = mda.get_output(inputs)

            sq_error = mda.evaluate_square_error(inputs, reconstruction)
            print "MSE %d: %f " %(i, sq_error)
            inputs = reconstruction
            self.reconstruction = reconstruction

    def _build_network(self, inputs, outputs, batch_size, learning_rate):
        n_hidden = int(inputs.shape[1] / 1.5)        
        self.input_layer = lasagne.layers.InputLayer(shape=(batch_size, inputs.shape[1]))
        self.hidden_layer = lasagne.layers.DenseLayer(self.input_layer, num_units=n_hidden, 
                        nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.Uniform(), b=lasagne.init.Uniform())
        self.output_layer = lasagne.layers.DenseLayer(self.hidden_layer, num_units=outputs.shape[1],
                        nonlinearity=None, W=lasagne.init.Uniform(), b=lasagne.init.Uniform())

        network_inputs = T.matrix('input')
        target_outputs = T.matrix('output')
        self.inputs_shared = theano.shared( np.zeros((batch_size, inputs.shape[1]), dtype=theano.config.floatX))
        self.outputs_shared = theano.shared( np.zeros((batch_size, outputs.shape[1]), dtype=theano.config.floatX))

        
        self.network_outputs = lasagne.layers.get_output(self.output_layer, {self.input_layer : network_inputs})

        error = T.sum( T.mean((target_outputs - self.network_outputs)**2, axis=0) )
        self.all_parameters = lasagne.layers.helper.get_all_params(self.output_layer)


        updates = lasagne.updates.momentum(error, self.all_parameters, learning_rate)
        self._train = theano.function([], [error], updates=updates, givens={network_inputs:self.inputs_shared, target_outputs:self.outputs_shared})
        self._get_output = theano.function([], [self.network_outputs], givens={network_inputs:self.inputs_shared})

    def train(self, inputs, outputs):
        for i, layer in enumerate(self.layers):
            inputs = layer.get_output(inputs) if i == len(self.layers)-1 else layer.get_output(inputs, nonlinearity=np.tanh)

        ones = np.ones( (inputs.shape[0], 1) ) #[n_samples, n_features]
        inputs = np.hstack( (inputs, ones) )    
        if self.prediction_task == CLASSIFICATION:
            self.classifier = svm.SVC()
            self.classifier.fit(inputs, outputs)
        else:
            inputs = np.asarray(inputs, dtype=theano.config.floatX)
            outputs = np.asarray(outputs, dtype=theano.config.floatX)
            self.batch_size = 32
            training_epochs = 1000
            self._build_network(inputs, outputs, self.batch_size, learning_rate=0.01)
            n_batches = inputs.shape[0] / self.batch_size
            for epoch in range(training_epochs):
                error = 0.0
                for i in range(n_batches):
                    batch_input = inputs[i*self.batch_size : (i+1)*self.batch_size]
                    batch_output = outputs[i*self.batch_size : (i+1)*self.batch_size]
                    if len(batch_output.shape) == 1:
                        batch_output = batch_output.reshape( (self.batch_size, 1) )

                    if batch_input.shape[0] == self.batch_size:
                        self.inputs_shared.set_value( batch_input )
                        self.outputs_shared.set_value( batch_output )
                        error += self._train()[0]

                print "Training epoch %d: error %f" %(epoch, error/n_batches)
            

    def get_reconstruction(self, inputs):
        for layer in self.layers:
            if layer is self.layers[-1]:
                reconstruction = layer.get_output(inputs)
            else:
                inputs = layer.get_output(inputs, nonlinearity=np.tanh)

        return reconstruction

    def get_output(self, inputs):
        ones = np.ones( (inputs.shape[0], 1) ) #[n_samples, n_features]
        inputs = np.hstack( (inputs, ones) )    
        if self.prediction_task == CLASSIFICATION:        
            return self.classifier.predict( inputs )
        else:
            inputs = np.asarray(inputs, dtype=theano.config.floatX)
            n_batches = inputs.shape[0] / self.batch_size
            prediction = None
            for i in range(n_batches):
                batch_input = inputs[i*self.batch_size : (i+1)*self.batch_size]
                if batch_input.shape[0] == self.batch_size:
                    self.inputs_shared.set_value( batch_input )
                    if prediction == None:
                        prediction = self._get_output()[0]
                    else:
                        prediction = np.vstack( (prediction, self._get_output()[0]) )
                        

            return np.asarray(prediction)
