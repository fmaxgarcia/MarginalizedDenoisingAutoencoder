import numpy as np

class MarginalizedDenoisingAutoencoder():

    def __init__(self, inputs, corruption_prob):
        inputs = inputs.T
        ones = np.ones( (1, inputs.shape[1]) )        
        inputs = np.vstack( (inputs, ones) )

        d = inputs.shape[0]
        q = np.ones( (d, 1)) * (1 - corruption_prob)        
        q[ q.shape[0]-1 ] = 1.0
        
        S = inputs.dot(inputs.T)
        E_Q = np.zeros( S.shape )
        E_P = np.zeros( S.shape )
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if i == j:
                    E_Q[i,j] = S[i,j] * q[i]
                else:
                    E_Q[i,j] = S[i,j] * q[i] * q[j]
                
                E_P[i,j] = S[i,j] * q[j]

        E_Q += (np.eye( E_Q.shape[0] ) + 1e-5)
        self.W = E_P.dot( np.linalg.inv(E_Q) )

    def get_output(self, inputs, nonlinearity=None):
        inputs = inputs.T
        ones = np.ones( (1, inputs.shape[1]) )
        inputs = np.vstack( (inputs, ones) )

        output = self.W.dot(inputs).T
        output = np.delete(output, output.shape[1]-1, 1)
        if nonlinearity is not None:
            output = nonlinearity(output)
        return output

    def evaluate_square_error(self, inputs, reconstruction):
        error = np.mean( (inputs - reconstruction)**2 )
        return error


