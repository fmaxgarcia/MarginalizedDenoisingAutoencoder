from MarginalizedDenoisingAutoencoder import MarginalizedDenoisingAutoencoder
import numpy as np
from load_mnist import load_data
from utils import *
from PIL import Image

CORRUPTION_LEVEL = 0.3
LEARNING_RATE = 0.1
TRAINING_EPOCHS = 15
BATCH_SIZE = 20
DATASET = '../Datasets/mnist.pkl.gz'

import timeit

if __name__ == '__main__':

    datasets = load_data(DATASET)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / BATCH_SIZE
    train_x = train_set_x.get_value()

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################
    start_time = timeit.default_timer()
    train_x = train_x[:1000,:]
    mda = MarginalizedDenoisingAutoencoder(inputs=train_x, corruption_prob=CORRUPTION_LEVEL)
    reconstruction = mda.get_output(train_x)

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    sq_error = mda.evaluate_square_error(train_x, reconstruction)

    print reconstruction.shape
    print "Running time ", training_time
    print "MSE ", sq_error

    print("Saving original and reconstructed images")

    tiled_image = tile_raster_images(X=train_x[:100,:], img_shape=(28, 28), 
        tile_shape=(10, 10), tile_spacing=(1, 1))
    image = Image.fromarray(tiled_image)
    image.save('OriginalImage.png')

    tiled_image = tile_raster_images(X=reconstruction[:100,:], img_shape=(28, 28), 
        tile_shape=(10, 10), tile_spacing=(1, 1))
    image = Image.fromarray(tiled_image)
    image.save('ReconstructedImage.png')
