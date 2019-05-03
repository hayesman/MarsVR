#!/usr/bin/env python

# Eric Hayes
# January 2019

# Description:
# Program that trains an image colourizing neural network using Google Cloud

# Based on code written by Emil Wallner
# https://github.com/emilwallner/Coloring-greyscale-images

# Helpful guides:
# https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/
# https://medium.freecodecamp.org/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d

# Sample execution:

#    export BUCKET_NAME=marsvr
#    export JOB_NAME="colourizer_train_$(date +%Y%m%d_%H%M%S)"
#    export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
#    export REGION=us-east1

#    gcloud ml-engine jobs submit training $JOB_NAME \
#      --job-dir gs://$BUCKET_NAME/$JOB_NAME \
#      --runtime-version 1.0 \
#      --module-name trainer.colourizer-keras \
#      --package-path ./trainer \
#      --region $REGION \
#      --config=trainer/cloudml-gpu.yaml

from tensorflow.python.lib.io import file_io
from keras.layers import Conv2D, UpSampling2D, Input, Add, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import argparse

# Using 489 colour (Mastcam) training images

IMAGE_SIZE = 1024     # Image resolution is 1024 x 1024
EPOCHS = 300          # Number of times entire dataset is passed through network
STEPS_PER_EPOCH = 163 # Number of batches (num images/batch size)
BATCH_SIZE = 3        # Number of training images per batch

def main(job_dir,**args):

    # Reset everything to rerun in jupyter
    tf.reset_default_graph()

    # Setting up the path for saving logs
    logs_path = job_dir + '/logs/'

    # Create the model using Keras functional API
    # input is 2D tensor of L* (lightness) grayscale values
    input_img = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    conv_1_downsample = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(conv_1)
    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_1_downsample)
    conv_2_downsample = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(conv_2)
    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_2_downsample)
    conv_3_downsample = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(conv_3)
    conv_4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_3_downsample)
    conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_4)
    conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_5)
    conv_6_upsample = UpSampling2D((2, 2))(conv_6)
    conv_7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_6_upsample)
    conv_7_upsample = UpSampling2D((2, 2))(conv_7)
    conv_8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_7_upsample)
    conv_9 = Conv2D(2, (3, 3), activation='tanh', padding='same')(conv_8)
    conv_9_upsample = UpSampling2D((2, 2))(conv_9)
    model = Model(input_img, conv_9_upsample)

    print "*** Compile Model ***"
    model.compile(optimizer='rmsprop', loss='mse')

    print "*** Get Training Images: ***"

    # Training image URIs provided in a text file on Google Cloud
    image_paths_file = 'gs://mars-vr-3/image_URIs.txt'
    input_file = file_io.FileIO(image_paths_file, mode='r')
    contents =  input_file.read()
    paths = contents.split('\n')

    # Read training image data and load it into NumPy array
    X = []
    for filepath in paths:
        if filepath != "":
            file = file_io.FileIO(filepath, mode='r').read()
            X.append(np.array(tf.image.decode_jpeg(file).eval(session=tf.Session())))
            print filepath
    X = np.array(X, dtype=float)

    # Scale RGB train data to be -1.0 to 1.0
    split = int(0.95*len(X))
    Xtrain = X[:split]
    Xtrain = 1.0/255*Xtrain

    datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True)

    # Generator function that yields training data (inputs, targets) in lab colourspace
    def image_a_b_gen(batch_size):
        for batch in datagen.flow(Xtrain, batch_size=batch_size):
            lab_batch = rgb2lab(batch)

            # select grayscale layer
            X_batch = lab_batch[:,:,:,0]

            # select the a (green-red) and b (blue-yellow) layers
            Y_batch = lab_batch[:,:,:,1:] / 128

            # yield a tuple of (inputs, targets)
            yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
 
    # Print the model summary
    model.summary()

    # Add the callback for TensorBoard and History
    tensorboard = TensorBoard(
                log_dir=logs_path, 
                histogram_freq=0, 
                write_graph=True, 
                write_images=True)

    print "*** Train Model ***\n"

    # Train the model with TensorFlow
    model.fit_generator(
        image_a_b_gen(BATCH_SIZE), 
        callbacks=[tensorboard], 
        epochs=EPOCHS, 
        steps_per_epoch=STEPS_PER_EPOCH)

    # Save the model weights
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    
    # Print the loss value and metrics values for the model in test mode
    Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
    Ytest = Ytest / 128
    print(model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE))

    # Save model.h5 to specified job directory
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

    print "*** Model Saved as model.h5 ***\n"


# Run the training app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
