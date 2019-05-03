#!/usr/bin/env python

# Eric Hayes - Image colourizing model

# Based on code written by Emil Wallner
# https://github.com/emilwallner/Coloring-greyscale-images

# Sample execution (saves images to ./results)
# img_colourizer.py -i image_directory -m model_weights

from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Add, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer, Input
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


IMAGE_SIZE = 1024   # Image resolution is 1024 x 1024
EXTENSION = ".png"  # Output image format


def main(args):

    # Alert the user if args are missing
    if not args.model:
        raise ValueError("must provide a trained model")
    if not args.input:
        raise ValueError("no photos to colourize")

    print "*** Create Model ***"

    # Create a model that is the same as the model used in training
    # (model could also be saved and loaded from a file)
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

    print "*** Load Model Weights ***"

    # Load the weights from the trained model
    model.load_weights(args.model)

    print "*** Load Images to Colourize ***"

    # Read in images to colourize, convert to lab colourspace
    color_me = []
    for filename in os.listdir(args.input):
        color_me.append(img_to_array(load_img(os.path.join(args.input, filename))))
    color_me = np.array(color_me, dtype=float)

    # Select grayscale layer
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))

    print "*** Predict Colours With Model ***"

    # Predict colours
    output = model.predict(color_me)
    output = output * 128
    
    print "*** Colourized Photos: ***"

    # Create output dir if it doesn't exist
    if not os.path.exists("result"):
        os.makedirs("result")

    # Output colourized images
    for i in range(len(output)):
        cur = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave("result/img_"+str(i)+EXTENSION, lab2rgb(cur))
        print "img_"+str(i)+EXTENSION


# Run the colourizer app
if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='images to colourize')
    parser.add_argument('-m', '--model', type=str, help='weights of trained model')
    args = parser.parse_args()

    main(args)
