# NN_predict.py
# Yu Fu, Tianyang Jin, Sheng Chen
#
# Given a new logo image
# Apply segmentation to extract logo part,
# convert to greyscale,
# extract HOG features,
# use trained neural network to predict type.
#
# Call predict(theta, X_new) to give y_new.
#

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import Base
from skimage import io, transform, morphology
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import rgb2grey, label2rgb
from skimage.util import pad
import imageProcess


def predictImage(img_path, theta_path):
    brands = Base.logos
    imageProcess.processOneImage(img_path, './temp.jpg')
    image = io.imread('./temp.jpg')
    image = transform.resize(image, (400, 400))

    features = np.array([hog(image, orientations=8, pixels_per_cell=(20, 20), cells_per_block=(1, 1))])
    thetas = np.transpose(np.load(theta_path))
    os.remove('./temp.jpg')

    prediction = predict(thetas, features)
    return brands[prediction]


def predict(final_t, X):
    INPUT_LAYER_SIZE = Base.INPUT_LAYER_SIZE
    HIDDEN_LAYER_SIZE = Base.HIDDEN_LAYER_SIZE
    OUTPUT_LAYER_SIZE = Base.OUTPUT_LAYER_SIZE
    theta1 = np.reshape(final_t[0:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)],
                        (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE + 1), order='F')
    theta2 = np.reshape(final_t[HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1):],
                    (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE + 1), order='F')
    m = np.size(X, 0)
    p = np.zeros((m, 1))

    h1 = sigmoid(np.c_[np.ones((m, 1)), X].dot(np.transpose(theta1)))
    h2 = sigmoid(np.c_[np.ones((m, 1)), h1].dot(np.transpose(theta2)))

    p = np.amax(h2, 1)
    dummy = np.argmax(h2, 1)
    return dummy

def sigmoid(z):
    return np.divide(1.0, 1.0 + np.exp(-1 * z))
