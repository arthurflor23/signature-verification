from signet.cnn_model import CNNModel
import signet.signet_spp as signet_spp
import matplotlib.pyplot as plt    
import pdi.segmentation as segmentation
import util.path as path
import numpy as np
import sys

def sig_preprocess(img_arr):
    for (i, img) in enumerate(img_arr):
        img_arr[i] = segmentation.otsu(grayscale(img))

def grayscale(arr):
    return np.dot(arr[...,:3], [0.299, 0.587, 0.114])

def normalize(arr):
    return np.divide(arr, 255)

def bbox(arr, min_width=None):
    back = int(arr[0][0])
    img = (arr == np.logical_not(back))
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y_min, y_max = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    x_min, x_max = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))

    new = arr[y_min:y_max, x_min:x_max]
    min_padding = int(np.ceil((min_width - np.min(new.shape))/2))

    if (min_width is not None and min_padding > 0):
        return padding(arr[y_min:y_max, x_min:x_max], back=back, padding=min_padding)
    return new

def padding(c_x, back=0, padding=0):
    m, n = c_x.shape
    c_y = np.full((m+2*padding, n+2*padding), back)
    c_y[padding:-padding, padding:-padding] = c_x
    return c_y

def histogram(arr):
    h_arr = np.zeros(256, dtype=int)
    for y in range(len(arr)):
        for x in range(len(arr[0])):
            h_arr[int(arr[y,x])] += 1
    return h_arr

def extract_features(extract, img_arr):
    if (extract == "cnn"):
        model = CNNModel(signet_spp, path.model())
        return [model.get_feature_vector(img) for (i, img) in enumerate(img_arr)]
    elif (extract == "mi_hu"):

        return
    else:
        print("Extractor option doesn't exist")
        sys.exit(1)

def plot(img, cmap="Greys_r"):
    plt.imshow(img, cmap=cmap)
    plt.show()