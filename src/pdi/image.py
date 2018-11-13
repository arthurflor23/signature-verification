from multiprocessing import Pool
from signet.cnn_model import CNNModel
import signet.signet_spp as signet_spp
import matplotlib.pyplot as plt    
import pdi.segmentation as segmentation
import util.path as path
import numpy as np
import sys

def preprocess(img_arr):
    return __pool_process__(__pp__, img_arr)

def __pp__(img):
    return bbox(segmentation.otsu(grayscale(img)))

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

    if (min_width is not None):
        min_padding = int(np.ceil((min_width - np.min(new.shape))/2))
        if (min_padding > 0): return padding(arr[y_min:y_max, x_min:x_max], back=back, padding=min_padding)
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
        return [model.get_feature_vector(bbox(img, 128)) for (i, img) in enumerate(img_arr)]
    elif (extract == "mi_hu"):
        return __pool_process__(huMoments, img_arr)
    else:
        print("Extractor option doesn't exist")
        sys.exit(1)

def huMoments(arr):
    n00 = centralMoment(arr, 0, 0)
    n11 = centralMoment(arr, 1, 1) / (n00 ** 2)
    n12 = centralMoment(arr, 1, 2) / (n00 ** 2.5)
    n21 = centralMoment(arr, 2, 1) / (n00 ** 2.5)
    n02 = centralMoment(arr, 0, 2) / (n00 ** 2)
    n03 = centralMoment(arr, 0, 3) / (n00 ** 2.5)
    n20 = centralMoment(arr, 2, 0) / (n00 ** 2)
    n30 = centralMoment(arr, 3, 0) / (n00 ** 2.5)
    
    i1 = (n20 + n02)
    i2 = (n20 - n02)**2 + 4*((n11)**2)
    i3 = (n30 - (3*n12))**2 + ((3*n21) - n03)**2
    i4 = (n30 + n12)**2 + (n21 - n03)**2
    i5 = (n30 - (3*n12))*(n30 + n12)*((n30+n12)**2 - 3*((n21+n03)**2)) + ((3*n21) - n03)*(n21 + n03)*(3*((n30 + n12)**2) - (n21 + n03)**2)
    i6 = (n20 - n02)*( ((n30+n12)**2) - (n21 + n03)**2 ) + 4*n11*(n30 + n12)*(n21 + n03)
    i7 = ((3*n21) - n03)*(n30 + n12)*(((n30 + n12)**2) - 3*((n21 + n03)**2)) + ((3*n12) - n30)*(n21 + n03)*(3*((n30 + n12)**2) - (n21 + n03)**2)

    return np.array([i1, i2, i3, i4, i5, i6, i7])

def centralMoment(arr, p, q):
    momCen, momPQ = 0, [0, 0, 0]

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            momPQ[0] += (x**0) * (y**0) * arr[y, x]
            momPQ[1] += (x**1) * (y**0) * arr[y, x]
            momPQ[2] += (x**0) * (y**1) * arr[y, x]

    moment = [momPQ[1]/momPQ[0], momPQ[2]/momPQ[0]]

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            momCen += ((x - moment[0])**p) * ((y - moment[1])**q) * arr[y, x]

    return momCen

def plot(img, cmap="Greys_r"):
    plt.imshow(img, cmap=cmap)
    plt.show()

def __pool_process__(function, arr):
    pool = Pool()
    n_arr = pool.map(function, arr)
    pool.close()
    pool.join()
    return n_arr