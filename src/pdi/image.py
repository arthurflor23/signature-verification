import numpy as np
import pdi.segmentation as segmentation
    
def sig_preprocess(img_arr):
    for (i, img) in enumerate(img_arr):
        img_arr[i] = bbox(segmentation.otsu(grayscale(img)))

def grayscale(arr):
    return np.dot(arr[...,:3], [0.299, 0.587, 0.114])

def normalize(arr):
    return np.divide(arr, 255)

def bbox(arr):
    img = (arr == 1)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y_min, y_max = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    x_min, x_max = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
    return arr[y_min:y_max, x_min:x_max]

def histogram(arr):
    h_arr = np.zeros(256, dtype=int)
    for y in range(len(arr)):
        for x in range(len(arr[0])):
            h_arr[int(arr[y,x])] += 1
    return h_arr