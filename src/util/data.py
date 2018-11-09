import os
import sys
import cv2
import util.path as path
import numpy as np

def fetchFromPath(origin, suborigin=""):
    p = path.fetch(origin)
    dt_dir = os.path.join(p, suborigin)
    data = listFolder(dt_dir, origin)
    data[0] = fetchFromArray(data[0])
    return data

def fetchFromArray(arr):
    return [np.array(cv2.imread(x)) for x in arr]

def listFolder(folder, category="", data=None):
    try:
        folders, files = path.list(folder)

        if data is None:
            data = [[],[]]

        if (len(folders) > 0):
            for _, item in enumerate(folders):
                listFolder(os.path.join(folder, item), item, data)

        elif (len(files) > 0):
            data[0].extend([os.path.join(folder, item) for _, item in enumerate(files)])
            data[1].extend([category] * len(files))

        return data
    except:
        print("Diretório não existe:", folder)
        sys.exit(1)