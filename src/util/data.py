from sklearn.tree import export_graphviz
import graphviz
import os
import sys
import cv2
import util.path as path
import numpy as np

def fetchFromPath(origin, suborigin=""):
    p = os.path.join(path.data(), origin)
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
        print("Directory doesn't exist:", folder)
        sys.exit(1)

def saveVariable(destination, name, variable):
    os.makedirs(destination, exist_ok=True)

    if (not isinstance(variable, str)):
        variable = "\n".join(variable)

    with open(os.path.join(destination, name + ".txt"), "w") as variable_file:
        variable_file.write(variable)

def saveGraph(destination, name, tree, classes):
    os.makedirs(destination, exist_ok=True)
    dot_data = export_graphviz(tree, out_file=None, class_names=classes, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render(os.path.join(destination, name))

def compare(arr_1, arr_2):
    log, matched = [], []

    for (i, j) in zip(arr_1, arr_2):
        m = "FAIL"
        if (i == j):
            matched.append(j)
            m = "OK"
        log.append("%s \t:\t %s \t|\t %s" % (i, j, m))

    m = len(matched)
    t = len(arr_1)
    p = len(matched)/len(arr_1)
    log.append("\nAccuracy: %s/%s (%f)" % (m, t, p))
    return [log, matched, t, m, p]