from sklearn.tree import export_graphviz
import graphviz
import time
import os
import sys
import cv2
import numpy as np
import util.path as path
import pdi.image as image

DATASET = ""

def loadDataset(dataset, extract):    
    print("Loading data...")
    images, labels = fetchFromPath(dataset)

    print("Preprocessing data...")
    images = image.preprocess(images)

    print("Extracting data features...")
    features = image.extract_features(extract, images)
    del images
    return features, labels

def fetchFromPath(origin, suborigin=""):
    p = os.path.join(path.data(), origin)
    dt_dir = os.path.join(p, suborigin)
    data = listFolder(dt_dir, origin)
    data[0] = fetchFromArray(data[0])
    return data

def fetchFromArray(arr):
    resize = lambda img: cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
    return [np.array(resize(cv2.imread(x))) for x in arr]

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

def saveNodeTree(destination, name, tree):
    cl_root = "#f9bd77"
    cl_left = "#fefedf"
    cl_right = "#63a2d8"
    cl_leaf = "#4a6f46"
    cl_border = "#20202050"

    def classesText(object_key, breakline=''):
        return ''.join(['%s (%s) %s' % (key, object_key[key], breakline) for key in object_key])

    def plotNodes(node, dot, route, identifier=0, color=cl_root):
        if (node.results != None):
            dot.node(str(identifier), classesText(node.results, '\n'), shape="egg", style="filled", color=cl_border, fillcolor=cl_leaf, fontcolor="white")
        else:
            decision = 'i%s: x >= %s ?' % (node.col, node.value)
            dot.node(str(identifier), decision, shape="box", style="filled", color=cl_border, fillcolor=color)
            
            leftID = identifier + 1 + time.time()
            rightID = identifier + 1001 + time.time()

            route.append([False, identifier, leftID])
            route.append([True, identifier, rightID])

            plotNodes(node.false_branch, dot, route, leftID, cl_left)
            plotNodes(node.true_branch, dot, route, rightID, cl_right)

    def plotEdges(dot, route):
        for x in range(len(route)):
            dot.edge(str(route[x][1]), str(route[x][2]), label=str(route[x][0]))

    graph = graphviz.Digraph()
    graph.format = 'png'
    route = []

    plotNodes(tree, graph, route)
    plotEdges(graph, route)
    graph.render(os.path.join(destination, name))