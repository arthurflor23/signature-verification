import os

def data():
    return os.path.join("..", "data")

def out():
    return os.path.join("..", "out")

def model(m=None):
    model = "signet_spp.pkl" if m is None else m 
    return os.path.join("..", "model", model)

def fetch(path):
    return os.path.join(data(), path)

def list(path):
    l = [x for x in next(os.walk(path))]
    return sorted(l[1]), sorted(l[2])