from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import util.data as data
import util.path as path
import os

def random_split(features, labels):
    features_tr, features_te, labels_tr, labels_te = train_test_split(features, labels, test_size=0.25)
    return features_tr, labels_tr, features_te, labels_te

class RandomForest():

    def __init__(self):
        self.name = "random_forest_"
        self.clf = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=0)

    def training(self, features_tr, labels_tr):
        self.clf.fit(features_tr, labels_tr)

    def predict(self, features):
        return self.clf.predict(features)

    def save(self, report, extension="", version=0, graph=False):
        f_name = self.name + extension + "_" + str(version) 
        destination = os.path.join(path.out(), self.name + extension)
        data.saveVariable(destination, f_name, report.log)
        if not graph: data.saveGraph(destination, f_name, self.clf.estimators_[0], self.clf.classes_)

class CART():

    def __init__(self):
        self.name = "cart_"
        self.clf = DecisionTreeClassifier(criterion="entropy", random_state=0)

    def training(self, features_tr, labels_tr):
        self.clf.fit(features_tr, labels_tr)

    def predict(self, features):
        return self.clf.predict(features)

    def save(self, report, extension="", version=0, graph=False):
        f_name = self.name + extension + "_" + str(version) 
        destination = os.path.join(path.out(), self.name + extension)
        data.saveVariable(destination, f_name, report.log)
        if not graph: data.saveGraph(destination, f_name, self.clf, self.clf.classes_)