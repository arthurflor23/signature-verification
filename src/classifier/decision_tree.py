from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import util.data as data
import util.path as path
import os

class RandomForest():

    def __init__(self):
        self.name = "random_forest"
        self.clf = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=0)

        # self.clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
        #         max_depth=10, max_features='auto', max_leaf_nodes=None,
        #         min_impurity_decrease=0.0, min_impurity_split=None,
        #         min_samples_leaf=1, min_samples_split=2,
        #         min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
        #         oob_score=False, random_state=0, verbose=0, warm_start=False)

    def randomSplit(self, features, labels):
        features_tr, features_te, labels_tr, labels_te = train_test_split(features, labels, test_size=0.25)
        return features_tr, labels_tr, features_te, labels_te

    def training(self, features_tr, labels_tr):
        self.clf.fit(features_tr, labels_tr)

    def predict(self, features):
        return self.clf.predict(features)

    def save(self, report, extension=""):
        f_name = self.name + extension
        destination = os.path.join(path.out(), self.name)
        data.saveVariable(destination, f_name, report.log)