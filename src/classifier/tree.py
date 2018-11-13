from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import util.data as data
import util.path as path
import numpy as np
import os

def random_split(test_size, features, labels):
    features_tr, features_te, labels_tr, labels_te = train_test_split(features, labels, test_size=test_size)
    return features_tr, labels_tr, features_te, labels_te

class RandomForest():

    def __init__(self, len_features):
        self.name = "random_forest_(" + str(len_features) + "f)"
        self.clf = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=0)

    def save(self, report, version, graph):
        f_name = self.name + "_" + str(version) 
        destination = os.path.join(path.out(), data.DATASET, self.name)
        data.saveVariable(destination, f_name, report.log)
        if graph: data.saveGraph(destination, f_name, self.clf.estimators_[0], self.clf.classes_)

    def training(self, features_tr, labels_tr):
        print("Training...")
        self.clf.fit(features_tr, labels_tr)

    def predict(self, features):
        print("Classifying...")
        return self.clf.predict(features)

class CART():

    def __init__(self, len_features):
        self.name = "cart_(" + str(len_features) + "f)"
        self.clf = DecisionTreeClassifier(criterion="entropy", random_state=0)

    def save(self, report, version, graph):
        f_name = self.name + "_" + str(version) 
        destination = os.path.join(path.out(), data.DATASET, self.name)
        data.saveVariable(destination, f_name, report.log)
        if graph: data.saveGraph(destination, f_name, self.clf, self.clf.classes_)

    def training(self, features_tr, labels_tr):
        print("Training...")
        self.clf.fit(features_tr, labels_tr)

    def predict(self, features):
        print("Classifying...")
        return self.clf.predict(features)


class C45():

    def __init__(self, len_features):
        self.name = "c45_(" + str(len_features) + "f)"
        self.criterion = self.entropy
        self.min_gain = 0.5
        self.clf = None

    def save(self, report, version, graph):
        f_name = self.name + "_" + str(version) 
        destination = os.path.join(path.out(), data.DATASET, self.name)
        data.saveVariable(destination, f_name, report.log)
        if graph: data.saveNodeTree(destination, f_name, self.clf)

    def training(self, features_tr, labels_tr):
        features_tr = np.array(features_tr).tolist()

        for i in range(len(features_tr)):
            features_tr[i].append(labels_tr[i])

        print("Training...")
        self.clf = self.__training__(list(features_tr), self.criterion)
        self.prune(self.clf, self.criterion, self.min_gain)

    def predict(self, features):
        features = np.array(features).tolist()
        json_arr = [self.__predict__(self.clf, item) for item in features]
        print("Classifying...")
        return self.predictToStandart(json_arr)

    def predictToStandart(self, json_arr):
        arr = []
        for item in json_arr:
            keys = list(item.keys())
            quant = [item[x] for x in keys]
            i_max = np.argmax(quant)
            arr.append(keys[i_max])
        return arr

    def __training__(self, rows, criterion):
        if len(rows) == 0: return Node()
        current_score = criterion(rows)
        best_gain = 0.0
        best_attr = None
        best_sets = None
        column_count = len(rows[0]) - 1

        for col in range(0, column_count):
            columnValues = [row[col] for row in rows]

            for value in columnValues:
                (set1, set2) = self.divideSet(rows, col, value)
                p = float(len(set1)) / len(rows)
                gain = current_score - (p*criterion(set1)) - ((1-p)*criterion(set2))
                
                if (gain > best_gain and len(set1) > 0 and len(set2) > 0):
                    best_gain = gain
                    best_attr = (col, value)
                    best_sets = (set1, set2)

        if (best_gain > 0):
            true_branch = self.__training__(best_sets[0], criterion)
            false_branch = self.__training__(best_sets[1], criterion)
            return Node(col=best_attr[0], value=best_attr[1], true_branch=true_branch, false_branch=false_branch)
        else:
            return Node(results=self.uniqueCounts(rows))

    def divideSet(self, rows, column, value):
        splittingFunction = None
        if (isinstance(value, int) or isinstance(value, float)):
            splittingFunction = lambda row: row[column] >= value
        else: 
            splittingFunction = lambda row: row[column] == value
        list1 = [row for row in rows if splittingFunction(row)]
        list2 = [row for row in rows if not splittingFunction(row)]
        return (list1, list2)

    def uniqueCounts(self, rows):
        results = {}
        for row in rows:
            r = row[-1]
            if (r not in results): results[r] = 0
            results[r] += 1
        return results
    
    def entropy(self, rows):
        log2 = lambda x: np.log(x)/np.log(2)
        results = self.uniqueCounts(rows)
        entr = 0.0
        for r in results:
            p = float(results[r])/len(rows)
            entr -= p*log2(p)
        return entr
    
    def gini(self, rows):
        total = len(rows)
        counts = self.uniqueCounts(rows)
        imp = 0.0
        for k1 in counts:
            p1 = float(counts[k1])/total  
            for k2 in counts:
                if k1 == k2: continue
                p2 = float(counts[k2])/total
                imp += p1*p2
        return imp

    def prune(self, tree, criterion, min_gain):
        if (tree.true_branch.results == None): 
            self.prune(tree.true_branch, criterion, min_gain)
        if (tree.false_branch.results == None): 
            self.prune(tree.false_branch, criterion, min_gain)

        if (tree.true_branch.results != None and tree.false_branch.results != None):
            tb, fb = [], []
            for v, c in tree.true_branch.results.items(): tb += [[v]] * c
            for v, c in tree.false_branch.results.items(): fb += [[v]] * c

            p = float(len(tb)) / len(tb + fb)
            delta = criterion(tb+fb) - p*criterion(tb) - (1-p)*criterion(fb)

            if (delta < min_gain):
                print('A branch was pruned: gain ~ %f' % delta)
                tree.true_branch, tree.false_branch = None, None
                tree.results = self.uniqueCounts(tb + fb)

    def __predict__(self, tree, features):        
        if (tree.results != None):
            return tree.results
        else:
            v = features[tree.col]
            branch = None
            if (isinstance(v, int) or isinstance(v, float)):
                if (v >= tree.value): branch = tree.true_branch
                else: branch = tree.false_branch
            else:
                if (v == tree.value): branch = tree.true_branch
                else: branch = tree.false_branch
        return self.__predict__(branch, features)

class Node():

    def __init__(self, col=-1, value=None, true_branch=None, false_branch=None, results=None):
        self.col = col
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.results = results