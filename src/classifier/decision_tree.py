import util.data as data
import pdi.image as image
from util.report import Report
import classifier.tree as tree
import time

def random_forest(features, labels, repeat=1, save=False):
    __classifier__(tree.RandomForest, features, labels, repeat, save)

def cart(features, labels, repeat=1, save=False):
    __classifier__(tree.CART, features, labels, repeat, save)

def c45(features, labels, repeat=1, save=False):
    __classifier__(tree.C45, features, labels, repeat, save)

def __classifier__(tree_class, features, labels, repeat, save):

    for i in range(repeat):
        print("\n> %s" % (i+1))
        cl_tree = tree_class(len(features[0]))

        report = Report(console=save)
        report.information("Data size: %s" % (len(features)))

        ### random split (training, test)
        features_tr, labels_tr, features_te, labels_te = tree.random_split(0.3333, features, labels)

        report.information("Sample size: %s (training), %s (test)" % (len(features_tr), len(features_te)))
        report.information("Number of features: %s" % (len(features_tr[0])))

        training_start = time.time()
        cl_tree.training(features_tr, labels_tr)
        training_end = time.time() - training_start

        predict_start = time.time()
        classified = cl_tree.predict(features_te)
        predict_end = time.time() - predict_start
        predict_end = time.time() - predict_start

        report.information("\nTraining time: %f s" % training_end)
        report.information("Predict time: %f s\n" % predict_end)

        result = check_classified(labels_te, classified)
        report.finish(result)

        if save:
            cl_tree.save(report, (i+1), (i==0))
        else:
            for x in report.log: print(x)

def check_classified(classes, classified):
    log, matched = [], []
    for (i, j) in zip(classes, classified):
        m = "FAIL"
        if (i == j):
            matched.append(j)
            m = "OK"
        log.append("%s \t:\t %s \t|\t %s" % (i, j, m))
    return [log, matched, len(classes), len(matched), (len(matched)/len(classes))]