import util.data as data
import pdi.image as image
from util.report import Report
import classifier.decision_tree as decision_tree

def random_forest(dataset, extract, random_split=False, repeat=1, save=False):

    for i in range(repeat):
        report = Report(console=save)
        random_forest = decision_tree.RandomForest()

        data_tr, labels_tr = data.fetchFromPath(dataset, "training")
        data_te, labels_te = data.fetchFromPath(dataset, "test")

        image.sig_preprocess(data_tr, extract)
        image.sig_preprocess(data_te, extract)

        report.information("Data: %s (training), %s (test)" % (len(data_tr), len(data_te)))

        features_tr = image.extract_features(extract, data_tr)
        features_te = image.extract_features(extract, data_te)

        del data_tr
        del data_te

        ### BEGIN ~ RANDOM SPLIT ###
        if random_split: 
            features_tr, labels_tr, features_te, labels_te = decision_tree.random_split(features_tr+features_te, labels_tr+labels_te)
        ### END ~ RANDOM SPLIT ###

        report.information("Features (%s): %s (training), %s (test)" % (len(features_tr[0]), len(features_tr), len(features_te)))

        random_forest.training(features_tr, labels_tr)

        del features_tr
        del labels_tr

        classified = random_forest.predict(features_te)

        compare = data.compare(labels_te, classified)
        report.add(compare[0])
        report.finish(compare[2], compare[3], compare[4])

        if save:
            random_forest.save(report, extract, i, random_split)
        else:
            for x in report.log: print(x)

        del features_te
        del labels_te

def cart(dataset, extract, random_split=False, repeat=1, save=False):

    for i in range(repeat):
        report = Report(console=save)
        cart = decision_tree.CART()

        data_tr, labels_tr = data.fetchFromPath(dataset, "training")
        data_te, labels_te = data.fetchFromPath(dataset, "test")

        image.sig_preprocess(data_tr, extract)
        image.sig_preprocess(data_te, extract)

        report.information("Data: %s (training), %s (test)" % (len(data_tr), len(data_te)))

        features_tr = image.extract_features(extract, data_tr)
        features_te = image.extract_features(extract, data_te)

        del data_tr
        del data_te

        ### BEGIN ~ RANDOM SPLIT ###
        if random_split: 
            features_tr, labels_tr, features_te, labels_te = decision_tree.random_split(features_tr+features_te, labels_tr+labels_te)
        ### END ~ RANDOM SPLIT ###

        report.information("Features (%s): %s (training), %s (test)" % (len(features_tr[0]), len(features_tr), len(features_te)))

        cart.training(features_tr, labels_tr)

        del features_tr
        del labels_tr

        classified = cart.predict(features_te)

        compare = data.compare(labels_te, classified)
        report.add(compare[0])
        report.finish(compare[2], compare[3], compare[4])

        if save:
            cart.save(report, extract, i, random_split)
        else:
            for x in report.log: print(x)

        del features_te
        del labels_te