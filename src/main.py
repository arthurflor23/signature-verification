import classifier.decision_tree as decision_tree
import util.data as data

def main():
    
    ### Decision Tree
    # data.DATASET = "signatures"
    # data.DATASET = "characters"
    # repeat = 30
    # save = True

    # features, labels = data.loadDataset(data.DATASET, "mi_hu")
    # decision_tree.random_forest(features, labels, repeat, save)
    # decision_tree.cart(features, labels, repeat, save)
    # decision_tree.c45(features, labels, repeat, save)
    # del features, labels

    # features, labels = data.loadDataset(data.DATASET, "cnn")
    # decision_tree.random_forest(features, labels, repeat, save)
    # decision_tree.cart(features, labels, repeat, save)
    # decision_tree.c45(features, labels, repeat, save)
    # del features, labels

    ### Word Beam Search
    data.DATASET = "iam"

    


if __name__ == '__main__':
    main()