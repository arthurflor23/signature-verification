import classifier.decision_tree as decision_tree
import util.data as data

def main():
    
    # dataset, repeat, save = "signatures", 30, True
    data.DATASET, repeat, save = "characters", 30, True

    features, labels = data.loadDataset(data.DATASET, "mi_hu")

    decision_tree.random_forest(features, labels, repeat, save)
    decision_tree.cart(features, labels, repeat, save)
    decision_tree.c45(features, labels, repeat, save)

    features, labels = data.loadDataset(data.DATASET, "cnn")

    decision_tree.random_forest(features, labels, repeat, save)
    decision_tree.cart(features, labels, repeat, save)
    decision_tree.c45(features, labels, repeat, save)

if __name__ == '__main__':
    main()