import classifier.run as run

def main():

    run.random_forest(dataset="characters", extract="cnn", random_split=False, repeat=1, save=False)

if __name__ == '__main__':
    main()