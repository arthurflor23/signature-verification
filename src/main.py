import util.data as data
import pdi.image as image
import signet.cnn_model as cnn

def main():
    # origin = "signatures"
    origin = "characters"

    dt_training, cl_training = data.fetchFromPath(origin, "training")
    print("training", len(dt_training), len(cl_training))

    image.sig_preprocess(dt_training)
    dt_training_features = cnn.extract_features(dt_training)
    del dt_training

    print("training features", len(dt_training_features[0][0]))

    # ... cl_training
    del cl_training

    dt_test, cl_test = data.fetchFromPath(origin, "test")
    print("test", len(dt_test), len(cl_test))

    image.sig_preprocess(dt_test)
    dt_test_features = cnn.extract_features(dt_test)
    del dt_test

    print("test features", len(dt_test_features[0][0]))

    # ... cl_test
    del cl_test

    # import matplotlib.pyplot as plt
    # plt.imshow(image.bbox(dt_training[0]), cmap="Greys_r")
    # plt.show()

if __name__ == '__main__':
    main()