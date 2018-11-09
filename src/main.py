import util.data as data
import pdi.image as image

import matplotlib.pyplot as plt

def main():
    # origin = "signatures"
    origin = "characters"

    dt_training, cl_training = data.fetchFromPath(origin, "training")
    print("training", len(dt_training), len(cl_training))

    image.sig_preprocess(dt_training)

    del dt_training
    del cl_training

    dt_test, cl_test = data.fetchFromPath(origin, "test")
    print("test", len(dt_test), len(cl_test))

    # plt.imshow(image.bbox(dt_training[0]), cmap="Greys_r")
    # plt.show()

if __name__ == '__main__':
    main()