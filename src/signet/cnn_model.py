from six.moves import cPickle
from theano import tensor as T
import signet.signet_spp as signet_spp
import util.path as path
import lasagne
import theano
import numpy as np

def extract_features(img_arr):
    model = CNNModel(signet_spp, path.model())
    return [model.get_feature_vector(img) for (i, img) in enumerate(img_arr)]

class CNNModel:
    """ Represents a model trained with the Lasagne library. """

    def __init__(self, model_factory, model_weight_path):
        """
        Parameters:
            model_factory (module): An object containing a "build_architecture" function.
            model_weights_path (str): A file containing the trained weights
        """
        with open(model_weight_path, 'rb') as f:
            model_params = cPickle.load(f, encoding='latin1')

        self.input_size = model_params['input_size']
        self.img_size = model_params['img_size']

        net_input_size = (None, 1, self.input_size[0], self.input_size[1])
        self.model = model_factory.build_architecture(net_input_size, model_params['params'])

        self.forward_util_layer = {}

    def get_feature_vector(self, image, layer='fc2'):
        """ 
        Runs forward propagation until a desired layer, for one input image
        Parameters:
            image (numpy.ndarray): The input image
            layer (str): The desired output layer
        """

        assert len(image.shape) == 2, "Input should have two dimensions: H x W"
        input = image[np.newaxis, np.newaxis]

        if layer not in self.forward_util_layer:
            inputs = T.tensor4('inputs')
            outputs = lasagne.layers.get_output(self.model[layer], inputs=inputs, deterministic=True)
            self.forward_util_layer[layer] = theano.function([inputs], outputs)

        out = self.forward_util_layer[layer](input)
        return out

    def get_feature_vector_multiple(self, images, layer='fc2'):
        """ 
        Runs forward propagation until a desired layer, for one input image
        Parameters:
            images (numpy.ndarray): The input images. 
                Should have three dimensions: N x H x W, where N: number of images, H: height, W: width
            layer (str): The desired output layer
        """

        images = np.asarray(images)
        assert len(images.shape) == 3, "Input should have three dimensions: N x H x W"

        input = np.expand_dims(images, axis=1)

        if layer not in self.forward_util_layer:
            inputs = T.tensor4('inputs')
            outputs = lasagne.layers.get_output(self.model[layer], inputs=inputs, deterministic=True)
            self.forward_util_layer[layer] = theano.function([inputs], outputs)

        out = self.forward_util_layer[layer](input)
        return out    