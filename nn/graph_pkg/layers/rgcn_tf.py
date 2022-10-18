import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, initializers, regularizers


class GraphConvolution(Layer):

    def __init__(self,
                 output_dim, support=1, featureless=False,
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, num_bases=-1,
                 b_regularizer=None, bias=False, dropout=0., **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.featureless = featureless  # use / ignore input features
        self.dropout = dropout

        assert support >= 1

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights
        self.num_bases = num_bases

        # defined during build()
        self.input_dim = None
        self.W = None
        self.W_comp = None
        self.num_nodes = None

        super(GraphConvolution, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        feature_shapes = input_shape[0]
        output_shape = (feature_shapes[0], self.output_dim)
        return output_shape

    def build(self, input_shape):
        pass
