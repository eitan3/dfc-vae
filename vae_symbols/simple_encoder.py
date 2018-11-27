from mxnet import gluon
from common_blocks import ConvBlock

class Encoder_Module(gluon.HybridBlock):
    def __init__(self, encoder_args):
        super(Encoder_Module, self).__init__()
        self.encoder_args = encoder_args
        self.features = gluon.nn.HybridSequential(prefix='')
        with self.features.name_scope():

            # Input: (batch_size, 3, 256, 256)
            # Output: (batch_size, 128, 128, 128)
            self.features.add(ConvBlock(64, 3, 3, 1, 1, "conv_b1", True, 'relu'))
            self.features.add(ConvBlock(64, 64, 3, 1, 1, "conv_b2", True, 'relu'))
            self.features.add(ConvBlock(128, 64, 3, 1, 1, "conv_b3", True, 'relu'))
            self.features.add(gluon.nn.MaxPool2D(pool_size=2, strides=2, padding=0))

            # Input: (batch_size, 128, 128, 128)
            # Output: (batch_size, 256, 64, 64)
            self.features.add(ConvBlock(128, 128, 3, 1, 1, "conv_b4", True, 'relu'))
            self.features.add(ConvBlock(128, 128, 3, 1, 1, "conv_b5", True, 'relu'))
            self.features.add(ConvBlock(256, 128, 3, 1, 1, "conv_b6", True, 'relu'))
            self.features.add(gluon.nn.MaxPool2D(pool_size=2, strides=2, padding=0))

            # Input: (batch_size, 256, 64, 64)
            # Output: (batch_size, 512, 32, 32)
            self.features.add(ConvBlock(256, 256, 3, 1, 1, "conv_b7", True, 'relu'))
            self.features.add(ConvBlock(256, 256, 3, 1, 1, "conv_b8", True, 'relu'))
            self.features.add(ConvBlock(512, 256, 3, 1, 1, "conv_b9", True, 'relu'))
            self.features.add(gluon.nn.MaxPool2D(pool_size=2, strides=2, padding=0))

            # Input: (batch_size, 512, 32, 32)
            # Output: (batch_size, 1024, 16, 16)
            self.features.add(ConvBlock(512, 512, 3, 1, 1, "conv_b10", True, 'relu'))
            self.features.add(ConvBlock(512, 512, 3, 1, 1, "conv_b11", True, 'relu'))
            self.features.add(ConvBlock(1024, 512, 3, 1, 1, "conv_b12", True, 'relu'))
            self.features.add(gluon.nn.MaxPool2D(pool_size=2, strides=2, padding=0))

            # Input: (batch_size, 1024, 16, 16)
            # Output: (batch_size, 1024, 8, 8)
            self.features.add(ConvBlock(1024, 1024, 3, 1, 1, "conv_b13", True, 'relu'))
            self.features.add(ConvBlock(1024, 1024, 3, 1, 1, "conv_b14", True, 'relu'))
            # self.features.add(gluon.nn.MaxPool2D(pool_size=4, strides=2, padding=1))

            # Input: (batch_size, 1024, 8, 8)
            # Output: (batch_size, 1024, 8, 8)
            # self.features.add(ConvBlock(1024, 1024, 3, 1, 1, "conv_b15", True, 'relu'))
            # self.features.add(ConvBlock(1024, 1024, 3, 1, 1, "conv_b16", True, 'relu'))

            # Input: (batch_size, 512, 8, 8)
            # Output: (batch_size, 512)
            self.features.add(gluon.nn.Flatten())
            self.features.add(gluon.nn.Dense(self.encoder_args['num_fc_hidden_units'], activation='relu'))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x