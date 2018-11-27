from mxnet import gluon
from common_blocks import ConvBlock

class DFC_Module(gluon.HybridBlock):
    def __init__(self):
        super(DFC_Module, self).__init__(prefix="dfc_module_")
        with self.name_scope():

            # Input: (batch_size, 3, 256, 256)
            # Output: (batch_size, 128, 128, 128)
            self.cnn1 = ConvBlock(32, 3, 3, 1, 1, "conv_b1", True, 'relu')
            self.cnn2 = ConvBlock(64, 32, 3, 1, 1, "conv_b2", True, 'relu')
            self.pool1 = gluon.nn.MaxPool2D(pool_size=2, strides=2, padding=0)

            # Input: (batch_size, 128, 128, 128)
            # Output: (batch_size, 256, 64, 64)
            self.cnn3 = ConvBlock(64, 64, 3, 1, 1, "conv_b3", True, 'relu')
            self.cnn4 = ConvBlock(128, 64, 3, 1, 1, "conv_b4", True, 'relu')
            self.pool2 = gluon.nn.MaxPool2D(pool_size=2, strides=2, padding=0)

            # Input: (batch_size, 256, 64, 64)
            # Output: (batch_size, 512, 32, 32)
            self.cnn5 = ConvBlock(128, 128, 3, 1, 1, "conv_b5", True, 'relu')
            self.cnn6 = ConvBlock(256, 128, 3, 1, 1, "conv_b6", True, 'relu')
            self.pool3 = gluon.nn.MaxPool2D(pool_size=2, strides=2, padding=0)

            # Input: (batch_size, 512, 32, 32)
            # Output: (batch_size, 512, 32, 32)
            self.cnn7 = ConvBlock(256, 256, 3, 1, 1, "conv_b7", True, 'relu')

    def hybrid_forward(self, F, x):
        outs = []
        x = self.cnn1(x)
        outs.append(x)
        x = self.cnn2(x)
        x = self.pool1(x)

        x = self.cnn3(x)
        outs.append(x)
        x = self.cnn4(x)
        x = self.pool2(x)

        x = self.cnn5(x)
        outs.append(x)
        x = self.cnn6(x)
        x = self.pool3(x)

        x = self.cnn7(x)
        outs.append(x)
        return outs
