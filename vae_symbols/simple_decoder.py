from mxnet import gluon
from common_blocks import ConvBlock, DeConvBlock

class Decoder_Module(gluon.HybridBlock):
    def __init__(self, decoder_args):
        super(Decoder_Module, self).__init__()
        self.decoder_args = decoder_args
        with self.name_scope():
            self.fc1 = gluon.nn.Dense(decoder_args['num_fc_hidden_units'])
            self.fc2 = gluon.nn.Dense(1024 * 8 * 8)

            # Input: (batch_size, 1024, 8, 8)
            # Output: (batch_size, 1024, 16, 16)
            # self.up1 = DeConvBlock(1024, 1024, 4, 2, 1, 'up1', True, 'relu')
            # self.cnn1 = ConvBlock(1024, 1024, 3, 1, 1, "conv_b1", True, 'relu')

            # Input: (batch_size, 1024, 16, 16)
            # Output: (batch_size, 512, 32, 32)
            self.up2 = DeConvBlock(512, 1024, 4, 2, 1, 'up2', True, 'relu')
            self.cnn2_1 = ConvBlock(512, 512, 3, 1, 1, "conv_b2_1", True, 'relu')
            self.cnn2_2 = ConvBlock(512, 512, 3, 1, 1, "conv_b2_2", True, 'relu')

            # Input: (batch_size, 512, 32, 32)
            # Output: (batch_size, 256, 64, 64)
            self.up3 = DeConvBlock(256, 512, 4, 2, 1, 'up3', True, 'relu')
            self.cnn3_1 = ConvBlock(256, 256, 3, 1, 1, "conv_b3_1", True, 'relu')
            self.cnn3_2 = ConvBlock(256, 256, 3, 1, 1, "conv_b3_2", True, 'relu')

            # Input: (batch_size, 256, 64, 64)
            # Output: (batch_size, 128, 128, 128)
            self.up4 = DeConvBlock(128, 256, 4, 2, 1, 'up4', True, 'relu')
            self.cnn4_1 = ConvBlock(128, 128, 3, 1, 1, "conv_b4_1", True, 'relu')
            self.cnn4_2 = ConvBlock(128, 128, 3, 1, 1, "conv_b4_2", True, 'relu')

            # Input: (batch_size, 128, 128, 128)
            # Output: (batch_size, 64, 256, 256)
            self.up5 = DeConvBlock(64, 128, 4, 2, 1, 'up5', True, 'relu')
            self.cnn5_1 = ConvBlock(64, 64, 3, 1, 1, "conv_b5_1", True, 'relu')
            self.cnn5_2 = ConvBlock(64, 64, 3, 1, 1, "conv_b5_2", True, 'relu')

            self.output = ConvBlock(3, 64, 1, 1, 0, "output", False, 'tanh')

    def hybrid_forward(self, F, x):
        # x = self.fc0(x)
        # x = F.Activation(x, 'relu')

        x = self.fc1(x)
        x = F.Activation(x, 'relu')

        x = self.fc2(x)
        x = F.Activation(x, 'relu')
        x = F.reshape(x, (0, 1024, 8, 8))

        # x = self.up1(x)
        # x = self.cnn1(x)

        x = self.up2(x)
        x = self.cnn2_1(x)
        x = self.cnn2_2(x)

        x = self.up3(x)
        x = self.cnn3_1(x)
        x = self.cnn3_2(x)

        x = self.up4(x)
        x = self.cnn4_1(x)
        x = self.cnn4_2(x)

        x = self.up5(x)
        x = self.cnn5_1(x)
        x = self.cnn5_2(x)

        x = self.output(x)
        return x