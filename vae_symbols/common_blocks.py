import mxnet as mx
from mxnet import gluon


class ConvBlock(gluon.HybridBlock):
    def __init__(self, filters, in_channels, kernel, stride, padding, block_prefix, bn, activation, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.enable_bn = bn
        self.activation = activation
        with self.name_scope():
            self.conv = gluon.nn.Conv2D(filters, kernel, strides=stride, padding=padding, in_channels=in_channels,
                          prefix=block_prefix + '_conv')
            if self.enable_bn:
                self.bn = gluon.nn.BatchNorm(in_channels=filters, center=True, scale=True, prefix=block_prefix + '_bn')
            if activation is not None:
                self.act = gluon.nn.Activation(activation, prefix=block_prefix + '_act')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.enable_bn:
            x = self.bn(x)
        if self.activation is not None:
            x = self.act(x)
        return x


class DeConvBlock(gluon.nn.HybridBlock):
    def __init__(self, filters, in_channels, kernel, stride, padding, block_prefix, bn, activation,
                 bilinear=False, deconv_grand=True, **kwargs):
        super(DeConvBlock, self).__init__(**kwargs)
        self.enable_bn = bn
        self.act_type = activation
        with self.name_scope():
            if bilinear:
                weight_initializer = mx.init.Bilinear()
            else:
                weight_initializer = None
            self.upsampler = gluon.nn.Conv2DTranspose(channels=filters, kernel_size=kernel, strides=stride,
                                                      padding=padding, use_bias=True, in_channels=in_channels,
                                                      weight_initializer=weight_initializer, prefix=block_prefix + "_deconv")
            if deconv_grand is False:
                self.upsampler.collect_params().setattr('gred_req', 'null')
            if self.enable_bn:
                self.bn = gluon.nn.BatchNorm(in_channels=filters, center=True, scale=True, prefix=block_prefix + '_bn')
            if self.act_type is not None:
                self.act = gluon.nn.Activation(self.act_type, prefix=block_prefix + '_act')

    def hybrid_forward(self, F, x):
        x = self.upsampler(x)
        if self.enable_bn:
            x = self.bn(x)
        if self.act_type is not None:
            x = self.act(x)
        return x