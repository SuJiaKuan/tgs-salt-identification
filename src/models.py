from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import SpatialDropout2D
from keras.layers.merge import concatenate
from keras.models import Model

from resnet import ResNet50


def _conv_block(prelayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters,
                  (3, 3),
                  padding='same',
                  kernel_initializer='he_normal',
                  strides=strides,
                  name='{}_conv'.format(prefix))(prelayer)
    conv = BatchNormalization(name='{}_bn'.format(prefix))(conv)
    conv = Activation('relu', name='{}_activation'.format(prefix))(conv)

    return conv


def unet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape)

    for layer in resnet_base.layers:
        layer.trainable = True

    conv1 = resnet_base.get_layer('activation_1').output
    conv2 = resnet_base.get_layer('activation_10').output
    conv3 = resnet_base.get_layer('activation_22').output
    conv4 = resnet_base.get_layer('activation_40').output
    conv5 = resnet_base.get_layer('activation_49').output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = _conv_block(up6, 256, 'conv6_1')
    conv6 = _conv_block(conv6, 256, 'conv6_2')

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = _conv_block(up7, 192, 'conv7_1')
    conv7 = _conv_block(conv7, 192, 'conv7_2')

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = _conv_block(up8, 128, 'conv8_1')
    conv8 = _conv_block(conv8, 128, 'conv8_2')

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = _conv_block(up9, 64, 'conv9_1')
    conv9 = _conv_block(conv9, 64, 'conv9_2')

    up10 = UpSampling2D()(conv9)
    conv10 = _conv_block(up10, 32, 'conv10_1')
    conv10 = _conv_block(conv10, 32, 'conv10_2')
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation='sigmoid', name='prediction')(conv10)

    model = Model(resnet_base.input, x)

    return model
