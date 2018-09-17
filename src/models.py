from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from keras.models import Model


def _unet_conv(pre_layer, filters, kernel_size, activation, padding):
    conv = Conv2D(filters,
                  kernel_size,
                  activation=activation,
                  padding=padding)(pre_layer)

    return conv


def _unet_up_conv(pre_layer,
                  copied_layer,
                  filters,
                  kernel_size,
                  strides,
                  padding):
    up = Conv2DTranspose(filters,
                         kernel_size,
                         strides=strides,
                         padding=padding)(pre_layer)
    up = Concatenate()([up, copied_layer])

    return up


def _unet_pool(pre_layer, pool_size):
    pool = MaxPooling2D(pool_size)(pre_layer)

    return pool


def unet(input_shape,
         kernel_size=(3, 3),
         activation='relu',
         padding='same',
         pool_size=(2, 2)):
    inputs = Input(input_shape)

    conv1_1 = _unet_conv(inputs, 64, kernel_size, activation, padding)
    conv1_2 = _unet_conv(conv1_1, 64, kernel_size, activation, padding)
    pool1 = _unet_pool(conv1_2, pool_size)

    conv2_1 = _unet_conv(pool1, 128, kernel_size, activation, padding)
    conv2_2 = _unet_conv(conv2_1, 128, kernel_size, activation, padding)
    pool2 = _unet_pool(conv2_2, pool_size)

    conv3_1 = _unet_conv(pool2, 256, kernel_size, activation, padding)
    conv3_2 = _unet_conv(conv3_1, 256, kernel_size, activation, padding)
    pool3 = _unet_pool(conv3_2, pool_size)

    conv4_1 = _unet_conv(pool3, 512, kernel_size, activation, padding)
    conv4_2 = _unet_conv(conv4_1, 512, kernel_size, activation, padding)
    pool4 = _unet_pool(conv4_2, pool_size)

    conv5_1 = _unet_conv(pool4, 1024, kernel_size, activation, padding)
    conv5_2 = _unet_conv(conv5_1, 1024, kernel_size, activation, padding)

    up6 = _unet_up_conv(conv5_2, conv4_2, 512, kernel_size, pool_size, padding)
    conv6_1 = _unet_conv(up6, 512, kernel_size, activation, padding)
    conv6_2 = _unet_conv(conv6_1, 512, kernel_size, activation, padding)

    up7 = _unet_up_conv(conv6_2, conv3_2, 256, kernel_size, pool_size, 'valid')
    conv7_1 = _unet_conv(up7, 256, kernel_size, activation, padding)
    conv7_2 = _unet_conv(conv7_1, 256, kernel_size, activation, padding)

    up8 = _unet_up_conv(conv7_2, conv2_2, 128, kernel_size, pool_size, padding)
    conv8_1 = _unet_conv(up8, 128, kernel_size, activation, padding)
    conv8_2 = _unet_conv(conv8_1, 128, kernel_size, activation, padding)

    up9 = _unet_up_conv(conv8_2, conv1_2, 64, kernel_size, pool_size, 'valid')
    conv9_1 = _unet_conv(up9, 64, kernel_size, activation, padding)
    conv9_2 = _unet_conv(conv9_1, 64, kernel_size, activation, padding)

    conv10 = _unet_conv(conv9_2, 1, (1, 1), 'sigmoid', padding)

    model = Model(inputs=inputs, outputs=conv10)

    return model
