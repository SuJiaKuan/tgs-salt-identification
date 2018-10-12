from keras import layers
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file
from keras import backend as K


WEIGHTS_PATH_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

BN_AXIS = 3 if K.image_data_format() == 'channels_last' else 1


def _name(base, identity):
    return '{}{}'.format(base, identity)


def _conv_block(input_tensor,
                kernel_size,
                filters,
                stage,
                block,
                shortcut_conv,
                shortcut_conv_strides=(2, 2)):
    """A ResNet convolution block."""
    filters1, filters2, filters3 = filters
    conv_name_base = 'res{}{}_branch'.format(stage, block)
    bn_name_base = 'bn{}{}_branch'.format(stage, block)

    conv2a_strides = shortcut_conv_strides if shortcut_conv else (1, 1)

    x = Conv2D(filters1,
               (1, 1),
               strides=conv2a_strides,
               name=_name(conv_name_base, '2a'))(input_tensor)
    x = BatchNormalization(axis=BN_AXIS,
                           name=_name(bn_name_base, '2a'))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2,
               kernel_size,
               padding='same',
               name=_name(conv_name_base, '2b'))(x)
    x = BatchNormalization(axis=BN_AXIS,
                           name=_name(bn_name_base, '2b'))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,
               (1, 1),
               name=_name(conv_name_base, '2c'))(x)
    x = BatchNormalization(axis=BN_AXIS,
                           name=_name(bn_name_base, '2c'))(x)

    if shortcut_conv:
        shortcut = Conv2D(filters3,
                          (1, 1),
                          strides=shortcut_conv_strides,
                          name=_name(conv_name_base, '1'))(input_tensor)
        shortcut = BatchNormalization(axis=BN_AXIS,
                                      name=_name(bn_name_base, '1'))(shortcut)
    else:
        shortcut = input_tensor

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             classes=1000):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either `None` '
                         '(random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(64,
               (7, 7),
               strides=(2, 2),
               padding='same',
               name='conv1')(img_input)
    x = BatchNormalization(axis=BN_AXIS, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = _conv_block(x, 3, [64, 64, 256], 2, 'a', True, shortcut_conv_strides=(1, 1))
    x = _conv_block(x, 3, [64, 64, 256], 2, 'b', False)
    x = _conv_block(x, 3, [64, 64, 256], 2, 'c', False)

    x = _conv_block(x, 3, [128, 128, 512], 3, 'a', True)
    x = _conv_block(x, 3, [128, 128, 512], 3, 'b', False)
    x = _conv_block(x, 3, [128, 128, 512], 3, 'c', False)
    x = _conv_block(x, 3, [128, 128, 512], 3, 'd', False)

    x = _conv_block(x, 3, [256, 256, 1024], 4, 'a', True)
    x = _conv_block(x, 3, [256, 256, 1024], 4, 'b', False)
    x = _conv_block(x, 3, [256, 256, 1024], 4, 'c', False)
    x = _conv_block(x, 3, [256, 256, 1024], 4, 'd', False)
    x = _conv_block(x, 3, [256, 256, 1024], 4, 'e', False)
    x = _conv_block(x, 3, [256, 256, 1024], 4, 'f', False)

    x = _conv_block(x, 3, [512, 512, 2048], 5, 'a', True)
    x = _conv_block(x, 3, [512, 512, 2048], 5, 'b', False)
    x = _conv_block(x, 3, [512, 512, 2048], 5, 'c', False)

    # Ensure that the model takes into account any potential predecessors of
    # `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_TOP,
                                    cache_subdir='__pretrained_models__',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='__pretrained_models__',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path, by_name=True)

    return model
