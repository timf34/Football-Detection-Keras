from keras import Input, layers, Model, optimizers
#from keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU
import tensorflow as tf


def conv_layer(input, filters, kernel_size, padding='same', strides=(1, 1), pool='True'):

    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if pool == 'True':
        x = layers.MaxPool2D(pool_size=(2,2))(x)
    return x

def onebyoneconv(input):

    x = layers.Conv2D(filters=32, kernel_size=1, strides=(1,1), padding='same')(input)

    return x

def up_conv(input, filters, name):

    x = layers.Conv2DTranspose(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu', name=name)(input)

    return x

def create_model(input_size = (360, 640, 3)):
    inputs = Input(input_size)

    #block1
    x = conv_layer(inputs, 16, 3, pool='False')
    x = conv_layer(x, 16, 3, pool='True')
    #conv1 = x

    #block2
    x = conv_layer(x, 32, 3, pool='False')
    x = conv_layer(x, 32, 3, pool='True')
    conv2 = x

    #block3
    x = conv_layer(x, 64, 3, pool='False')
    x = conv_layer(x, 64, 3, pool='True')
    conv3 = x

    #conv4
    x = conv_layer(x, 64, 3, pool='False')
    x = conv_layer(x, 64, 3, pool='False')
    conv4 = x

    #upsampling&concatenation

    #conv4 upsampling
    conv4 = onebyoneconv(conv4)
    #conv4_upsamp=layers.UpSampling2D(size=(2,2))(conv4)
    up_con4 = up_conv(conv4, 32, 'up_conv4')

    #conv3upsampling
    conv3 = onebyoneconv(conv3)
    #conv3_upsamp=layers.UpSampling2D(size=(2,2))(conv3)
    up_conv3 = up_conv(conv3, 32, 'upconv3')

    #conv2 pass through/ skip connection
    conv2 = onebyoneconv(conv2)

    concat = layers.Concatenate(axis=-1)([up_con4, up_conv3, conv2])

    final_layer = layers.Conv2D(96, 3, strides=(1,1), padding='same')(concat)
    output_layer = layers.Conv2D(2, 3, strides=(1,1), activation='softmax', padding='same')(final_layer)


    model = Model(inputs, output_layer)
    #model.compile(optimizer=optimizers.SGD(lr=1e-2), loss='binary_crossentropy')

    model.summary()
    return model

create_model()
