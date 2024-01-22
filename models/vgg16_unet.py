from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications import VGG16
from tensorflow.keras.initializers import GlorotUniform


class VGG16_UNet:
    def __init__(self, inputs, doBatchNorm: bool, doDropout: bool = True, dropout_rate: float = 0.3) -> None:
        ''' initialization of the network '''
        # 1 class: human
        self.classes = 1
        # inputs of form Input(shape=x)
        self.inputs = inputs
        # doDropout: disable or enable droput layer
        self.dropout = doDropout
        # dropout_rate: rate of dropout, deafult 0.3
        self.dropout_rate = dropout_rate
        # doBatchNorm: disable or enable batch normalization
        self.batch_norm = doBatchNorm
        # importing VGG-16 pretrained network without top 3 layers
        self.vgg16 = VGG16(include_top=False,
                           weights="imagenet", input_tensor=self.inputs)

    def double_conv_block(self, inputs, n_filters: int, kernel_size=3):
        ''' double convolution block consisting of 3×3 convolution layers
            each layer followed by batch normalization and relu activation function 
            beacuse of relu, he_normal weights initialization has been used '''
        # ---------- # conv 3x3, ReLU
        x = Conv2D(n_filters, kernel_size, padding='same',
                   kernel_initializer='he_normal')(inputs)
        # optional batch normalization
        if self.batch_norm:
            x = BatchNormalization()(x)
        x = relu(x)
        # ---------- # conv 3x3, ReLU
        x = Conv2D(n_filters, kernel_size, padding='same',
                   kernel_initializer='he_normal')(x)
        if self.batch_norm:
            # optional batch normalization
            x = BatchNormalization()(x)
        x = relu(x)
        # ----------
        return x

    def expand_block(self, inputs, skip_connection, n_filters: int, kernel_size=3):
        ''' /decoder block/
        consists of 2×2 Transpose Convolution layer followed by concetenate where skip connections from
        encoder block are incorporated '''

        x = Conv2DTranspose(n_filters, (2, 2), strides=(
            2, 2), padding='same')(inputs)
        x = Concatenate()([x, skip_connection])

        # optional dropout
        if self.dropout:
            x = Dropout(self.dropout_rate)(x)

        # double convolution
        x = self.double_conv_block(x, n_filters, kernel_size)
        return x

    def network(self):
        # contracting path - VGG16
        skip1 = self.vgg16.get_layer("block1_conv2").output
        skip2 = self.vgg16.get_layer("block2_conv2").output
        skip3 = self.vgg16.get_layer("block3_conv3").output
        skip4 = self.vgg16.get_layer("block4_conv3").output

        c5 = self.vgg16.get_layer("block5_conv3").output

        # expansive path - UNET
        # shapes of encoder outputs are used
        # incorporating skip features from last contracting step
        e6 = self.expand_block(c5, skip4, skip4.shape[1])
        # incorporating skip features from 3rd contracting step
        e7 = self.expand_block(e6, skip3, skip3.shape[1])
        # incorporating skip features from 2nd contracting step
        e8 = self.expand_block(e7, skip2, skip2.shape[1])
        # incorporating skip features from 1st contracting step
        e9 = self.expand_block(e8, skip1, skip1.shape[1])

        # outputs
        ''' using sigmoid activation layer for binary classification together with xavier weight initialization '''
        outputs = Conv2D(self.classes, 1, activation='sigmoid',
                         kernel_initializer=GlorotUniform())(e9)  # 1x1 convolution
        model = Model(inputs=[self.inputs], outputs=[outputs])
        return model
