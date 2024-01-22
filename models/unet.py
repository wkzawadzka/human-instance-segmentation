from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import GlorotUniform


class Binary_UNet:
    def __init__(self, inputs, conv_filters: int, doBatchNorm: bool = True, doDropout: bool = True, dropout_rate: float = 0.3) -> None:
        ''' initialization of the network '''
        self.filters = conv_filters
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

    def contract_block(self, inputs, n_filters, kernel_size=3, max_pooling=True):
        # double convolution block
        x = self.double_conv_block(inputs, n_filters, kernel_size)
        next_layer = x

        # optional droupout layer
        if self.dropout:
            x = Dropout(self.dropout_rate)(x)
        if max_pooling:
            # a 2x2 max pooling operation with stride 2 for downsampling
            next_layer = MaxPooling2D((2, 2))(x)

        # returning next_layer and x (skip connection later incorporated in expanding path)
        return (next_layer, x)

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
        # contracting path
        ''' /encoder/ 
        doubling number of filters in each contacting path step '''
        c1, skip1 = self.contract_block(self.inputs, self.filters * 1)
        c2, skip2 = self.contract_block(c1, self.filters * 2)
        c3, skip3 = self.contract_block(c2, self.filters * 4)
        c4, skip4 = self.contract_block(c3, self.filters * 8)
        # max_pooling false, stop downsampling process
        c5, skip5 = self.contract_block(
            c4, self.filters * 16, max_pooling=False)

        # expansive path
        ''' /decoder/
        halfing number of filters in each expansive path step '''
        e6 = self.expand_block(
            c5, skip4, self.filters * 8)  # incorporating skip features from last contracting step
        # incorporating skip features from 3rd contracting step
        e7 = self.expand_block(e6, skip3, self.filters * 4)
        # incorporating skip features from 2nd contracting step
        e8 = self.expand_block(e7, skip2, self.filters * 2)
        # incorporating skip features from 1st contracting step
        e9 = self.expand_block(e8, skip1, self.filters * 1)

        # outputs
        ''' using sigmoid activation layer for binary classification together with xavier weight initialization '''
        outputs = Conv2D(self.classes, 1, activation='sigmoid',
                         kernel_initializer=GlorotUniform())(e9)  # 1x1 convolution
        model = Model(inputs=[self.inputs], outputs=[outputs])
        return model
