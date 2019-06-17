from keras.models import Model
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten,Dropout
from keras.layers import  Input
#from Configuration import cnn_N_filt,cnn_len_filt,cnn_max_pool_len,cnn_use_batchnorm,fc_use_laynorm_inp,cnn_use_laynorm,fs,fc_use_batchnorm,fc_use_laynorm,fc_drop,fc_lay
from Configuration import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras import initializers
import numpy as np
import math


def debug_print(*objects):
    if debug:
        print(objects)


class Layer_Normalization(Layer):
    def __init__(self, scale_initializer='ones', bias_initializer='zeros', **kwargs):
        super(Layer_Normalization, self).__init__(**kwargs)
        self.epsilon = 1e-6
        self.scale_initializer = initializers.get(scale_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],),
                                     initializer=self.scale_initializer,
                                     trainable=True,
                                     name='{}_scale'.format(self.name))

        self.bias = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.bias_initializer,
                                    trainable=True,
                                    name='{}_bias'.format(self.name))

        #self.built = True
        super (Layer_Normalization, self).build(input_shape)

    def call(self, inputs, mask=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        norm = (inputs - mean) / (std + self.epsilon)
        return norm * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

#Sinc Convolution implemented here
class Sinc_Conv_Layer(Layer):
    def __init__(self, N_filt, Filt_dim, fs, **kwargs):
        """

        :type N_filt: No of filter. int
        :type Filt_dim: Length of filter, int
        :type fs: Sampling frequency, int
        """
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

        super(Sinc_Conv_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # the fileres are trainable parameters
        self.filt_b1 = self.add_weight(name='filt_b1',
                        shape=(self.N_filt,),
                        initializer='uniform',
                        trainable=True)

        self.filt_band = self.add_weight(name='filt_band',
                        shape=(self.N_filt,),
                        initializer='uniform',
                        trainable=True)

        #MEL initialization of filter banks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700)) # convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1/self.freq_scale, (b2-b1)/self.freq_scale])

        super (Sinc_Conv_Layer, self).build(input_shape) #Compulsory to call this func

    def call(self, inputs, **kwargs):

        debug_print("Call")

        #Obtain beginning and end frequencies of the filters
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = K.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (K.abs(self.filt_band) + min_band / self.freq_scale)

        #Hamming window
        n = np.linspace(0, self.Filt_dim, self.Filt_dim)
        window = 0.54 - 0.46 * K.cos(2 * math.pi * n / self.Filt_dim)
        window = K.cast(window, "float32")
        window = K.variable(window)
        debug_print("  window", window)

        # TODO what is this?
        t_right_linspace = np.linspace(1, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2))
        t_right = K.variable(t_right_linspace / self.fs)
        debug_print("  t_right", t_right)

        # Compute the filters.
        output_list = []
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i] * sinc(filt_beg_freq[i] * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i] * sinc(filt_end_freq[i] * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)
            band_pass = band_pass / K.max(band_pass)
            output_list.append(band_pass * window)

        filters = K.stack(output_list) #(80,251)
        filters = K.transpose(filters) #(251, 80)
        filters = K.reshape(filters, (self.Filt_dim, 1,self.N_filt)) #(251,1,80) in TF: (filter_width, in_channels, out_channels)
        """
        Given an input tensor of shape [ batch, in_width, in_channels ] if data format is "NWC"
        or [ batch, in_channels, in_width ] if data format is "NCW"
        and a filter/kernel tensor of shape [ filter_width, in_channels, output_channels],
        this op reshapes the arguments to pass them to conv1d to perform equivalent convolution operations.
        Internally, this op reshapes the input tensors and invokes tf.nn.conv2d. For example, if data_format does not start with "NC", 
        a tensor of shape [batch, in_width, in_channels] is reshaped to [batch, 1, in_width, in_channels], and the filter is reshaped to 
        [1, filter_width, in_channels, out_channels]. The result is then reshaped back to [batch, out_width, out_channels] 
        (where out_width is a function of the stride and padding as in conv2d) and returned to the caller.
        
        """

        #Perform convolution
        debug_print("Call")
        debug_print("X", inputs)
        debug_print(" Filters", filters)

        out = K.conv1d(inputs, kernel=filters)

        debug_print("Output", out)
        return out

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1],
            self.Filt_dim,
            padding="valid",
            stride=1,
            dilation=1)
        return (input_shape[0],) + (new_size,) + (self.N_filt,)


def sinc(band, t_right):
    y_right = K.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = K.reverse(y_right, 0)
    y = K.concatenate([y_left, K.variable(K.ones(1)), y_right])
    return y


#stacking model
def get_model(input_shape, out_dim):

    #sinc layer
    inputs = Input(input_shape)
    #debug_print(inputs)
    x = BatchNormalization(momentum = 0.05)(inputs)
    x = Sinc_Conv_Layer(cnn_N_filt[0], cnn_len_filt[0], fs)(x)

    x = MaxPooling1D(pool_size= cnn_max_pool_len[0])(x)
    if cnn_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05)(x)
    if cnn_use_laynorm[0]:
        x = Layer_Normalization()(x)
    x = LeakyReLU(alpha=0.2)(x)


    #cnn1
    x = Conv1D(cnn_N_filt[1], cnn_len_filt[1], strides=1, padding='valid')(x)
    x = MaxPooling1D(pool_size= cnn_max_pool_len[1])(x)
    if cnn_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05)(x)
    if cnn_use_laynorm[1]:
        x = Layer_Normalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    #cnn2
    x = Conv1D(cnn_N_filt[2], cnn_len_filt[2], strides=1, padding='valid')(x)
    x = MaxPooling1D(pool_size=cnn_max_pool_len[2])(x)
    if cnn_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05)(x)
    if cnn_use_laynorm[2]:
        x = Layer_Normalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)

    #dnn1
    if fc_use_laynorm_inp:
        x =Layer_Normalization()(x)
    x = Dense(fc_lay[0])(x)
    if fc_use_batchnorm[0]:
        x = BatchNormalization(momentum=0.05)(x)
    if fc_use_laynorm[0]:
        x = Layer_Normalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)
    #dnn2
    x = Dense(fc_lay[1])(x)
    if fc_use_batchnorm[1]:
        x = BatchNormalization(momentum=0.05)(x)
    if fc_use_laynorm[1]:
        x = Layer_Normalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    #dnn2
    x = Dense(fc_lay[2])(x)
    if fc_use_batchnorm[2]:
        x = BatchNormalization(momentum=0.05)(x)
    if fc_use_laynorm[2]:
        x = Layer_Normalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    # DNN final
    prediction = Dense(out_dim, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=prediction)
    return model

