import os
import tensorflow as tf
from pathlib import Path
from keras.layers import Dense
from models._base_ import TensorflowModel
from keras.layers import AveragePooling1D
from keras.layers import Layer
from activations.RevIN import RevNorm 
from layers.Time2Vec import Time2Vec
import numpy as np
from keras.initializers import GlorotUniform

class MovingAvg__Tensorflow(Layer):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg__Tensorflow, self).__init__()
        self.kernel_size = kernel_size
        self.avg = AveragePooling1D(pool_size=kernel_size, strides=stride, padding='valid')

    def call(self, x):
        # padding on the both ends of time series
        front = tf.tile(x[:, 0:1, :], multiples=[1, (self.kernel_size - 1) // 2, 1])
        end = tf.tile(x[:, -1:, :], multiples=[1, (self.kernel_size - 1) // 2, 1])
        x = tf.concat([front, x, end], axis=1)
        x = self.avg(x)
        return x

class SeriesDecomp__Tensorflow(Layer):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp__Tensorflow, self).__init__()
        self.moving_avg = MovingAvg__Tensorflow(kernel_size, stride=1)

    def call(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear__Tensorflow(tf.keras.Model):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, channels_in, channels_out, individual, activation=None, seed=941):
        super(DLinear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.seed = seed

        # Decompsition Kernel Size
        # print(f'{self.seq_len = }')
        if self.seq_len[0] >= 7:
            kernel_min = 3
            kernel_size = int(self.seq_len[0] // kernel_min) + 1 if (self.seq_len[0] // kernel_min) % 2 == 0 else int(self.seq_len[0] // kernel_min)
        else:
            kernel_size = 25
        # print(f'{kernel_size = }')
        self.decomposition = SeriesDecomp__Tensorflow(kernel_size)
        
        if self.individual:
            self.Linear_Seasonal = []
            self.Linear_Trend = []
            for i in range(self.channels_in):
                self.Linear_Seasonal.append(Dense(self.pred_len, activation=activation, kernel_initializer=GlorotUniform(seed=self.seed)))
                self.Linear_Trend.append(Dense(self.pred_len, activation=activation, kernel_initializer=GlorotUniform(seed=self.seed)))
                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
                # self.Linear_Trend[i].kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
        else:
            self.Linear_Seasonal = Dense(self.pred_len, activation=activation, kernel_initializer=GlorotUniform(seed=self.seed))
            self.Linear_Trend = Dense(self.pred_len, activation=activation, kernel_initializer=GlorotUniform(seed=self.seed))
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
            # self.Linear_Trend.kernel = tf.Variable((1/self.seq_len)*tf.ones([self.seq_len, self.pred_len]))
        self.final_layer = Dense(self.channels_out, kernel_initializer=GlorotUniform(seed=self.seed))

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)

        if self.individual:
            seasonal_output = tf.concat([tf.expand_dims(self.Linear_Seasonal[i](x[:,:,i]), axis=-1) for i in range(self.channels_in)], axis=-1)
            trend_output = tf.concat([tf.expand_dims(self.Linear_Trend[i](x[:,:,i]), axis=-1) for i in range(self.channels_in)], axis=-1)
            x = seasonal_output + trend_output
        else:
            seasonal_init, trend_init = tf.transpose(seasonal_init, perm=[0,2,1]), tf.transpose(trend_init, perm=[0,2,1])
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            x = seasonal_output + trend_output
            x = tf.transpose(x, perm=[0,2,1]) # to [Batch, Output length, Channel]

        if self.channels_out==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        # print(f'{x.shape = }')
        return x # [Batch, Output length, Channel]

class NDLinearTime2VecRevIN__Tensorflow(DLinear__Tensorflow):
    """
    Decomposition-Linear with additional Time2Vec
    """
    def __init__(self, seq_len, pred_len, channels_in, channels_out, individual, activation=None, seed=941):
        # super(NDLinearTime2VecRevIN__Tensorflow, self).__init__()
        super(NDLinearTime2VecRevIN__Tensorflow, self).__init__(seq_len, pred_len, channels_in, channels_out, individual, activation=activation)
        self.time2vec_kernel_size = 128
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.seed = seed

        # Decompsition Kernel Size
        # print(f'{self.seq_len = }')
        if self.seq_len[0] >= 7:
            kernel_min = 3
            kernel_size = int(self.seq_len[0] // kernel_min) + 1 if (self.seq_len[0] // kernel_min) % 2 == 0 else int(self.seq_len[0] // kernel_min)
        else:
            kernel_size = 25
        print(f'{kernel_size = }')
        self.decomposition = SeriesDecomp__Tensorflow(kernel_size)
        self.time2vec = Time2Vec(kernel_size=self.time2vec_kernel_size)
        self.rev_norm = RevNorm(axis=-2)

    def call(self, x):
        x = self.time2vec(x)
        x = self.rev_norm(x, 'norm')
        seasonal_init, trend_init = self.decomposition(x)

        if self.individual:
            seasonal_output = tf.concat([tf.expand_dims(self.Linear_Seasonal[i](x[:,:,i]), axis=-1) for i in range(self.channels_in)], axis=-1)
            trend_output = tf.concat([tf.expand_dims(self.Linear_Trend[i](x[:,:,i]), axis=-1) for i in range(self.channels_in)], axis=-1)
            x = seasonal_output + trend_output
        else:
            seasonal_init, trend_init = tf.transpose(seasonal_init, perm=[0,2,1]), tf.transpose(trend_init, perm=[0,2,1])
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            x = seasonal_output + trend_output
            x = tf.transpose(x, perm=[0,2,1]) # to [Batch, Output length, Channel]

        x = self.rev_norm(x, 'denorm')
        if self.channels_out==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        return x # [Batch, Output length, Channel]

class NLinear__Tensorflow(tf.keras.Model):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, channels_in, channels_out, individual, activation=None, seed=941):
        super(NLinear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.seed = seed
        
        # Use this line if you want to visualize the weights
        # self.Linear.weights = (1/self.seq_len)*tf.ones([self.seq_len, self.pred_len])
        if self.individual:
            self.Linear = [Dense(self.pred_len, kernel_initializer=GlorotUniform(seed=self.seed)) for _ in range(self.channels_in)]
        else:
            self.Linear = Dense(self.pred_len, kernel_initializer=GlorotUniform(seed=self.seed), activation=activation)
        self.final_layer = Dense(self.channels_out, kernel_initializer=GlorotUniform(seed=self.seed))

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:]
        x = x - seq_last
        if self.individual:
            x = tf.concat([tf.expand_dims(self.Linear[i](x[:,:,i]), axis=-1) for i in range(self.channels_in)], axis=-1)
        else:
            # print(x.shape)
            x = tf.transpose(x, perm=[0, 2, 1])
            x = self.Linear(x)
            x = tf.transpose(x, perm=[0, 2, 1])
        x = x + seq_last

        if self.channels_out==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        return x # [Batch, Output length, Channel]

class Linear__Tensorflow(tf.keras.Model):
    """
    Just one Linear layer
    """
    def __init__(self, seq_len, pred_len, channels_in, channels_out, individual, seed=941):
        super(Linear__Tensorflow, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.seed=seed

        if self.individual:
            self.Linear = [Dense(units=self.pred_len, kernel_initializer=GlorotUniform(seed=self.seed)) for _ in range(self.channels_in)]
        else:
            self.Linear = Dense(units=self.pred_len, kernel_initializer=GlorotUniform(seed=self.seed))
        self.final_layer = Dense(self.channels_out, kernel_initializer=GlorotUniform(seed=self.seed))

    def call(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            x = tf.concat([tf.expand_dims(self.Linear[i](x[:,:,i]), axis=-1) for i in range(self.channels_in)], axis=-1)
        else:
            x = tf.transpose(x, perm=[0, 2, 1])
            x = self.Linear(x)
            x = tf.transpose(x, perm=[0, 2, 1])
        if self.channels_out==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        return x # [Batch, Output length, Channel]

class LTSF_Linear_Base(TensorflowModel):
    def __init__(self, modelConfigs, input_shape, output_shape, save_dir, seed=941, **kwargs):
        super().__init__(modelConfigs=modelConfigs, 
                         input_shape=input_shape, 
                         output_shape=output_shape,
                         save_dir=save_dir,
                         seed=seed)
        self.individual = self.modelConfigs['individual']
        self.channels_in = input_shape[-1]
        self.channels_out = output_shape[-1] if isinstance(output_shape, int) != 1 else 1

        assert self.channels_in >= self.channels_out, 'check the inputs shape'

    def save(self, file_name:str):
        file_path = os.path.join(self.path_weight, file_name, "ckpt")
        self.model.save_weights(Path(file_path).absolute())
        # with open(os.path.join(self.path_architecture, f'{file_name}.json') , 'w') as outfile: json.dump(self.model.to_json(), outfile, indent=4)
        return file_path

    def callbacks(self, patience, min_delta=0.001, extension=''):
        return super().callbacks(patience=patience, min_delta=min_delta, extension="", best_weight_add='.index')

class LTSF_Linear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = Linear__Tensorflow(seq_len=self.input_shape, 
                                        pred_len=self.output_shape, 
                                        channels_in=self.channels_in, 
                                        channels_out=self.channels_out,  
                                        individual=self.individual)

class LTSF_NLinear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = NLinear__Tensorflow(seq_len=self.input_shape, 
                                         pred_len=self.output_shape, 
                                         channels_in=self.channels_in, 
                                         channels_out=self.channels_out, 
                                         individual=self.individual)

class LTSF_DLinear__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = DLinear__Tensorflow(seq_len=self.input_shape, 
                                         pred_len=self.output_shape, 
                                         channels_in=self.channels_in, 
                                         channels_out=self.channels_out, 
                                         individual=self.individual)

class LTSF_NDLinearTime2VecRevIN__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = NDLinearTime2VecRevIN__Tensorflow(seq_len=self.input_shape, 
                                                       pred_len=self.output_shape, 
                                                       channels_in=self.channels_in, 
                                                       channels_out=self.channels_out, 
                                                       individual=self.individual)

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Add
from keras.initializers import GlorotUniform

class ResLSTM(tf.keras.Model):
    def __init__(self, seq_len, pred_len, channels_in, channels_out, individual, activation=None, seed=941):
        super(ResLSTM, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.seed = seed

        # Define the LSTM layer with 128 units
        self.lstm = LSTM(48, return_sequences=True, kernel_initializer=GlorotUniform(seed=self.seed))
        
        # Define three fully connected (Dense) layers
        self.fc1 = Dense(184, activation='sigmoid', kernel_initializer=GlorotUniform(seed=self.seed))
        self.fc2 = Dense(46, activation='sigmoid', kernel_initializer=GlorotUniform(seed=self.seed))
        self.fc3 = Dense(self.pred_len, activation=None, kernel_initializer=GlorotUniform(seed=self.seed))
        self.fc4 = Dense(self.pred_len, activation=None, kernel_initializer=GlorotUniform(seed=self.seed))
        
        # Define the final Dense layer for prediction
        self.final_layer = Dense(self.channels_out, kernel_initializer=GlorotUniform(seed=self.seed))

    def call(self, inputs):
        # batch, seq_len, channels
        inputs = tf.transpose(inputs, perm=[0, 2, 1]) 
        # Forward pass through LSTM
        lstm_out = self.lstm(inputs)
        lstm_out = self.fc4(lstm_out)
        
        # Forward pass through fully connected layers
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # Add skip connection
        # print(f'{x.shape = }, 0')
        x = Add()([lstm_out, x])
        
        # Output layer for prediction
        x = tf.transpose(x, perm=[0, 2, 1])
        # print(f'{x.shape = }, 1')
        if self.channels_out==1: x = tf.squeeze(self.final_layer(x), axis=-1)
        # print(f'{x.shape = }, 2')
        return x

class ResLSTM__Tensorflow(LTSF_Linear_Base):
    def build(self):
        self.model = ResLSTM(seq_len=self.input_shape, 
                             pred_len=self.output_shape, 
                             channels_in=self.channels_in, 
                             channels_out=self.channels_out, 
                             individual=self.individual)
        # print(self.model.summary)
        # exit()
