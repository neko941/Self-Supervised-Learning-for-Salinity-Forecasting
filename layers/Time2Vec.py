import tensorflow as tf
from keras.layers import Layer
from keras.layers import Concatenate
from keras.initializers import RandomUniform

class Time2Vec(Layer):
    def __init__(self, kernel_size, seed=941):
        super(Time2Vec, self).__init__(trainable=True)     
        self.k = kernel_size
        self.seed = seed
    
    def build(self, input_shape):
        self.wb = self.add_weight(
            shape=(input_shape[1], 1),
            initializer=RandomUniform(seed=self.seed),
            trainable=True,
            name='wb_weight'
        )
        
        self.bb = self.add_weight(
            shape=(input_shape[1], 1),
            initializer=RandomUniform(seed=self.seed),
            trainable=True,
            name='bb_weight'
        )
        
        self.wa = self.add_weight(
            shape=(input_shape[-1], self.k),
            initializer=RandomUniform(seed=self.seed),
            trainable=True,
            name='wa_weight'
        )
        
        self.ba = self.add_weight(
            shape=(input_shape[1], self.k),
            initializer=RandomUniform(seed=self.seed),
            trainable=True,
            name='ba_weight'
        )
        
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs):
        bias = self.wb * inputs + self.bb
        wgts = tf.math.sin( tf.matmul(inputs, self.wa) + self.ba)
        return Concatenate(axis=-1)([wgts, bias])
    
    def get_config(self):
        config = super(Time2Vec, self).get_config()
        config.update({'kernel_size': self.k})
        return config