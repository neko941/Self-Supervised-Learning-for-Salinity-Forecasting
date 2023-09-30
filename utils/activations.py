from keras import backend as K
from keras.layers.core import Activation
from keras.utils import get_custom_objects

activation_dict = {
                    'xsinsquared': Activation(lambda x: x + (K.sin(x)) ** 2),
                    'xsin': Activation(lambda x: x + (K.sin(x))),
                    'snake_a.5': Activation(lambda x: SnakeActivation(x=x, alpha=0.5)),
                    'snake_a1': Activation(lambda x: SnakeActivation(x=x, alpha=1)),
                    'snake_a5': Activation(lambda x: SnakeActivation(x=x, alpha=5)),
                    # 'srs_a5_b3': Activation(lambda x: _SoftRootSign(x=x, alpha=5.0, beta=3.0)),
                  }

def get_custom_activations():
    get_custom_objects().update(activation_dict)  

def SnakeActivation(x, alpha: float = 0.5):
    return x - K.cos(2*alpha*x)/(2*alpha) + 1/(2*alpha)
