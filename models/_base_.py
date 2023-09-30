import os
from abc import abstractmethod
import json
import time

import tensorflow as tf

from keras.optimizers import SGD
from keras.optimizers import Ftrl
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import Adadelta
from tensorflow.keras.optimizers.experimental import AdamW

from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from keras.losses import MeanSquaredError
from keras.models import Sequential 
from keras.layers import Input

from utils.visuals import save_plot
from utils.metrics import score

from utils.general import convert_seconds
import numpy as np
from utils.general import yaml_load
from pathlib import Path
from pathlib import Path
from tensorflow.keras.utils import Sequence 

class BaseModel:
    def __init__(self, modelConfigs, save_dir='.'):
        self.history = None
        self.time_used = '0s'
        self.model = None
        self.modelConfigs = yaml_load(modelConfigs)

        self.dir_log          = 'logs'
        self.dir_plot         = 'plots'
        self.dir_value        = 'values'
        self.dir_model        = 'models'
        self.dir_weight       = 'weights'
        self.dir_architecture = 'architectures'
        self.mkdirs(path=save_dir)

        self.best_weight = None

    def mkdirs(self, path):
        path = Path(path)
        self.path_log          = path / self.dir_log
        self.path_plot         = path / self.dir_plot
        self.path_value        = path / self.dir_value
        self.path_model        = path / self.dir_model
        self.path_weight       = path / self.dir_weight
        self.path_architecture = path / self.dir_architecture

        for p in [self.path_log, self.path_plot, self.path_value, self.path_model, self.path_weight, self.path_architecture]: 
            p.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build(self, *inputs):
        raise NotImplementedError 

    @abstractmethod
    def preprocessing(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def save(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def load(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *inputs):
        raise NotImplementedError

    def plot(self, save_dir, y, yhat, dataset):
        try:
            save_plot(filename=os.path.join(self.path_plot, f'{self.__class__.__name__}-{dataset}.png'),
                      data=[{'data': [range(len(y)), y],
                             'color': 'green',
                             'label': 'y'},
                            {'data': [range(len(yhat)), yhat],
                             'color': 'red',
                             'label': 'yhat'}],
                      xlabel='Sample',
                      ylabel='Value')
        except: pass

    def score(self, 
              y, 
              yhat, 
              scaler=None,
              r=-1):
        return score(y=y, 
                     yhat=yhat, 
                     scaler=scaler,
                     r=r)

class TensorflowModel(BaseModel):
    def __init__(self, modelConfigs, input_shape, output_shape, save_dir, seed=941, **kwargs):
        super().__init__(modelConfigs=modelConfigs, save_dir=save_dir)
        self.function_dict = {
            'MSE'       : MeanSquaredError,
            'Adam'      : Adam,
            'SGD'       : lambda learning_rate: SGD(learning_rate=learning_rate, clipnorm=1.0),
            'AdamW'     : AdamW,
            'Nadam'     : Nadam,
            'RMSprop'   : RMSprop,
            'Adadelta'  : Adadelta,
            'Adagrad'   : Adagrad,
            'Adamax'    : Adamax,
            'Ftrl'      : Ftrl
        }
        self.units           = self.modelConfigs['units']
        self.activations     = [ele if ele != 'None' else None for ele in self.modelConfigs['activations']]
        self.dropouts        = self.modelConfigs['dropouts']
        self.seed            = seed
        self.input_shape     = input_shape
        self.output_shape    = output_shape
    
    class DataGenerator(Sequence):
        def __init__(self, x, y, batchsz):
            self.x, self.y = x, y
            self.batch_size = batchsz

        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

    def callbacks(self, patience, min_delta=0.00001, extension='.h5', best_weight_add=''):
        if extension != '.h5':
            best = Path(self.path_weight, self.__class__.__name__, 'best')
            last = Path(self.path_weight, self.__class__.__name__, 'last')
        else:
            best = Path(self.path_weight, self.__class__.__name__)
            last = best
        best.mkdir(parents=True, exist_ok=True)
        last.mkdir(parents=True, exist_ok=True)

        self.best_weight = best / f"{self.__class__.__name__}_best{extension}{best_weight_add}"

        return [EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta), 
                ModelCheckpoint(filepath=best / f"{self.__class__.__name__}_best{extension}",
                                save_best_only=True,
                                save_weights_only=True,
                                verbose=0), 
                ModelCheckpoint(filepath=last / f"{self.__class__.__name__}_last{extension}",
                                save_best_only=False,
                                save_weights_only=True,
                                verbose=0),
                ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=patience / 5,
                                  verbose=0,
                                  mode='auto',
                                  min_delta=min_delta * 10,
                                  cooldown=0,
                                  min_lr=0), 
                CSVLogger(filename=os.path.join(self.path_log, f'{self.__class__.__name__}.csv'), separator=',', append=False)]  
        
    def preprocessing(self, 
                      x, 
                      y, 
                      batchsz:int, 
                      buffer_size:int=512):
        data = tf.data.Dataset.from_tensor_slices((x, y))\
                              .shuffle(buffer_size=buffer_size, seed=self.seed, reshuffle_each_iteration=True)\
                              .batch(batchsz)\
                              .cache()\
                              .prefetch(buffer_size=tf.data.AUTOTUNE)
        return data

    def build(self):
        try:
            self.model = Sequential(layers=None, name=self.__class__.__name__)
            # Input layer
            self.model.add(Input(shape=self.input_shape, name='Input_layer'))
            self.body()
            self.model.summary()
        except Exception as e:
            print(e)
            self.model = None

    def fit(self, X_train, y_train, X_val, y_val, patience, learning_rate, epochs, batchsz, min_delta=0.00001, optimizer='Adam', loss='MSE', time_as_int=False, **kwargs):
        start = time.time()
        import datetime
        print(f'{datetime.datetime.now() = }')
        self.model.compile(optimizer=self.function_dict[optimizer](learning_rate=learning_rate), loss=self.function_dict[loss]())
        self.history = self.model.fit(self.DataGenerator(x=X_train, y=y_train, batchsz=batchsz), 
                                      validation_data=self.DataGenerator(x=X_val, y=y_val, batchsz=batchsz),
                                      epochs=epochs, 
                                      callbacks=self.callbacks(patience=patience, min_delta=min_delta))
        self.time_used = time.time() - start
        if not time_as_int:
            self.time_used = convert_seconds(self.time_used)
        loss = self.history.history.get('loss')
        val_loss = self.history.history.get('val_loss')
        if all([len(loss)>1, len(val_loss)>1]):
            save_plot(filename=os.path.join(self.path_plot, f'{self.__class__.__name__}-Loss.png'),
                      data=[{'data': [range(len(loss)), loss],
                              'color': 'green',
                              'label': 'loss'},
                          {'data': [range(len(val_loss)), val_loss],
                              'color': 'red',
                              'label': 'val_loss'}],
                      xlabel='Epoch',
                      ylabel='Loss Value')

    def predict(self, X, save=True, scaler=None):
        yhat = self.model.predict(X, verbose=0)
        if save:
            filename = self.path_value / f'yhat-{self.__class__.__name__}.npy'
            np.save(file=filename, 
                    arr=yhat, 
                    allow_pickle=True, 
                    fix_imports=True)
        return yhat
    
    def save(self, file_name:str, extension:str='.h5'):
        self.model.save_weights(Path(self.path_weight, file_name, f'{file_name}{extension}'))
        with open(os.path.join(self.path_architecture, f'{file_name}.json') , 'w') as outfile: json.dump(self.model.to_json(), outfile, indent=4)
        self.model.save(os.path.join(self.path_model, file_name))
        return os.path.join(self.path_weight, f'{file_name}{extension}')
    
    def load(self, weight):
        import shutil
        if os.path.exists(weight): 
            print('=' * 50)
            print(f'Loading weights => {weight}')
            # print(weight.replace(os.path.basename(weight), 'e'))
            if '.index' in str(weight):
                shutil.copyfile(weight, 
                                str(weight).replace(os.path.basename(weight), 
                                               os.path.basename(weight).replace('.index', '')))
                weight = str(weight).replace(os.path.basename(weight), os.path.basename(weight).replace('.index', ''))
            self.model.load_weights(os.path.normpath(weight))
            print('=' * 50)
