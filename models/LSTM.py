# import tensorflow as tf
# from tensorflow.keras.layers import Input, LSTM, Dense, Add
# from keras.initializers import GlorotUniform

# class ResLSTM(tf.keras.Model):
#     def __init__(self, seq_len, pred_len, channels_in, channels_out, individual, activation=None, seed=941):
#         super(ResLSTM, self).__init__()
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.individual = individual
#         self.channels_in = channels_in
#         self.channels_out = channels_out
#         self.seed = seed

#         # Define the LSTM layer with 128 units
#         self.lstm = LSTM(48, return_sequences=True)
        
#         # Define three fully connected (Dense) layers
#         self.fc1 = Dense(self.pred_len, activation='sigmoid', kernel_initializer=GlorotUniform(seed=self.seed))
#         self.fc2 = Dense(46, activation='sigmoid', kernel_initializer=GlorotUniform(seed=self.seed))
#         self.fc3 = Dense(23, activation=None, kernel_initializer=GlorotUniform(seed=self.seed))
#         self.fc4 = Dense(23, activation=None, kernel_initializer=GlorotUniform(seed=self.seed))
        
#         # Define the final Dense layer for prediction
#         self.final_layer = Dense(self.channels_out, kernel_initializer=GlorotUniform(seed=self.seed))

#     def call(self, inputs):
#         # Forward pass through LSTM
#         lstm_out = self.lstm(inputs)
#         lstm_out = self.fc4(lstm_out)
        
#         # Forward pass through fully connected layers
#         x = self.fc1(inputs)
#         x = self.fc2(x)
#         x = self.fc3(x)
        
#         # Add skip connection
#         x = Add()([lstm_out, x])
        
#         # Output layer for prediction
#         return self.final_layer(x)

# class ResLSTM__Tensorflow(LTSF_Linear_Base):
#     def build(self):
#         self.model = ResLSTM(seq_len=self.input_shape, 
#                                                        pred_len=self.output_shape, 
#                                                        channels_in=self.channels_in, 
#                                                        channels_out=self.channels_out, 
#                                                        individual=self.individual)