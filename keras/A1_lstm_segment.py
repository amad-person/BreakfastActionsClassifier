import os
import time
import csv
import numpy as np
import pickle

import matplotlib.pyplot as plt
from keras import optimizers
from keras import regularizers
from keras.layers import Bidirectional, Dense, LSTM, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from dataset_generator_segment import BreakfastActionTrainDataGenerator, BreakfastActionTestDataGenerator
from utils import read_dict

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K


# Check if program is running on GPU
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(device_lib.list_local_devices())
K.tensorflow_backend._get_available_gpus()

# Set CPU/GPU cores (keras does it automatically)
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
# sess = tf.Session(config=config) 
# keras.backend.set_session(sess)

DIR_PATH = ''
PARTITION_PATH = os.path.join(DIR_PATH, 'data/segment_partition.csv')
SEGMENT_LABELS_PATH = os.path.join(DIR_PATH, 'data/segment_labels.csv')

# Values for model architecture
batch_size = 16  # number of segments for an iteration of training
input_dim = 400  # dimension of an i3D video frame
hidden_dim = 400  # dimension of RNN hidden state
layer_dim = 1  # number of hidden RNN layers
output_dim = 48  # number of sub-action labels
num_epochs = 25

# Define LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True), input_shape=(None, input_dim)))
model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))
model.add(MaxPooling1D(pool_size=6000, strides=None, padding='valid', data_format='channels_last'))
model.add(Dense(output_dim, activation = 'softmax'))

# Values for experimenting with different optimizers
# learning_rate = 0.02
# decay = 0.0001
# weight_decay = 0.005
# momentum = 0.3
# beta_1 = 0.95
# beta_2 = 0.999
# l2_weight_penalty = 0.001

# Define optimizers to experiment with
# adam = optimizers.Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)
# rms_prop = optimizers.RMSprop(lr=learning_rate)
# adagrad = optimizers.Adagrad(lr=learning_rate)
# nesterov_sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True)

# Checkpoint callback
checkpoint_filename = "./runs/segment-lstm-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
callbacks_list = [checkpoint]

# Compile model (use default Adam optimizer)
model.compile('adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# Load indices for training, testing, and validation
partition = read_dict(PARTITION_PATH)

# Load labels
labels = read_dict(SEGMENT_LABELS_PATH)

# Data generators for train/validation
training_generator = BreakfastActionTrainDataGenerator(partition['training'],
                                                       labels=labels,
                                                       batch_size=batch_size,
                                                       input_dim=input_dim,
                                                       output_dim=output_dim,
                                                       shuffle=True)
validation_generator = BreakfastActionTrainDataGenerator(partition['validation'],
                                                         labels=labels,
                                                         batch_size=batch_size,
                                                         input_dim=input_dim,
                                                         output_dim=output_dim,
                                                         shuffle=True)

# Train model
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=True,
                              workers=4,
                              epochs=num_epochs,
                              verbose=1,
                              callbacks=callbacks_list)

# Evaluate model
val_loss, val_acc = model.evaluate_generator(generator=validation_generator,
                                             use_multiprocessing=True,
                                             workers=4,
                                             verbose=1)
print("val_acc: ", val_acc)

# Save model
timestr = time.strftime("%Y%m%d_%H%M%S_")
model_filename = "./runs/final-segment-lstm_" + timestr + str(round(val_acc, 3)) + ".h5"
model.save(model_filename)

# Save history
history_file = open('/runs/history/segment-lstm-history.p', 'wb')
pickle.dump(history.history, history_file)

# Save accuracy plot
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("./runs/figures/final-segment-lstm_" + timestr + str(round(val_acc, 3)) + "_acc" + ".png")

# Save loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("./runs/figures/final-segment-lstm_" + timestr + str(round(val_acc, 3)) + "_loss" + ".png")