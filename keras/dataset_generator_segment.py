"""
Data generator for segment to label problem.
Each segment is padded to length 6000, and labels are one-hot encoded.
"""
import numpy as np
from keras.utils import Sequence, to_categorical
from keras.preprocessing.sequence import pad_sequences

import os.path
DIR_PATH = ''


"""
Generator for training and validation.
"""
class BreakfastActionTrainDataGenerator(Sequence):
    def __init__(self, segment_ids, labels, batch_size=100, input_dim=400, output_dim=48, shuffle=True):
        self.segment_ids = segment_ids
        self.labels = labels
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.ceil(len(self.segment_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of train/validation data.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        segment_ids_temp = [self.segment_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(segment_ids_temp)

        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.segment_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, segment_ids_temp):
        """
        Generates data containing batch_size samples.
        """
        x = []
        y = []

        # Generate data
        for i, ID in enumerate(segment_ids_temp):
            # Store sample
            curr_path = os.path.join(DIR_PATH, 'data/segments/', (ID + '.npy'))
            x.append(np.load(curr_path))

            # Store class
            y.append([self.labels[ID]])

        padded_x = pad_sequences(x, padding='post', maxlen=6000, truncating='post', value=0.0)
        encoded_y = np.array([to_categorical(i, self.output_dim) for i in y])
        return padded_x, encoded_y


"""
Generator for testing.
"""
class BreakfastActionTestDataGenerator(Sequence):
    def __init__(self, segment_ids, batch_size=100, input_dim=400):
        self.segment_ids = segment_ids
        self.indexes = np.arange(len(self.segment_ids))
        self.batch_size = batch_size
        self.input_dim = input_dim

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.ceil(len(self.segment_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates one batch of test data.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        segment_ids_temp = [self.segment_ids[k] for k in indexes]

        # Generate data
        x = self.__data_generation(segment_ids_temp)

        return x

    def __data_generation(self, segment_ids_temp):
        """
        Generates data containing batch_size samples.
        """
        x = []

        # Generate data
        for i, ID in enumerate(segment_ids_temp):
            # Store sample
            curr_path = os.path.join(DIR_PATH, 'data/segments/', (ID + '.npy'))
            x.append(np.load(curr_path))

        padded_x = pad_sequences(x, padding='post')
        return padded_x