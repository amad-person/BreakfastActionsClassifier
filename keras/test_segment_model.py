import os
import csv
import argparse
from pathlib import Path

import numpy as np
from keras.models import load_model 

from dataset_generator_segment import BreakfastActionTrainDataGenerator, BreakfastActionTestDataGenerator
from utils import read_dict


DIR_PATH = ''
PARTITION_PATH = os.path.join(DIR_PATH, 'data/segment_partition.csv')

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=Path)
p = parser.parse_args()

# Load model
if (p.file_path.exists()):
    model = load_model(p.file_path.as_posix())
    model.summary()
else:
    exit("The given file path does not exist: ", p.file_path)

# Data generator for test
input_dim = 400
partition = read_dict(PARTITION_PATH)
test_generator = BreakfastActionTestDataGenerator(partition['testing'],
                                                  batch_size=1,
                                                  input_dim=input_dim)

# Predict using model (returns probabilities)
print("Getting predictions...")
predictions = model.predict_generator(test_generator,
                                      use_multiprocessing=True,
                                      workers=4,
                                      verbose=2)

# Save raw predictions
model_name = p.file_path.as_posix().split("runs/", 1)[1] # model name will have the .hdf5 extension
timestr = time.strftime("%Y%m%d_%H%M%S")
print("Writing predictions...")
prediction_file_path = os.path.join(DIR_PATH, 'results/predictions_' + model_name + timestr + '.npy')
np.save(prediction_file_path, predictions)
print("predictions saved at ", prediction_file_path)

# Get final predictions (labels)
prediction_labels = np.argmax(predictions, axis=2)

# Create file according to submission format
print("Writing prediction labels...")
SUBMISSION_PATH = os.path.join(DIR_PATH, 'results/predictions_' + model_name + timestr + '.csv')
with open(SUBMISSION_PATH, 'w', newline='') as submission_file:
    writer = csv.writer(submission_file)
    writer.writerow(["Id", "Category"])
    for (i, label) in enumerate(prediction_labels):
        writer.writerow([i, label[0]])
submission_file.close()
print("Saved predictions to: ", SUBMISSION_PATH)