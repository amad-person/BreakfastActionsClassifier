import os
import csv
import argparse
from pathlib import Path

import numpy as np
from keras.models import load_model 

from dataset_generator_segment import BreakfastActionTrainDataGenerator, BreakfastActionTestDataGenerator
from utils import read_dict


DIR_PATH = ''
PARTITION_PATH = os.path.join(DIR_PATH, 'data/frame_partition.csv')
TEST_FILENAME_TO_SEGMENT_DATA = os.path.join(DIR_PATH, 'data/filename_to_segment_ids.csv')

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

model_name = p.file_path.as_posix().split("runs/", 1)[1] # model name will have the .hdf5 extension
timestr = time.strftime("_%Y%m%d_%H%M%S")

# Save raw predictions
print("Writing predictions...")
prediction_file_path = os.path.join(DIR_PATH, 'results/predictions_' + model_name + timestr + '.npy')
np.save(prediction_file_path, predictions)
print("predictions saved at ", prediction_file_path)

# Get final predictions by voting
prediction_labels = []
test_segment_splits = read_dict(TEST_FILENAME_TO_SEGMENT_DATA)
for (test_vid_idx, vid_prediction) in enumerate(predictions):
    frame_prediction_labels = [np.argmax(x) for x in vid_prediction]
    prediction_probs = [np.max(x) for x in vid_prediction]

    # Get segment split for test video
    test_id = 'TESTVID_' + str(test_vid_idx + 1)
    split = test_segment_splits[test_id]
    offset = split[0] # this is needed because we don't store the SIL frames in our video data
    
    tail = 0.15
    reduced_weight = 1

    # Get label for each segment by voting
    for i in range(len(split) - 1):
        start_idx = split[i] - offset
        end_idx = split[i + 1] - offset
        segment_labels = np.array(frame_prediction_labels[start_idx:end_idx])
        prediction_probs_section = np.array(prediction_probs[start_idx:end_idx])

        # simple weighted voting
        # for unweighted voting remove weights param from np.bincount
        # most_common_label = np.argmax(np.bincount(segment_labels, weights=prediction_probs_section))

        # soft voting with probabilties
        weights = np.ones_like(segment_labels, dtype=float) * 2
        weights[0:int(tail*len(segment_labels))] = reduced_weight
        weights[int((1-tail)*len(segment_labels)):len(segment_labels)] = reduced_weight

        vid_predictions_section = vid_prediction[start_idx:end_idx]
        vid_predictions_section_t = vid_predictions_section.T
        weighted_sums = np.dot(vid_predictions_section_t, weights)
        
        most_common_label = np.argmax(weighted_sums)
        prediction_labels.append(most_common_label)


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