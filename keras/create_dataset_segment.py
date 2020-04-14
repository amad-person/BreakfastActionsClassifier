# -*- coding: utf-8 -*-
"""
Creates dataset for the segment to label prediction problem.
Stores the segment data in ./data/segments
Stores the segment labels in ./data/segment_labels.csv
Stores the partition data in ./data/segment_partition.csv
"""
import os
import torch
import numpy as np
import os.path
import csv

from sklearn.model_selection import train_test_split

# Paths for data
DATA_DIR_PATH = './data/'
SEGMENTS_DIR_PATH = os.path.join(DATA_DIR_PATH, 'segments/')
LABELS_DATA = os.path.join(DATA_DIR_PATH, 'segment_labels.csv')
PARTITION_DATA = os.path.join(DATA_DIR_PATH, 'segment_partition.csv')
LENGTH_DATA = os.path.join(DATA_DIR_PATH, 'segment_lengths.csv')
TEST_FILENAME_TO_SEGMENT_DATA = os.path.join(DATA_DIR_PATH, 'filename_to_segment_ids.csv')

# Paths for given segment split data
COMP_PATH = './breakfast-actions-classifier-data/'
TRAINING_SEGMENTS_PATH = os.path.join(COMP_PATH, 'training_segment.txt') 
TESTING_SEGMENTS_PATH = os.path.join(COMP_PATH, 'testing_segment.txt') 

partition_dict = {
    "training": [],
    "validation": [],
    "testing": []
}

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def load_data(split_load, actions_dict, GT_folder, DATA_folder, datatype='training', ):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]  # because first line is #bundle and last line is blank
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

    if datatype == 'training':
        print("==================================================")
        print("CREATING RAW TRAINING DATA")

        data_breakfast = []
        labels_breakfast = []

        # read content of train segment splits
        train_segments_file = open(TRAINING_SEGMENTS_PATH, 'r')
        segment_ids = train_segments_file.read().split('\n')[:-1]  # last line is blank

        training_segment_uids = []

        labels_data_file = open(LABELS_DATA, 'w')
        labels_data_csv_writer = csv.writer(labels_data_file)

        segment_lengths = dict()
        lengths_data_file = open(LENGTH_DATA, 'w')
        lengths_data_csv_writer = csv.writer(lengths_data_file)
        num_segments = 1
        for (idx, content) in enumerate(content_all):
            file_ptr = open(GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]
            label_seq, length_seq = get_label_length_seq(curr_gt)

            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'

            # load data into memory
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            label_curr_video = []
            for iik in range(len(curr_gt)):
                label_curr_video.append(actions_dict[curr_gt[iik]])
            data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
            labels_breakfast.append(label_curr_video)

            # dump (segment, label) into data directory
            curr_segment_ids = segment_ids[idx].split()
            for i in range(len(curr_segment_ids) - 1):
                start_segment_idx = int(curr_segment_ids[i])
                end_segment_idx = int(curr_segment_ids[i + 1]) # exclusive so this is actually (end + 1)

                curr_segment_frames = curr_data[start_segment_idx:end_segment_idx]
                curr_segment_label = label_curr_video[start_segment_idx]

                curr_segment_uid = "TRAINSEG_" + str(num_segments)

                # store training segment id in list
                training_segment_uids.append(curr_segment_uid)

                # save segment numpy file
                curr_segment_full_path = os.path.join(SEGMENTS_DIR_PATH, (curr_segment_uid + '.npy'))
                np.save(curr_segment_full_path, curr_segment_frames)

                # write segment label
                labels_data_csv_writer.writerow([curr_segment_uid, str(curr_segment_label)])

                # save and write segment length
                curr_segment_length = end_segment_idx - start_segment_idx
                segment_lengths[curr_segment_uid] = curr_segment_length
                lengths_data_csv_writer.writerow([curr_segment_uid, str(curr_segment_length)])

                num_segments = num_segments + 1

            print(f'[{idx}] {content} contents dumped')

        # split training ids into 80-20
        dummy_array = [0] * len(training_segment_uids)
        final_training_segment_uids, final_validation_segment_uids, _, _ = train_test_split(training_segment_uids,
                                                                                         dummy_array,
                                                                                         test_size=0.2,
                                                                                         random_state=42)

        partition_dict['training'] = final_training_segment_uids
        partition_dict['validation'] = final_validation_segment_uids

        print("Finished loading the training data and labels!")

        # close files
        train_segments_file.close()
        labels_data_file.close()
        lengths_data_file.close()

        return data_breakfast, labels_uniq

    if datatype == 'test':
        print("==================================================")
        print("CREATING TESTING DATA FILE")

        data_breakfast = []

        # read content of test segment splits
        test_segments_file = open(TESTING_SEGMENTS_PATH, 'r')
        segment_ids = test_segments_file.read().split('\n')[:-1]  # last line is blank

        testing_segment_uids = []
        lengths_data_file = open(LENGTH_DATA, 'a')
        lengths_data_csv_writer = csv.writer(lengths_data_file)
        filename_to_segments_file = open(TEST_FILENAME_TO_SEGMENT_DATA, 'w')
        filename_to_segments_csv_writer = csv.writer(filename_to_segments_file)
        num_segments = 1
        for (idx, content) in enumerate(content_all):
            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'

            # load data into memory
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))

            # dump (segment, label) into data directory
            curr_segment_ids = segment_ids[idx].split()
            curr_file_segment_ids = []
            for i in range(len(curr_segment_ids) - 1):
                start_segment_idx = int(curr_segment_ids[i])
                end_segment_idx = int(curr_segment_ids[i + 1])  # exclusive so this is actually (end + 1)

                curr_segment_frames = curr_data[start_segment_idx:end_segment_idx]

                curr_segment_uid = "TESTSEG_" + str(num_segments)

                # store testing segment id in list
                testing_segment_uids.append(curr_segment_uid)

                # store testing segment id for current file's segment id list
                curr_file_segment_ids.append(curr_segment_uid)

                # save segment numpy file
                curr_segment_full_path = os.path.join(SEGMENTS_DIR_PATH, (curr_segment_uid + '.npy'))
                np.save(curr_segment_full_path, curr_segment_frames)

                # write segment length
                curr_segment_length = end_segment_idx - start_segment_idx
                lengths_data_csv_writer.writerow([curr_segment_uid, str(curr_segment_length)])

                num_segments = num_segments + 1

            filename_to_segments_csv_writer.writerow([content, curr_file_segment_ids])
            print(f'[{idx}] {content} contents dumped')

        partition_dict['testing'] = testing_segment_uids

        print("Finished loading the test data!")

        # close files
        test_segments_file.close()
        filename_to_segments_file.close()
        lengths_data_file.close()

        return data_breakfast


def get_label_bounds(data_labels):
    labels_uniq = []
    labels_uniq_loc = []
    for kki in range(0, len(data_labels)):
        uniq_group, indc_group = get_label_length_seq(data_labels[kki])
        labels_uniq.append(uniq_group[1:-1])
        labels_uniq_loc.append(indc_group[1:-1])
    return labels_uniq, labels_uniq_loc


def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    return label_seq, length_seq


def get_maxpool_lstm_data(cData, indices):
    list_data = []
    for kkl in range(len(indices) - 1):
        cur_start = indices[kkl]
        cur_end = indices[kkl + 1]
        if cur_end > cur_start:
            list_data.append(torch.max(cData[cur_start:cur_end, :],
                                       0)[0].squeeze(0))
        else:
            list_data.append(torch.max(cData[cur_start:cur_end + 1, :],
                                       0)[0].squeeze(0))
    list_data = torch.stack(list_data)
    return list_data


def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]

    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

if __name__ == "__main__":
    COMP_PATH = './breakfast-actions-classifier-data/'
    train_split = os.path.join(COMP_PATH, 'splits/train.split1.bundle')
    test_split = os.path.join(COMP_PATH, 'splits/test.split1.bundle')
    GT_folder = os.path.join(COMP_PATH, 'groundTruth/')
    DATA_folder = os.path.join(COMP_PATH, 'data/')
    mapping_loc = os.path.join(COMP_PATH, 'splits/mapping_bf.txt')

    actions_dict = read_mapping_dict(mapping_loc)

    # split = 'training'
    data_feat, data_labels = load_data(train_split, actions_dict, GT_folder, DATA_folder, datatype=split)

    split = 'test'
    data_feat = load_data(test_split, actions_dict, GT_folder, DATA_folder, datatype=split)

    # save partition data into csv file
    partition_data_file = open(PARTITION_DATA, 'w')
    partition_data_csv_writer = csv.writer(partition_data_file)
    partition_data_csv_writer.writerow(['training', partition_dict['training']])
    partition_data_csv_writer.writerow(['validation', partition_dict['validation']])
    partition_data_csv_writer.writerow(['testing', partition_dict['testing']])