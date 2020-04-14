# -*- coding: utf-8 -*-
"""
Creates dataset for the frame to label prediction problem.
Stores the video data in ./data/vids
Stores the video frame labels in ./data/frame_labels.csv
Stores the partition data in ./data/frame_partition.csv
"""
import os
import torch
import numpy as np
import os.path
import csv

from sklearn.model_selection import train_test_split

# Paths for data
DATA_DIR_PATH = './data'
VIDEOS_DIR_PATH = os.path.join(DATA_DIR_PATH, 'vids/')
LABELS_DATA = os.path.join(DATA_DIR_PATH, 'frame_labels.csv')
PARTITION_DATA = os.path.join(DATA_DIR_PATH, 'frame_partition.csv')

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

        training_video_uids = []

        train_segments_file = open(TRAINING_SEGMENTS_PATH, 'r')
        segment_ids = train_segments_file.read().split('\n')[:-1]  # last line is blank

        labels_data_file = open(LABELS_DATA, 'w')
        labels_data_csv_writer = csv.writer(labels_data_file)

        num_videos = 1
        for idx, content in enumerate(content_all):
            file_ptr = open(GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]

            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'

            # load data into memory
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            label_curr_video = []

            curr_segment_ids = segment_ids[idx].split()
            start_idx = int(curr_segment_ids[0]) + 1
            end_idx = int(curr_segment_ids[-1])
            
            curr_gt = curr_gt[start_idx:end_idx]
            curr_data = curr_data[start_idx:end_idx]
            
            for iik in range(len(curr_gt)):
                label_curr_video.append(actions_dict[curr_gt[iik]])
            data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
            labels_breakfast.append(label_curr_video)

            curr_video_uid = "TRAINVID_" + str(num_videos)
            training_video_uids.append(curr_video_uid)

            curr_video_full_path = os.path.join(VIDEOS_DIR_PATH, (curr_video_uid + '.npy'))
            np.save(curr_video_full_path, curr_data)

            labels_data_csv_writer.writerow([curr_video_uid, label_curr_video])

            print("training video %d data saved in %s" % (num_videos, curr_video_full_path))
            num_videos += 1

        # split training ids into 80-20
        dummy_array = [0] * len(training_video_uids)
        final_training_video_uids, final_validation_segment_uids, _, _ = train_test_split(training_video_uids,
                                                                                         dummy_array,
                                                                                         test_size=0.2,
                                                                                         random_state=42)

        partition_dict['training'] = final_training_video_uids
        partition_dict['validation'] = final_validation_segment_uids

        print("Finished loading the training data and labels!")

        # close files
        labels_data_file.close()

        return data_breakfast, labels_breakfast

    if datatype == 'test':
        print("==================================================")
        print("CREATING TESTING DATA FILE")

        data_breakfast = []
        testing_video_uids = []
        num_videos = 1

        test_segments_file = open(TESTING_SEGMENTS_PATH, 'r')
        segment_ids = test_segments_file.read().split('\n')[:-1]  # last line is blank

        for idx, content in enumerate(content_all):
            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'

            # load data into memory
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')

            curr_segment_ids = segment_ids[idx].split()
            start_idx = int(curr_segment_ids[0]) + 1
            end_idx = int(curr_segment_ids[-1]) + 1

            curr_data = curr_data[start_idx:end_idx]
            
            data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
            
            curr_video_uid = "TESTVID_" + str(num_videos)

            # add the uid to the list of all test video uids
            testing_video_uids.append(curr_video_uid)

            # save the current video data to a .npy file
            curr_video_full_path = os.path.join(VIDEOS_DIR_PATH, (curr_video_uid + '.npy'))
            np.save(curr_video_full_path, curr_data)

            print("testing video %d data saved in %s" % (num_videos, curr_video_full_path))
            num_videos += 1

        partition_dict['testing'] = testing_video_uids

        print("Finished loading the test data!")

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

    split = 'training'
    data_feat, data_labels = load_data(train_split, actions_dict, GT_folder, DATA_folder, datatype=split)

    split = 'test'
    data_feat = load_data(test_split, actions_dict, GT_folder, DATA_folder, datatype=split)

    # save partition data into csv file
    partition_data_file = open(PARTITION_DATA, 'w')
    partition_data_csv_writer = csv.writer(partition_data_file)
    partition_data_csv_writer.writerow(['training', partition_dict['training']])
    partition_data_csv_writer.writerow(['validation', partition_dict['validation']])
    partition_data_csv_writer.writerow(['testing', partition_dict['testing']])
