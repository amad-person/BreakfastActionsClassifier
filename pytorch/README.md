# Breakfast Actions Classifier Data

This folder contains the following:

- `./data`: The `.gz` files that contain the I3D features for the videos, which can be downloaded [here](https://drive.google.com/drive/folders/1KtpuFYRGXByf_9ICPsCbGRBoR_hLsruh?usp=drive_open).
- `./splits`: The training-testing split information.
- `./groundTruth`: The ground truth labels for the videos.
- `training_segment.txt`: The segment split information for the training videos.
- `testing_segment.txt`: The segment split information for the testing videos.
- `dataset_generator.py`: Processes the `.gz` data files and creates the training, validation and test data in the format used by the models in the Jupyter notebooks.
- `A2_bigru_video_model.ipynb`: A Bidirectional GRU model that uses entire videos and segment indices for predicting segment labels.
- `A3_stacked_bigru_video_model.ipynb`: A stacked Bidirectional GRU model that uses entire videos and segment indices for predicting segment labels.
