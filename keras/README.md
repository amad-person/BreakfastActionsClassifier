# Breakfast Actions Classification (Keras Code)

The task is to perform video action classification on the [Breakfast actions dataset](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/). This dataset includes 1712 videos and shows activities related to breakfast preparation.

## Data

The data given to us is stored in the `./breakfast-actions-classifier-data` directory.

- `./data`: The `.gz` files that contain the I3D features for the videos. The video data can be downloaded from [here](https://drive.google.com/drive/folders/1KtpuFYRGXByf_9ICPsCbGRBoR_hLsruh).
- `./splits`: The training-testing split information.
- `./groundTruth`: The ground truth labels for the videos.
- `training_segment.txt`: The segment split information for the training videos.
- `testing_segment.txt`: The segment split information for the testing videos.

## Usage

## Setting up

Main dependencies:

- Python 3.5.2
- Keras: `Keras==2.2.4`
- Tensorflow: `tensorflow==1.12.0`, tensorflow-gpu==1.12.0
- Numpy: `numpy==1.16.3`

The `requirements.txt` file is provided for convenience. Create a virtual environment and install the dependencies.

```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Data processing

Run `create_dataset_frame.py` and `create_dataset_segment.py` to process the raw I3D data files. 

```sh
python3 create_dataset_frame.py
python3 create_dataset_segment.py
```

The following outputs will be stored in the `.data/` directory:

- `./segments`: The `.gz` files that contain the I3D features for the videos.
- `./vids`: The training-testing split information.
- `frame_labels.csv`: Frame labels for the frame to label problem.
- `frame_partition.csv`: Train-validation-test video IDs for the frame to label problem.
- `segment_labels.csv`: Segment labels for the segment to label problem.
- `segment_partition.csv`: Train-validation-test segment IDs for the segment to label problem.
- `segment_lengths.csv`: Helper file to store segment lengths.
- `filename_to_segment_ids`: Helper file to store segment splits according to videos.

### Training

Training scripts for both the frame to label and segment to label problems are provided.

```sh
python3 <train_script_name>.py
```

#### Frame to Label

- `dnn_frame.py`: Trains a baseline DNN.
- `lstm_frame.py`: Trains a bidirectional LSTM.

#### Segment to Label

- `lstm_segment.py`: Trains a bidirectional LSTM with max-pooling.

### Testing

Testing scripts for both the frame to label and segment to label problems are provided.

```sh
python3 <test_script_name>.py [.HDF5_MODEL_FILEPATH]
```

#### Frame to Label

- `test_model_frame.py`: Generates and saves the final prediction based on soft voting on the prediction probabilities.

#### Segment to Label

- `test_model_segment.py`: Generates and saves the final prediction by taking argmax of the prediction probabilities.

## References

[H. Kuehne, A. B. Arslan and T. Serre. The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities. CVPR, 2014.](https://serre-lab.clps.brown.edu/wp-content/uploads/2014/05/paper_cameraReady-2.pdf)

I3D features: [Carreira J, Zisserman A. Quo vadis, action recognition? a new model and the kinetics dataset. IEEE Conference on Computer Vision and Pattern Recognition. 2017](https://arxiv.org/pdf/1705.07750.pdf)