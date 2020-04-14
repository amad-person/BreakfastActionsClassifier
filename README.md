# CS5242 Project

## Results

|                       | Test Accuracy | Notes                 |
|-----------------------|---------------|-----------------------|
| Segment To Label      |               |                       |
| A1. BiLSTM            |    49.610%    | Segment as input      |
| A2. BiGRU             |    64.797%    | Entire video as input |
| A3. Multi-level BiGRU |    **70.015%**    | Entire video as input |
| Frame to Label        |               |                       |
| B1. DNN               |    23.286%    | Entire video as input |
| B2. BiLSTM            |    **52.803%**    | Entire video as input |

## Code

- The `keras` folder contains the code the baseline DNN and frame/segment to label LSTM models.
- The `pytorch` folder contains the code for the best GRU model.

```sh
├── README.md
├── keras
│   ├── README.md
│   ├── breakfast-actions-classifier-data
│   │   ├── README.md
│   │   ├── data
│   │   │   └── README.md
│   │   ├── groundTruth
│   │   ├── splits
│   │   │   ├── mapping_bf.txt
│   │   │   ├── test.split1.bundle
│   │   │   └── train.split1.bundle
│   │   ├── testing_segment.txt
│   │   └── training_segment.txt
│   ├── create_dataset_frame.py
│   ├── create_dataset_segment.py
│   ├── data
│   │   ├── README.md
│   │   ├── filename_to_segment_ids.csv
│   │   ├── frame_labels.csv
│   │   ├── frame_partition.csv
│   │   ├── segment_labels.csv
│   │   ├── segment_lengths.csv
│   │   ├── segment_partition.csv
│   │   ├── segments
│   │   └── vids
│   ├── dataset_generator_frame.py
│   ├── dataset_generator_segment.py
│   ├── dnn_frame.py
│   ├── lstm_frame.py
│   ├── lstm_segment.py
│   ├── requirements.txt
│   ├── results
│   ├── runs
│   │   ├── figures
│   │   └── history
│   ├── test_frame_model.py
│   ├── test_segment_model.py
│   └── utils.py
└── pytorch
```
