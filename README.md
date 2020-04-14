# CS5242 Project

## Results

|                       | Test Accuracy | Notes                 |
|-----------------------|---------------|-----------------------|
| Segment To Label      |               |                       |
| A1. BiLSTM            |    49.610%    | Segment as input      |
| A2. BiGRU             |    64.797%    | Entire video as input |
| A3. Stacked BiGRU |    **70.482%**    | Entire video as input |
| Frame to Label        |               |                       |
| B1. DNN               |    23.286%    | Entire video as input |
| B2. BiLSTM            |    **52.803%**    | Entire video as input |

## Code

The `best` folder contains the Jupyter notebook with the code and results of our best performing model, A3. Stacked BiGRU. The model alone is shown in 
`best_model(A3_stacked_bigru).py`.

For reference, we also provide the other models discussed in our report.
- The `keras` folder contains the code the baseline DNN and frame/segment to label LSTM models (B1., B2., A1.).
- The `pytorch` folder contains the Jupyter notebooks with the code for the GRU models (A2. and A3.), and the python script used for dataset generation.

```sh
.
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
│   │   ├── A1_lstm_segment_test_predictions.csv
│   │   ├── B1_dnn_frame_test_predictions.csv
│   │   └── B2_lstm_frame_test_predictions.csv
│   ├── runs
│   │   ├── figures
│   │   └── history
│   ├── test_frame_model.py
│   ├── test_segment_model.py
│   └── utils.py
└── pytorch
```
