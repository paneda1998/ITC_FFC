
# ITC_FFC

## Overview

This repository contains code for training, testing, and cross-validating deep learning and machine learning models on various datasets. The supported models include:
- Byte RCNN
- Decision Tree
- Random Forest



## Requirements

- Python 3.7+
- TensorFlow 2.4+
- NumPy
- pandas
- scikit-learn
- h5py
- matplotlib
- argparse

## Installation

1. Clone the repository:
   ```sh
   https://github.com/paneda1998/ITC_FFC.git
   cd ITC_FFC
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Configuration
Before running the scripts, ensure you have a configuration file for your model. Below is an example configuration file for a Byte RCNN model:
```
{
  "model_type": "byte_rcnn",
  "scenario_to_run": "scenario_1",
  "maxlen": 100,
  "batch_size": 32,
  "output": "output_dir",
  "train_data_path": "path/to/train_data",
  "val_data_path": "path/to/val_data",
  "test_data_path": "path/to/test_data",
  "olab_data_path": "path/to/olab_data",
  "lr": 0.001,
  "embed_dim": 128,
  "kernels": [3, 4, 5],
  "cnn_size": 64,
  "rnn_size": 128,
  "epochs": 10,
  "checkpoint_dir": "path/to/checkpoints",
  "k_folds": 5
}

```

### Training

To train a model, use the following command:
```sh
python train.py train path/to/config.json
```


### Testing

To test a trained model, use the following command:


```sh
python train.py test path/to/config.json
```

### Cross-Validation


To perform k-fold cross-validation, use the following command:

```sh
python train.py cross_validate path/to/config.json
```

## Explanation of Configuration Parameters
- model_type: The type of model to train (byte_rcnn, decision_tree, random_forest).
- scenario_to_run: A scenario identifier for the current run.
- maxlen: Maximum length of input sequences (specific to Byte RCNN).
- batch_size: Batch size for training.
- output: Directory to save model outputs and reports.
- train_data_path: Path to the training data.
- val_data_path: Path to the validation data.
- test_data_path: Path to the test data.
- olab_data_path: Path to any additional data (specific to Byte RCNN).
- lr: Learning rate (specific to Byte RCNN).
- embed_dim: Embedding dimension (specific to Byte RCNN).
- kernels: List of kernel sizes for convolutional layers (specific to Byte RCNN).
- cnn_size: Number of filters in convolutional layers (specific to Byte RCNN).
- rnn_size: Number of units in recurrent layers (specific to Byte RCNN).
- epochs: Number of epochs for training.
- checkpoint_dir: Directory to save checkpoints during training.
- k_folds: Number of folds for k-fold cross-validation.

## Output
During training and testing, the following files will be generated in the output directory:

- byte_rcnn_model.h5: Trained Byte RCNN model (for Byte RCNN).
- decision_tree_model.pkl: Trained Decision Tree model (for Decision Tree).
- random_forest_model.pkl: Trained Random Forest model (for Random Forest).
- classification_report.csv: Classification report.
- confusion_matrix.csv: Confusion matrix.


For cross-validation, additional files for each fold will be generated:

- byte_rcnn_model_foldX.h5: Trained Byte RCNN model for fold X.
- classification_report_foldX.csv: Classification report for fold X.
- confusion_matrix_foldX.csv: Confusion matrix for fold X.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Contact
For questions or comments, please contact baneshi.alireza@gmail.com.

