
# ByteRCNN Model for Classification

This project implements a Byte-level Recurrent Convolutional Neural Network (ByteRCNN) for classification tasks using TensorFlow and Keras. The model combines RNN and CNN layers to effectively process sequential data. This repository includes scripts for training, evaluating, and testing the model, along with data loaders and utility functions.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

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
   git clone https://github.com/yourusername/bytercnn.git
   cd bytercnn
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### GUI
simply run:
```
python gui.py
```

Or you can run everything mannually
### Training

To train the ByteRCNN model, use the \`train\` command. You need to specify the paths to your training and validation datasets, along with other hyperparameters.

```sh
python train.py train \
  --train_data_path "path/to/train_data.h5" \
  --val_data_path "path/to/val_data.h5" \
  --scenario_to_run 1 \
  --maxlen 4096 \
  --lr 0.001 \
  --embed_dim 16 \
  --batch_size 200 \
  --kernels 9 27 40 65 \
  --cnn_size 128 \
  --rnn_size 64 \
  --epochs 30 \
  --output "output_directory/"
```

### Evaluation

To evaluate a trained model on a test dataset, use the \`evaluate\` command. Specify the path to the test dataset and the saved model.

```sh
python train.py evaluate \
  --test_data_path "path/to/test_data.h5" \
  --scenario_to_run 1 \
  --maxlen 4096 \
  --batch_size 200 \
  --output "output_directory/" \
  --model_path "path/to/saved_model"
```

### Testing

To test the model on an additional dataset (e.g., OLAB dataset), use the \`test\` command. Specify the path to the OLAB dataset and the saved model.

```sh
python train.py test \
  --olab_data_path "path/to/olab_data/" \
  --scenario_to_run 1 \
  --maxlen 4096 \
  --batch_size 200 \
  --output "output_directory/" \
  --model_path "path/to/saved_model"
```

## Project Structure

```
byttercnn/
│
├── bytercnn_models.py        # Contains the ByteRCNN model definition
├── data_loader.py            # Data loader and generator classes
├── train.py                  # Main script to train, evaluate, and test the model
├── utility.py                # Utility functions
├── requirements.txt          # List of required Python packages
└── README.md                 # This README file
```

### bytercnn_models.py
Defines the ByteRCNN model and a function to load the model.

### data_loader.py
Contains the \`NPZBatchGenerator\` class for loading batches of data and the \`ByteRCNNDataLoader\` class for handling different data formats.

### train.py
Main script to handle training, evaluation, and testing of the ByteRCNN model.

### utility.py
Includes utility functions such as converting data formats and calculating accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify this README to better suit your project's specifics. If you have any questions or run into issues, please open an issue on the GitHub repository. Happy coding!
