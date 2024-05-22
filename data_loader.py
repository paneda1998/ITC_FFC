import numpy as np
import os
import json
import tensorflow as tf
import h5py

class NPZBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, npz_file_path, batch_size):
        self.file_path = npz_file_path
        self.batch_size = batch_size
        with h5py.File(self.file_path, 'r') as data:
            self.length = len(data['y'])

    def __len__(self):
        return int(np.ceil(self.length / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.length)

        with h5py.File(self.file_path, 'r') as data:
            batch_x = data['x'][start_idx:end_idx]
            batch_y = data['y'][start_idx:end_idx]

        return batch_x.astype('float16'), batch_y.astype('float16')

class ByteRCNNDataLoader():
    def __init__(self, data_type, data_path):
        self.data_type = data_type
        self.data_path = data_path

    def load_npz_data(self, scenario=1, block_size=4096, subset='train'):
        if block_size not in [512, 4096]:
            raise ValueError('Invalid block size!')
        if scenario not in range(1, 7):
            raise ValueError('Invalid scenario!')
        if subset not in ['train', 'val', 'test']:
            raise ValueError('Invalid subset!')

        with np.load(os.path.join(self.data_path, subset, '.npz'), mmap_mode='r') as data:
            if os.path.isfile('labels.json'):
                with open('labels.json') as json_file:
                    classes = json_file.read()
                    labels = classes[str(scenario)]
            else:
                raise FileNotFoundError('Please download classes.json to the current directory!')
            return data['x'], data['y'], labels

    def load_hd5_data(self, scenario=1, block_size=4096, subset='train', batch_size=200):
        return NPZBatchGenerator(os.path.join(self.data_path, subset, '.h5'), batch_size)
