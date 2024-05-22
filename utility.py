import numpy as np
import h5py

def convert_npz_to_hdf5(npz_path, hdf5_path):
    with np.load(npz_path, allow_pickle=True) as data:
        with h5py.File(hdf5_path, 'w') as hf:
            for key in data.keys():
                hf.create_dataset(key, data=data[key])


def acc_calc(row):
    if row.name == 0: return '-'
    correct = int(row[row.name])
    total = sum([int(x) for x in row[1:]])
    try:
        return (correct / total) * 100.0
    except:
        return 0