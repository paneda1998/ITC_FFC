import numpy as np
import pandas as pd
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

def npz_to_csv(npz_file_path, csv_save_path):
    
    # Load the NPZ file
    with np.load(npz_file_path) as data:
        array = data['data']  # Make sure to use the correct key

    # Convert the NumPy array to a DataFrame
    df = pd.DataFrame(array)

    # Save the DataFrame to a CSV file
    df.to_csv(save_path, index=False)

def csv_to_npz(csv_file_path, npz_save_path):
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Convert the DataFrame to a NumPy array
    data = df.values
    
    # Save the NumPy array to an NPZ file
    np.savez(npz_save_path, data=data)
