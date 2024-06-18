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


from scipy.io import savemat

#Dataset different version reading and saving is not completed yet!!!

#merging nested lists

#why 100 is not working!

#modular: df begir df khorooji bede

# func e joda bezan baraye tabdil

def process_dataset(df, merge_lables = None, merging_value = None, shuffle = False, select_features = None, 
                    select_classes = None, output_format = 'npz'):
    
    
    # merging all labels (or class names/values) with a given value
    
    # we assumed that merging_values is given as a list and they are going to be replaced with a specific value 
    # named merging value
    
    if merging_value and merging_value:
        df[df.columns[-1]] = df[df.columns[-1]].replace(merge_lables, merging_value)

        # Select only specified features and classes
    if select_features:
        select_features.append(df.columns[-1])
        df = df[select_features]
    
    if select_classes:
        df = df[df[df.columns[-1]].isin(select_classes)]
    
    # Shuffle the dataset if specified
    #oke
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    
    # Convert the DataFrame to the desired format    
    data = df.to_numpy()
    # Saving column names for further creating files
    columns_name = df.columns.to_numpy()
    
    
    if output_format == 'npz':
        
        # Reshape columns_name to (1, len(features)) so it can be stacked as a row
        columns_name_reshaped = columns_name.reshape(1, len(columns_name))
    
        # Stack arr1_reshaped on top of arr2
        data = np.vstack((columns_name_reshaped, data))
        
        # Now result is a (len(data)+1, len(features)) array with arr1 as the first row
        np.savez('output.npz', data=data)
        
        print("Data saved as output.npz")
        
    #should be checked    
    elif output_format == 'mat':
        savemat('output.mat', {'data': data})
        print("Data saved as output.mat")
        
    #should be checked    
    elif output_format == 'dat':
        data.tofile('output.dat')
        print("Data saved as output.dat")
        
    else:
        raise ValueError("Unsupported output format. Choose from 'NPZ', 'MAT', or 'DAT'.")
        
def read_npz(npz_path):
    
    #this function read a npz file and create a dataframe based on the npz file
    #we assumed that npz file consists of a column array following the data like this:
    
    #[['BFD_0' 'BFD_1' 'BFD_2' ... 'MP3_Sync' 'FLAC_Sync' 'Class Label (1']
    #[2.625 2.375 1.5625 ... 0.62521 1.0004 0.0]
    #[2.0625 1.8125 1.125 ... 12.629 6.5026 0.0]
    #...
    #[7.1875 2.0625 2.125 ... 0.37513 0.5002 2.0]
    #[1.5 1.6875 1.4375 ... 0.75025 0.5002 2.0]
    #[1.3125 2.25 1.75 ... 1.3755 1.5006 2.0]]
    
    loaded_data = np.load('output.npz', allow_pickle = True)

    arr_loaded = loaded_data['data']

    df = pd.DataFrame(arr_loaded)

    # Convert the loaded numpy array to a pandas DataFrame
    df.rename(columns = df.iloc[0], inplace = True)

    #dropping column names
    df.drop(df.index[0], inplace = True)

    return df

def concat(df1, df2):
    
    frames = [df1,df2]
    
    result = pd.concat(frames)
    
    return result
