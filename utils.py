import numpy as np
import pandas as pd
import random
import pywt
from PIL import Image
from glob import glob
import re
from scipy.signal import find_peaks
import math


def read_label(df_path):
    """
    Reads the labels of each house from the labels.dat files and returns a dictionary where the keys are the house numbers 
    and the values are dictionaries where the keys are the appliance numbers and the values are the names of the appliances.
    
    Args:
        df_path (str): The path to the directory containing the labels files.
        
    Returns:
        dict: A dictionary where the keys are the house numbers and the values are dictionaries where the keys are the 
        appliance numbers and the values are the names of the appliances. Example: {1: {1: 'mains_1', 2: 'mains_2', 3: 'oven_3', 4: 'oven_4', 5: 'refrigerator_5', ...}}
    """
    label = {}
    for i in range(1, 7):
        hi = str(df_path/'low_freq/house_{}/labels.dat').format(i)
        label[i] = {}
        with open(hi) as f:
            for line in f:
                splitted_line = line.split(' ')
                label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
    return label


def read_merge_data(df_path, labels, house):
    """
    Reads the data from the channel files of a specific house and merges them into a single DataFrame. The function takes 
    three arguments: `df_path`, the path to the directory containing the data files; `labels`, a dictionary containing the 
    labels of each appliance in each house; and `house`, the number of the house to read the data from.
    
    Args:
        df_path (str): The path to the directory containing the data files.
        labels (dict): A dictionary containing the labels of each appliance in each house.
        house (int): The number of the house to read the data from.
        
    Returns:
        pandas.DataFrame: A DataFrame containing the data from all the channel files of the specified house, merged into 
        a single DataFrame.
    """
    path = df_path/'low_freq/house_{}/'.format(house)
    file = str(path/'channel_1.dat')
    df = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][1]], 
                                       dtype = {'unix_time': 'int64', labels[house][1]:'float64'}) 
    
    num_apps = len(glob(str(path/'channel*')))
    for i in range(2, num_apps + 1):
        file = str(path/'channel_{}.dat'.format(i))
        data = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][i]], 
                                       dtype = {'unix_time': 'int64', labels[house][i]:'float64'})
        df = pd.merge(df, data, how = 'inner', on = 'unix_time')
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df = df.set_index(df['timestamp'].values)
    df.drop(['unix_time','timestamp'], axis=1, inplace=True)
    return df


def get_label_pairs(labels):
    """
    Takes a dictionary of labels as input and returns a dictionary where the keys are the building numbers and the values 
    are lists of tuples. Each tuple contains the name of an appliance and its corresponding class label.
    
    Args:
        labels (dict): A dictionary where the keys are the building numbers and the values are dictionaries where the keys 
        are the appliance numbers and the values are the names of the appliances.
        
    Returns:
        dict: A dictionary where the keys are the building numbers and the values are lists of tuples. Each tuple contains 
        the name of an appliance and its corresponding class label. Example: {1: [('oven_3', 'oven'), ('oven_4', 'oven'), ('refrigerator_5', 'refrigerator'), ...]}
    """
    label_pairs = {}
    for building in labels.keys():
        building_labels = [re.match(r'(^(.+)\_\d+$)', l).groups() for l in labels[building].values() if not l.startswith('mains')]
        label_pairs[building] = building_labels
    return label_pairs


def get_min_meter_active(df, labels, label_pairs, threshold_bias=0.7):
    """
    Calculates the minimum power consumption of meter to consider active for each appliance in the dataset.

    Args:
        df (pandas.DataFrame): The dataset containing the power consumption data.
        labels (dict): A dictionary where the keys are the building numbers and the values are the labels for each appliance.
        label_pairs (dict): A dictionary where the keys are the building numbers and the values are lists of tuples where the 
        first element is the name of an appliance and the second element is the corresponding class label for that appliance.
        threshold_bias (float): A value between 0 and 1 that determines the threshold for detecting active power, the bias helps
        to avoid false positives. The default value is 0.7.

    Returns:
        dict: A dictionary where the keys are the class labels for each appliance and the values are the minimu power consumption
        of meter to consider active for that appliance. Example: {'oven': 1148.7541666666666, 'refrigerator': 114.08638776614207, ...}
    """
    min_meter_active = {}
    for building in labels.keys():
        dfb = df[building]
        for key, label in label_pairs[building]:
            peaks, _ = find_peaks(dfb[key], distance=1, height=25)
            threshold = np.mean(dfb[key][peaks])*threshold_bias
            if not math.isnan(threshold):
                min_meter_active.setdefault(label,[]).append(threshold)
    for key in min_meter_active.keys():
        min_meter_active[key] = np.mean(min_meter_active[key])
    return min_meter_active


def append_arr(a, b):
    """
    Appends two numpy arrays vertically.

    Args:
        a (numpy.ndarray): The first array to append.
        b (numpy.ndarray): The second array to append.

    Returns:
        numpy.ndarray: The vertically stacked array.
    """
    if len(a) == 0: return b
    else: return np.vstack((a, b))


def get_image_array_from_cwtmatr(cwtmatr):
    """
    Converts a continuous wavelet transform matrix to an image array.

    Args:
        cwtmatr (numpy.ndarray): The continuous wavelet transform matrix.

    Returns:
        numpy.ndarray: The image array.
    """
    sca = np.abs(cwtmatr)
    img = np.interp(sca, (sca.min(), sca.max()), (0, 255)).astype(np.uint8)
    return img


def save_wavelets(swt_data):
    """
    Saves the continuous wavelet transform matrices of two signals to a file.

    Args:
        swt_data (tuple): A tuple containing the two signals to transform, the destination file path, and the window size.

    Returns:
        None
    """
    crop_mains1, crop_mains2, dst_file, window_size = swt_data
    cwtmatr1, _ = pywt.cwt(crop_mains1, np.arange(1, window_size), 'morl', 1)
    cwtmatr2, _ = pywt.cwt(crop_mains2, np.arange(1, window_size), 'morl', 1)
    save_wavelet(cwtmatr1, cwtmatr2, dst_file)


def save_wavelet(cwtmatr1, cwtmatr2, path):
    """
    Saves two continuous wavelet transform matrices as an image file.

    Args:
        cwtmatr1 (numpy.ndarray): The first continuous wavelet transform matrix.
        cwtmatr2 (numpy.ndarray): The second continuous wavelet transform matrix.
        path (str): The file path to save the image.

    Returns:
        PIL.Image: The saved image.
    """
    imarr1 = get_image_array_from_cwtmatr(cwtmatr1)
    imarr2 = get_image_array_from_cwtmatr(cwtmatr2)
    imarr = np.concatenate((imarr1, imarr2), axis=0)
    img = Image.fromarray(imarr)
    img.save(path)
    return img


def encode_labels(min_meter_active, labels):
    """
    Encode the labels as one-hot arrays.

    Args:
    - min_meter_active: dictionary of minimum meter activity
    - labels: string of labels separated by spaces

    Returns:
    - encoded_labels: numpy array of encoded labels as one-hot arrays.
    """
    encoded_labels = np.zeros(len(min_meter_active))
    for label in labels.split(' '):
        if label in list(min_meter_active.keys()):
            label_idx = list(min_meter_active.keys()).index(label)
            encoded_labels[label_idx] = 1
    return encoded_labels


def roll_meter(meter, interval=(200, 400)):
    """
    Roll the meter in the time series by a random interval.

    Args:
    - meter: numpy array containing the meter data
    - interval: tuple containing the range of the random interval (default=(200, 400))

    Returns:
    - rolled_meter: numpy array containing the rolled meter data
    """
    rolled_meter = np.roll(meter, int(random.randrange(*interval)))
    return rolled_meter


def get_valid_labels(dst_path, valid_idx):
    """
    Get the set of valid labels for a specific validation index.

    Args:
    - valid_idx: integer of the validation index

    Returns:
    - set of valid labels
    """
    labels = []
    df = pd.read_csv(dst_path/f'data-split-valid{valid_idx}.csv')
    for l in df[df.is_valid==True].labels:
        labels.extend(l.split(' '))
    return set(labels)

