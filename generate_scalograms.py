import numpy as np
import pandas as pd
import random
import os
from scipy.signal import find_peaks
import math
from multiprocessing.dummy import Pool as ThreadPool
import logging

from utils import append_arr, save_wavelets, roll_meter, encode_labels, get_valid_labels


def generate_scalograms(df, label_pairs, building, dst_path, min_meter_active, is_valid, window_size, shift):
    """
    Generates scalograms of the mains data for a specific building and saves them as images. Finds the labels for each scalogram and encodes them as binary vectors.

    Args:
    - df: pandas DataFrame containing the mains data
    - label_pairs: dictionary of label pairs
    - building: building number
    - dst_path: path to save the images
    - min_meter_active: dictionary of minimum meter activity
    - is_valid: boolean indicating whether the data is valid
    - window_size: window size
    - shift: shift value for the sliding window

    Returns:
    - annotations: list of dictionaries containing information about each scalogram,
    example [{'fname': 'house2-sca-0.png', 'labels': 'refrigerator', 'keys': 'refrigerator_9', 'is_valid': False, 'building': 2, 'start': 0, 'end': 600}, ...]
    - X: input data, example [[ 15.71 ...  22.82 ], [ 15.76 ...  22.81 ], ...]
    - y: output data, example [[0. 1. 0. ... 0. 0. 0.], [0. 0. 0. ... 1. 0. 0.], ...]
    """
    
    # Get the mains data for the specified building
    dfb = df[building]
    mains1, mains2 = dfb['mains_1'], dfb['mains_2']
    X, y = np.array([]), np.array([])
    (dst_path/'images').mkdir(parents=True, exist_ok=True)

    annotations = []
    swts_data = []
    
    logging.info(f'Generating scalograms of building {building}...')
    
    # Iterate over the mains data in sliding windows with a given shift
    for i in range(0, len(mains1), shift):
        crop_mains1 = mains1[i:i+window_size]
        crop_mains2 = mains2[i:i+window_size]
        
        # Check if the cropped mains data has the correct window size
        if len(crop_mains1) == window_size and len(crop_mains2) == window_size:
            fname = f'house{building}-sca-{i}.png'
            dst_file = dst_path/'images'/fname
            encoded_labels = np.zeros(len(min_meter_active))
            labels = []
            keys = []
            
            # Add to pool of processes to generate the scalogram image in dst_path/images if it doesn't exist
            if not os.path.isfile(dst_file):
                swts_data.append((crop_mains1, crop_mains2, dst_file, window_size))
            
            # Find the labels for the scalogram
            for key, label in label_pairs[building]:
                meter = dfb[key]
                crop_meter = meter[i:i+window_size]
                peaks, _ = find_peaks(crop_meter, distance=1, height=25)
                mean_peaks = np.mean(crop_meter[peaks]) if len(peaks) > 0 else 0

                # Check if the mean peak value is greater than the minimum meter activity for the label
                if label in min_meter_active and mean_peaks >= min_meter_active[label]:
                    labels.append(label)
                    keys.append(key)
                    label_idx = list(min_meter_active.keys()).index(label)
                    encoded_labels[label_idx] = 1
            
            # Append the input and output data to X and y, respectively
            X = append_arr(X, np.append(crop_mains1, crop_mains2))
            y = append_arr(y, np.array(encoded_labels))
            
            # Append information about the scalogram to the annotations list
            annotations.append({
                'fname': fname,
                'labels': ' '.join(labels) if len(labels) > 0 else 'none',
                'keys': ' '.join(keys) if len(keys) > 0 else 'none',
                'is_valid': is_valid,
                'building': building,
                'start': i,
                'end': i+window_size
            })
    
    logging.info(f'Generating wavelets...')
    # Use multithreading to generate the wavelets, saving the scalogram images in the process
    pool = ThreadPool(8)
    pool.map(save_wavelets, swts_data)
    pool.close()
    pool.join()
    return annotations, X, y
    

def save_scalograms(df, label_pairs, train_buildings, valid_buildings, dst_path, min_meter_active, window_size=600, shift=200):
    """
    Generate scalograms of mains data for train and valid buildings and save them as images.
    This function make use of the generate_scalograms(), itereating over the train and valid buildings.
    Find the labels for each scalogram and encode them as binary vectors.
    Return a tuple containing the annotations, the input data (X), and the output data (y).

    Args:
    - df: pandas DataFrame containing the mains data
    - label_pairs: dictionary of label pairs
    - train_buildings: list of building numbers to use for training
    - valid_buildings: list of building numbers to use for validation
    - path: path to save the images and annotations
    - min_meter_active: dictionary of minimum power consumption for each meter to be considered active
    - window_size: size of the sliding window (default=600)
    - shift: shift value for the sliding window (default=200)

    Returns:
    - annotations_df: pandas DataFrame containing the annotations,
    example [{'fname': 'house2-sca-0.png', 'labels': 'refrigerator', 'keys': 'refrigerator_9', 'is_valid': False, 'building': 2, 'start': 0, 'end': 600}, ...]
    """

    # Initialize empty lists and arrays to store annotations and input/output data
    annotations = []
    X_train, y_train, X_test, y_test = np.array([]), np.array([]), np.array([]), np.array([])
    
    # Loop through each building in the training set
    for building in train_buildings:
        # Generate scalograms for the current building and append the annotations and data to the respective lists/arrays
        building_annotations, building_X, building_y = generate_scalograms(df, label_pairs, building, dst_path, min_meter_active, is_valid=False, window_size=window_size, shift=shift)
        annotations.extend(building_annotations)
        X_train = append_arr(X_train, building_X)
        y_train = append_arr(y_train, building_y)

    # Save the training data to disk
    with open(str(dst_path/f'X_train-split-valid{",".join([str(b) for b in valid_buildings])}.npy'), 'wb') as f:
        np.save(f, X_train)
    with open(str(dst_path/f'y_train-split-valid{",".join([str(b) for b in valid_buildings])}.npy'), 'wb') as f:
        np.save(f, y_train)
    
    # Loop through each building in the validation set
    for building in valid_buildings:
        # Generate scalograms for the current building and append the annotations and data to the respective lists/arrays
        building_annotations, building_X, building_y = generate_scalograms(df, label_pairs, building, dst_path, min_meter_active, is_valid=True, window_size=window_size, shift=shift)
        annotations.extend(building_annotations)
        X_test = append_arr(X_test, building_X)
        y_test = append_arr(y_test, building_y)
    
    # Save the validation data to disk
    with open(str(dst_path/f'X_test-split-valid{",".join([str(b) for b in valid_buildings])}.npy'), 'wb') as f:
        np.save(f, X_test)
    with open(str(dst_path/f'y_test-split-valid{",".join([str(b) for b in valid_buildings])}.npy'), 'wb') as f:
        np.save(f, y_test)
    
    # Convert the annotations list to a pandas DataFrame and save it to disk
    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv(dst_path/f'data-split-valid{",".join([str(b) for b in valid_buildings])}.csv', index=False)

    # Return the annotations DataFrame, input data, and output data as a tuple
    return annotations_df


def generate_daug_scalograms(df, label, positions, valid_buildings, dst_path, min_meter_active, window_size, aug_factor):
    """
    Generate augmented scalograms of the specified label.

    Args:
    - df: pandas DataFrame containing the mains data
    - label: string of the label to generate augmented scalograms for
    - positions: list of dictionaries containing the positions to generate scalograms for
    - valid_buildings: list of integers containing the buildings to generate scalograms for
    - path: string of the path to save the images
    - min_meter_active: dictionary of minimum meter activity
    - window_size: integer of the size of the window
    - aug_factor: integer of the augmentation factor

    Returns:
    - annotation_df_daug: pandas DataFrame containing the annotations for the augmented scalograms
    - X: input data, example [[ 15.71 ...  22.82 ], [ 15.76 ...  22.81 ], ...]
    - y: output data, example [[0. 1. 0. ... 0. 0. 0.], [0. 0. 0. ... 1. 0. 0.], ...]
    """
    # Create a directory for the images
    (dst_path/f'images-split-valid{",".join([str(b) for b in valid_buildings])}').mkdir(parents=True, exist_ok=True)
    
    # Load the annotation data
    annotation_path = str(dst_path/f'data-split-valid{",".join([str(b) for b in valid_buildings])}.csv')
    annotation_df = pd.read_csv(annotation_path)
    training_annotation_df = annotation_df[annotation_df['is_valid'] == False]

    # Create an empty DataFrame to store the augmented annotations
    annotation_df_daug = pd.DataFrame()
    
    swts_data = []
    # Load the input data
    X_path = str(dst_path/f'X_train-split-valid{",".join([str(b) for b in valid_buildings])}.npy')
    X = np.load(X_path)
    X_daug = np.array([])
    # Load the output data
    y_path = str(dst_path/f'y_train-split-valid{",".join([str(b) for b in valid_buildings])}.npy')
    y = np.load(y_path)
    y_daug = np.array([])
    
    # Log the start of the process
    logging.info(f'Generating daug scalograms of label {label} with an aug_factor of {aug_factor}...')

    # Loop through each position
    for pos in positions:        
        for key in pos['keys']:
            # Check if the building is not in the list of valid buildings
            if pos['building'] not in valid_buildings:
                # Get the meter data for the current position
                meter = df[pos['building']].iloc[pos['start']:pos['end']][key]
                # Loop n times for the augmentation factor
                for aug_i in range(aug_factor):
                    # Roll the meter data by a random interval
                    meter_aug = roll_meter(meter)

                    # Get a random sample from the training data
                    random_idx = int(random.random()*len(training_annotation_df))
                    ramdom_sample = training_annotation_df.iloc[random_idx]
                    mains1 = df[ramdom_sample['building']]['mains_1']
                    mains2 = df[ramdom_sample['building']]['mains_2']
                    crop_mains1 = mains1[ramdom_sample['start']:ramdom_sample['end']]
                    crop_mains2 = mains2[ramdom_sample['start']:ramdom_sample['end']]

                    # Choose a random main to add the meter data to, between mains1 and mains2
                    main_idx = random.sample([1,2], k=1)[0]

                    # Create a filename for the image
                    fname = f"house{ramdom_sample['building']}-sca-{ramdom_sample['start']}-daug{key}-main{main_idx}-{aug_i}.png"
                    dst_file = dst_path/f'images-split-valid{",".join([str(b) for b in valid_buildings])}'/fname

                    # Add the meter data to the chosen main
                    if main_idx == 1 and len(crop_mains1) > 0 and len(crop_mains1) == len(meter_aug):
                        crop_mains1_daug = np.add(np.array(crop_mains1), np.array(meter_aug))
                        swts_data.append((crop_mains1_daug, crop_mains2, dst_file, window_size))
                        X_daug = append_arr(X_daug, np.append(crop_mains1_daug, crop_mains2))
                    elif main_idx == 2 and len(crop_mains2) > 0 and len(crop_mains2) == len(meter_aug):
                        crop_mains2_daug = np.add(np.array(crop_mains2), np.array(meter_aug))
                        swts_data.append((crop_mains1, crop_mains2_daug, dst_file, window_size))
                        X_daug = append_arr(X_daug, np.append(crop_mains1, crop_mains2_daug))

                    # Update the labels with the augmented label
                    labels = ramdom_sample['labels'].split(' ')
                    labels.append(label)
                    keys = ramdom_sample['keys'].split(' ')
                    keys.append(key)

                    # Remove the 'none' label if it is present
                    if 'none' in labels:
                        labels.remove('none')
                    if 'none' in keys:
                        keys.remove('none')

                    # Encode the labels as one-hot arrays
                    encoded_labels = encode_labels(min_meter_active, ramdom_sample['labels'])
                    label_idx = list(min_meter_active.keys()).index(label)
                    encoded_labels[label_idx] = 1
                    y_daug = append_arr(y_daug, np.array(encoded_labels))

                    # Add a new row to the annotation data
                    new_row = pd.DataFrame.from_records([{
                        'fname': fname,
                        'labels': ' '.join(labels),
                        'keys': ' '.join(keys),
                        'is_valid': False,
                        'building': ramdom_sample['building'],
                        'start': ramdom_sample['start'],
                        'end': ramdom_sample['end']
                    }])
                    annotation_df_daug = pd.concat([annotation_df_daug, new_row], ignore_index=True)
            
    # Use multithreading to generate the wavelets, saving the scalogram images in the process
    logging.info(f'Generating wavelets...')
    pool = ThreadPool(8)
    pool.map(save_wavelets, swts_data)
    pool.close()
    pool.join()

    # Update the annotation data
    annotation_df = pd.concat([annotation_df, annotation_df_daug], sort=False)
    annotation_df.to_csv(annotation_path.replace('.csv', '-daug.csv'), index=False)
    
    # Save the augmented input and output data
    with open(X_path.replace('.npy', '-daug.npy'), 'wb') as f:
        np.save(f, append_arr(X, X_daug))
    with open(y_path.replace('.npy', '-daug.npy'), 'wb') as f:
        np.save(f, append_arr(y, y_daug))

    # Return the augmented annotation, input and output data
    return annotation_df_daug, X_daug, y_daug


def augment_house(df, dst_path, min_meter_active, idx_to_augment, min_augmentations=3000, max_occurrences=10000):
    """
    This function generates augmented scalograms for all labels that need to be augmented.
    
    Args:
    - df: A pandas DataFrame containing the data to be augmented.
    - min_meter_active: An integer representing the minimum number of active meters required for a label to be considered valid.
    - idx_to_augment: An integer representing the index of the house to be augmented.
    - min_augmentations: An integer representing the minimum number of augmentations to be generated for a label.
    - max_occurrences: An integer representing the maximum number of occurrences of a label to be augmented.
    
    Returns:
    None
    """
    # Get the set of valid labels for the given index
    valid_labels = get_valid_labels(dst_path, idx_to_augment)
    if 'none' in valid_labels:
        valid_labels.remove('none')
    
    # Read in the CSV file and create a dictionary of positions by labels
    split = pd.read_csv(dst_path/f'data-split-valid{idx_to_augment}.csv')
    pos_by_labels = {}
    for _, row in split.iterrows():
        for label in row['labels'].split(' '):
            pos_by_labels.setdefault(label, []).append({
                'building': row['building'],
                'start': row['start'],
                'end': row['end'],
                'keys': [k for k in row['keys'].split(' ') if label in k]
            })
    
    labels_to_augment = []
    # Generate a list of labels to augment and the corresponding augmentation factor
    for label, positions in pos_by_labels.items():
        # Only augment labels that are valid and have less than max occurrences
        if len(positions) < max_occurrences and label in valid_labels:
            labels_to_augment.append((label, math.ceil(min_augmentations/len(positions))))
    logging.info(f'Labels to augment: {labels_to_augment}')
    
    # Read in the CSV and NPY files
    split_daug = pd.read_csv(dst_path/f'data-split-valid{idx_to_augment}.csv')
    X_daug = np.load(dst_path/f'X_train-split-valid{idx_to_augment}.npy')
    y_daug = np.load(dst_path/f'y_train-split-valid{idx_to_augment}.npy')

    # Generate augmented scalograms for each label
    for label, aug_factor in labels_to_augment:
        split_daug_label, X_daug_label, y_daug_label = generate_daug_scalograms(df, label,
                                               pos_by_labels[label],
                                               valid_buildings = [idx_to_augment],
                                               dst_path = dst_path,
                                               min_meter_active=min_meter_active,
                                               window_size = 600,
                                               aug_factor = aug_factor)
        split_daug = pd.concat([split_daug, split_daug_label])
        X_daug = append_arr(X_daug, X_daug_label)
        y_daug = append_arr(y_daug, y_daug_label)

    # Save the augmented CSV and NPY files
    split_daug.to_csv(dst_path/f'data-split-valid{idx_to_augment}-daug.csv', index=False)

    with open(dst_path/f'X_train-split-valid{idx_to_augment}-daug.npy', 'wb') as f:
        np.save(f, X_daug)
    with open(dst_path/f'y_train-split-valid{idx_to_augment}-daug.npy', 'wb') as f:
        np.save(f, y_daug)
