import argparse
from pathlib import Path
from fastprogress.fastprogress import progress_bar
import logging

from utils import read_label, read_merge_data, get_label_pairs, get_min_meter_active
from generate_scalograms import save_scalograms, augment_house

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def main(df_path, dst_path, houses):
    """
    Generates and saves scalograms for the specified valid houses.

    Args:
    - df_path: A string representing the path to the CSV file containing the data to be processed.
    - dst_path: A string representing the path to the directory where the generated scalograms will be saved.
    - houses: A list of integers representing the indices of the houses to be processed as evaluation.

    Returns:
    - None
    """
    # Convert df_path to a Path object and read the labels from the CSV file
    df_path = Path(df_path)
    labels = read_label(df_path)

    # Convert dst_path to a Path object and create the destination path if it doesn't exist
    dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Create a dictionary of DataFrames for each building
    df = {}
    for i in range(1,7):
        df[i] = read_merge_data(df_path, labels, i)

    # Get the label pairs and minimum number of active meters
    label_pairs = get_label_pairs(labels)
    min_meter_active = get_min_meter_active(df, labels, label_pairs)

    # Generate and save scalograms for each valid building
    for valid_building in progress_bar(houses):
        train_buildings = [i for i in range(1,7) if i is not valid_building]
        save_scalograms(df, label_pairs, train_buildings=train_buildings, valid_buildings=[valid_building], dst_path=dst_path, min_meter_active=min_meter_active, window_size=600, shift=200)
        augment_house(df, dst_path, min_meter_active, valid_building)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_path', type=str, default='data.csv', help='Path to the input data folder')
    parser.add_argument('--dst_path', type=str, default='output.csv', help='Path to the output data folder')
    parser.add_argument('--houses', nargs='+', type=int, default=range(1,7), help='List of houses')
    args = parser.parse_args()

    main(args.df_path, args.dst_path, args.houses)

    # example of use
    # python main_generate.py --df_path /home/jossalgon/datasets/NILM/redd/ --dst_path /home/jossalgon/datasets/NILM-scalograms --houses 1 2 3 4 5 6