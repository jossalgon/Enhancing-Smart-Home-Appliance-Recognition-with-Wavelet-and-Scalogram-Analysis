# Enhancing-Smart-Home-Appliance-Recognition-with-Wavelet-and-Scalogram-Analysis

Repository of the paper "Enhancing Smart Home Appliance Recognition with Wavelet and Scalogram Analysis". This repository contains the code for the experiments and the dataset used in the paper.

## Requirements:
1. Install conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
2. Install the requirements using the command:
```
conda env create -f environment.yml
```

## How to use the code

1. Download the dataset REDD, available at: http://redd.csail.mit.edu/.
2. Generate the scalogram images using the script main_generate.py. An example of the command is:
```
python main_generate.py --df_path /home/jossalgon/datasets/NILM/redd/ --dst_path /home/jossalgon/datasets/NILM-scalograms --houses 1 2 3 4 5 6
```
3. Run the training scripts.
