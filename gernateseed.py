import random
import os
import glob
import pandas as pd

def get_subfolder_names(directory):
    # List comprehension to get directories only
    subfolders = [int(f.name) for f in os.scandir(directory) if f.is_dir()]
    return subfolders

# Example usage
# directory = "log/base/cifar100/"
# subfolders = get_subfolder_names(directory)
# print(subfolders)


def getseeds():
    seeds = [random.randint(0, 2**32 - 1)]

    print(seeds)
def find_average():
    file_path = 'results/Cifar_100/all_3/wrn/*.xlsx'  # Adjust this path
    # Read all Excel files and store them in a list
    all_dataframes = []
    all_dataframes = [pd.read_excel(file,sheet_name="cor_base_2", index_col=0) for file in glob.glob(file_path)]
    print(all_dataframes)

    sum_df = sum(all_dataframes)  # Element-wise sum of all DataFrames
    average_df = sum_df / len(all_dataframes) 


    print("Average values:")
    print(average_df)
    average_df.to_excel('cor_average_base_all_1.xlsx',index=False)
# find_average()


def check_sheet_names():
    file_path = 'results/Cifar_10/wrn/*.xlsx'  # Adjust this path

    # Loop through all Excel files
    for file in glob.glob(file_path):
        try:
            # Use pd.ExcelFile to get the sheet names
            xls = pd.ExcelFile(file)
            print(f"File: {file} - Sheet names: {xls.sheet_names}")
        except Exception as e:
            print(f"Could not read {file}: {e}")

# Run the function
# check_sheet_names()

import torch

# Load the saved model
model = torch.load('cifar100_4232357360_vgg_2_all_3_ts_0.2.pytorch')
for name, param in model['state_dict'].items():  # or use the key in checkpoint that stores the model
    if 'fc' in name and 'weight' in name:  # look for FC layer weights specifically
        print(f"{name}: Number of neurons = {param.shape[0]} (output size), input size = {param.shape[1]}")
# Print out the model's structure to inspect layer parameters
# print(model)