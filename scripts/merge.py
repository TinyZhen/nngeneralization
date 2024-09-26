import ast
import pandas as pd
import os
import glob

def extract_aasr_3(df):
    df['aasr'] = df['aasr'].apply(ast.literal_eval)
    df['std'] = df['std'].apply(ast.literal_eval)
    for index, value in enumerate(df['aasr']):
        if not isinstance(value, list) or len(value) != 3:
            print(f"Entry at index {index} does not have 3 items: {value}, seed: {df['seed'].iloc[index]}")    
    aasr_df = pd.DataFrame(df['aasr'].values.tolist(), index=df.index, columns=['aasr_1', 'aasr_2', 'aasr_3'])
    std_df = pd.DataFrame(df['std'].values.tolist(), index=df.index, columns=['std_1', 'std_2', 'std_3'])
    return aasr_df, std_df

sz = input("Enter size: ")

# Collect all CSV file paths
csv_files = glob.glob("results/threshold_0.2/tmp/*.csv")

# Lists to store DataFrames
base_dfs = []
free_dfs = []

# Loop through the CSV files and read them
for file in csv_files:
    if "base" in file:
        df = pd.read_csv(file) 
        base_dfs.append(df)  # Append to the base list
    elif "free" in file:
        df = pd.read_csv(file) 
        free_dfs.append(df)  # Append to the free list
    else:
        print(file)

# Concatenate the DataFrames after reading all files
if base_dfs:
    base_combined_df = pd.concat(base_dfs, ignore_index=True)
else:
    raise ValueError("No base files found!")

if free_dfs:
    free_combined_df = pd.concat(free_dfs, ignore_index=True)
else:
    raise ValueError("No free files found!")

# Create Excel writer
base_writer = pd.ExcelWriter(f'cor_dataframe_base_{sz}x{sz}.xlsx', engine='xlsxwriter')
print(base_combined_df.head())
# Extract aasr and std
base_aasr_df, base_std_df = extract_aasr_3(base_combined_df)
free_aasr_df, free_std_df = extract_aasr_3(free_combined_df)

# Drop the original 'aasr' and 'std' columns
base_combined_df = base_combined_df.drop(columns=['aasr', 'std','timestamp'])
free_combined_df = free_combined_df.drop(columns=['aasr', 'std','timestamp'])
base_combined_df['test-train'] = base_combined_df['test loss']-base_combined_df['train loss']
free_combined_df['test-train'] = free_combined_df['test loss']-free_combined_df['train loss']

# Concatenate the extracted columns back to the DataFrame
cor_base_df = pd.concat([base_combined_df, base_aasr_df, base_std_df], axis=1)
cor_base_df = cor_base_df.corr()  # Compute correlation
cor_base_df = round(cor_base_df, 3)  # Round to 3 decimal places
cor_base_df.to_excel(base_writer, sheet_name='cor_base')

cor_free_df = pd.concat([free_combined_df, free_aasr_df, free_std_df], axis=1)
cor_free_df = cor_free_df.corr()  # Compute correlation
cor_free_df = round(cor_free_df, 3)  # Round to 3 decimal places
cor_free_df.to_excel(base_writer, sheet_name='cor_free')

# Don't forget to save the Excel file
base_writer.close()
