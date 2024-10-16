import argparse
import ast
from datetime import datetime
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

def extract_aasr_3(df):
    df['aasr'] = df['aasr'].apply(ast.literal_eval)
    df['std'] = df['std'].apply(ast.literal_eval)
    aasr_df = pd.DataFrame(df['aasr'].values.tolist(), index=df.index, columns=['aasr_1', 'aasr_2', 'aasr_3', 'aasr_4','aasr_5'])
    std_df = pd.DataFrame(df['std'].values.tolist(), index=df.index, columns=['std_1', 'std_2', 'std_3', 'std_4','std_5'])
    return aasr_df,std_df

base_combined_df = pd.read_csv("dataframe0_base_8x8.csv")
free_combined_df = pd.read_csv("dataframe0_freeze_8x8.csv")


base_aasr_df, base_std_df = extract_aasr_3(base_combined_df)
free_aasr_df,free_std_df = extract_aasr_3(free_combined_df)

base_combined_df = base_combined_df.drop(columns=['aasr','std'])
free_combined_df = free_combined_df.drop(columns=['aasr','std'])
# print(base_std_df.head)
print(base_aasr_df.head)

cor_base_df = pd.concat([base_combined_df, base_aasr_df,base_std_df],axis=1)
base_combined_df = base_combined_df.sort_values(by='test-train')
free_combined_df = free_combined_df.sort_values(by='test-train')

# # print(base_combined_df['train loss'])
fig, ax = plt.subplots(2, 2)

colors_1 = ['blue', 'red', 'green', 'pink','grey']  # Base colors
colors_2 = ['orange', 'purple', 'yellow', 'cyan','brown']  # Freeze colors

for i in range(1, 6):
    # Plot for Base AASR
    color = colors_1[i-1]
    ax[1,0].plot(base_aasr_df[f'aasr_{i}'], 
                base_combined_df['test loss'] - base_combined_df['train loss'], 
                label=f'Base AASR {i}', 
                linestyle='-', 
                color=color)

    
    # Plot for Freeze AASR
    color = colors_2[i-1]
    ax[1,0].plot(free_aasr_df[f'aasr_{i}'], 
                free_combined_df['test loss'] - free_combined_df['train loss'], 
                label=f'Freeze AASR {i}', 
                linestyle='-', 
                color=color)
    


# Set labels and title
ax[1,0].set_xlabel('AASR')
ax[1,0].set_ylabel('Loss')
ax[1,0].set_title('Comparison of AASR ')

# Add legend
ax[1,0].legend(loc='lower left',fontsize='xx-small',ncols=3) 

base_combined_df = base_combined_df.sort_values(by='orig_model_acc')
free_combined_df = free_combined_df.sort_values(by='orig_model_acc')
for i in range(1, 6):
    # Plot for Base AASR
    color = colors_1[i-1]
    ax[1,1].plot(base_aasr_df[f'aasr_{i}'], 
                base_combined_df['orig_model_acc'], 
                label=f'Base AASR {i}', 
                linestyle='-', 
                color=color)

    
    # Plot for Freeze AASR
    color = colors_2[i-1]
    ax[1,1].plot(free_aasr_df[f'aasr_{i}'], 
                free_combined_df['orig_model_acc'], 
                label=f'Freeze AASR {i}', 
                linestyle='-', 
                color=color)
    


# Set labels and title
ax[1,1].set_xlabel('AASR')
ax[1,1].set_ylabel('acc')
ax[1,1].set_title('Comparison of AASR ')

# Add legend
ax[1,1].legend(loc='lower left',fontsize='xx-small',ncols=3) 


plt.tight_layout()

fig.savefig(f'combined_plot_1.png')
