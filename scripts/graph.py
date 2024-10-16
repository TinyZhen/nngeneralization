import argparse
import ast
from datetime import datetime
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats

args = argparse.ArgumentParser(allow_abbrev=False)
args.add_argument("--sz", type=int, default=0)
args.add_argument("--file", type=str, default="")
args.add_argument("--infile",type=str,default="")
args = args.parse_args()


sz = args.sz
pd.set_option('display.float_format', '{:.6e}'.format)

def extract_info_from_log(log_file):
    polytope_color_sz = None
    total_samples = None
    average_samples = None
    std_samples = None
    orig_model_acc = None
    num_poly = None
    seed = None
    aasr = []
    std = []
    with open(log_file, 'r') as file:
        for line in file:
            i = 0
            if "polytope colours sz" in line:
                polytope_color_sz = int(re.search(r'polytope colours sz: (\d+)', line).group(1))
            elif "total samples" in line:
                total_samples = int(re.search(r'total samples: (\d+)', line).group(1))
            elif "average number of samples per polytope" in line:
                average_samples = float(re.search(r'average number of samples per polytope: ([\d\.]+)', line).group(1))
            elif "standard deviation" in line:
                std_samples = float(re.search(r'standard deviation: ([\d\.]+)', line).group(1))
            elif "orig model acc" in line:
                orig_model_acc = float(re.search(r'orig model acc: ([\d\.]+)', line).group(1))
            elif "num populated polytopes" in line:
                num_poly = int(re.search(r'num populated polytopes: (\d+)',line).group(1))
            elif "seed" in line:
                seed = int(re.search(r'seed: (\d+)',line).group(1))
            elif "Average AASR"in line:
                aasr_value = float(re.search(r'Average AASR: ([\d\.eE\+-]+)', line).group(1))
                aasr.append(aasr_value)
            elif "std" in line:
                std_value = float(re.search(r'std: ([[\d\.eE\+-]+)', line).group(1))
                std.append(std_value)

    return {
        "polytope_color_sz": polytope_color_sz,
        "num populated polytopes": num_poly,
        "total_samples": total_samples,
        "average_samples": average_samples,
        "std_samples": std_samples,
        "orig_model_acc": orig_model_acc,
        "seed": seed,
        "aasr": aasr,
        "std": std
    }

def get_timestamp_from_filename(filename):
    match = re.search(r'cifar10_mlp_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\.txt', filename)
    return match.group(1) if match else None

def extract_aasr_3(df):
    aasr_df = pd.DataFrame(df['aasr'].values.tolist(), index=df.index, columns=['aasr_1', 'aasr_2', 'aasr_3', 'aasr_4','aasr_5'])
    std_df = pd.DataFrame(df['std'].values.tolist(), index=df.index, columns=['std_1', 'std_2', 'std_3', 'std_4','std_5'])
    return aasr_df,std_df

def extract_aasr_1(df):
    for i, x in enumerate(df['aasr']):
        # Check if x is not a list or is an empty list
        if not isinstance(x, list) or len(x) == 0:
            print(f"Row {i} has an invalid 'aasr': {x}")
    aasr_df = pd.DataFrame([x[0] for x in df['aasr']], index=df.index, columns=['aasr_1'])
    std_df = pd.DataFrame([x[0] for x in df['std']], index=df.index, columns=['std_1'])
    print(aasr_df.head())
    return aasr_df, std_df


def save_file(df,train_loss,test_loss,i,mode):
    combined_df = pd.DataFrame(df)
    combined_df = combined_df.sort_values(by='timestamp',ignore_index=True)
    print(f"{i} {len(base_train_loss)}")
    last_train_loss = train_loss[-1]
    last_test_loss = test_loss[-1]
    train_loss.append(last_train_loss)
    test_loss.append(last_test_loss)
    combined_df['train loss'] = train_loss
    combined_df['test loss'] = test_loss
    combined_df['test-train'] = np.array(test_loss)-np.array(train_loss)
    combined_df.to_csv(f'dataframe{i}_{mode}_{sz}x{sz}_{combined_df.loc[1,"seed"]}_{args.file}.csv')
    return combined_df


base_root = "log/base/cifar10/"

free_root = "log/freeze/cifar10/"

base_dirs = {}
try:
    for subdir, dirs, files in os.walk(base_root):
        # print(f"dirs: {dirs}")
        # print(f"sub: {subdir}")
        # print(f"file: {files}")
        i = 0
        for dir_name in dirs:
            # if dir_name == '179665917':
                    dir_path = os.path.join(subdir, dir_name,f"[{sz}, {sz}, {sz}, {sz}]",args.infile)
                    base_dirs[dir_name] = dir_path
                    # base_dirs = subdir
                    if not os.path.exists(dir_path):
                        continue

    for subdir, dirs, files in os.walk(free_root):

        # print(files)
        for dir_name in dirs:
            # print(dir_name)
            if dir_name in base_dirs:
                # if dir_name == '179665917':
                    base_dir_path = base_dirs[dir_name]
                    free_dir_path = os.path.join(subdir, dir_name,f"[{sz}, {sz}, {sz}, {sz}]",args.infile)
                    if not os.path.exists(free_dir_path):
                        continue
                    print(base_dir_path)
                    print(free_dir_path)

                    base_files = [os.path.join(base_dir_path, f) for f in os.listdir(base_dir_path) if os.path.isfile(os.path.join(base_dir_path, f))]
                    free_files = [os.path.join(free_dir_path, f) for f in os.listdir(free_dir_path) if os.path.isfile(os.path.join(free_dir_path, f))]

                    if base_files:

                        base_dfs = []
                        free_dfs = []
                        
                        last_train_loss = None
                        last_test_loss = None
                        
                        base_train_loss = []
                        base_test_loss = [] 
                        for file in base_files:
                            # print(file)
                            if "loss" in file:
                                with open(file, 'r') as file:
                                    for line in file:
                                        if "train loss" in line:
                                                    pattern = r'train loss: ([\d\.]+), test loss: ([\d\.]+)'
                                                    match = re.search(pattern, line)
                                                    if match:
                                                        base_train_loss.append(float(match.group(1)))
                                                        base_test_loss.append(float(match.group(2)))
                            else:
                                match = re.search(r'cifar10_mlp_(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}).txt', file)
                                if match:
                                    timestamp_str = match.group(1)
                                    df = extract_info_from_log(file)

                                    df['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                                else:
                                    df['timestamp'] = None       
                            base_dfs.append(df) 

                        free_train_loss = []
                        free_test_loss = [] 
                        for file in free_files:

                            if "loss" in file:
                                with open(file, 'r') as file:
                                    for line in file:
                                        if "train loss" in line:
                                                    pattern = r'train loss: ([\d\.]+), test loss: ([\d\.]+)'
                                                    match = re.search(pattern, line)
                                                    if match:
                                                        free_train_loss.append(float(match.group(1)))
                                                        free_test_loss.append(float(match.group(2)))
                                
                            else:
                                match = re.search(r'cifar10_mlp_(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}).txt', file)
                                if match:
                                    timestamp_str = match.group(1)
                                    df = extract_info_from_log(file) 

                                    df['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                                else:
                                    df['timestamp'] = None        
                            free_dfs.append(df)

                        if base_dfs:
                            base_combined_df = save_file(base_dfs,base_train_loss,base_test_loss,i,"base")

                            # print("Base Combined DataFrame:")
                            # print(base_combined_df)
                        if free_dfs:
                            free_combined_df = save_file(free_dfs,free_train_loss,free_test_loss,i,"free")

                            # print("Free Combined DataFrame:")
                            # print(free_combined_df)
                        base_writer = pd.ExcelWriter(f'cor_dataframe_base_{sz}x{sz}_{base_combined_df.loc[1,"seed"]}_{args.file}.xlsx', engine='xlsxwriter')
                        # base_aasr_df, base_std_df = extract_aasr_3(base_combined_df)
                        # free_aasr_df,free_std_df = extract_aasr_3(free_combined_df)
                        base_aasr_df, base_std_df = extract_aasr_3(base_combined_df)
                        free_aasr_df,free_std_df = extract_aasr_3(free_combined_df)

                        base_combined_df = base_combined_df.drop(columns=['aasr','std'])
                        free_combined_df = free_combined_df.drop(columns=['aasr','std'])
                        # print(base_std_df.head)
                        print(base_aasr_df.head)
                        
                        cor_base_df = pd.concat([base_combined_df, base_aasr_df,base_std_df],axis=1)

                        # print(cor_base_df.head)
                        cor_base_df = cor_base_df.corr()
                        cor_base_df = round(cor_base_df, 3)
                        # cor_base_df.to_csv('cor_base_{i}_4096.csv')
                        cor_base_df.to_excel(base_writer, sheet_name=f'cor_base_{i}')

                        cor_free_df = pd.concat([free_combined_df, free_aasr_df,free_std_df],axis=1)
                        cor_free_df = cor_free_df.corr()
                        cor_free_df = round(cor_free_df, 3)
                        cor_free_df.to_excel(base_writer, sheet_name=f'cor_free_{i}')


                        # # print(base_combined_df['train loss'])
                        fig, ax = plt.subplots(2,2)


                        ax[0,0].plot(base_combined_df.index, base_combined_df['num populated polytopes'], linestyle='-', color='b', label='Base Data')

                        ax[0,0].plot(free_combined_df.index, free_combined_df['num populated polytopes'], linestyle='-', color='r', label='Free Data')

                        ax[0,0].set_xlabel('epochs')
                        ax[0,0].set_ylabel('polytopes')
                        ax[0,0].set_title('Comparison of Polytopes')
                        ax[0,0].legend()

                        # print(type(base_combined_df['aasr']))
                        ax[0,1].plot(base_combined_df.index, base_combined_df['orig_model_acc'], linestyle='-', color='b', label='Base Data')

                        ax[0,1].plot(free_combined_df.index, free_combined_df['orig_model_acc'], linestyle='-', color='r', label='Free Data')

                        ax[0,1].set_xlabel('epochs')
                        ax[0,1].set_ylabel('orig_model_acc')
                        ax[0,1].set_title('Comparison of acc')

                        colors_1 = ['blue', 'red', 'green', 'pink','grey']  # Base colors
                        colors_2 = ['orange', 'purple', 'yellow', 'cyan','brown']  # Freeze colors

                        for i in range(1, 6):
                            # Plot for Base AASR
                            color = colors_1[i-1]
                            ax[1,0].plot(base_combined_df.index, 
                                        base_aasr_df[f'aasr_{i}'], 
                                        label=f'Base AASR {i}', 
                                        linestyle='-', 
                                        color=color)
                            
                            ax[1,0].fill_between(base_combined_df.index, 
                                                base_aasr_df[f'aasr_{i}'] - base_std_df[f'std_{i}'], 
                                                base_aasr_df[f'aasr_{i}'] + base_std_df[f'std_{i}'], 
                                                color=color, 
                                                alpha=0.1)
                            
                            # Plot for Freeze AASR
                            color = colors_2[i-1]
                            ax[1,0].plot(free_combined_df.index, 
                                        free_aasr_df[f'aasr_{i}'], 
                                        label=f'Freeze AASR {i}', 
                                        linestyle='-', 
                                        color=color)
                            
                            ax[1,0].fill_between(free_combined_df.index, 
                                                free_aasr_df[f'aasr_{i}'] - free_std_df[f'std_{i}'], 
                                                free_aasr_df[f'aasr_{i}'] + free_std_df[f'std_{i}'], 
                                                color=color, 
                                                alpha=0.1)

                        # Set labels and title
                        ax[1,0].set_xlabel('Epochs')
                        ax[1,0].set_ylabel('AASR')
                        ax[1,0].set_title('Comparison of AASR with Error Bars')

                        # Add legend
                        ax[1,0].legend(loc='lower left',fontsize='xx-small',ncols=3)  # Adjust the location as necessary


                        # print(base_combined_df['test loss'])
                        ax[1,1].plot(base_combined_df.index, base_combined_df['test loss'] - base_combined_df['train loss'], linestyle='-', color='b', label='Base Data')

                        ax[1,1].plot(free_combined_df.index, free_combined_df['test loss'] - free_combined_df['train loss'], linestyle='-', color='r', label='Free Data')

                        ax[1,1].set_xlabel('epochs')
                        ax[1,1].set_ylabel('loss')
                        ax[1,1].set_title('Comparison of loss')
                        

                        plt.tight_layout()

                        fig.savefig(f'combined_plot_{sz}x{sz}_{base_combined_df.loc[1,"seed"]}_{args.file}.png')
                        i += 1
                        base_writer.close()
        break

    # if __name__ == "__main__":
    #     log_files = glob.glob("log/base/cifar10/0828/cifar10_mlp_*.txt")
    #     # print(log_files)
    #     data = extract_info_from_logs(log_files)
    #     save_to_excel(data, "output_1.xlsx")
except Exception as e:
     print(e)