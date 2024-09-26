from datetime import datetime
import glob
import os
import re
import pandas as pd

def extract_info_from_log(log_file):
    polytope_color_sz = None
    total_samples = None
    average_samples = None
    std_samples = None
    orig_model_acc = None
    num_poly = None
    seed = None

    with open(log_file, 'r') as file:
        for line in file:
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

    return {
        "polytope_color_sz": polytope_color_sz,
        "num populated polytopes": num_poly,
        "total_samples": total_samples,
        "average_samples": average_samples,
        "std_samples": std_samples,
        "orig_model_acc": orig_model_acc,
        "seed": seed
    }


def extract_info_from_logs(log_files):
    data = []
    for log_file in log_files:
        info = extract_info_from_log(log_file)
        if info is not None:  # Only add the info if the file was not empty
            info['file'] = os.path.basename(log_file)
            # Extract the timestamp from the filename
            match = re.search(r'cifar10_mlp_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\.txt', log_file)
            if match:
                timestamp_str = match.group(1)
                info['timestamp'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            else:
                info['timestamp'] = None  # Or handle it another way if you prefer
            data.append(info)
    return data

def save_to_excel(data, output_file):
    df = pd.DataFrame(data)
    df = df.sort_values(by='timestamp')  # Sort the DataFrame by the timestamp
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    log_files = glob.glob("log/base/cifar10/0828/cifar10_mlp_*.txt")
    # print(log_files)
    data = extract_info_from_logs(log_files)
    save_to_excel(data, "output_1.xlsx")