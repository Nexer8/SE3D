import argparse
from glob import glob

import numpy as np

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--metadata_root', type=str, default='.',
                    help="Root folder of metadata.")
parser.add_argument('--model_path', type=str, default='3D_CNN',
                    help="Root folder of the model to evaluate.")
args = parser.parse_args()

# Project Config
BASE_PATH = args.metadata_root
MODEL_PATH = args.model_path

# Get the results
full_results = {}
METHODS = ['GradCAM', 'HiResCAM', 'GradCAMPlusPlus', 'SaliencyTubes']

for method in METHODS:
    # find relevant numpy files in the directory using glob
    results_files = glob(f'{MODEL_PATH}/**/{method}.npz')
    results = {}
    for results_file in results_files:
        result = np.load(results_file, allow_pickle=True)['data'].item()
        for metric, performance in result.items():
            if metric not in results.keys():
                results[metric] = []
            results[metric].append(performance)
    # average over the results
    # also save the std error

    std_err = {}
    for metric, performance in results.items():
        results[metric] = np.average(performance, axis=0)
        std_err[metric] = np.std(performance, axis=0) / np.sqrt(len(performance))

    full_results[method] = results
    full_results[f'{method}_std_err'] = std_err


# Create Markdown Table
def create_table(results, std_err):
    table = '| Method | MaxBoxAcc (IoU: 0.5) | MaxBoxAccV2 (IoU: [0.3, 0.5, 0.7]) | VxAP | Max3DBoxAcc (IoU: 0.5) | Max3DBoxAccV2 (IoU: [0.3, 0.5, 0.7]) |\n'
    table += '|--------|-----------|--------------|------|------------|---------------|\n'
    for cam_method, result in results.items():
        table += f'| {cam_method} | '
        for metric, performance in result.items():
            if metric == 'VxAP':
                table += f'{performance:.2f} +- {std_err[cam_method][metric]:.2f} | '
            elif metric in ['MaxBoxAcc', 'Max3DBoxAcc']:
                middle = len(performance) // 2
                table += f'{performance[middle]:.2f} +- {std_err[method][metric][middle]:.2f} | '
            else:
                table += '['
                for value, error in zip(performance, std_err[cam_method][metric]):
                    table += f'{value:.2f} +- {error:.2f}, '
                table = f'{table[:-2]}] | '
        table += '\n'
    return table


# Calculate Model's Accuracy Across all 5 Folds
accuracies = []
for fold in range(5):
    with open(f'{MODEL_PATH}/fold{fold}/cr{fold}.txt', 'r') as f:
        for line in f:
            if 'accuracy' in line:
                accuracies.append(float(line.split()[-2]))
                break

# calculate the mean and std error
mean_accuracy = np.mean(accuracies)
std_err_accuracy = np.std(accuracies) / np.sqrt(len(accuracies))
accuracy_info = f'Accuracy (*mean &plusmn; std_err*): {mean_accuracy * 100:.2f} &plusmn; {std_err_accuracy * 100:.2f}%.'

# save the table in a file
t_full_std_err = {k.replace('_std_err', ''): v for k, v in full_results.items() if 'std_err' in k}
t_full_results = {k: v for k, v in full_results.items() if 'std_err' not in k}

with open(f'{MODEL_PATH}/results.md', 'w') as f:
    f.write(create_table(t_full_results, t_full_std_err).replace('+-', '&plusmn;'))
    f.write('\n')
    f.write(accuracy_info)
    f.write('\n')
