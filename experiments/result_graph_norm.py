import os
import json

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.normalize_list import normalize_list, standardize_list

import ipdb



pwd = os.getcwd()

data_path = os.path.join(pwd, f'experiments/final_result_consitency/compare_final_200.json')

with open (data_path, 'r') as f:
    datas = json.load(f)

print(f'The num of data is {len(datas)}.')

a_human_ratings = []
a_gpt_ratings = []
c_human_ratings = []
c_gpt_ratings = []
r_human_ratings = []
r_gpt_ratings = []
avg_human_ratings = []
avg_gpt_ratings = []

for data in datas:
    a_human_ratings.append(float(data['human']['a_score']))
    a_gpt_ratings.append(float(data['gpt']['a_score']))
    c_human_ratings.append(float(data['human']['c_score']))
    c_gpt_ratings.append(float(data['gpt']['c_score']))
    r_human_ratings.append(float(data['human']['r_score']))
    r_gpt_ratings.append(float(data['gpt']['r_score']))

    avg_human_ratings.append((a_human_ratings[-1] + c_human_ratings[-1] + r_human_ratings[-1]) / 3.)
    avg_gpt_ratings.append((a_gpt_ratings[-1] + c_gpt_ratings[-1] + r_gpt_ratings[-1]) / 3.)

def drawing_results_as_heapmap(participant_ratings, predicted_ratings, x_label = 'GPT-4 Ratings', y_label = 'Human Ratings', title = ''):


    # Number of bins
    num_bins = 5

    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.1])

    participant_indices = np.digitize(participant_ratings, bins) - 1
    # ipdb.set_trace()
    predicted_indices = np.digitize(predicted_ratings, bins) - 1

    # Initialize the heatmap data array
    heatmap_data = np.zeros((num_bins, num_bins))

    for p_idx, pred_idx in zip(participant_indices, predicted_indices):
        heatmap_data[p_idx, pred_idx] += 1


    row_sums = heatmap_data.sum(axis=1) #+ epsilon
    row_sums = row_sums[:, None]
    heatmap_data /= row_sums


    bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

    # Create the heatmap
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(heatmap_data, annot=True, cmap='Blues', cbar=False, xticklabels=bin_labels, yticklabels=bin_labels)

    # Set labels and titles
    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)

    # ax.set_title(title, fontsize=14)

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()




drawing_results_as_heapmap(normalize_list(standardize_list(a_human_ratings)), normalize_list(standardize_list(a_gpt_ratings)), x_label = 'GPT-4 Ratings', y_label = 'Human Ratings', title = 'Attribution Score Analysis')

drawing_results_as_heapmap(normalize_list(standardize_list(c_human_ratings)), normalize_list(standardize_list(c_gpt_ratings)), x_label = 'GPT-4 Ratings', y_label = 'Human Ratings', title = 'Counterfactual Score Analysis')

drawing_results_as_heapmap(normalize_list(standardize_list(r_human_ratings)), normalize_list(standardize_list(r_gpt_ratings)), x_label = 'GPT-4 Ratings', y_label = 'Human Ratings', title = 'Rebuttal Score Analysis')

drawing_results_as_heapmap(normalize_list(standardize_list(avg_human_ratings)), normalize_list(standardize_list(avg_gpt_ratings)), x_label = 'GPT-4 Ratings', y_label = 'Human Ratings', title = 'Average Score Analysis')





