import os
import ipdb
import pandas as pd
import random
import csv

from utils.ratio_sampling import three_ratio_sampling

SAMPlE_NUM = 500

cwd = os.getcwd()
valuenet_path_addition = 'datasets/valuenet_original/'
valuenet_save_path_addition = 'datasets/sampling_data/valuenet_'

VALUE_TYPES = ['ACHIEVEMENT', 'BENEVOLENCE', 'CONFORMITY', 'HEDONISM', 'POWER', 'SECURITY', 'SELF-DIRECTION', 'STIMULATION', 'TRADITION', 'UNIVERSALISM']

data_paths = [os.path.join(cwd, valuenet_path_addition + value_type + '.csv') for value_type in VALUE_TYPES]
data_save_paths = [os.path.join(cwd, valuenet_save_path_addition + value_type + '.csv') for value_type in VALUE_TYPES]

def sampling_labeling_valuenet_data(data_path):
    df = pd.read_csv(data_path)
    valuenet_datas = df.values.tolist()  # col0 = label, col1 = input

    if len(valuenet_datas) <= SAMPlE_NUM:
        return valuenet_datas

    # Use for statistic analysis
    pos_set = []
    neu_set = []
    neg_set = []

    for ele in valuenet_datas:
        if ele[-1] == 1:
            pos_set.append(ele)
        elif ele[-1] == -1:
            neg_set.append(ele)
        else:
            neu_set.append(ele)
    valuenet_sample_list = three_ratio_sampling(pos_set, neu_set, neg_set, len(pos_set), len(neu_set), len(neg_set), SAMPlE_NUM)
    random.shuffle(valuenet_sample_list)

    return valuenet_sample_list

def save_data_as_txt(save_path, save_data):
    with open(save_path, 'a+') as f:
        for ele in save_data:
            f.write(str(ele) + '\n')

def save_data_as_csv(save_path, save_data):
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["input", "label"])
        # write each row of the 2d list
        for row in save_data:
            writer.writerow(row[2:])  # only need input and label

    print(f"CSV file '{save_path}' has been created.")


for i in range(len(data_paths)):
    sample_data = sampling_labeling_valuenet_data(data_paths[i])
    save_data_as_csv(data_save_paths[i], sample_data)












