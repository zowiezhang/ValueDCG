import os
import ipdb
import pandas as pd
import random
import csv

from utils.ratio_sampling import ratio_sampling

SAMPlE_NUM = 500

cwd = os.getcwd()
# print(cwd)

ETHICS_TYPES = ['commonsense', 'deontology', 'justice', 'virtue']

data_paths = [os.path.join(cwd, 'datasets/ethics/data/' + ethics_type + '/train.csv') for ethics_type in ETHICS_TYPES]
data_save_paths = [os.path.join(cwd, 'datasets/sampling_data/ethics_' + ethics_type + '.csv') for ethics_type in ETHICS_TYPES]


def sampling_labeling_ethics_data(data_path):
    df = pd.read_csv(data_path)
    ethics_datas = df.values.tolist()  # col0 = label, col1 = input

    if len(ethics_datas) <= SAMPlE_NUM:
        return ethics_datas

    # Use for statistic analysis
    pos_set = []
    neg_set = []

    for ele in ethics_datas:
        if ele[0] == 1:
            pos_set.append(ele)
        else:
            neg_set.append(ele)
    # print('pos num:', len(pos_set))
    # print('neg num:', len(neg_set))
    ethics_sample_list = ratio_sampling(pos_set, neg_set, len(pos_set), len(neg_set), SAMPlE_NUM)
    random.shuffle(ethics_sample_list)

    return ethics_sample_list

def save_data_as_txt(save_path, save_data):
    with open(save_path, 'a+') as f:
        for ele in save_data:
            f.write(str(ele) + '\n')

def save_data_as_csv(save_path, save_data, current_ethics_type):
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["input", "label"])
        # write each row of the 2d list
        for row in save_data:
            if current_ethics_type == 'commonsense':
                writer.writerow([row[1], row[0]])
            elif current_ethics_type == 'deontology':
                writer.writerow([row[1] + ' ' + row[2], row[0]])
            else:
                writer.writerow([row[1], row[0]])

    print(f"CSV file '{save_path}' has been created.")


for i in range(len(data_paths)):
    sample_data = sampling_labeling_ethics_data(data_paths[i])
    save_data_as_csv(data_save_paths[i], sample_data, ETHICS_TYPES[i])












