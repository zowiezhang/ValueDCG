import json
import os

from utils.normalize_list import normalize_list, standardize_list

import ipdb

pwd = os.getcwd()

with open(os.path.join(pwd, 'response/achievement.json'), 'r') as f:
    datas = json.load(f)

# ipdb.set_trace()

gpt_answers = []
a_scores = []
c_scores = []
r_scores = []
avg_scores = []

for data in datas:
    gpt_answers.append(data['gpt_eval_result'])
    a_scores.append(data['gpt_eval_result']['a_score'])
    c_scores.append(data['gpt_eval_result']['c_score'])
    r_scores.append(data['gpt_eval_result']['r_score'])

    avg_scores.append((a_scores[-1] + c_scores[-1] + r_scores[-1]) / 3.)

# norm_a_scores = normalize_list(standardize_list(a_scores))
# norm_c_scores = normalize_list(standardize_list(c_scores))
# norm_r_scores = normalize_list(standardize_list(r_scores))
# norm_avg_scores = normalize_list(standardize_list(avg_scores))

a_score_rate = (sum(a_scores) / len(a_scores)) / 5.
c_score_rate = (sum(c_scores) / len(c_scores)) / 5.
r_score_rate = (sum(r_scores) / len(r_scores)) / 5.
avg_score_rate = (sum(avg_scores) / len(avg_scores)) / 5.

print('a_score_rate:', a_score_rate)
print('c_score_rate:', c_score_rate)
print('r_score_rate:', r_score_rate)
print('avg_score_rate:', avg_score_rate)


# ipdb.set_trace()








