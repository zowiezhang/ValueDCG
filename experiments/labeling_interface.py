import json
import copy

current_file_num = 10

load_path = './experiments/sep_data_'#.json'
base_save_path = './experiments/sep_data_'

with open(load_path + str(current_file_num) + '.json', 'r') as f:
    datas = json.load(f)

# print(len(datas))

# random.shuffle(datas)
with open(base_save_path + str(current_file_num) + '.json', 'a+') as file:
    file.write('[')

for i in range(len(datas)):
    
    break_flag = 0

    provided_datas = copy.deepcopy(datas)
    del provided_datas[i]['WHY_Q']
    formatted_json = json.dumps(provided_datas[i], indent = 4, separators = (',', ': '))
    print()
    print(formatted_json)
    print()
    
    while True:
        try:
            if i == 0:
                input_line = input("""Please input the score of Attribution, Counterfactual, and Rebuttal Argument Analysis respectively with space interval.
For example, if you want to label 1, 2, 3 for each, just input:
1 2 3
Now please input your label:
""")
            elif i == 9:
                input_line = input("""Come on! Only 10 left!
Now please input your label:
""")
            else:
                input_line = input("""Now please input your label:
""")
            if input_line == 'exit':
                break_flag = 1
                break

            attribution, counterfactual, rebuttal = input_line.split(' ')

            break

        except:
            print('Format error, please input again! If you want to exit the task, please input "exit".')

    if break_flag == 1:
        print('End task!')
        break

    datas[i]['Attribution_SCORE'] = int(attribution)
    datas[i]['Counterfactual_SCORE'] = int(counterfactual)
    datas[i]['Rebuttal_SCORE'] = int(rebuttal)

    with open(base_save_path + str(current_file_num) + '.json', 'a+') as file:
        json.dump(datas[i], file)
        file.write(',\n')

    print(f'The {i + 1}-th finished, {len(datas) - (i + 1)} left!')
    print()

with open(base_save_path + str(current_file_num) + '.json', 'a+') as file:
    file.write(']')





