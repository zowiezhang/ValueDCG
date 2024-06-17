import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

from base_prompts import VALUE_INTRO, ETHICS_VALUES, LABEL_TYPES, base_eval_prompt
from openai_key_info import OPENAI_API_KEY, OPENAI_BASE_URL

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

import ipdb
pwd = os.getcwd()

MODEL_NAMES = ['Llama-2-70b-chat-hf', 'Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf', 'Meta-Llama-3-8B-Instruct']
VALUE_TYPES = ['ACHIEVEMENT', 'BENEVOLENCE', 'commonsense', 'CONFORMITY', 'deontology', 'HEDONISM', 'justice', 'POWER', 'SECURITY', 'SELF-DIRECTION', 'STIMULATION', 'TRADITION', 'UNIVERSALISM']

class GPTEvaluator:
    def __init__(self, eval_model = "gpt-4o-2024-05-13", tested_model_name = '', load_value_type = ''):
        self.eval_model = eval_model
        self.pwd = pwd
        self.client = OpenAI()
        self.tested_model_name = tested_model_name
        self.load_value_type = load_value_type.lower()
        self.base_load_path = os.path.join(self.pwd, 'response/' + tested_model_name + '/' + load_value_type + '.json')

    def gen_chatgpt_outputs(self, batch_prompt, \
        max_token = 200, temperature = 0, top_p = 0.95, seed = 42):
            # ipdb.set_trace()
            outputs = []
            for cur_prompt in tqdm(batch_prompt):
                if isinstance(cur_prompt, dict):
                    cur_prompt = [cur_prompt]
                elif isinstance(cur_prompt, str):
                    cur_prompt = [{'role': 'user', 'content': cur_prompt}]
                else:
                    raise ValueError("Invalid input type")
                while True:
                    try:
                        completion = self.client.chat.completions.create(
                            # model="gpt-3.5-turbo-0125",
                            model = self.eval_model,
                            messages = cur_prompt,
                            max_tokens = max_token,
                            temperature = temperature,
                            top_p = top_p,
                            seed = seed
                        )
                        break
                    except Exception as e:
                        print(e)
                        time.sleep(5)
                outputs.append(completion.choices[0].message.content)

            return outputs

    def get_value_type(self, label, value_name):

        if (value_name in base_eval_prompt) and label == 0:
            label = -1

        return LABEL_TYPES[label] + value_name


    def load_data(self):#, s_num = 0, e_num = 512):

        '''
        data json example:
        {"label": 0, 
        "input": "effeminate", 
        "WHY_Q": ".[/INST]", 
        "WHY_A": " Sure", 
        "model_name": "58f9420c015b8d5", 
        "value": "virtue", 
        "Attribution_SCORE": 1,
        "Counterfactual_SCORE": 3, 
        "Rebuttal_SCORE": 5}
        '''

        with open(self.base_load_path, 'r') as f:
            datas = json.load(f)

        return datas#[s_num:e_num]

    def save_eval_data(self, save_path, save_data):

        with open(save_path, 'w+') as file:
            json.dump(save_data, file)

    def get_input_prompt(self, data_json):
        
        input_text_content = data_json['input']
        value_name = self.get_value_type(data_json['label'], self.load_value_type.lower())
        value_defination = VALUE_INTRO[self.load_value_type.lower()]
        model_answer = data_json['WHY_A']

        return base_eval_prompt.format(input_text_content, value_name, value_defination, model_answer)

    def evaluate_and_save_datas(self):#, s_num = 0, use_data_num = 512):

        datas = self.load_data()#(s_num = s_num, e_num = use_data_num)

        input_prompts = [self.get_input_prompt(data) for data in datas]

        print('Data loaded!')

        outputs = self.gen_chatgpt_outputs([{'role': 'user', 'content': text} for text in input_prompts])

        print(outputs)

        legal_datas = []
        compare_labels = []
        
        for i in range(len(datas)):
            # datas[i]['gpt_eval_result'] = outputs[i]
            # print(outputs[i])
            
            try:
                output = eval(outputs[i])
                datas[i]['gpt_eval_result'] = {'a_score': float(output['a_score']), 'c_score': float(output['c_score']), 'r_score': float(output['r_score'])}
                legal_datas.append(datas[i])
            except:
                print(f'######### Format error id {i} ###########')
                continue

        gpt_evaluator.save_eval_data(os.path.join(pwd, f'know_why_save/{self.tested_model_name}/{self.load_value_type}.json'), legal_datas)


for model_name in MODEL_NAMES:
    for value_file in VALUE_TYPES:
        gpt_evaluator = GPTEvaluator(tested_model_name = model_name, load_value_type = value_file)
        gpt_evaluator.evaluate_and_save_datas()#0,2)







