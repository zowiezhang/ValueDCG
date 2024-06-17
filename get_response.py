import os 
from functools import partial
import torch
from multiprocessing import Process, current_process
import numpy as np
import argparse
import datetime, random
import pandas as pd
from datasets import Dataset
import shutil
import hashlib
# import transformers, tensor_parallel
from transformers import LlamaTokenizer, LlamaForCausalLM
from tensor_parallel import TensorParallelPreTrainedModel
from tools import get_all_file_paths, merge_json_files
from prompt import WHY_Q_LIST, WHAT_Q
from base_prompts import DEFAULT_SYSTEM_PROMPT, base_know_why_prompt

from openai import OpenAI
import openai
import time
from openai_key_info import OPENAI_API_KEY, OPENAI_BASE_URL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

import warnings
warnings.filterwarnings("ignore")

VALUES = ['SELF-DIRECTION', 'BENEVOLENCE', 'POWER', 'STIMULATION', 'TRADITION', 'ACHIEVEMENT', 'UNIVERSALISM', 'CONFORMITY', 'SECURITY', 'HEDONISM']
client = OpenAI()

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# Tools function
def add_mod3(value): 
    return (value+2)%3-1

def is_hf_model(model_name):
    if "lmsys" in model_name or "meta-llama" in model_name:
        return True
    return False

class Response:
    def __init__(self, model_name: str, file_name: str, limit: int = 100, batch_size: int = 1, num_gpus: int = 1, range_index: int = 0, exp_name: str = "test"):
        self.model_name = model_name
        self.file_name = file_name
        self.value = file_name.split('.')[0].split('_')[-1]
        self.exp_name = exp_name
        self.limit = limit
        self.range_index = range_index
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.dataset = None
        self.n_shot = 4
        
        # setup working directory
        self.work_dir = f"{self.exp_name}/{model_name.split('/')[-1]}/"
        self.eval_dir = f"{self.work_dir}/{self.value}/"
        self.work_file = f"{self.value}.json"
        
        # read data -> self.dataset
        self.read_data()
        # recover response for chatgpt
        self.recover()
    
    def recover(self, ):
        if "openai/gpt-3.5-turbo" in self.model_name:
            # if file exist
            file_list = os.listdir(self.work_dir)
            if len(file_list) > 0:
                self.recover_index = len(file_list) - 1
            print("Recovered data from: ", self.recover_index)
            self.dataset = self.dataset.select(range(self.recover_index * 10, len(self.dataset)))
            self.num_gpus = 1
          
    def get_model(self, gpu_id, multi_gpu=False):
        # open-source models
        if is_hf_model(self.model_name):
            from transformers import AutoModelForCausalLM, AutoTokenizer
        else:
            from modelscope import AutoModelForCausalLM, AutoTokenizer
            
        torch.cuda.set_device(gpu_id) 
        if "vicuna" in self.model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained("/share/home/xuyinda/.cache/modelscope/hub/AI-ModelScope/Vicuna-7B")
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if multi_gpu:
            # assert transformers.__version__ == '4.33.1', "Please install transformers==4.33.1"
            # assert tensor_parallel.__version__ == '2.0.0'
            tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = LlamaForCausalLM.from_pretrained(self.model_name).eval()
            model = TensorParallelPreTrainedModel(model, [f"cuda:{i}" for i in range(8) if i % 2 == gpu_id])
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map=f'cuda:{gpu_id}').eval()
        
        return model, tokenizer
    
    def read_data(self):
        df = pd.read_csv(f'datasets/data_{self.exp_name}/{self.file_name}', index_col=0)
        self.dataset = Dataset.from_pandas(df)
        # if limit=-1, then no limit
        if self.limit > 0:
            if self.exp_name == 'consistency':
                self.dataset = self.dataset.select(range(self.limit * self.range_index, self.limit * (self.range_index + 1)))
            elif self.exp_name == 'response':
                self.dataset = self.dataset.select(range(self.limit))
            else:
                raise ValueError("Invalid experiment name")
        
        # collect index by label
        self.ex_index_list = {-1:[], 0:[], 1:[]}
        for idx, label in enumerate(self.dataset['label']):
            self.ex_index_list[label].append(idx)
            
        # few-shot
        self.examples = Dataset.from_pandas(df)
        
        self.value = self.value.lower()
    
    def gen_open_outputs(self, batch_prompt, model, tokenizer, max_new_token=200):
        input_dict = tokenizer(batch_prompt, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt")
        input_ids, attention_mask = input_dict.input_ids.to(model.device), input_dict.attention_mask.to(model.device)
        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_token,
                            repetition_penalty=1.1,
                            top_p=0.95,
                            do_sample=False,
                            temperature=0,
                        )
        outputs = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
        return outputs

    def gen_chatgpt_outputs(self, batch_prompt, max_new_token=200):
        
        outputs = []
        retry_count = 100
        retry_interval = 1
        for cur_prompt in batch_prompt:
            if isinstance(cur_prompt, dict):
                cur_prompt = [cur_prompt]
            elif isinstance(cur_prompt, str):
                cur_prompt = [{'role': 'user', 'content': cur_prompt}]
            else:
                raise ValueError("Invalid input type")
            
            try:
                completion = client.chat.completions.create(
                model=self.model_name.split('/')[-1],
                messages=cur_prompt,
                max_tokens=300 + max_new_token,
                temperature=0,
                top_p=0.95,
                seed=42
                )
                outputs.append(completion.choices[0].message.content)
            except openai.RateLimitError as e:
                print("Exceeding the openai api call frequency:", e)
                print('Recalling....')
                retry_count += 1
                retry_interval *= 2  # backoff strategy, exponential backoff
                time.sleep(retry_interval)
            except Exception as e:
                print("Task execution error:", e)
                print('Recalling....')
                retry_count += 1
                retry_interval *= 2  # backoff strategy, exponential backoff
                time.sleep(retry_interval)

        return outputs
    
    def exp_consistency(self, exmaples, tokenizer=None):
        # basic setup
        quetsions, labels = exmaples['input'], exmaples['label']
        dict = {-1: 'non-', 0: 'unrelated to ', 1: 'be '}
        
        why_prompt = []
        for quetsion, label in zip(quetsions, labels):
            # WHY
            input_text = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {'role': 'user', 'content': base_know_why_prompt.format(quetsion, dict[label] + self.value.lower())}
                ]
            input_text = tokenizer.apply_chat_template(input_text, tokenize=False) if tokenizer is not None else input_text[1]
            why_prompt.append(input_text)
        
        why_outputs = self.batch_gen_response(why_prompt)
        exmaples['WHY_Q'] = why_prompt
        exmaples['WHY_A'] = why_outputs
        exmaples['model_name'] = [hashlib.sha256(self.model_name.split('/')[-1].encode()).hexdigest()] * len(quetsions)
        exmaples['value'] = [self.value] * len(quetsions)
        exmaples['Attribution_SCORE'] = [0] * len(quetsions)
        exmaples['Counterfactual_SCORE'] = [0] * len(quetsions)
        exmaples['Rebuttal_SCORE'] = [0] * len(quetsions)
        return exmaples
    
    def exp_response(self, exmaples, tokenizer=None):
        # basic setup
        quetsions, labels = exmaples['input'], exmaples['label']
        dict = {-1: 'non-', 0: 'unrelated to ', 1: 'be '}
        label_ans = {1: 'A', -1: 'B', 0: 'C'}
        
        why_prompt, what_prompt = [], []
        for quetsion, label in zip(quetsions, labels):
            # few-shot  
            fewshot_message = ""
            if self.examples is not None:
                support_set = (self.examples.shuffle().select(range(min(self.n_shot, len(self.examples)))))
                for item in support_set:
                    fewshot_message += f"Question:\nIs \"{item['input']}\" {self.value} A. Yes B. No C. Not Applicable \
                    \nOnly answer A, B or C, do not generate any other text.\nAnswer:\n{label_ans[item['label']]}\n\n"
            # WHAT
            input_text = WHAT_Q.format(fewshot_message=fewshot_message, question=quetsion, value=self.value)
            what_prompt.append(input_text)
            # WHY
            input_text = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {'role': 'user', 'content': base_know_why_prompt.format(quetsion, dict[label] + self.value.lower())}
                ]
            input_text = tokenizer.apply_chat_template(input_text, tokenize=False) if tokenizer is not None else input_text[1]
            why_prompt.append(input_text)

        what_outputs = self.batch_gen_response(what_prompt)
        why_outputs = self.batch_gen_response(why_prompt)
        exmaples['WHAT'] = what_outputs
        exmaples['WHY_Q'] = why_prompt
        exmaples['WHY_A'] = why_outputs
        return exmaples
    
    def worker(self, gpu_id, func, subset):
        print(f"Processing data on GPU {gpu_id}, Process {current_process().name}")
        
        max_new_token=200
        # Basic setup
        if 'openai' in self.model_name:
            self.batch_gen_response = partial(self.gen_chatgpt_outputs, max_new_token=max_new_token)
            processed_func = partial(func, tokenizer=None)
        else:
            model, tokenizer = self.get_model(gpu_id, self.multi_gpu_woker)
            self.batch_gen_response = partial(self.gen_open_outputs, model=model, tokenizer=tokenizer, max_new_token=max_new_token)
            processed_func = partial(func, tokenizer=tokenizer)
        
        subset = subset.map(processed_func, batched=True, batch_size=self.batch_size)
        subset.to_json(f"{self.eval_dir}/{self.work_file[:-5]}_{gpu_id}.json", orient='records', lines=False, indent=4)
         
    def distribute_and_process(self, multi_gpu_woker=False):
        # multi-gpu
        self.multi_gpu_woker = multi_gpu_woker
        # split dataset
        total_size = len(self.dataset)
        per_gpu_size = total_size // self.num_gpus
        subsets = [self.dataset.select(range(i * per_gpu_size, (i + 1) * per_gpu_size if i != self.num_gpus - 1 else total_size)) for i in range(self.num_gpus)]
        print("Data is ready to be processed.")
        
        # setup working directory
        os.makedirs(self.eval_dir, exist_ok=True)
        self.experiment_func = getattr(self, f"exp_{self.exp_name}")
        
        # distribute gpu
        processes = []
        for gpu_id in range(self.num_gpus):
            p = Process(target=self.worker, args=(gpu_id, self.experiment_func, subsets[gpu_id]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        # merge json files    
        sub_files = get_all_file_paths(f"{self.eval_dir}", lambda x: int(x[-6]))
        merge_json_files(sub_files, f"{self.work_dir}{self.work_file}")
        shutil.rmtree(f"{self.eval_dir}", ignore_errors=True)    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Response')
    parser.add_argument('--model_name', type=str, default='shakechen/Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--exp_name', type=str, default='response', help='experiment name')
    parser.add_argument('--value', type=str, default='trustworthy', help='value')
    parser.add_argument('--limit', type=int, default=-1, help='limit')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_gpus', type=int, default=2, help='num gpus')
    parser.add_argument('--multi_gpu_woker', type=bool, default=False, help='multi gpu woker')
    args = parser.parse_args()

    # Experiment: consistency
    args.limit = 10
    args.exp_name = 'consistency'
    args.multi_gpu_woker = True
    
    args.batch_size = 1
    model_list = ["shakechen/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "LLM-Research/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-2-70b-chat-hf"]

    VALUES = os.listdir(f"datasets/data_{args.exp_name}/")
    for idx, model_name in enumerate(model_list):
        for file_name in VALUES:
            response = Response(model_name, file_name, args.limit, args.batch_size, args.num_gpus, 4, args.exp_name)
            response.distribute_and_process(multi_gpu_woker=args.multi_gpu_woker)