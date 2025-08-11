import json
from transformers import AutoTokenizer

import re
import importlib.util
import os
import argparse

import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb
import pdb


def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./Data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--generation_path", default="test", type=str)

    parser.add_argument("--prompt_type", default="qwen-base", type=str)

    args = parser.parse_args()
    


    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format

def read_jsonl(file_path):

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:

            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data





def infer(args):
    examples = load_data(args.data_name, args.split, args.data_dir)
    file_outputs = read_jsonl(args.generation_path)
    
    
    print("llm generate done")
    print(len(file_outputs))
    
    pass_at_k_list = []
    k = args.k
    
    correct_cnt = 0
    for i in tqdm(range(len(file_outputs)), "check correct..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses = file_outputs[i]['generated_responses']
        
        
        generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
        is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            #print(i)
            correct_cnt += 1
        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list
        
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)
                
    
    print(f"correct cnt / total cnt: {correct_cnt}/{len(file_outputs)}")
    print(f"Acc: {correct_cnt / len(file_outputs):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"Pass@1: {correct_cnt}/{len(file_outputs)} = {correct_cnt / len(file_outputs):.4f}")



    response_length = []
    token_num = []
    wait_num = []
    alt_num = []

    test_num = len(file_outputs)
    correct_num = 0
    for data in file_outputs:
        response_length.append(len(data['generated_responses'][0].split()))
        tokens_response_len = len(tokenizer(data['generated_responses'][0])['input_ids'])
        token_num.append(tokens_response_len)
        

    avg_response_length = sum(response_length) / test_num
    avg_token_num = sum(token_num) / test_num

    print("length:", avg_response_length)
    print('token_num:', avg_token_num)


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    infer(args)