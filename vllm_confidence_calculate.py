import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

import os
import json
import time
import argparse
import sys
import torch
import torch.nn.functional as F

from typing import Any, Dict, List
from nltk import ngrams
from collections import Counter

from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
import pdb

import math
import numpy as np
import random

def append_jsonl(data, file_path):
    """Append results in the list to a .jsonl file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
def write_jsonl(data, file_path):
    """Write results in the list to a .jsonl file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
def read_jsonl(file_path):
    """Read .jsonl file and return a list of dictionaries."""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found at {file_path}")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
def set_seeds(seed=42):
    # Set Python built-in random seed
    random.seed(seed)
    # Set NumPy random seed
    np.random.seed(seed)
    # Set PyTorch CPU random seed
    torch.manual_seed(seed)
    # If using GPU (especially CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # Set seed for current GPU
        torch.cuda.manual_seed_all(seed)       # Also effective for multi-GPU
        # For better reproducibility, enable cudnn determinism mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Optional: Set generator (for DataLoader with multi-threading)
    g = torch.Generator()
    g.manual_seed(seed)
def parse_args():
    parser = argparse.ArgumentParser()
    # 模型与数据路径
    parser.add_argument('--model_name_or_path', type=str, default='DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument('--dataset_dir', type=str, default="./Data/")
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default='Math_Math500')
    # 模型参数
    parser.add_argument('--dtype', type=str, default="bfloat16")
    parser.add_argument('--method', type=str, default="original")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--trust_remote_code', action="store_true")
    # token长度限制
    parser.add_argument('--max_model_len', '--model-context-len', type=int, default=40000, dest="model_context_len",
                        help="Model context limit (input + output).")
    parser.add_argument('--max_generated_tokens', '--max-len', type=int, default=16384, dest="max_len",
                        help="Maximum number of tokens to generate.")
    # 解码参数
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    # vLLM批处理,设置为比数据集大的值
    parser.add_argument('--batch_size', type=int, default=2000)
    args = parser.parse_args()
    return args


from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput
from transformers import AutoTokenizer
import torch, math, os, sys, json
from tqdm import tqdm

def calculate_average_max_prob_from_logprobs(logprobs_list, policy='avg2') -> float:
    num_tokens = len(logprobs_list)
    start_index = 1
    end_index = num_tokens
    if num_tokens < 1:
        return 0.0
    total_prob_sum, log_prob_sum, count_for_average = 0.0, 0.0, 0
    min_prob = 1.0
    for i in range(start_index, end_index):
        if i < len(logprobs_list) and logprobs_list[i]:
            try:
                logprob_obj = list(logprobs_list[i].values())[0]
                if hasattr(logprob_obj, 'logprob'):
                    prob = torch.exp(torch.tensor(logprob_obj.logprob)).item()
                    min_prob = min(min_prob, prob)
                    total_prob_sum += prob
                    log_prob_sum += math.log(max(prob, 1e-10))
                    count_for_average += 1
            except:
                continue
    if count_for_average == 0:
        return 0.0
    return {
        'min': min_prob,
        'avg1': total_prob_sum / count_for_average,
        'avg2': math.exp(log_prob_sum / count_for_average)
    }.get(policy, 0.0)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--max_generated_tokens', '--max-len', type=int, default=16384, dest="max_len",
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

    llm_engine = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')),
        dtype=args.dtype,
        max_model_len=args.max_len + 8000,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code
    )

    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    with open(dataset_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        questions_json = [json.loads(first_line)]
        #questions_json = [json.loads(line) for line in f]

    output_dir = os.path.join(args.output_path, os.path.basename(args.model_name_or_path), args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'origin_temp{args.temperature}_maxlen{args.max_len}.jsonl')

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    sampling_params = SamplingParams(
        max_tokens=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=[tokenizer.eos_token],
        logprobs=True
    )

    final_results = []
    pbar = tqdm(total=len(questions_json), desc="Origin Inference")

    for q in questions_json:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q['problem']}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm_engine.generate([prompt], sampling_params, use_tqdm=False)

        output = outputs[0].outputs[0]
        response_text = output.text
        token_ids = output.token_ids
        logprobs = output.logprobs

        # Step 1: 从字符串切分为段落（以 \n\n 为断句）
        # Step 1: 只保留 </think> 前的文本
        think_end_pos = response_text.find("</think>")
        if think_end_pos != -1:
            truncated_text = response_text[:think_end_pos]
        else:
            truncated_text = response_text  # fallback
        text_segments = truncated_text.split('\n\n')

        # Step 2: 按段落重新分 token，计算每段的 logprobs 区间
        sentence_confidences = []
        start = 0
        for segment in text_segments:
            # 用 tokenizer 获取当前段落的 token 数
            seg_token_ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
            seg_len = len(seg_token_ids)

            # 获取该段的 logprobs（注意：超出 token 长度则跳过）
            end = start + seg_len
            if end > len(logprobs):
                break  # 防止 logprobs 不足导致溢出

            segment_logprobs = logprobs[start:end]
            conf = calculate_average_max_prob_from_logprobs(segment_logprobs, policy='avg2')
            sentence_confidences.append(conf)

            start = end  # 更新起点

        # Step 3: 保存结果
        final_results.append({
            "question": q["problem"],
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "sentence_confidences": sentence_confidences
        })
        pbar.update(1)

    pbar.close()

    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in final_results:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ Origin evaluation complete. Saved to: {output_file}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()

