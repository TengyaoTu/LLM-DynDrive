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
    parser.add_argument('--model_name_or_path', type=str, default='/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
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
def main():
    args = parse_args()
    args.model_context_len = args.max_len + 8000
    print(f"Using vLLM LLM object for direct inference (batch processing)")
    print(f"Model path: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max total generated tokens: {args.max_len}")
    print("\nInitializing vLLM LLM engine...")
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    # 模型下载
    from modelscope import snapshot_download

    # 下载模型（你也可以修改成别的模型名）
    #model_dir='/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    #model_dir='/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    try:
        llm_engine = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus),
            dtype=args.dtype,
            max_model_len=args.max_len + 8000,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        print("vLLM LLM engine initialized successfully.")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        print(f"Successfully loaded tokenizer: {args.model_name_or_path}")
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print(f"Tokenizer using pad_token_id: {tokenizer.pad_token_id}")

    except Exception as e:
        print(f"Failed to initialize vLLM LLM engine or load tokenizer: {e}")
        sys.exit(1)

    # 加载数据
    dataset_path = f'{args.dataset_dir}/{args.dataset}/test.jsonl'
    try:
        questions_json = read_jsonl(dataset_path)
        if not questions_json:
            print(f"Error: No questions loaded from {dataset_path}.")
            sys.exit(1)
        print(f"Loaded {len(questions_json)} questions from {dataset_path}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    # 输出文件
    model_dir_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = f'{args.output_path}/{model_dir_name}/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/origin_temp{args.temperature}_maxlen{args.max_len}.jsonl'

    # Prompt 基础设定
    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    sampling_params = SamplingParams(
        max_tokens=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=[tokenizer.eos_token]
    )

    # 推理主循环
    final_results = []
    pbar = tqdm(total=len(questions_json), desc="Origin Inference")

    for i, question_data in enumerate(questions_json):
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question_data['problem']}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = llm_engine.generate([prompt], sampling_params, use_tqdm=False)

            if outputs and outputs[0].outputs:
                response_text = outputs[0].outputs[0].text
            else:
                response_text = "[Error: Empty generation]"
            final_results.append({
                "question": question_data["problem"],
                "generated_responses": [response_text],
                "gold_answer": question_data.get("answer", "")
            })
        except Exception as e:
            print(f"\nError processing question {i}: {e}")
            final_results.append({
                "question": question_data["problem"],
                "generated_responses": [f"[Error]: {e}"],
                "gold_answer": question_data.get("answer", "")
            })
        pbar.update(1)
    pbar.close()
    # 排序 + 保存
    problem_to_index = {item['problem']: i for i, item in enumerate(questions_json)}
    final_results.sort(key=lambda x: problem_to_index.get(x['question'], len(questions_json)))
    print("\nSaving origin results...")
    try:
        write_jsonl(final_results, output_file)
    except Exception as e:
        print(f"Error saving results: {e}")
    print(f"✅ Origin evaluation complete. Saved to: {output_file}")
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm.outputs import CompletionOutput
    from vllm import LLM, SamplingParams
    # 读取原始 JSON 文件（包含一个 list of dicts）
    set_seeds(42)
    main()


