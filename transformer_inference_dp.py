# -*- coding: utf-8 -*-
"""
Data-parallel version of transformer_inference.py

- Preserves ALL core logic from file one:
  * AutoModelForCausalLM + AutoTokenizer
  * sampling (temperature/top_p)
  * confidence computation per \n\n segment before </think>
  * hidden state saving for step tokens (save_qwen2_think_split_tokens_only)

- Adds dataset sharding across GPUs using torch.multiprocessing (similar to file two).
  Each rank loads its own model copy on its assigned GPU and writes to a shard file:
    <output_dir>/origin_temp{T}_maxlen{L}.shard{rank}.jsonl
  and hidden states to:
    <output_dir>/hidden_rank{rank}_{k}.pt
"""
import warnings
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import os
import json
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
from typing import List
from re import split as rsplit


def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_avg_logprob(logits, token_ids, policy='avg2'):
    probs = F.softmax(logits, dim=-1)
    chosen_probs = probs[range(len(token_ids)), token_ids]
    min_prob = chosen_probs.min().item() if len(chosen_probs) > 0 else 0.0
    avg1 = chosen_probs.mean().item() if len(chosen_probs) > 0 else 0.0
    avg2 = torch.exp(torch.mean(torch.log(chosen_probs + 1e-10))).item() if len(chosen_probs) > 0 else 0.0
    return {'min': min_prob, 'avg1': avg1, 'avg2': avg2}.get(policy, 0.0)


def load_model_and_tokenizer_single_gpu(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def save_qwen2_think_split_tokens_only(model, tokenizer, input_ids, full_text, save_path):
    import torch as _torch
    import os as _os
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    input_ids_flat = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    try:
        think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
        think_end_idx = input_ids_flat.index(think_end_token)
    except ValueError:
        think_end_idx = len(input_ids_flat)
    step_positions = []
    for i in range(think_end_idx - 1):
        if tokens[i] == ".ĊĊ" or tokens[i] == "ĊĊ" or "ĊĊ" in tokens[i]:
            step_positions.append(i + 1)
    step_indices = _torch.tensor(step_positions, dtype=_torch.long)
    hidden_dict = {}
    sample_id = 0
    for layer_id, layer_h in enumerate(hidden_states):
        h = layer_h.squeeze(0).cpu()
        step_h = h[step_indices] if len(step_indices) > 0 else _torch.empty((0, h.shape[1]))
        hidden_dict[layer_id] = {
            sample_id: {
                "step": step_h
            }
        }
    _os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _torch.save(hidden_dict, save_path)


def load_existing_indices(shard_file):
    existing_idx = set()
    existing_q = set()
    num_lines = 0
    if os.path.exists(shard_file):
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                num_lines += 1
                try:
                    obj = json.loads(s)
                    if "idx" in obj:
                        existing_idx.add(int(obj["idx"]))
                    if "question" in obj:
                        existing_q.add(obj["question"])
                except Exception:
                    continue
    return existing_idx, existing_q, num_lines


def worker(rank, world_size, args):
    set_seeds(42 + rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    questions = read_jsonl(dataset_path)

    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    base_name = f'origin_temp{args.temperature}_maxlen{args.max_generated_tokens}'
    shard_file = os.path.join(output_dir, f'{base_name}.shard{rank}.jsonl')

    existing_idx, existing_q, num_lines = load_existing_indices(shard_file)
    # ⬇️ 不再使用 hidden_offset
    # hidden_offset = num_lines

    model, tokenizer = load_model_and_tokenizer_single_gpu(args, device)

    my_indices = [i for i in range(len(questions)) if (i % world_size) == rank]

    print(f"[rank {rank}] world_size={world_size}")
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing: idx={len(existing_idx)}, question={len(existing_q)}, lines={num_lines}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    pbar = tqdm(
        total=len(my_indices),
        desc=f"Rank {rank} DP Inference",
        position=rank,
        leave=True
    )

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    for i in my_indices:
        q = questions[i]

        if (i in existing_idx) or (q.get("problem") in existing_q):
            pbar.update(1)
            continue

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q['problem']}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_generated_tokens,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[OOM][rank {rank}] idx={i} : {q['problem'][:80]}... skipping.")
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        response_text = tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        full_text = prompt + response_text
        full_inputs = tokenizer(full_text, return_tensors="pt").to(device)

        # ✅ 仅按问题编号保存隐藏层：hidden_{idx}.pt
        hidden_save_path = os.path.join(output_dir, f"hidden_{i}.pt")
        try:
            # 如果已经存在同名文件，可选择跳过，避免重复写入
            if not os.path.exists(hidden_save_path):
                save_qwen2_think_split_tokens_only(
                    model, tokenizer, full_inputs["input_ids"], full_text, hidden_save_path
                )
            else:
                print(f"[rank {rank}] hidden exists, skip: {hidden_save_path}")
        except Exception as he:
            print(f"[WARN][rank {rank}] hidden save failed at idx={i}: {he}")

        scores = torch.stack(output.scores, dim=1)[0]
        token_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]
        text_before_think = response_text.split('</think>')[0]
        text_segments = rsplit(r'\n\n+', text_before_think)  # 保持你原来的切分逻辑不动

        confidences = []
        start = 0
        for segment in text_segments:
            seg_ids = tokenizer(segment, add_special_tokens=False)['input_ids']
            end = start + len(seg_ids)
            if end > len(token_ids):
                break
            if end > start:
                conf = calculate_avg_logprob(scores[start:end], token_ids[start:end])
                confidences.append(conf)
            start = end

        result_entry = {
            "idx": i,
            "question": q.get("problem", ""),
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "sentence_confidences": confidences
        }

        with open(shard_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

        # ⬇️ 不再使用 hidden_offset
        # hidden_offset += 1
        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_generated_tokens', type=int, default=512)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()

    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()
