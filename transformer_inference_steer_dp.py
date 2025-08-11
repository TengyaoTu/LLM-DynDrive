
import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from re import split as rsplit
import torch.multiprocessing as mp

# NOTE: keep your custom Qwen import path if you have a patched model
# e.g., your custom class that supports set_steering_flag/start_new_round
from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def calculate_avg_logprob(logits, token_ids):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    chosen_probs = probs[range(len(token_ids)), token_ids]
    return torch.exp(torch.mean(torch.log(chosen_probs + 1e-10))).item()


def load_model_and_tokenizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()
    return model, tokenizer


def build_output_paths(args):
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    run_prefix = (args.run_id.strip() + "_") if args.run_id else ""
    base_name = f"{run_prefix}steer_temp{args.temperature}_maxlen{args.max_generated_tokens}"
    return output_dir, base_name


def load_existing_indices(shard_file):
    """
    Robust checkpoint loader. Returns a set of completed sample indices.
    Falls back to 'question' for backward compatibility.
    """
    existing_idx = set()
    existing_q = set()
    if os.path.exists(shard_file):
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "idx" in obj:
                        existing_idx.add(int(obj["idx"]))
                    elif "question" in obj:
                        existing_q.add(obj["question"])
                except Exception:
                    # Skip malformed line
                    continue
    return existing_idx, existing_q


def worker(rank, world_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    questions = read_jsonl(dataset_path)

    # Output paths
    output_dir, base_name = build_output_paths(args)
    shard_file = os.path.join(output_dir, f"{base_name}.shard{rank}.jsonl")

    # Resume from checkpoint
    existing_idx, existing_q = load_existing_indices(shard_file)

    # Load model/tokenizer per rank
    model, tokenizer = load_model_and_tokenizer(args, device)

    # Load steer vector
    steer_vector = torch.load(args.steer_vector_path, map_location="cpu")
    steer_vector = steer_vector.to(device)

    # Enable steering (custom API in your patched model)
    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.steer_layer,
        steer_vec=steer_vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer
    )

    # Partition: each rank handles indices congruent to its rank
    my_indices = [i for i in range(len(questions)) if (i % world_size) == rank]

    # Logging
    print(f"[rank {rank}] world_size={world_size}")
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing: idx={len(existing_idx)}, question={len(existing_q)}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    pbar = tqdm(
        total=len(my_indices),
        desc=f"Rank {rank} DP Inference",
        position=rank,      # show both progress bars
        leave=True
    )

    for i in my_indices:
        q = questions[i]

        # Skip if already done (prefer idx; fallback to question text for old files)
        if (i in existing_idx) or (q.get('problem') in existing_q):
            pbar.update(1)
            continue

        model.start_new_round()

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q["problem"]}
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
            print(f"[OOM][rank {rank}] Skipping idx={i} : {q['problem'][:80]}...")
            torch.cuda.empty_cache()
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

        # Confidence by segments (<= '</think>' and split by blank lines)
        scores = torch.stack(output.scores, dim=1)[0]
        token_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]
        text_before_think = response_text.split('</think>')[0]
        text_segments = rsplit(r'\n\n+', text_before_think)

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

        result = {
            "idx": i,  # store idx for robust checkpointing
            "question": q.get("problem", ""),
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "sentence_confidences": confidences
        }

        # Append to shard file
        with open(shard_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--steer_vector_path', type=str, required=True)
    parser.add_argument('--steer_layer', type=int, default=22)
    parser.add_argument('--steer_coef', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_generated_tokens', type=int, default=512)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--run_id', type=str, default="", help="Optional tag to separate different runs in filenames")
    args = parser.parse_args()

    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()
