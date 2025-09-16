# transformer_inference_steer_dp_code.py
# -*- coding: utf-8 -*-

import os
import json
import argparse
import gc
import random
from re import split as rsplit

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- silence all warnings/logging (保持与现有DP脚本一致) ----
import warnings, logging
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
try:
    import datasets
    datasets.utils.logging.set_verbosity_error()
except Exception:
    pass
try:
    import numpy as np
    np.seterr(all="ignore")
except Exception:
    pass
# logging.disable(logging.CRITICAL)
# ---- end silence ----

# 你自定义的Qwen2（支持 set_steering_flag / start_new_round）
from modeling_utils.modeling_qwen2_dynamic import Qwen2ForCausalLM

# 复用你提供的 code_evaluation 工具
from code_evaluation import (
    load_code_generation_dataset,
    get_deepseekcode_question_template_answer,
    extract_code,
    codegen_metrics,
    extract_instance_results,
)

def set_seed(seed=42):
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_output_paths(args):
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    run_prefix = (args.run_id.strip() + "_") if args.run_id else ""
    base_name = f"{run_prefix}steer_temp{args.temperature}_maxlen{args.max_generated_tokens}"
    return output_dir, base_name


def load_existing_indices(shard_file):
    """
    断点续跑：返回已完成的 idx 集合（兼容旧字段 question）
    """
    existing_idx, existing_q = set(), set()
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
                    elif "prompt" in obj:
                        existing_q.add(obj["prompt"])
                except Exception:
                    continue
    return existing_idx, existing_q


def load_model_and_tokenizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 代码任务常用左侧padding + 保证pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)

    # 选择注意力实现：优先FA2，回落SDPA
    try:
        use_fa2 = False
        try:
            from transformers.utils import is_flash_attn_2_available
            use_fa2 = bool(is_flash_attn_2_available())
            if use_fa2:
                import flash_attn_2_cuda  # 确认扩展存在
        except Exception:
            use_fa2 = False

        if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
            if use_fa2:
                model.config.attn_implementation = "flash_attention_2"
                print("[INFO] Using flash_attention_2")
            else:
                model.config.attn_implementation = "sdpa"
                torch.backends.cuda.matmul.allow_tf32 = True
                if torch.cuda.is_available():
                    torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=False,
                        enable_mem_efficient=True
                    )
                print("[INFO] Using PyTorch SDPA (flash preferred, mem_efficient fallback)")
    except Exception as e:
        print(f"[WARN] attention backend setup failed: {e}")

    model.eval()
    return model, tokenizer, dtype


def is_code_dataset(name: str) -> bool:
    return name.lower().startswith("code_")


def dataset_to_release(dataset_name: str) -> str:
    # "Code_livecodebenchv2" -> "livecodebenchv2"
    return dataset_name.split("_", 1)[1] if "_" in dataset_name else dataset_name


def prepare_code_benchmark(args, tokenizer):
    """
    加载代码评测数据集 + 构造 prompts（采用 chat 模版）
    返回：
      benchmark: list[Instance]（code_evaluation定义的对象）
      prompts:  list[str]
    """
    release = dataset_to_release(args.dataset)
    benchmark = load_code_generation_dataset(release_version=release)

    prompts = []
    for ex in benchmark:
        prompt = get_deepseekcode_question_template_answer(ex)
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 兼容 remove_bos 语义（你在eval_code_steering里默认去掉）
        if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)

    return benchmark, prompts


def worker(rank, world_size, args):
    """
    每个rank负责生成自己的切片，并将结果写入 shard 文件（仅包含 idx 与 output_text）。
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 输出路径/分片文件
    output_dir, base_name = build_output_paths(args)
    shard_file = os.path.join(output_dir, f"{base_name}.shard{rank}.jsonl")

    # 已完成检查
    existing_idx, existing_prompt = load_existing_indices(shard_file)

    # 模型与tokenizer
    model, tokenizer, dtype = load_model_and_tokenizer(args, device)

    # Steering向量
    steer_vector = torch.load(args.steer_vector_path, map_location="cpu")
    steer_vector = steer_vector.to(device, dtype=dtype)
    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.steer_layer,
        steer_vec=steer_vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer
    )

    # 仅实现“代码评测”分支（你当前需求）
    if not is_code_dataset(args.dataset):
        raise ValueError(f"This script targets code evaluation datasets only (dataset startswith 'Code_'). Got: {args.dataset}")

    benchmark, prompts = prepare_code_benchmark(args, tokenizer)
    N = len(prompts)

    my_indices = [i for i in range(N) if (i % world_size) == rank]

    print(f"[rank {rank}] world_size={world_size}")
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing idx={len(existing_idx)}")
    print(f"[rank {rank}] will process {len(my_indices)} items of total {N}")

    pbar = tqdm(total=len(my_indices), desc=f"Rank {rank} Code Inference", position=rank, leave=True)

    for i in my_indices:
        if i in existing_idx:
            pbar.update(1)
            continue

        model.start_new_round()

        prompt = prompts[i]
        encoded = tokenizer([prompt], return_tensors="pt", padding=True).to(device)

        try:
            with torch.no_grad():
                # 代码任务：默认贪心（与eval_code_steering一致）
                gen = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_generated_tokens,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id
                )
            prompt_len = encoded["input_ids"].shape[1]
            decoded = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM][rank {rank}] Skipping idx={i}...")
            torch.cuda.empty_cache()
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        rec = {
            "idx": i,
            "prompt": prompt,
            "output_text": decoded
        }
        with open(shard_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 主动释放显存碎片
        del encoded, gen
        torch.cuda.empty_cache()
        gc.collect()

        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")


def aggregate_and_evaluate(args):
    """
    主进程在所有rank完成后：
      1) 读取所有shard，按idx聚合为输出列表
      2) 用 code_evaluation 抽取代码、评测 pass@1
      3) 保存 predictions / metrics / code_eval
    """
    print("[main] Aggregating shards and running evaluation...")

    output_dir, base_name = build_output_paths(args)
    shard_files = [
        os.path.join(output_dir, f"{base_name}.shard{r}.jsonl")
        for r in range(max(1, args.num_gpus))
        if os.path.exists(os.path.join(output_dir, f"{base_name}.shard{r}.jsonl"))
    ]
    # 读入所有分片
    idx2text = {}
    for sf in shard_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    idx = int(obj["idx"])
                    out = obj["output_text"]
                    # 若重复以最后一次为准
                    idx2text[idx] = out
                except Exception:
                    continue

    # 重新加载benchmark（用于评测对象与抽样）
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    bench, prompts = prepare_code_benchmark(args, tokenizer)
    N = len(bench)

    # 缺失样本提醒
    missing = sorted(set(range(N)) - set(idx2text.keys()))
    if missing:
        print(f"[main][WARN] Missing {len(missing)} / {N} indices. They will be ignored in evaluation.")

    # 只评测已生成的样本
    paired = sorted(((i, idx2text[i]) for i in idx2text.keys()), key=lambda x: x[0])
    outputs = [[txt] for _, txt in paired]  # list[list[str]] for pass@1
    extracted = [[extract_code(txt) for txt in lst] for lst in outputs]

    # 对齐到相同的实例子集
    # 注意：code_evaluation 的 Instance 依赖其内部索引顺序
    sub_bench = [bench[i] for i, _ in paired]

    # 保存 predictions
    save_results = [
        inst.insert_output(outputs_list, extracted_list)
        for inst, (outputs_list, extracted_list) in zip(sub_bench, zip(outputs, extracted))
    ]
    pred_path = os.path.join(output_dir, f"{base_name}.predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # 评测
    eval_samples = [inst.get_evaluation_sample() for inst in sub_bench]
    generations = extracted  # [[code_str]]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=12,
        timeout=50,
    )
    pass_at_1 = metrics[0].get("pass@1", None)
    print(f"[main] pass@1 = {pass_at_1}")

    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    save_eval_results = [
        inst.insert_output_evaluation(out_list, ext_list, grade_list, metadata=meta)
        for inst, out_list, ext_list, grade_list, meta in zip(
            sub_bench, outputs, extracted, graded, metadatas
        )
    ]

    metrics_path = os.path.join(output_dir, f"{base_name}.metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    code_eval_path = os.path.join(output_dir, f"{base_name}.code_eval.jsonl")
    with open(code_eval_path, "w", encoding="utf-8") as f:
        json.dump(save_eval_results, f, indent=2, ensure_ascii=False)

    print(f"[main] Saved predictions to: {pred_path}")
    print(f"[main] Saved metrics to: {metrics_path}")
    print(f"[main] Saved per-instance eval to: {code_eval_path}")


def main():
    parser = argparse.ArgumentParser()
    # 通用
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)  # 兼容你的CLI，代码评测不使用此目录
    parser.add_argument('--dataset', type=str, required=True)      # 形如 Code_livecodebenchv2
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--run_id', type=str, default="")
    # 生成与采样
    parser.add_argument('--max_generated_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--remove_bos', action="store_true", default=True)
    # Steering
    parser.add_argument('--steer_vector_path', type=str, required=True)
    parser.add_argument('--steer_layer', type=int, default=22)
    parser.add_argument('--steer_coef', type=float, default=0.0)

    args = parser.parse_args()
    set_seed(42)

    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))

    # 汇总并评测（父进程在所有rank结束后执行）
    aggregate_and_evaluate(args)


if __name__ == "__main__":
    main()
