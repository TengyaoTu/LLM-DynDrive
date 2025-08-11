import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM
from accelerate import dispatch_model, infer_auto_device_map
from re import split as rsplit


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def calculate_avg_logprob(logits, token_ids):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    chosen_probs = probs[range(len(token_ids)), token_ids]
    return torch.exp(torch.mean(torch.log(chosen_probs + 1e-10))).item()


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    if args.num_gpus > 1:
        print(f"🚀 Using {args.num_gpus} GPUs with Accelerate dispatch")
        max_memory = {i: "30GiB" for i in range(args.num_gpus)}
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Qwen2DecoderLayer"]
        )
        model = dispatch_model(model, device_map=device_map)
    else:
        print("🚀 Using single GPU.")
        model = model.cuda()

    model.eval()
    return model, tokenizer


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
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)

    # 设置 steer 向量
    steer_vector = torch.load(args.steer_vector_path)
    if hasattr(model, 'device'):
        steer_vector = steer_vector.to(model.device)
    else:
        # 多 GPU 情况：放入目标层设备
        target_layer = model.model.layers[args.steer_layer]
        steer_vector = steer_vector.to(next(target_layer.parameters()).device)

    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.steer_layer,
        steer_vec=steer_vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer
    )

    # 加载数据集
    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    questions = read_jsonl(dataset_path)
    output_dir = os.path.join(args.output_path, os.path.basename(args.model_name_or_path), args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'steer_temp{args.temperature}_maxlen{args.max_generated_tokens}.jsonl')

    # 支持断点续跑
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = set(json.loads(line)['question'] for line in f if line.strip())
    else:
        existing = set()

    pbar = tqdm(total=len(questions), desc="Steer Inference")

    for q in questions:
        if q['problem'] in existing:
            pbar.update(1)
            continue

        model.start_new_round()

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q["problem"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
            print(f"[OOM] Skipping: {q['problem'][:60]}...")
            torch.cuda.empty_cache()
            break

        response_text = tokenizer.decode(output.sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        scores = torch.stack(output.scores, dim=1)[0]
        token_ids = output.sequences[0][inputs["input_ids"].shape[-1]:]

        # 信心值评估
        text_segments = rsplit(r'\n\n+', response_text.split('</think>')[0])
        confidences = []
        start = 0
        for segment in text_segments:
            seg_ids = tokenizer(segment, add_special_tokens=False)['input_ids']
            end = start + len(seg_ids)
            if end > len(token_ids): break
            conf = calculate_avg_logprob(scores[start:end], token_ids[start:end])
            confidences.append(conf)
            start = end

        result = {
            "question": q["problem"],
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "sentence_confidences": confidences
        }

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        pbar.update(1)

    pbar.close()
    print(f"✅ 推理完成，结果保存在 {output_file}")


if __name__ == "__main__":
    main()

