# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from typing import Optional

import torch
from torch.utils.data import TensorDataset, ConcatDataset

# =========================
# 词表（命中即视为“包含词汇”）
# 来自关键词脚本，保留原有模式与正则构造
# =========================
LEXICON_BASE = [
    # 不确定/反思类的低信心词汇////计算类/不熟悉类的低信心词汇
    "alternatively", "alternative", "another", "perhaps", "maybe", "wait", "but",
    "think again", "make sure", "just to ensure", "there any other", "some other",
    "should consider", "about whether", "if they have", "i was", "any errors",
    "or something", "let me check", "hold on", "double check", "however",
    "confusing", "differently", "careful", "sometimes", "alternate"
]

def _normalize_text(s: str) -> str:
    # 引号 & 破折号/连字符统一
    return (s or "")\
        .replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')\
        .replace("–", "-").replace("—", "-").replace("-", "-")  # en/em/nb-hyphen

def _token_pattern(tok: str) -> str:
    """
    把一个词/短语转成更鲁棒的 regex 片段：
    - 空白折叠为 \s+
    - 允许 'double[-\s]check' 这种用空格或连字符连接
    - 给常见词尾加可选变体
    """
    tok = _normalize_text(tok).lower().strip()
    parts = re.split(r"\s+", tok)

    def word2regex(w: str) -> str:
        if w == "verify":
            return r"verif(?:y|ies|ied|ying|ication(?:s)?)"
        if w == "alternative":
            return r"alternative(?:s|ly)?"
        if w == "confusing":
            return r"confus(?:e|es|ed|ing)"
        if w == "differently":
            return r"different(?:ly)?"
        if w == "careful":
            return r"careful(?:ly)?"
        if w == "sometimes":
            return r"sometimes?"
        if w == "alternate":
            return r"alternat(?:e|es|ed|ing)"
        # 默认严格词形
        return re.escape(w)

    if len(parts) >= 2:
        pieces = [word2regex(p) for p in parts]
        between = r"(?:[-\s]+)"  # 空格或连字符
        body = between.join(pieces)
    else:
        body = word2regex(parts[0])

    return rf"\b{body}\b"

def _compile_lexicon_regex(lexicon_base):
    expanded = []
    for term in lexicon_base:
        t = term.strip()
        if not t:
            continue
        expanded.append(_token_pattern(t))
    pattern = "|".join(expanded)
    return re.compile(pattern, flags=re.IGNORECASE)

LEXICON_RE = _compile_lexicon_regex(LEXICON_BASE)

def has_lexicon_hit(text: str) -> int:
    return 1 if LEXICON_RE.search(_normalize_text(text).lower()) else 0


# ----------------------
# Utilities
# ----------------------
def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def split_segments_for_conf(resp_text: str):
    """
    仅统计 </think> 之前的内容；再按连续空行分段。
    """
    text = resp_text or ""
    if "</think>" in text:
        text = text.split("</think>")[0]
    segs = re.split(r"\n\n+", text)
    segs = [s.strip() for s in segs if len(s.strip()) > 0]
    return segs


# ----------------------
# Dataset builder: (feature, label_mixed)
# label_mixed = 1 当 (关键词命中) OR (conf < threshold)，否则 0
# 对齐策略与两个来源脚本一致，做最小合并：
#   - V = hidden['step'].shape[0]
#   - C_raw = len(sentence_confidences)
#   - 期望 V == C_raw - expected_offset
#   - 关键词分段来自 generated_responses 的 </think> 前内容
#   - 最终长度 m = min(V, len(lex_labels), len(confs_aligned))
# ----------------------
def build_dataset_from_layer_mixed(
    layer_id: int,
    json_item: dict,
    hidden_path: str,
    threshold: float = 0.7,
    expected_offset: int = 1,
    verbose: bool = False
) -> Optional[TensorDataset]:
    confs_raw = json_item.get("sentence_confidences", [])
    if not confs_raw or len(confs_raw) <= expected_offset:
        if verbose:
            print("⛔ 跳过：sentence_confidences 为空或过短")
        return None

    if not os.path.exists(hidden_path):
        if verbose:
            print(f"❌ 跳过：文件不存在 {hidden_path}")
        return None

    try:
        data = torch.load(hidden_path, map_location='cpu', weights_only=False)
    except Exception as e:
        if verbose:
            print(f"❌ 加载失败 {hidden_path}: {e}")
        return None

    if layer_id not in data:
        if verbose:
            print(f"❌ 跳过：layer {layer_id} 不存在于文件中")
        return None

    layer_data = data[layer_id]
    sample_id = 0
    if sample_id not in layer_data or 'step' not in layer_data[sample_id]:
        if verbose:
            print(f"❌ 跳过：缺失 sample_id={sample_id} 或 step")
        return None

    tensor = layer_data[sample_id]['step']  # [V, hidden_dim]
    V = tensor.shape[0]
    C_raw = len(confs_raw)

    if V != C_raw - expected_offset:
        if verbose:
            print(f"❌ 向量数 {V} 与原始信心数 {C_raw} 不满足差 {expected_offset} -> 跳过")
        return None

    # 1) 关键词标签（与关键词脚本保持一致）
    responses_text = (json_item.get("generated_responses") or [""])[0]
    segs = split_segments_for_conf(responses_text)
    if expected_offset > 0 and len(segs) >= expected_offset:
        segs = segs[expected_offset:]
    lex_labels = [float(has_lexicon_hit(s)) for s in segs]  # 0/1

    # 2) 置信度标签（与置信度脚本保持一致：低信心=1，高信心=0）
    confs = confs_raw[expected_offset:]
    # 截断保证不会超出 V
    confs = confs[:V]
    conf_labels = [1.0 if float(c) < threshold else 0.0 for c in confs]

    # 3) 对齐长度 m
    m = min(V, len(lex_labels), len(conf_labels))
    if m <= 0:
        if verbose:
            print("❌ 对齐后长度为 0 -> 跳过")
        return None

    features = tensor[:m].to(dtype=torch.float32)
    lex_labels_t = torch.tensor(lex_labels[:m], dtype=torch.float32)
    conf_labels_t = torch.tensor(conf_labels[:m], dtype=torch.float32)

    # 4) Mixed 规则：只要命中关键词 OR 低信心即为 1
    labels_mixed = torch.clamp(lex_labels_t + conf_labels_t, max=1.0)

    return TensorDataset(features, labels_mixed)


def batch_build_all_mixed(
    layer_id: int,
    jsonl_path: str,
    hidden_dir: str,
    threshold: float = 0.7,
    max_files: int = 100,
    expected_offset: int = 1,
    verbose: bool = True
) -> ConcatDataset:
    data_json = read_jsonl(jsonl_path)
    datasets = []
    total = min(max_files, len(data_json))
    kept = 0
    for i in range(total):
        hidden_path = os.path.join(hidden_dir, f"hidden_{i}.pt")
        ds = build_dataset_from_layer_mixed(
            layer_id=layer_id,
            json_item=data_json[i],
            hidden_path=hidden_path,
            threshold=threshold,
            expected_offset=expected_offset,
            verbose=False
        )
        if ds is not None:
            datasets.append(ds)
            kept += len(ds)
        else:
            if verbose:
                print(f"[skip] index={i}")
        if (i + 1) % 50 == 0 and verbose:
            print(f"[progress] processed {i+1}/{total}, collected samples: {kept}")

    if not datasets:
        raise RuntimeError("❌ 没有可用的数据集被成功构建")
    merged = ConcatDataset(datasets)
    if verbose:
        print(f"\n🎉 合并完成，共 {len(merged)} 条样本，文件数 {len(datasets)} / {total}")
    return merged


# ----------------------
# 唯一的 steer：S = mean(hit) - mean(nonhit)
# ----------------------
def build_steer_vector_mean_only(merged_dataset: ConcatDataset) -> torch.Tensor:
    hits = []
    nonhits = []
    for i in range(len(merged_dataset)):
        feat, y = merged_dataset[i]   # feat: [hidden_dim], y: {0.,1.}
        if int(y.item()) == 1:
            hits.append(feat)
        else:
            nonhits.append(feat)

    if len(hits) == 0 or len(nonhits) == 0:
        raise RuntimeError("需要同时包含 命中(1) 与 未命中(0) 的样本才能计算差向量。")

    mu_hit = torch.stack(hits, dim=0).mean(dim=0)
    mu_non = torch.stack(nonhits, dim=0).mean(dim=0)
    S = mu_hit - mu_non  # 不做任何归一化
    return S, len(hits), len(nonhits)


# ----------------------
# Main (CLI)
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build steer vector (mixed): mean(H_hit) - mean(H_nonhit), hit = (lexicon OR conf<threshold)"
    )
    parser.add_argument("--layer_id", type=int, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--hidden_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="低信心阈值：conf < threshold 视为低信心")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--expected_offset", type=int, default=1,
                        help="期望 V = len(sentence_confidences) - expected_offset")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    merged = batch_build_all_mixed(
        layer_id=args.layer_id,
        jsonl_path=args.jsonl_path,
        hidden_dir=args.hidden_dir,
        threshold=args.threshold,
        max_files=args.max_files,
        expected_offset=args.expected_offset,
        verbose=True
    )

    S, n_hit, n_non = build_steer_vector_mean_only(merged)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(S.cpu(), args.save_path)

    l2 = float(torch.linalg.norm(S))
    print(f"✅ 已保存 steer 向量到 {args.save_path}")
    print(f"   dim={S.numel()} | ||S||_2 = {l2:.6f} | hit={n_hit} | nonhit={n_non}")

if __name__ == "__main__":
    main()
