# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from typing import List, Optional

import torch
from torch.utils.data import TensorDataset, ConcatDataset

# =========================
# 词表（命中即视为“包含词汇”）
# =========================
# =========================
# 词表（基础词干/短语）
# =========================
LEXICON_BASE = [
    # 不确定/反思类的低信心词汇
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
    import re

    tok = _normalize_text(tok).lower().strip()
    # 先分词（按空白）
    parts = re.split(r"\s+", tok)

    def word2regex(w: str) -> str:
        # 为关键词补充常见派生（按需扩展）
        if w == "verify":
            return r"verif(?:y|ies|ied|ying|ication(?:s)?)"
        if w == "alternative":
            return r"alternative(?:s|ly)?"
        if w == "another":
            return r"another"
        if w == "perhaps":
            return r"perhaps"
        if w == "maybe":
            return r"maybe"
        if w == "wait":
            return r"wait"
        if w == "but":
            return r"but"
        if w == "however":
            return r"however"
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

    # 处理双词/多词短语的连接符：允许空格或连字符
    # 例如 "double check" -> r"double(?:[-\s]+)check"
    if len(parts) >= 2:
        pieces = [word2regex(p) for p in parts]
        between = r"(?:[-\s]+)"  # 空格或连字符
        body = between.join(pieces)
    else:
        body = word2regex(parts[0])

    # 两侧词边界
    return rf"\b{body}\b"

def _compile_lexicon_regex(lexicon_base):
    import re
    # 专门为“double check / double-check”这种再做一个短语同义归一
    expanded = []
    for term in lexicon_base:
        t = term.strip()
        if not t:
            continue  # ✅ 删除空字符串带来的全局命中
        # 特例：双词 & 三词中出现 known 连接词时自动扩展
        if t in ["double check", "make sure", "hold on", "think again",
                 "just to ensure", "there any other", "some other",
                 "should consider", "about whether", "or something",
                 "let me check", "if they have", "i was", "any errors"]:
            expanded.append(_token_pattern(t))
        else:
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
# Dataset builder: (feature, label_lex)
# ----------------------
def build_dataset_from_layer(
    layer_id: int,
    json_item: dict,
    hidden_path: str,
    expected_offset: int = 1,
    verbose: bool = False
) -> Optional[TensorDataset]:
    """
    返回每条样本二元组: (feature, label_lex)
    约定：
    - V = hidden['step'].shape[0]
    - C_raw = len(sentence_confidences)
    - 默认检查： V == C_raw - expected_offset
    """
    confs_raw = json_item.get("sentence_confidences", [])
    if not confs_raw or len(confs_raw) <= expected_offset:
        if verbose:
            print("⛔ 跳过：标签为空或过短")
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

    # 词表标签：从分段文本生成，并与 expected_offset 对齐
    responses_text = (json_item.get("generated_responses") or [""])[0]
    segs = split_segments_for_conf(responses_text)
    if expected_offset > 0 and len(segs) >= expected_offset:
        segs = segs[expected_offset:]
    lex_labels = [float(has_lexicon_hit(s)) for s in segs]  # 0/1

    # 对齐长度
    m = min(V, len(lex_labels))
    if m <= 0:
        if verbose:
            print("❌ 对齐后长度为 0 -> 跳过")
        return None

    features = tensor[:m].to(dtype=torch.float32)
    labels_lex = torch.tensor(lex_labels[:m], dtype=torch.float32)
    return TensorDataset(features, labels_lex)

def batch_build_all(
    layer_id: int,
    jsonl_path: str,
    hidden_dir: str,
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
        ds = build_dataset_from_layer(layer_id, data_json[i], hidden_path,
                                      expected_offset=expected_offset, verbose=False)
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
    return S

# ----------------------
# Main (CLI)
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Build steer vector = mean(hit) - mean(nonhit) from hidden states")
    parser.add_argument("--layer_id", type=int, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--hidden_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--expected_offset", type=int, default=1, help="期望 V = C_raw - expected_offset")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    merged = batch_build_all(
        layer_id=args.layer_id,
        jsonl_path=args.jsonl_path,
        hidden_dir=args.hidden_dir,
        max_files=args.max_files,
        expected_offset=args.expected_offset,
        verbose=True
    )

    S = build_steer_vector_mean_only(merged)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(S.cpu(), args.save_path)

    print(f"✅ 已保存 steer 向量到 {args.save_path}")
    print(f"   dim={S.numel()} | ||S||_2 = {float(torch.linalg.norm(S)):.6f}")

if __name__ == "__main__":
    main()
