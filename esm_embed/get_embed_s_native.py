import os
import os.path as osp
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import esm
from tqdm import tqdm


def load_p_map(root: str) -> Dict[str, str]:
    """读取 Uniprot -> 序列 的映射."""
    path = osp.join(root, "p_map.pkl")
    with open(path, "rb") as f:
        p_map = pickle.load(f)
    return p_map


def clean_sequence(seq: str) -> str:
    """
    对原始序列做最小清洗：
    - 去掉首尾空白
    - 将 U / O / B / Z / J 映射为 X（未知氨基酸）
    其它字符先保留，交给 ESM 自己处理（ESM 原生支持 X / B / Z 等）。
    """
    seq = seq.strip().upper()
    aa_map = str.maketrans({
        "U": "X",  # Selenocysteine
        "O": "X",  # Pyrrolysine
        "B": "X",  # Asx (D/N)
        "Z": "X",  # Glx (E/Q)
        "J": "X",  # Leu/Ile
    })
    return seq.translate(aa_map)


def build_sequence_list(p_map: Dict[str, str]) -> List[Tuple[str, str]]:
    seqs: List[Tuple[str, str]] = []
    for uid, seq in p_map.items():
        if not isinstance(seq, str):
            continue
        seq_clean = clean_sequence(seq)
        if len(seq_clean) == 0:
            continue
        seqs.append((uid, seq_clean))
    return seqs


def compute_esm2_embeddings(
    seqs: List[Tuple[str, str]],
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, np.ndarray]:
    """
    使用原生 ESM2 接口对一批 (uniprot, seq) 计算图级 embedding（mean pooling over residues）。
    默认使用 650M 模型（通过 esm.pretrained.load_model_and_alphabet 加载），
    并在其上加载本地的 esm_650m_s.pth S 版权重。
    返回 {uniprot: D-dim float32 向量（D 由模型隐藏维度决定，650M 为 1280）}.
    """
    model_dir = "./model"
    
    # 1. 使用 esm 官方接口加载 ESM-2 基模（会从本地 cache / 网络获取）
    print(f"Loading ESM model from pretrained name: {model_name}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)

    # 2. 如果本地有 S 版权重，则在基模上加载（保持与原来 ESM-S 一致）
    s_weight_path = osp.join(model_dir, "esm_650m_s.pth")
    if osp.exists(s_weight_path):
        print(f"Loading S weights from {s_weight_path}")
        state_dict = torch.load(s_weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    layer = model.num_layers

    embeddings: Dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(seqs), batch_size), desc="Embedding"):
        batch = seqs[i : i + batch_size]
        data = [(uid, s) for uid, s in batch]
        labels, strs, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[layer], return_contacts=False)
        token_reprs = out["representations"][layer]  # [B, L, C]

        # 去掉特殊 token（CLS / EOS / PAD），对剩余残基做 mean pooling
        special_idx = {alphabet.cls_idx, alphabet.eos_idx, alphabet.padding_idx}
        special_idx_tensor = torch.tensor(list(special_idx), device=tokens.device)

        for (uid, _), toks, rep in zip(batch, tokens, token_reprs):
            mask = ~torch.isin(toks, special_idx_tensor)
            seq_repr = rep[mask]
            if seq_repr.numel() == 0:
                # 极端情况下（全是特殊 token），跳过
                continue
            emb = seq_repr.mean(dim=0).cpu().numpy().astype("float32")
            embeddings[uid] = emb

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="ESM2 native embedding (string input)")
    parser.add_argument(
        "--root",
        type=str,
        default="../data/TPDdb",
        help="Path to the protein data (containing p_map.pkl)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="ESM2 model name in esm.pretrained",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for ESM forward",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="esm_s_map.pkl",
        help="Output pkl filename saved under root",
    )
    args = parser.parse_args()

    root = os.path.expanduser(args.root)

    # 1. 读取 Uniprot -> 序列
    p_map = load_p_map(root)
    seqs = build_sequence_list(p_map)
    print(f"Total sequences in p_map: {len(p_map)}")
    print(f"Sequences to embed after cleaning: {len(seqs)}")

    if not seqs:
        print("No valid sequences to embed, exit.")
        return

    # 2. 计算 ESM2 embedding
    esm_map = compute_esm2_embeddings(
        seqs,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    # 3. 保存
    out_path = osp.join(root, args.output_name)
    with open(out_path, "wb") as f:
        pickle.dump(esm_map, f)

    print(f"Saved {len(esm_map)} embeddings to {out_path}")


if __name__ == "__main__":
    main()


