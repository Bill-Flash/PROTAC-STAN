import os
import time
import copy

import toml
import torch
import wandb

from data_loader import PROTACLoader, collate_fn
from model import PROTAC_STAN
from main import setup_seed, train


def load_cfg_two_stages(cfg_path: str):
    """
    从配置文件读取基础配置，并构造：
      - model_cfg_stage1: 只训练下游（固定 ESM+LoRA，可缓存）
      - model_cfg_stage2: 只训练 LoRA（不缓存）
    """
    cfg = toml.load(cfg_path)

    model_cfg_base = cfg["model"]
    train_cfg = cfg["train"]

    # === Stage 1: use_lora=True, freeze_esm=True（固定 ESM+LoRA，可缓存） ===
    model_cfg_s1 = copy.deepcopy(model_cfg_base)
    protein_s1 = model_cfg_s1["protein"]
    protein_s1["use_lora"] = True
    protein_s1["freeze_esm"] = True

    # === Stage 2: use_lora=True, freeze_esm=False（只训 LoRA，不缓存） ===
    model_cfg_s2 = copy.deepcopy(model_cfg_base)
    protein_s2 = model_cfg_s2["protein"]
    protein_s2["use_lora"] = True
    # freeze_esm 对 LoRA 来说只影响 base 模型，真实是否缓存由参数的 requires_grad 决定
    protein_s2["freeze_esm"] = False

    return cfg, model_cfg_s1, model_cfg_s2, train_cfg


def freeze_for_stage1(model: torch.nn.Module) -> None:
    """
    LoRA-only 训练：
      - 只训练 LoRA 参数（名称中包含 "lora_"）
      - 冻结：ESM base（.esm.）以及所有 STAN / 下游模块
    """
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def freeze_for_stage2_downstream_only(model: torch.nn.Module) -> None:
    """
    下游-only 训练：
      - 只训练下游模块（PROTAC GNN、蛋白 adapter/fc、TAN、MLP）
      - 冻结：ESM base + LoRA（名称中包含 ".esm." 或 "lora_" 的参数）

    在这种设置下，OnlineESMProteinEncoder 会检测到 ESM(+LoRA) 全部冻结，
    从而启用序列级缓存，加速训练。
    """
    for name, p in model.named_parameters():
        if (".esm." in name) or ("lora_" in name):
            p.requires_grad = False
        else:
            p.requires_grad = True


def load_stage1_weights_into_stage2(
    stage1_state_path: str,
    model_stage2: torch.nn.Module,
) -> None:
    """
    将 Stage 1 训练好的下游权重，尽量拷贝到带 LoRA 的 Stage 2 模型中：
      - 只加载 shape 完全一致的参数（主要是 PROTAC GNN / adapter / TAN / MLP）
      - LoRA / ESM 主干中 shape 不匹配的键直接跳过
    """
    state_s1 = torch.load(stage1_state_path, map_location="cpu")
    state_s2 = model_stage2.state_dict()

    loaded, skipped = 0, 0
    for k, v in state_s1.items():
        if k in state_s2 and state_s2[k].shape == v.shape:
            state_s2[k] = v
            loaded += 1
        else:
            skipped += 1

    model_stage2.load_state_dict(state_s2)
    print(f"[Stage2 init] 从 Stage1 加载参数 {loaded} 个，跳过 {skipped} 个（多为 LoRA / ESM 权重）。")


def main() -> None:
    # === 路径准备 ===
    model_dir = f"saved_models/{time.strftime('%Y%m%d')}/{time.strftime('%H%M%S')}"
    os.makedirs(model_dir, exist_ok=True)

    # === 读取 2-stage 专用配置，构造两阶段 config ===
    # 不影响原来的 config.toml + main.py 单阶段训练
    cfg, model_cfg_s1, model_cfg_s2, train_cfg = load_cfg_two_stages("config_2stage.toml")

    setup_seed(cfg["model"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    protein_base = cfg["model"]["protein"]
    use_online_esm = protein_base.get("mode", "legacy") == "online_esm"

    train_loader, test_loader = PROTACLoader(
        root="data/protacdb3",
        name="protac_maccs",
        batch_size=train_cfg["batch_size"],
        collate_fn=collate_fn,
        train_ratio=train_cfg["train_ratio"],
        seed=cfg["model"]["seed"],
        use_online_esm=use_online_esm,
    )

    # === Weights & Biases ===
    wandb.init(
        mode="online",
        project="protac-stan",
        config=cfg,
        group=f"two_stage_cache_lora_bz{train_cfg['batch_size']}_lr{train_cfg['learning_rate']}",
    )
    wandb.run.summary["model_dir"] = model_dir

    # =========================================================
    # Stage 1：固定 ESM+LoRA（启用缓存），只训练下游
    # =========================================================
    print("\n========== Stage 1: 固定 ESM+LoRA（启用缓存），只训练下游 ==========")
    model_s1 = PROTAC_STAN(model_cfg_s1)
    print(model_s1)
    wandb.watch(model_s1)

    # Stage 1：只训练下游模块，冻结 ESM+LoRA
    freeze_for_stage2_downstream_only(model_s1)

    model_s1 = train(
        model_s1,
        train_loader,
        test_loader,
        device,
        lr=train_cfg["learning_rate"],          # 下游学习率
        num_epochs=train_cfg["num_epochs"],
        lora_lr=None,                           # 仅训练下游，不给 LoRA 单独 lr
    )

    stage1_state_path = os.path.join(model_dir, "stage1_state_dict.pt")
    torch.save(model_s1.state_dict(), stage1_state_path)
    print(f"Stage 1 完成，state_dict 保存到: {stage1_state_path}")

    # =========================================================
    # Stage 2：只训练 LoRA（不缓存），下游与 ESM base 冻结
    # =========================================================
    print("\n========== Stage 2: 只训练 LoRA（ESM+下游冻结，不使用缓存） ==========")
    model_s2 = PROTAC_STAN(model_cfg_s2)
    print(model_s2)

    # 用 Stage1 训练好的下游权重初始化 Stage2，LoRA 初值保持一致
    load_stage1_weights_into_stage2(stage1_state_path, model_s2)

    # Stage 2：只训练 LoRA，冻结 ESM base + 下游
    freeze_for_stage1(model_s2)

    model_s2 = train(
        model_s2,
        train_loader,
        test_loader,
        device,
        lr=train_cfg["learning_rate"],
        num_epochs=train_cfg["num_epochs"],
        lora_lr=train_cfg.get("lora_learning_rate", 1e-5),  # LoRA 小学习率
    )

    final_full_path = os.path.join(model_dir, "model_two_stage_cache_lora.pt")
    final_state_path = os.path.join(model_dir, "model_two_stage_cache_lora_state_dict.pt")

    torch.save(model_s2, final_full_path)
    torch.save(model_s2.state_dict(), final_state_path)

    # 覆盖 inference 使用的默认路径
    final_infer_path = "saved_models/protac-stan.pt"
    os.makedirs(os.path.dirname(final_infer_path), exist_ok=True)
    torch.save(model_s2.state_dict(), final_infer_path)
    print(f"最终两阶段（Stage1: 下游微调 + ESM+LoRA 缓存；Stage2: 仅 LoRA 微调）模型 state_dict 已保存到 {final_infer_path}")

    wandb.finish()


if __name__ == "__main__":
    main()


