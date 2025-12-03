import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model


class ESM2Base150M(nn.Module):
    """
    使用 HuggingFace ESM2 base (150M) 的最简在线推理封装。

    - 默认使用 facebook/esm2_t30_150M_UR50D
    - forward / encode(seq_list) 返回每条序列的 [CLS/mean] 级别表示 (batch, hidden)
    - 通过 freeze 控制是否参与微调
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        device: str | None = None,
        fp16: bool = False,
        freeze: bool = True,
        pooling: str = "mean",
        max_length: int = 1022,
        # LoRA 相关配置
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        base_model = AutoModel.from_pretrained(model_name)

        # ===== LoRA 接入逻辑 =====
        # - 若 use_lora=True：在注意力投影层上挂 LoRA 适配器
        #   EsmModel 的注意力实现为 Bert 风格，线性层命名为 query/key/value/dense
        # - 默认只训练 LoRA 参数，冻结 base_model 的原始权重，显著降低可训练参数量
        if use_lora:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                # ESM2 (HuggingFace EsmModel) 中注意力层的线性层命名为 query/key/value/dense
                # 对应你想要的 q_proj/k_proj/v_proj/out_proj 功能位置
                target_modules=["query", "key", "value", "dense"],
                task_type="SEQ_CLS",  # 这里只做序列级表征/下游分类
            )
            # 冻结 base 模型权重，仅训练 LoRA 层
            for p in base_model.parameters():
                p.requires_grad = False
            self.model = get_peft_model(base_model, peft_config)
        else:
            self.model = base_model
            # 不使用 LoRA 时，可选地整体冻结 ESM2
            if freeze:
                for p in self.model.parameters():
                    p.requires_grad = False

        self.hidden_size = self.model.config.hidden_size
        self.fp16 = fp16
        self.pooling = pooling
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    def _pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        将 token 级别表示聚合为序列级别表示。
        默认使用非 padding token 的 mean pooling。
        """
        if self.pooling == "cls":
            # 一般 ESM2 的 first token 可以视作 CLS
            return token_embeddings[:, 0]

        # mean pooling over valid tokens
        mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
        masked = token_embeddings * mask
        summed = masked.sum(dim=1)  # (B, H)
        lengths = mask.sum(dim=1).clamp(min=1)  # 避免除 0
        return summed / lengths

    @torch.no_grad()
    def encode(self, sequences: list[str]) -> torch.Tensor:
        """
        推理（不求导）模式下，将一批氨基酸序列转为 ESM2 表示。

        Args:
            sequences: List[str]，每个是氨基酸序列，如 "MSEQNNTEMTFQIQRIYTK..."

        Returns:
            Tensor, 形状为 (batch_size, hidden_size)
        """
        self.model.eval()
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.fp16 and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)

        token_embeddings = outputs.last_hidden_state  # (B, L, H)
        seq_embeddings = self._pool(token_embeddings, batch["attention_mask"])
        return seq_embeddings

    def forward(self, sequences: list[str]) -> torch.Tensor:
        """
        兼容 nn.Module 的 forward。
        - 如果 freeze=False，则会保留梯度，可与下游任务一起微调。
        """
        batch = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.fp16 and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)

        token_embeddings = outputs.last_hidden_state  # (B, L, H)
        seq_embeddings = self._pool(token_embeddings, batch["attention_mask"])
        return seq_embeddings


if __name__ == "__main__":
    # 最简在线推理示例
    esm2 = ESM2Base150M(freeze=True)  # 仅推理，不微调

    seqs = [
        "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWMEK",
        "MKKLLFAIPLVVPFYSHSAVSADKDNVVVIGAGPSGLGKT",
    ]

    with torch.no_grad():
        emb = esm2.encode(seqs)

    print("emb shape:", emb.shape)  # (batch, hidden_size)



