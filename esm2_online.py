import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


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
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.model.config.hidden_size
        self.fp16 = fp16
        self.pooling = pooling
        self.max_length = max_length

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

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



