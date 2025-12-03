import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, global_max_pool
from torch_geometric.utils import add_self_loops, degree

from tan import TAN
# from torch.nn.utils.weight_norm import weight_norm

from esm2_online import ESM2Base150M


class PROTAC_STAN(nn.Module):
    """
    主模型。
    兼容两种蛋白输入方式：
    1) legacy: 直接输入预先计算好的蛋白 embedding（与原始实现一致）
    2) online_esm: 直接输入氨基酸序列，内部调用 ESM2Base150M 进行编码（支持未来微调）
    """

    def __init__(self, cfg):
        super(PROTAC_STAN, self).__init__()
        fingerprint_dim = cfg['protac'].get('fingerprint_dim', 166)

        # small molecule encoder
        self.protac_encoder = MolecularEncoder(
            num_mol_features=cfg['protac']['feature'],
            embedding_dim=cfg['protac']['embed'],
            hidden_channels=cfg['protac']['hidden'],
            edge_dim=cfg['protac']['edge_dim'],
            fingerprint_dim=fingerprint_dim,  # MACCS指纹维度（从配置读取，默认166）
        )

        # === 蛋白编码部分 ===
        protein_mode = cfg['protein'].get('mode', 'legacy')
        self.protein_mode = protein_mode

        if protein_mode == 'legacy':
            # 与原实现完全一致：输入已经是 ESM embedding
            self.e3_ligase_encoder = ProteinEncoder(
                embedding_dim=cfg['protein']['embed'],
                hidden=cfg['protein']['hidden'],
                out_dim=cfg['protein']['out_dim'],
            )
            self.poi_encoder = ProteinEncoder(
                embedding_dim=cfg['protein']['embed'],
                hidden=cfg['protein']['hidden'],
                out_dim=cfg['protein']['out_dim'],
            )

        elif protein_mode == 'online_esm':
            # 在线 ESM2 编码 + 线性适配层，支持未来微调
            freeze_esm = cfg['protein'].get('freeze_esm', True)
            pooling = cfg['protein'].get('pooling', 'mean')

            self.e3_ligase_encoder = OnlineESMProteinEncoder(
                hidden=cfg['protein']['hidden'],
                out_dim=cfg['protein']['out_dim'],
                freeze_esm=freeze_esm,
                pooling=pooling,
            )
            self.poi_encoder = OnlineESMProteinEncoder(
                hidden=cfg['protein']['hidden'],
                out_dim=cfg['protein']['out_dim'],
                freeze_esm=freeze_esm,
                pooling=pooling,
            )
        else:
            raise ValueError(f"Unknown protein mode: {protein_mode}")

        self.tan = TAN(cfg['tan']['in_dims'], cfg['clf']['embed'], cfg['tan']['heads'])
        self.mlp = nn.Sequential(
            nn.Linear(cfg['clf']['embed'], cfg['clf']['hidden']),
            nn.BatchNorm1d(cfg['clf']['hidden']),
            nn.ReLU(),
            nn.Linear(cfg['clf']['hidden'], cfg['clf']['class']),
        )

    def forward(self, protac, e3_ligase, poi, mode='train', fingerprint=None):
        """
        Args:
            protac: 图数据（与原实现相同）
            e3_ligase:
                - 若 protein_mode=='legacy'：Tensor, 预计算好的 embedding
                - 若 protein_mode=='online_esm'：List[str]，E3 ligase 氨基酸序列
            poi:
                - 若 protein_mode=='legacy'：Tensor, 预计算好的 embedding
                - 若 protein_mode=='online_esm'：List[str]，POI 氨基酸序列
            fingerprint: MACCS 指纹（可选）
        """
        protac_embedding = self.protac_encoder(protac, fingerprint=fingerprint)

        if self.protein_mode == 'legacy':
            e3_ligase_embedding = self.e3_ligase_encoder(e3_ligase)
            poi_embedding = self.poi_encoder(poi)
        else:
            # online_esm: 直接从序列编码
            e3_ligase_embedding = self.e3_ligase_encoder(e3_ligase)
            poi_embedding = self.poi_encoder(poi)

        atts = None

        joint_embedding, atts = self.tan(
            protac_embedding.unsqueeze(2),
            e3_ligase_embedding.unsqueeze(2),
            poi_embedding.unsqueeze(2),
        )
        output = self.mlp(joint_embedding)

        pred = F.log_softmax(output, dim=1)

        if mode == 'train':
            return pred
        elif mode == 'eval':
            return pred, atts
        else:
            raise ValueError(f'Unknown mode: {mode}')
        

class EdgedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(EdgedGCNConv, self).__init__(aggr='add')
        
        self.node_lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.edge_lin = torch.nn.Linear(edge_dim, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)), dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        x = self.node_lin(x)
        edge_attr = self.edge_lin(edge_attr)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr, norm=norm)
        out += self.bias

        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

    def __repr__(self):
        return '{}(\n\t(node_lin): {}\n\t(edge_lin): {}\n)'.format(
            self.__class__.__name__, 
            self.node_lin, 
            self.edge_lin, 
        )


class MolecularEncoder(nn.Module):
    ## TODO: 需要修改，使用GINConv代替EdgedGCNConv
    def __init__(self, num_mol_features, embedding_dim, hidden_channels, edge_dim, fingerprint_dim=166):
        super(MolecularEncoder, self).__init__()
        self.lin = nn.Linear(num_mol_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.conv1 = EdgedGCNConv(embedding_dim, hidden_channels, edge_dim)
        self.conv2 = EdgedGCNConv(hidden_channels, embedding_dim, edge_dim)
        
        # MACCS指纹特征处理
        self.fingerprint_lin = nn.Linear(fingerprint_dim, embedding_dim)  # 将166维MACCS指纹映射到64维

    def forward(self, data, fingerprint=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.lin(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_max_pool(x, batch)  # [batch_size, 64]
        
        # 融合MACCS指纹特征
        if fingerprint is not None:
            fingerprint_embed = self.fingerprint_lin(fingerprint)  # [batch_size, 166] -> [batch_size, 64]
            x = x + fingerprint_embed  # 相加融合，保持64维输出
        
        return x


class ProteinEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden, out_dim):
        super(ProteinEncoder, self).__init__()
        self.adapter = nn.Linear(embedding_dim, hidden)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.adapter(x)
        x = F.relu(x)
        x = self.fc(x)
        return x


class OnlineESMProteinEncoder(nn.Module):
    """
    使用 ESM2Base150M 的在线蛋白编码器。

    - 输入：List[str]（一批氨基酸序列）
    - 内部：ESM2Base150M + 两层 MLP 适配到 out_dim
    - freeze_esm=True 时只训练适配层；False 时可端到端微调 ESM2
    """

    def __init__(self, hidden, out_dim, freeze_esm: bool = True, pooling: str = "mean"):
        super().__init__()
        self.esm = ESM2Base150M(freeze=freeze_esm, pooling=pooling)
        self.adapter = nn.Linear(self.esm.hidden_size, hidden)
        self.fc = nn.Linear(hidden, out_dim)

        # 缓存仅在 ESM 冻结时启用：每条唯一序列只计算一次 ESM 表征
        self.freeze_esm = freeze_esm
        self.embedding_cache = {} if freeze_esm else None  # dict[str, torch.Tensor]

    def forward(self, seqs: list[str]):
        """
        seqs: List[str]，当前 batch 的氨基酸序列
        """
        # 冻结 + 缓存模式：每条唯一序列只跑一次 ESM
        if self.freeze_esm and self.embedding_cache is not None:
            device = next(self.adapter.parameters()).device
            hidden_size = self.esm.hidden_size

            # 先为当前 batch 准备占位
            batch_embeddings = [None] * len(seqs)  # type: ignore[var-annotated]
            to_compute = []  # list of (idx, seq)

            for i, s in enumerate(seqs):
                if s in self.embedding_cache:
                    # 已缓存：直接使用（搬到当前设备）
                    batch_embeddings[i] = self.embedding_cache[s].to(device)
                else:
                    to_compute.append((i, s))

            if to_compute:
                # 对未缓存的序列一次性跑 ESM
                _, seq_list = zip(*to_compute)
                new_emb = self.esm(list(seq_list))  # (K, hidden_size)

                for row, (i, s) in enumerate(to_compute):
                    emb_i = new_emb[row]  # (hidden_size,)
                    # 缓存到 CPU，节省显存
                    self.embedding_cache[s] = emb_i.detach().cpu()
                    batch_embeddings[i] = emb_i.to(device)

            # 所有位置都应被填充
            x = torch.stack(batch_embeddings, dim=0)  # type: ignore[arg-type]
        else:
            # 非冻结模式不做缓存，确保梯度正确传回 ESM
            x = self.esm(seqs)

        # 下游适配
        x = self.adapter(x)
        x = F.relu(x)
        x = self.fc(x)
        return x
