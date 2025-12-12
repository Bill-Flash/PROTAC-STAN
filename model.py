import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, global_max_pool, GINConv
from torch_geometric.utils import add_self_loops, degree

# from torch.nn.utils.weight_norm import weight_norm


class PROTAC_STAN(nn.Module):
    ## TODO: 微调ESM小模型
    def __init__(self, cfg):
        super(PROTAC_STAN, self).__init__()
        fingerprint_dim = cfg['protac'].get('fingerprint_dim', 166)
        protac_embed_dim = cfg['protac']['embed']
        protein_out_dim = cfg['protein']['out_dim']

        self.protac_encoder = MolecularEncoder(
            num_mol_features=cfg['protac']['feature'],
            embedding_dim=protac_embed_dim,
            hidden_channels=cfg['protac']['hidden'],
            edge_dim=cfg['protac']['edge_dim'],
            fingerprint_dim=fingerprint_dim  # MACCS指纹维度（从配置读取，默认166）
        )
        self.e3_ligase_encoder = ProteinEncoder(
            embedding_dim=cfg['protein']['embed'],
            hidden=cfg['protein']['hidden'],
            out_dim=cfg['protein']['out_dim'],
        )
        self.poi_encoder = ProteinEncoder(
            embedding_dim=cfg['protein']['embed'],
            hidden=cfg['protein']['hidden'],
            out_dim=protein_out_dim,
        )

        # Baseline-B: 只使用两两 Hadamard 交互项拼接，得到 3 * 64 = 192 维
        self.mlp = nn.Sequential(
            nn.Linear(cfg['clf']['embed'], cfg['clf']['hidden']),
            nn.BatchNorm1d(cfg['clf']['hidden']),
            nn.ReLU(),
            nn.Linear(cfg['clf']['hidden'], cfg['clf']['class']),
        )

        # CLIP-style 对比学习投影头：将 PROTAC 和 (E3, POI) 映射到同一对比空间
        contrast_cfg = cfg.get('contrast', {})
        proj_dim = contrast_cfg.get('proj_dim', protein_out_dim)
        self.protac_proj = nn.Linear(protac_embed_dim, proj_dim)
        # (E3, POI) 先拼接成 2 * protein_out_dim
        self.et_proj = nn.Linear(2 * protein_out_dim, proj_dim)

    def forward(self, protac, e3_ligase, poi, mode='train', fingerprint=None, return_embeddings=False):
        protac_embedding = self.protac_encoder(protac, fingerprint=fingerprint)   # [B, 64]
        e3_ligase_embedding = self.e3_ligase_encoder(e3_ligase)                   # [B, 64]
        poi_embedding = self.poi_encoder(poi)                                     # [B, 64]

        # 显式交互：两两 Hadamard 乘积
        pe = protac_embedding * e3_ligase_embedding   # [B, 64]
        pp = protac_embedding * poi_embedding         # [B, 64]
        ep = e3_ligase_embedding * poi_embedding      # [B, 64]

        # 只拼接三对交互项，最终维度为 3 * 64 = 192，对齐 cfg['clf']['embed']
        joint_embedding = torch.cat([pe, pp, ep], dim=1)
        logits = self.mlp(joint_embedding)

        # CLIP-style 对比学习用的投影向量（L2 归一化）
        et_embedding = torch.cat([e3_ligase_embedding, poi_embedding], dim=1)  # [B, 2 * protein_out_dim]
        z_protac = F.normalize(self.protac_proj(protac_embedding), dim=-1)     # [B, proj_dim]
        z_et = F.normalize(self.et_proj(et_embedding), dim=-1)                 # [B, proj_dim]

        # 返回原始 logits，供 CrossEntropyLoss 使用；可选返回对比学习 embedding
        if return_embeddings:
            return logits, z_protac, z_et

        if mode == 'train':
            return logits
        elif mode == 'eval':
            # 兼容原有接口，注意力图用 None 占位
            return logits, None
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


class EdgedGINConv(MessagePassing):
    """GINConv with edge features: message = h_u + edge_mlp(edge_attr)"""
    def __init__(self, in_channels, out_channels, edge_dim, eps=0.0):
        super(EdgedGINConv, self).__init__(aggr='add')
        self.nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.edge_mlp = nn.Linear(edge_dim, in_channels)
        self.eps = eps

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)), dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        out = self.nn((1 + self.eps) * x + out)
        return out

    def message(self, x_j, edge_attr):
        return x_j + self.edge_mlp(edge_attr)


class MolecularEncoder(nn.Module):
    def __init__(self, num_mol_features, embedding_dim, hidden_channels, edge_dim, fingerprint_dim=166, dropout=0.1):
        super(MolecularEncoder, self).__init__()
        self.lin = nn.Linear(num_mol_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        # 原始 GIN：只使用节点特征，不使用边特征
        self.conv1 = GINConv(nn.Sequential(nn.Linear(embedding_dim, hidden_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_channels, embedding_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim)))
        self.fingerprint_lin = nn.Linear(fingerprint_dim, embedding_dim)

    def forward(self, data, fingerprint=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.lin(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_max_pool(x, batch)
        
        if fingerprint is not None:
            x = x + self.fingerprint_lin(fingerprint)
        
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
