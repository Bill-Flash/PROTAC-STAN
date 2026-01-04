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
        protac_embed_dim = cfg['protac']['embed']  # PROTAC encoder 输出维度
        protein_out_dim = cfg['protein']['out_dim']

        # Bag-of-atoms baseline：不做图卷积，只对节点特征做图级聚合 + MLP
        # 如需恢复原来的 GNN 编码器，可将下面一行改回 MolecularEncoder(...)
        self.protac_encoder = BagOfAtomsEncoder(
            num_mol_features=cfg['protac']['feature'],
            embedding_dim=protac_embed_dim,
            hidden_channels=cfg['protac']['hidden'],
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

        # CLIP-style 三模态对比学习投影头：
        # 将 PROTAC / E3 / POI 以及二元复合体分别映射到同一对比空间
        contrast_cfg = cfg.get('contrast', {})
        proj_dim = contrast_cfg.get('proj_dim', protein_out_dim)
        self.protac_proj = nn.Linear(protac_embed_dim, proj_dim)
        self.e3_proj = nn.Linear(protein_out_dim, proj_dim)
        self.poi_proj = nn.Linear(protein_out_dim, proj_dim)
        # 二元复合体模态：PE（PROTAC-E3）、PO（PROTAC-POI）
        # pe、pp 的维度与单体 embedding 一致（默认 64），因此也映射到相同 proj_dim
        self.pe_proj = nn.Linear(protein_out_dim, proj_dim)
        self.po_proj = nn.Linear(protein_out_dim, proj_dim)

    def forward(self, protac, e3_ligase, poi, mode='train', fingerprint=None, return_embeddings=False):
        """
        方案二：在共享 encoder 上“层内解耦”：
        - encoder 内部划分 shared_block / ce_block / contrast_block
        - 分类只用 ce_block 输出
        - 对比只用 contrast_block 输出
        """
        # -------------------------
        # 1) encoder 前向：得到 shared / ce / contrast 三路特征
        # -------------------------
        _, protac_ce, protac_contrast = self.protac_encoder(protac, fingerprint=fingerprint)  # [B, 64] * 2
        _, e3_ce, e3_contrast = self.e3_ligase_encoder(e3_ligase)                             # [B, 64] * 2
        _, poi_ce, poi_contrast = self.poi_encoder(poi)                                       # [B, 64] * 2

        # -------------------------
        # 2) 分类分支：只用 CE 头做两两 Hadamard
        # -------------------------
        pe_ce = protac_ce * e3_ce     # [B, 64]
        pp_ce = protac_ce * poi_ce    # [B, 64]
        ep_ce = e3_ce * poi_ce        # [B, 64]

        # 只拼接三对交互项，最终维度为 3 * 64 = 192，对齐 cfg['clf']['embed']
        joint_embedding = torch.cat([pe_ce, pp_ce, ep_ce], dim=1)
        logits = self.mlp(joint_embedding)

        # -------------------------
        # 3) 对比分支：只用 contrast 头 + 投影头
        # -------------------------
        # 单体模态
        z_protac = F.normalize(self.protac_proj(protac_contrast), dim=-1)    # [B, proj_dim]
        z_e3 = F.normalize(self.e3_proj(e3_contrast), dim=-1)                # [B, proj_dim]
        z_poi = F.normalize(self.poi_proj(poi_contrast), dim=-1)             # [B, proj_dim]

        # 复合模态：二元复合体基于 contrast 表征构造
        pe_contrast = protac_contrast * e3_contrast                          # [B, 64]
        po_contrast = protac_contrast * poi_contrast                         # [B, 64]
        z_pe = F.normalize(self.pe_proj(pe_contrast), dim=-1)                # [B, proj_dim]
        z_po = F.normalize(self.po_proj(po_contrast), dim=-1)                # [B, proj_dim]

        # 返回原始 logits，供 CrossEntropyLoss 使用；可选返回对比学习 embedding
        if return_embeddings:
            # 返回三单体 + 两个二元复合体的投影向量
            return logits, z_protac, z_e3, z_poi, z_pe, z_po

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


class BagOfAtomsEncoder(nn.Module):
    """
    Bag-of-atoms baseline:
    - 不做图卷积，只对节点特征做图级聚合（global pooling）再过 MLP
    - 输出维度与 MolecularEncoder 保持一致（embedding_dim），便于公平对比
    """
    def __init__(self, num_mol_features, embedding_dim, hidden_channels, fingerprint_dim=166, dropout=0.1):
        """
        在 Bag-of-atoms encoder 内部做“层内解耦”：
        - shared_block: 图级聚合 + 低层 MLP，输出 hidden 维 shared 表示
        - ce_block:  shared -> embedding_dim，分类专用
        - contrast_block: shared -> embedding_dim，对比专用
        """
        super(BagOfAtomsEncoder, self).__init__()
        hidden = hidden_channels if hidden_channels is not None else embedding_dim

        # shared_block：作用于 pooled 图级向量的低层 MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_mol_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ce_block：分类专用高层
        self.ce_mlp = nn.Linear(hidden, embedding_dim)
        # contrast_block：对比专用高层
        self.contrast_mlp = nn.Linear(hidden, embedding_dim)

        # 指纹映射到 shared 维度，在 shared 层融合
        self.fingerprint_lin = nn.Linear(fingerprint_dim, hidden)

    def forward(self, data, fingerprint=None):
        """
        返回三路特征：
        - shared:   encoder 共享底层表示（hidden 维）
        - ce_feat:  分类专用高层表示（embedding_dim 维）
        - con_feat: 对比专用高层表示（embedding_dim 维）
        """
        # 这里只使用节点特征，不使用 edge_index / edge_attr
        x, batch = data.x, data.batch                    # [num_nodes, num_mol_features], [num_nodes]

        # 图级聚合：bag-of-atoms，可以理解为对所有原子做 pooling 得到整体表示
        mol_x = global_max_pool(x, batch)                # [B, num_mol_features]
        # 如果你想做“求和池化”版本，可以把上面一行替换成 global_add_pool(x, batch)

        # shared_block：低层变换
        shared = self.shared_mlp(mol_x)                  # [B, hidden]

        # 可选：在 shared 层融入 MACCS 指纹信息
        if fingerprint is not None:
            shared = shared + self.fingerprint_lin(fingerprint)  # [B, hidden]

        # ce / contrast 两条私有高层头
        ce_feat = self.ce_mlp(shared)                    # [B, embedding_dim]
        con_feat = self.contrast_mlp(shared)             # [B, embedding_dim]

        return shared, ce_feat, con_feat


class ProteinEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden, out_dim):
        super(ProteinEncoder, self).__init__()
        """
        Protein encoder 的“层内解耦”版：
        - shared_block: adapter，将 ESM embedding 映射到 hidden
        - ce_block:     hidden -> out_dim，分类专用
        - contrast_block: hidden -> out_dim，对比专用
        """
        # shared_block
        self.adapter = nn.Linear(embedding_dim, hidden)
        # 分类专用高层
        self.ce_head = nn.Linear(hidden, out_dim)
        # 对比专用高层
        self.contrast_head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        """
        返回三路特征：
        - shared:  encoder 共享底层表示（hidden 维）
        - ce_out:  分类专用高层表示（out_dim 维）
        - con_out: 对比专用高层表示（out_dim 维）
        """
        shared = self.adapter(x)
        shared = F.relu(shared)

        ce_out = self.ce_head(shared)
        con_out = self.contrast_head(shared)

        return shared, ce_out, con_out
