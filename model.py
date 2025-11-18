import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, global_max_pool
from torch_geometric.utils import add_self_loops, degree

from tan import TAN
# from torch.nn.utils.weight_norm import weight_norm


class PROTAC_STAN(nn.Module):
    ## TODO: 微调ESM小模型
    def __init__(self, cfg):
        super(PROTAC_STAN, self).__init__()
        self.protac_encoder = MolecularEncoder(
            num_mol_features=cfg['protac']['feature'], 
            embedding_dim=cfg['protac']['embed'],
            hidden_channels=cfg['protac']['hidden'], 
            edge_dim=cfg['protac']['edge_dim']
        )
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

        self.tan = TAN(cfg['tan']['in_dims'], cfg['clf']['embed'], cfg['tan']['heads'])
        self.mlp = nn.Sequential(
            nn.Linear(cfg['clf']['embed'], cfg['clf']['hidden']),
            nn.BatchNorm1d(cfg['clf']['hidden']),
            nn.ReLU(),
            nn.Linear(cfg['clf']['hidden'], cfg['clf']['class']),
        )

    def forward(self, protac, e3_ligase, poi, mode='train'):
        protac_embedding = self.protac_encoder(protac)
        e3_ligase_embedding = self.e3_ligase_encoder(e3_ligase)
        poi_embedding = self.poi_encoder(poi)
        
        atts = None
        
        joint_embedding, atts = self.tan(
            protac_embedding.unsqueeze(2),
            e3_ligase_embedding.unsqueeze(2),
            poi_embedding.unsqueeze(2),
        )
        output = self.mlp(joint_embedding)

        pred =  F.log_softmax(output, dim=1)

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
    def __init__(self, num_mol_features, embedding_dim, hidden_channels, edge_dim):
        super(MolecularEncoder, self).__init__()
        self.lin = nn.Linear(num_mol_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.conv1 = EdgedGCNConv(embedding_dim, hidden_channels, edge_dim)
        self.conv2 = EdgedGCNConv(hidden_channels, embedding_dim, edge_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.lin(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_max_pool(x, batch)  # [batch_size, 64]
        
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
