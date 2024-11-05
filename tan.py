import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
# from torch.nn.utils.parametrizations import weight_norm
    
    
class TAN(nn.Module):
    '''
    in_dims: [tuple] (x_dim, y_dim, z_dim)
    h_dim: [int] size of fused representations, also rank. K = h_dim * k
    h_out: [int] multiheads
    '''
    def __init__(self, in_dims, h_dim, h_out, dropouts=(0.2, 0.2, 0.2), act='ReLU', k=1):
        super(TAN, self).__init__()

        self.k = k
        self.x_dim = in_dims[0]
        self.y_dim = in_dims[1]
        self.z_dim = in_dims[2]
        self.h_dim = h_dim
        self.h_out = h_out

        self.x_net = FCNet([self.x_dim, h_dim * k], act=act, dropout=dropouts[0])
        self.y_net = FCNet([self.y_dim, h_dim * k], act=act, dropout=dropouts[1])
        self.z_net = FCNet([self.z_dim, h_dim * k], act=act, dropout=dropouts[2])
        
        if 1 < k:
            self.p_net = nn.AvgPool1d(kernel_size=k, stride=k) # pooling layer
            
        self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None) # K x H
        # self.h_net = nn.Linear(h_dim * self.k, h_out) # K x H
        
        self.bn = nn.BatchNorm1d(h_dim)
    
    def attention_pooling(self, x, y, z, att_map):
        xy = torch.einsum('bxk,byk->bxyk', (x, y))
        xy = xy.permute(0, 2, 1, 3).contiguous() # byxk
        fusion_logits = torch.einsum('byxk,bxyz,bzk->bk', (xy, att_map, z))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b1k
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, x, y, z, softmax=True):
        x_num = x.size(1)
        y_num = y.size(1)
        z_num = z.size(1)

        _x = self.x_net(x) # bxk
        _y = self.y_net(y) # byk
        _z = self.z_net(z) # bzk
        
        _xyz = torch.einsum('bxk,byk,bzk->bxyzk', (_x, _y, _z)) # bxyzk
        att_maps = self.h_net(_xyz) # bxyzh
        att_maps = att_maps.permute(0, 4, 1, 2, 3) # bhxyz
        
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, x_num * y_num * z_num), 2)
            att_maps = p.view(-1, self.h_out, x_num, y_num, z_num)
        logits = self.attention_pooling(_x, _y, _z, att_maps[:, 0, :, :, :])
        
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(_x, _y, _z, att_maps[:, i, :, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            # layers.append(nn.Linear(in_dim, out_dim))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        # layers.append(nn.Linear(dims[-2], dims[-1]))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.fcnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.fcnet(x)

if __name__ == '__main__':
    B, num_features = 8, (9, 10, 11)
    hidden, multiheads = 64, 2
    
    poi = torch.randn(B, num_features[0], 1)
    protac = torch.randn(B, num_features[1], 2)
    e3 = torch.randn(B, num_features[2], 3)
    
    print('Testing TAN')
    
    tan = TAN((poi.size(2), protac.size(2), e3.size(2)), hidden, multiheads)
    print(tan)
    
    output, att_maps = tan(poi, protac, e3, softmax=True)
    
    print(output.size())
    print(att_maps.size())