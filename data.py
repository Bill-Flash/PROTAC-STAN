import os.path as osp
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset

RDLogger.DisableLog('rdApp.*')


columns = [
            'Smiles',
            'Molecular Weight',
            'Exact Mass',
            'XLogP3',
            'Heavy Atom Count',
            'Ring Count',
            'Hydrogen Bond Acceptor Count',
            'Hydrogen Bond Donor Count',
            'Rotatable Bond Count',
            'Topological Polar Surface Area'
        ]

x_map = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

SMILES_CHAR =['[PAD]', 'C', '(', '=', 'O', ')', 'N', '[', '@', 'H', ']', '1', 'c', 'n', '/', '2', '#', 'S', 's', '+', '-', '\\', '3', '4', 'l', 'F', 'o', 'I', 'B', 'r', 'P', '5', '6', 'i', '7', '8', '9', '%', '0', 'p']


def trans_smiles(x, target_size=128, pad_value=0):
    temp = list(x)
    temp = [SMILES_CHAR.index(i) if i in SMILES_CHAR else len(SMILES_CHAR) for i in temp]
    if len(temp) < target_size:
        temp = temp + [pad_value] * (target_size - len(temp))
    else:
        temp = temp[:target_size]
    return temp


def data_to_graph(raw_data, with_hydrogen: bool = False, kekulize: bool = False) -> 'Data':
    """
    Convert raw data to a graph representation.
    """
    
    smiles = raw_data[columns[0]]
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    
    smiles_encoding = trans_smiles(smiles)
    mol_properties = [raw_data[column] for column in columns[1:]]
    
    # 解析MACCS Fingerprint（从CSV列读取）
    fp = []
    if 'MACCS_Fingerprint' in raw_data and pd.notna(raw_data.get('MACCS_Fingerprint')):
        try:
            fp_str = str(raw_data['MACCS_Fingerprint'])
            fp = [int(x) for x in fp_str.split(',')]
            if len(fp) != 166:
                # 如果长度不对，使用全零向量
                fp = [0] * 166
        except:
            fp = [0] * 166
    else:
        fp = [0] * 166
    
    # 合并节点特征（不包括指纹）
    global_feature = smiles_encoding + mol_properties  # 128 + 9 = 137维
    global_feature = torch.tensor(global_feature, dtype=torch.float).view(1, -1)
    
    # Morgan指纹单独保存
    fp = torch.tensor(fp, dtype=torch.float).view(1, -1)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)
    
    x = torch.tensor(xs, dtype=torch.float).view(-1, 9)
    x = torch.cat([x, global_feature.expand(x.size(0), -1)], dim=1)


    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]
    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 3)
    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, fingerprint=fp)

class PROTACData(InMemoryDataset):
    """
    PyTorch Geometric dataset for processing PROTAC data.
    
    Args:

    root (str): Root directory.
    name (str): Name of the raw file.
    """

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.raw_files = [osp.join(root, f'{name}.csv')]
        super(PROTACData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    # @property
    # def raw_dir(self):
    #     return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.name)
    
    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return [
            'protac.pt',
            'e3_ligase.pt',
            'poi.pt',
            'label.pt'
        ]

    def process(self):
        print(self.processed_dir)
        protac_df = pd.read_csv(self.raw_file_names[0])

        # Load ESM map to check which Uniprot IDs are available
        # 改进的数据过滤策略
        # 现在代码会同时过滤掉以下情况的样本：
        # Uniprot列为NaN的样本
        # E3 ligase Uniprot列为NaN的样本
        # Uniprot ID不在esm_map中的样本
        # E3 ligase Uniprot ID不在esm_map中的样本
        # label列为NaN的样本 ← 新增
        with open(osp.join(self.root, 'esm_s_map.pkl'), 'rb') as f:
            esm_map = pickle.load(f)
        
        # Filter out rows with NaN values and ensure Uniprot IDs exist in ESM map
        # Also drop rows where label is NaN
        valid_mask = (protac_df['Uniprot'].notna() & 
                     protac_df['E3 ligase Uniprot'].notna() &
                     protac_df['Uniprot'].isin(esm_map.keys()) &
                     protac_df['E3 ligase Uniprot'].isin(esm_map.keys()) &
                     protac_df['label'].notna())
        protac_df = protac_df[valid_mask].reset_index(drop=True)
        
        print(f"Filtered dataset: {len(protac_df)} valid samples out of {len(pd.read_csv(self.raw_file_names[0]))} total samples")

        # standardize the data
        protac_df[columns[1:]] = protac_df[columns[1:]].apply(lambda x: (x - x.mean()) / x.std())

        # protac
        data_list = []
        for _, row in protac_df.iterrows():
            raw_data = row[columns].to_list()
            raw_data = dict(zip(columns, raw_data))
            data_list.append(data_to_graph(raw_data))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # e3 ligase
        e3 = protac_df['E3 ligase Uniprot'].to_list()
        e3_data_list = [esm_map[id] for id in e3]
        e3_data_list = [torch.from_numpy(e3) for e3 in e3_data_list]

        torch.save(e3_data_list, self.processed_paths[1])

        # poi
        poi = protac_df['Uniprot'].to_list()
        poi_data_list = [esm_map[id] for id in poi]
        poi_data_list = [torch.from_numpy(poi) for poi in poi_data_list]

        torch.save(poi_data_list, self.processed_paths[2])

        # label
        if 'label' not in protac_df.columns:
            return
        # Since we already filtered out NaN values, we can directly convert to int
        label = protac_df['label'].astype(int).to_list()
        label = torch.tensor(label)
        torch.save(label, self.processed_paths[3])

    def __repr__(self):
        return '{}()'.format(self.name)
    

if __name__ == '__main__':
    root = 'data/protacdb3'
    dataset = PROTACData(root=root, name='protac_maccs')
    print("Dataset info:")
    print("Number of samples:", len(dataset))
