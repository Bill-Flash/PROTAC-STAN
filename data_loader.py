import os.path as osp
import pickle

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch
import pandas as pd

from data import PROTACData


def collate_fn(data_list):
    batch = {}
    protac = [item['protac'] for item in data_list]
    e3_ligase = [item['e3_ligase'] for item in data_list]
    poi = [item['poi'] for item in data_list]
    label = [item['label'] for item in data_list]

    batch['protac'] =  Batch.from_data_list(protac)
    # 从batch中提取MACCS指纹特征
    if hasattr(protac[0], 'fingerprint'):
        fingerprint_list = [item.fingerprint for item in protac]
        batch['fingerprint'] = torch.stack(fingerprint_list).squeeze(1)  # [batch_size, 166]
    else:
        batch['fingerprint'] = None
    
    # e3_ligase / poi 可能是 tensor（离线 ESM embedding）或 str（在线 ESM 序列）
    if isinstance(e3_ligase[0], torch.Tensor):
        batch['e3_ligase'] = torch.stack(e3_ligase)
    else:
        batch['e3_ligase'] = e3_ligase

    if isinstance(poi[0], torch.Tensor):
        batch['poi'] = torch.stack(poi)
    else:
        batch['poi'] = poi
    batch['label'] = torch.stack(label) if label[0] is not None else None

    return batch


def _sample_key(sample):
    smiles = sample['protac'].smiles
    # 优先使用 POI 的 Uniprot ID 作为去重键，更语义化且与表示形式无关
    if 'poi_uniprot' in sample and sample['poi_uniprot'] is not None:
        target = sample['poi_uniprot']
    else:
        # 兼容旧数据：退回到基于 poi 表示本身的 key
        poi_tensor = sample['poi']
        if isinstance(poi_tensor, torch.Tensor):
            target = poi_tensor.detach().cpu().numpy().tobytes()
        else:
            target = str(poi_tensor)
    return (smiles, target)


class PROTACDataset(Dataset):
    def __init__(self, protac, e3_ligase, poi, label, e3_uniprot=None, poi_uniprot=None):
        self.protac = protac
        self.e3_ligase = e3_ligase
        self.poi = poi
        if label is not None:
            self.label = label
        # 可选的 Uniprot ID 列表，用于去重或分析
        if e3_uniprot is not None:
            self.e3_uniprot = e3_uniprot
        if poi_uniprot is not None:
            self.poi_uniprot = poi_uniprot

    def __len__(self):
        return len(self.protac)

    def __getitem__(self, index):
        item = {
            'protac': self.protac[index],
            'e3_ligase': self.e3_ligase[index],
            'poi': self.poi[index],
            'label': self.label[index] if hasattr(self, 'label') else None
        }
        if hasattr(self, 'e3_uniprot'):
            item['e3_uniprot'] = self.e3_uniprot[index]
        if hasattr(self, 'poi_uniprot'):
            item['poi_uniprot'] = self.poi_uniprot[index]
        return item
    

def PROTACLoader(
    root='data/PROTAC-fine',
    name='protac-fine',
    batch_size=2,
    collate_fn=collate_fn,
    train_ratio=0.8,
    use_smiles_split=False,
    seed=None,
    save_split_csv=True,
    use_online_esm=False,
):
    """
    Args:
        use_smiles_split: 如果为 True，使用 train/test_compound_smiles.csv 进行划分
                          如果为 False，使用随机划分（原始行为）
        seed: 随机种子，用于确保随机划分的可复现性
        save_split_csv: 如果为 True，保存划分结果到CSV文件
        use_online_esm: 如果为 True，则使用 Uniprot ID 从 p_map.pkl 读取氨基酸序列，
                        由下游模型（ESM2Base150M）在线编码，而不是使用预计算 embedding
    """
    protac = PROTACData(root, name=name) # name: raw file name
    with open(f'{root}/processed/{name}/e3_ligase.pt', 'rb') as f:
        e3_ligase = torch.load(f)
    with open(f'{root}/processed/{name}/poi.pt', 'rb') as f:
        poi = torch.load(f)
    try:
        with open(f'{root}/processed/{name}/label.pt', 'rb') as f:
            label = torch.load(f)
    except:
        label = None

    # 可选：加载 E3 / POI 的 Uniprot ID（若存在）
    try:
        with open(f'{root}/processed/{name}/e3_uniprot.pt', 'rb') as f:
            e3_uniprot = torch.load(f)
        with open(f'{root}/processed/{name}/poi_uniprot.pt', 'rb') as f:
            poi_uniprot = torch.load(f)
    except FileNotFoundError:
        e3_uniprot = None
        poi_uniprot = None

    # 如果开启在线 ESM 模式，则使用 Uniprot ID 从 p_map.pkl 中恢复氨基酸序列
    if use_online_esm:
        if poi_uniprot is None:
            raise RuntimeError("use_online_esm=True 但未找到 poi_uniprot.pt / e3_uniprot.pt，请先重新处理数据集。")

        p_map_path = osp.join(root, 'p_map.pkl')
        if not osp.exists(p_map_path):
            raise RuntimeError(f"use_online_esm=True 但未找到 {p_map_path}，无法从 Uniprot 恢复序列。")

        with open(p_map_path, 'rb') as f:
            p_map = pickle.load(f)

        # 根据 Uniprot ID 映射得到序列列表
        e3_ligase = [p_map[uid] for uid in e3_uniprot]
        poi = [p_map[uid] for uid in poi_uniprot]

    dataset = PROTACDataset(protac, e3_ligase, poi, label, e3_uniprot=e3_uniprot, poi_uniprot=poi_uniprot)

    # 使用 SMILES CSV 文件进行划分
    if use_smiles_split:
        train_csv_path = f'{root}/train_compound_smiles.csv'
        test_csv_path = f'{root}/test_compound_smiles.csv'
        
        try:
            train_df = pd.read_csv(train_csv_path)
            test_df = pd.read_csv(test_csv_path)
            
            train_smiles_set = set(train_df['SMILES'].tolist())
            test_smiles_set = set(test_df['SMILES'].tolist())
            
            # 找到匹配的索引
            train_indices = []
            test_indices = []
            
            for idx in range(len(dataset)):
                sample_smiles = dataset[idx]['protac'].smiles
                if sample_smiles in train_smiles_set:
                    train_indices.append(idx)
                elif sample_smiles in test_smiles_set:
                    test_indices.append(idx)
            
            train_dataset = Subset(dataset, train_indices) if train_indices else None
            test_dataset = Subset(dataset, test_indices)
            
            print('Using fixed split from CSV files:')
            if train_dataset:
                print(f'Train size: {len(train_dataset)}')
            print(f'Test size: {len(test_dataset)}')
            
            # 创建 DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) if train_dataset else None
            test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
            
            return train_loader, test_loader
            
        except FileNotFoundError as e:
            print(f'Warning: Split CSV files not found ({e}), using random split')
            use_smiles_split = False

    # 原始随机划分逻辑
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    if train_ratio > 0.0:
        print('Cleaned Dataset: ')
        print('Total size: ', len(dataset))
        print('Train size: ', train_size)
        print('Test size: ', test_size)
    else:
        print('Test Dataset: ')
        print('Total size: ', len(dataset))

    if train_size == 0:
        test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        return None, test_loader

    # 使用 seed 创建 generator 以确保可复现性
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    # Drop overlapping data in test set from train set
    train_keys = {_sample_key(data) for data in train_dataset}
    test_dataset = [data for data in test_dataset if _sample_key(data) not in train_keys]

    print('Dropped overlapping:')
    print('Train size: ', len(train_dataset))
    print('Test size: ', len(test_dataset))

    # 保存划分结果到CSV文件
    if save_split_csv:
        try:
            # 读取原始CSV文件以获取更多信息（如Compound ID等）
            raw_csv_path = osp.join(root, f'{name}.csv')
            if osp.exists(raw_csv_path):
                raw_df = pd.read_csv(raw_csv_path)
                # 确保有Smiles列（注意大小写）
                smiles_col = 'Smiles' if 'Smiles' in raw_df.columns else 'SMILES'
                
                # 收集train和test的SMILES
                train_smiles_list = [data['protac'].smiles for data in train_dataset]
                test_smiles_list = [data['protac'].smiles for data in test_dataset]
                
                # 从原始CSV中匹配并提取数据
                train_df = raw_df[raw_df[smiles_col].isin(train_smiles_list)].copy()
                test_df = raw_df[raw_df[smiles_col].isin(test_smiles_list)].copy()
                
                # 如果原始CSV有Compound ID，使用它；否则只保存SMILES
                if 'Compound ID' in train_df.columns:
                    train_save_df = train_df[['Compound ID', smiles_col]].copy()
                    train_save_df.columns = ['Compound ID', 'SMILES']
                else:
                    train_save_df = pd.DataFrame({'SMILES': train_smiles_list})
                
                if 'Compound ID' in test_df.columns:
                    test_save_df = test_df[['Compound ID', smiles_col]].copy()
                    test_save_df.columns = ['Compound ID', 'SMILES']
                else:
                    test_save_df = pd.DataFrame({'SMILES': test_smiles_list})
                
                # 保存CSV文件
                train_csv_path = f'{root}/train_compound_smiles.csv'
                test_csv_path = f'{root}/test_compound_smiles.csv'
                train_save_df.to_csv(train_csv_path, index=False)
                test_save_df.to_csv(test_csv_path, index=False)
                print(f'Saved train split to {train_csv_path}')
                print(f'Saved test split to {test_csv_path}')
            else:
                # 如果原始CSV不存在，只保存SMILES
                train_smiles_list = [data['protac'].smiles for data in train_dataset]
                test_smiles_list = [data['protac'].smiles for data in test_dataset]
                
                train_save_df = pd.DataFrame({'SMILES': train_smiles_list})
                test_save_df = pd.DataFrame({'SMILES': test_smiles_list})
                
                train_csv_path = f'{root}/train_compound_smiles.csv'
                test_csv_path = f'{root}/test_compound_smiles.csv'
                train_save_df.to_csv(train_csv_path, index=False)
                test_save_df.to_csv(test_csv_path, index=False)
                print(f'Saved train split to {train_csv_path}')
                print(f'Saved test split to {test_csv_path}')
        except Exception as e:
            print(f'Warning: Failed to save split CSV files: {e}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = PROTACLoader()

    for item in train_loader:
        print(item)
        break
    
    for item in test_loader:
        print(item)
        break