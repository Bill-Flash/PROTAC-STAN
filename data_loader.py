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
    batch['e3_ligase'] = torch.stack(e3_ligase)
    batch['poi'] = torch.stack(poi)
    batch['label'] = torch.stack(label) if label[0] is not None else None

    return batch


class PROTACDataset(Dataset):
    def __init__(self, protac, e3_ligase, poi, label):
        self.protac = protac
        self.e3_ligase = e3_ligase
        self.poi = poi
        if label is not None:
            self.label = label

    def __len__(self):
        return len(self.protac)

    def __getitem__(self, index):
        item = {
            'protac': self.protac[index],
            'e3_ligase': self.e3_ligase[index],
            'poi': self.poi[index],
            'label': self.label[index] if hasattr(self, 'label') else None
        }
        return item
    

def PROTACLoader(root='data/PROTAC-fine', name='protac-fine', batch_size=2, collate_fn=collate_fn, train_ratio=0.8, use_smiles_split=False, seed=None, save_split_csv=True):
    """
    Args:
        use_smiles_split: 如果为 True，使用 train/test_compound_smiles.csv 进行划分
                          如果为 False，使用随机划分（原始行为）
        seed: 随机种子，用于确保随机划分的可复现性
        save_split_csv: 如果为 True，保存划分结果到CSV文件
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

    dataset = PROTACDataset(protac, e3_ligase, poi, label)

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
    train_smiles = set([data['protac'].smiles for data in train_dataset])
    test_dataset = [data for data in test_dataset if data['protac'].smiles not in train_smiles]

    print('Dropped overlapping:')
    print('Train size: ', len(train_dataset))
    print('Test size: ', len(test_dataset))

    # 保存划分结果到CSV文件
    if save_split_csv:
        try:
            # 读取原始CSV文件以获取更多信息（如Compound ID等）
            import os.path as osp
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