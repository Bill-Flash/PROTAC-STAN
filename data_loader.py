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
    

def PROTACLoader(root='data/PROTAC-fine', name='protac-fine', batch_size=2, collate_fn=collate_fn, train_ratio=0.8, use_smiles_split=False):
    """
    Args:
        use_smiles_split: 如果为 True，使用 train/test_compound_smiles.csv 进行划分
                          如果为 False，使用随机划分（原始行为）
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

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Drop overlapping data in test set from train set
    train_smiles = set([data['protac'].smiles for data in train_dataset])
    test_dataset = [data for data in test_dataset if data['protac'].smiles not in train_smiles]

    print('Dropped overlapping:')
    print('Train size: ', len(train_dataset))
    print('Test size: ', len(test_dataset))

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