import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

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
    batch['label'] = torch.stack(label)

    return batch


class PROTACDataset(Dataset):
    def __init__(self, protac, e3_ligase, poi, label):
        self.protac = protac
        self.e3_ligase = e3_ligase
        self.poi = poi
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        item = {
            'protac': self.protac[index],
            'e3_ligase': self.e3_ligase[index],
            'poi': self.poi[index],
            'label': self.label[index]
        }
        return item
    

def PROTACLoader(root='data/PROTAC-fine', name='protac-fine', batch_size=2, collate_fn=collate_fn, train_ratio=0.8):
    protac = PROTACData(root, name=name) # name: raw file name
    with open(f'{root}/processed/{name}/e3_ligase.pt', 'rb') as f:
        e3_ligase = torch.load(f)
    with open(f'{root}/processed/{name}/poi.pt', 'rb') as f:
        poi = torch.load(f)
    with open(f'{root}/processed/{name}/label.pt', 'rb') as f:
        label = torch.load(f)

    dataset = PROTACDataset(protac, e3_ligase, poi, label)

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