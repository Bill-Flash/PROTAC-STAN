import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


def _sample_key(sample):
    smiles = sample['protac'].smiles
    poi_tensor = sample['poi']
    if isinstance(poi_tensor, torch.Tensor):
        target_bytes = poi_tensor.detach().cpu().numpy().tobytes()
    else:
        target_bytes = bytes(poi_tensor)
    return (smiles, target_bytes)


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
    
    
def PROTACLoader(
    root='data/TPDdb',
    name='protac_maccs',
    batch_size=2,
    collate_fn=collate_fn,
    train_ratio=0.8,
    val_ratio=0.1,
    use_smiles_split=False,
    seed=None,
    save_split_csv=True
):
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

            # 在基于 SMILES 的固定划分下，从训练集中再划出验证集（保持整体约 8:1:1）
            val_dataset = None
            if train_dataset is not None and len(train_dataset) > 0 and val_ratio > 0.0:
                val_size = int(len(train_dataset) * val_ratio / (train_ratio + 1e-8))
                val_size = max(1, val_size) if len(train_dataset) > 1 else 0
                if val_size > 0 and val_size < len(train_dataset):
                    train_size = len(train_dataset) - val_size
                    generator = None
                    if seed is not None:
                        generator = torch.Generator().manual_seed(seed)
                    train_dataset, val_dataset = torch.utils.data.random_split(
                        train_dataset, [train_size, val_size], generator=generator
                    )

            # 按 SMILES+POI 去重：确保验证集/测试集中不包含训练集中已出现的样本
            if train_dataset is not None:
                train_keys = {_sample_key(data) for data in train_dataset}
                if val_dataset is not None:
                    val_dataset = [data for data in val_dataset if _sample_key(data) not in train_keys]
                test_dataset = [data for data in test_dataset if _sample_key(data) not in train_keys]
            
            print('Using fixed split from CSV files (after de-dup w.r.t. train):')
            if train_dataset:
                print(f'Train size: {len(train_dataset)}')
            if val_dataset:
                print(f'Val size: {len(val_dataset)}')
            print(f'Test size: {len(test_dataset)}')
            
            # 创建 DataLoader
            # 训练集使用 drop_last=True，避免出现 batch_size=1 导致 BatchNorm 报错
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True
            ) if train_dataset else None
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            ) if val_dataset is not None else None
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            return train_loader, val_loader, test_loader
            
        except FileNotFoundError as e:
            print(f'Warning: Split CSV files not found ({e}), using random split')
            use_smiles_split = False

    # 原始随机划分逻辑：支持 train/val/test（默认 8:1:1），并按 label 分层采样
    total_size = len(dataset)
    if train_ratio <= 0.0 or label is None:
        # 没有训练集或没有标签信息时，退化为只有测试集
        print('Test Dataset: ')
        print('Total size: ', total_size)
        test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        return None, None, test_loader

    # 第一步：先从全数据中分出「临时集」：如 80% 训练 + 20% 临时，使用 stratify 保证分层
    indices = list(range(total_size))
    # label.pt 一般是 tensor，这里统一转成一维 numpy 数组 / list
    if isinstance(label, torch.Tensor):
        all_labels = label.cpu().numpy()
    else:
        all_labels = torch.tensor(label).cpu().numpy()

    temp_ratio = 1.0 - train_ratio
    if temp_ratio <= 0:
        # 没有临时集，直接全部作为训练集
        train_indices = indices
        temp_indices = []
        y_train = all_labels
        y_temp = []
    else:
        train_indices, temp_indices, y_train, y_temp = train_test_split(
            indices,
            all_labels,
            test_size=temp_ratio,
            stratify=all_labels,
            random_state=seed
        )

    # 第二步：再把这 20% 临时集按 1:1（或根据 val_ratio）划成验证和测试，同样分层
    if temp_indices:
        # 目标：val 占全体的 val_ratio，其余作为 test
        # 在临时集中的相对比例：
        relative_val_ratio = val_ratio / max(temp_ratio, 1e-8)
        # 防止数值问题，限制到 (0,1) 内
        relative_val_ratio = max(0.0, min(1.0, relative_val_ratio))
        relative_test_ratio = 1.0 - relative_val_ratio

        if relative_val_ratio == 0.0:
            val_indices = []
            test_indices = temp_indices
        elif relative_test_ratio == 0.0:
            val_indices = temp_indices
            test_indices = []
        else:
            val_indices, test_indices, y_val, y_test = train_test_split(
                temp_indices,
                y_temp,
                test_size=relative_test_ratio,
                stratify=y_temp,
                random_state=seed
            )
    else:
        val_indices, test_indices = [], []

    # ========== 第三步：按 SMILES+POI 去重，并把重复样本从 val/test 移回 train ==========
    # 为当前 train 构建 key 集合
    train_keys = set()
    for idx in train_indices:
        sample = dataset[idx]
        train_keys.add(_sample_key(sample))

    overlap_val = []
    overlap_test = []
    val_clean_indices = []
    test_clean_indices = []

    # 找到 val 中与 train 重复的样本
    for idx in val_indices:
        key = _sample_key(dataset[idx])
        if key in train_keys:
            overlap_val.append(idx)
        else:
            val_clean_indices.append(idx)

    # 找到 test 中与 train 重复的样本
    for idx in test_indices:
        key = _sample_key(dataset[idx])
        if key in train_keys:
            overlap_test.append(idx)
        else:
            test_clean_indices.append(idx)

    # 把重复样本并回 train
    train_indices_extended = train_indices + overlap_val + overlap_test

    print('Cleaned Dataset with stratified split (before rebalancing): ')
    print('Total size: ', total_size)
    print('Train size: ', len(train_indices_extended))
    print('Val size: ', len(val_clean_indices))
    print('Test size: ', len(test_clean_indices))

    # ========== 第四步：从 train 中按 label 分层抽样，补足 val/test 至至少 target_size ==========
    target_size = 1000
    current_val = len(val_clean_indices)
    current_test = len(test_clean_indices)

    need_val = max(0, target_size - current_val)
    need_test = max(0, target_size - current_test)

    # 可用的 train 池
    train_pool_indices = train_indices_extended
    total_need = need_val + need_test

    # 如果 train 数量不足以完全满足需求，按比例缩放抽样数量
    if total_need > 0 and len(train_pool_indices) < total_need:
        scale = len(train_pool_indices) / float(total_need)
        alloc_val = int(round(need_val * scale))
        alloc_test = len(train_pool_indices) - alloc_val
    else:
        alloc_val = need_val
        alloc_test = need_test

    def stratified_sample(indices, y_all, n_samples, seed=None):
        """
        从给定 indices 中按 label 分层抽样 n_samples 条，
        返回 (剩余 indices, 抽出的 indices)。
        """
        if n_samples <= 0 or len(indices) == 0:
            return indices, []

        indices_array = np.array(indices)
        y_subset = y_all[indices_array]

        if n_samples >= len(indices_array):
            # 需要数量大于等于池子大小，直接全部拿走
            return [], list(indices_array)

        try:
            # 使用 sklearn 的 train_test_split 进行分层抽样
            from sklearn.model_selection import train_test_split as _tts

            remain_idx, sample_idx = _tts(
                indices_array,
                test_size=n_samples,
                stratify=y_subset,
                random_state=seed
            )
            return list(remain_idx), list(sample_idx)
        except ValueError as e:
            # 极端情况下某些 label 样本过少时，分层可能失败，退化为随机抽样
            print(f'Warning: stratified sampling failed ({e}), fallback to random sampling.')
            import random
            rng = random.Random(seed)
            sampled = rng.sample(list(indices_array), n_samples)
            remain = [idx for idx in indices_array if idx not in sampled]
            return remain, sampled

    # 先从 train 中补 val
    final_train_indices = train_pool_indices
    val_additional = []
    if alloc_val > 0:
        final_train_indices, val_additional = stratified_sample(
            final_train_indices,
            all_labels,
            alloc_val,
            seed=seed
        )

    # 再从更新后的 train 中补 test
    test_additional = []
    if alloc_test > 0:
        # 为了避免和 val 使用完全相同的随机划分，简单地对 seed 做一个偏移
        test_seed = None if seed is None else seed + 1
        final_train_indices, test_additional = stratified_sample(
            final_train_indices,
            all_labels,
            alloc_test,
            seed=test_seed
        )

    final_val_indices = val_clean_indices + val_additional
    final_test_indices = test_clean_indices + test_additional

    print('Final split after de-dup and rebalancing:')
    print('Train size: ', len(final_train_indices))
    print('Val size: ', len(final_val_indices))
    print('Test size: ', len(final_test_indices))

    # 根据最终索引构建 Subset
    train_dataset = Subset(dataset, final_train_indices)
    val_dataset = Subset(dataset, final_val_indices) if len(final_val_indices) > 0 else []
    test_dataset = Subset(dataset, final_test_indices) if len(final_test_indices) > 0 else []

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
                
                # 收集 train / val / test 的 SMILES
                train_smiles_list = [data['protac'].smiles for data in train_dataset]
                val_smiles_list = [data['protac'].smiles for data in val_dataset]
                test_smiles_list = [data['protac'].smiles for data in test_dataset]
                
                # 从原始CSV中匹配并提取数据
                train_df = raw_df[raw_df[smiles_col].isin(train_smiles_list)].copy()
                val_df = raw_df[raw_df[smiles_col].isin(val_smiles_list)].copy()
                test_df = raw_df[raw_df[smiles_col].isin(test_smiles_list)].copy()
                
                # 如果原始CSV有Compound ID，使用它；否则只保存SMILES
                if 'Compound ID' in train_df.columns:
                    train_save_df = train_df[['Compound ID', smiles_col]].copy()
                    train_save_df.columns = ['Compound ID', 'SMILES']
                else:
                    train_save_df = pd.DataFrame({'SMILES': train_smiles_list})
                
                if 'Compound ID' in val_df.columns:
                    val_save_df = val_df[['Compound ID', smiles_col]].copy()
                    val_save_df.columns = ['Compound ID', 'SMILES']
                else:
                    val_save_df = pd.DataFrame({'SMILES': val_smiles_list})
                
                if 'Compound ID' in test_df.columns:
                    test_save_df = test_df[['Compound ID', smiles_col]].copy()
                    test_save_df.columns = ['Compound ID', 'SMILES']
                else:
                    test_save_df = pd.DataFrame({'SMILES': test_smiles_list})
                
                # 保存CSV文件
                train_csv_path = f'{root}/train_compound_smiles.csv'
                val_csv_path = f'{root}/val_compound_smiles.csv'
                test_csv_path = f'{root}/test_compound_smiles.csv'
                train_save_df.to_csv(train_csv_path, index=False)
                val_save_df.to_csv(val_csv_path, index=False)
                test_save_df.to_csv(test_csv_path, index=False)
                print(f'Saved train split to {train_csv_path}')
                print(f'Saved val split to {val_csv_path}')
                print(f'Saved test split to {test_csv_path}')
            else:
                # 如果原始CSV不存在，只保存SMILES
                train_smiles_list = [data['protac'].smiles for data in train_dataset]
                val_smiles_list = [data['protac'].smiles for data in val_dataset]
                test_smiles_list = [data['protac'].smiles for data in test_dataset]
                
                train_save_df = pd.DataFrame({'SMILES': train_smiles_list})
                val_save_df = pd.DataFrame({'SMILES': val_smiles_list})
                test_save_df = pd.DataFrame({'SMILES': test_smiles_list})
                
                train_csv_path = f'{root}/train_compound_smiles.csv'
                val_csv_path = f'{root}/val_compound_smiles.csv'
                test_csv_path = f'{root}/test_compound_smiles.csv'
                train_save_df.to_csv(train_csv_path, index=False)
                val_save_df.to_csv(val_csv_path, index=False)
                test_save_df.to_csv(test_csv_path, index=False)
                print(f'Saved train split to {train_csv_path}')
                print(f'Saved val split to {val_csv_path}')
                print(f'Saved test split to {test_csv_path}')
        except Exception as e:
            print(f'Warning: Failed to save split CSV files: {e}')

    # 训练集使用 drop_last=True，避免出现 batch_size=1 导致 BatchNorm 报错
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = PROTACLoader()

    if train_loader is not None:
        for item in train_loader:
            print('Train batch example:', item)
            break
    
    if val_loader is not None:
        for item in val_loader:
            print('Val batch example:', item)
            break
    
    for item in test_loader:
        print('Test batch example:', item)
        break