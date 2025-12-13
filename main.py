import copy
import os
import random
import time

import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

import wandb
from data_loader import PROTACLoader, collate_fn
from model import PROTAC_STAN


def setup_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(model, test_loader, device): 
    model = model.to(device)
    model.eval()

    losses = []
    labels = []
    predictions = []

    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()

        for data in test_loader:
            protac_data = data['protac'].to(device)
            e3_ligase_data = data['e3_ligase'].to(device)
            poi_data = data['poi'].to(device)
            label = data['label'].to(device)
            fingerprint = data.get('fingerprint', None)
            if fingerprint is not None:
                fingerprint = fingerprint.to(device)

            # 测试只需要分类 logits，不需要对比学习 embedding
            logits = model(protac_data, e3_ligase_data, poi_data, fingerprint=fingerprint)
            _, predicted = torch.max(logits.data, dim=1)

            loss = criterion(logits, label)
            losses.append(loss.item())
            labels.extend(label.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    loss = sum(losses)/len(losses)
    accuracy = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return accuracy, loss, roc_auc, f1


def clip_contrastive_loss(z_a, z_b, labels, temperature=0.1, only_positive=True):
    """
    CLIP-style 对比损失（任意两模态之间）：
    - anchor: 第一个模态表征 z_a
    - 对齐对象: 第二个模态表征 z_b
    - 只在 label == 1 的样本上计算（only_positive=True）
    """
    device = z_a.device

    if only_positive:
        pos_mask = (labels == 1)
        num_pos = pos_mask.sum().item()
        # 正样本太少（<2）时跳过对比损失，避免退化为无负样本的情况
        if num_pos < 2:
            return torch.tensor(0.0, device=device)
        z_a = z_a[pos_mask]
        z_b = z_b[pos_mask]

    # [N, d] @ [d, N] -> [N, N]
    sim = torch.matmul(z_a, z_b.t()) / temperature
    targets = torch.arange(sim.size(0), device=device)

    # 模态 A -> 模态 B
    loss_a2b = F.cross_entropy(sim, targets)
    # 模态 B -> 模态 A
    loss_b2a = F.cross_entropy(sim.t(), targets)

    loss = 0.5 * (loss_a2b + loss_b2a)
    return loss


def train(model, train_loader, test_loader, device, lr=0.001, num_epochs=10,
          contrast_weight=0.0, contrast_temperature=0.1, contrast_only_positive=True,
          lambda_pe=1.0, lambda_po=1.0, lambda_eo=1.0,
          lambda_pe_poi=0.0, lambda_po_e3=0.0):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    patience = 30
    best_loss = float('inf')
    counter = 0
    best_model_wts = None
    best_roc_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_contrast_loss = 0.0
        for data in train_loader:
        # for data in tqdm(train_loader):
            protac_data = data['protac'].to(device)
            e3_ligase_data = data['e3_ligase'].to(device)
            poi_data = data['poi'].to(device)
            label = data['label'].to(device)
            fingerprint = data.get('fingerprint', None)
            if fingerprint is not None:
                fingerprint = fingerprint.to(device)

            optimizer.zero_grad()

            # 训练阶段：同时获取分类 logits 和三模态 + 复合模态 对比学习 embedding
            logits, z_protac, z_e3, z_poi, z_pe, z_po = model(
                protac_data, e3_ligase_data, poi_data,
                fingerprint=fingerprint,
                return_embeddings=True
            )

            ce_loss = criterion(logits, label)

            if contrast_weight > 0.0:
                # 三模态共享空间 + 三两成对对比（单体）
                loss_pe = clip_contrastive_loss(
                    z_protac, z_e3, label,
                    temperature=contrast_temperature,
                    only_positive=contrast_only_positive
                )
                loss_po = clip_contrastive_loss(
                    z_protac, z_poi, label,
                    temperature=contrast_temperature,
                    only_positive=contrast_only_positive
                )
                loss_eo = clip_contrastive_loss(
                    z_e3, z_poi, label,
                    temperature=contrast_temperature,
                    only_positive=contrast_only_positive
                )

                # 复合模态：二元复合体 ↔ 第三方
                # (PE) PROTAC-E3 ↔ POI
                loss_pe_poi = clip_contrastive_loss(
                    z_pe, z_poi, label,
                    temperature=contrast_temperature,
                    only_positive=contrast_only_positive
                )
                # (PO) PROTAC-POI ↔ E3
                loss_po_e3 = clip_contrastive_loss(
                    z_po, z_e3, label,
                    temperature=contrast_temperature,
                    only_positive=contrast_only_positive
                )

                # 根据权重做加权平均，保证整体尺度稳定
                total_lambda = (
                    lambda_pe + lambda_po + lambda_eo +
                    lambda_pe_poi + lambda_po_e3
                )
                if total_lambda > 0.0:
                    contrast_loss = (
                        lambda_pe * loss_pe +
                        lambda_po * loss_po +
                        lambda_eo * loss_eo +
                        lambda_pe_poi * loss_pe_poi +
                        lambda_po_e3 * loss_po_e3
                    ) / total_lambda
                else:
                    contrast_loss = torch.tensor(0.0, device=device)
            else:
                contrast_loss = torch.tensor(0.0, device=device)

            loss = ce_loss + contrast_weight * contrast_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_contrast_loss += contrast_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_ce_loss = running_ce_loss / len(train_loader)
        avg_contrast_loss = running_contrast_loss / len(train_loader)

        print(f'Epoch: {epoch+1}/{num_epochs}, '
              f'train loss: {avg_train_loss:.3f}, '
              f'ce: {avg_ce_loss:.3f}, '
              f'contrast: {avg_contrast_loss:.3f}')
        wandb.log({
            'train/epoch': epoch + 1,
            'train/loss': avg_train_loss,
            'train/ce_loss': avg_ce_loss,
            'train/contrast_loss': avg_contrast_loss,
        })
        
        model.eval()
        test_acc, test_loss, roc_auc, f1 = test(model, test_loader, device)

        if best_roc_auc < roc_auc:
            best_roc_auc = roc_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"Best model updated with roc_auc={roc_auc:.4f}!")
            wandb.run.summary['best_results'] = {
                'roc_auc': roc_auc,
                'f1_score': f1,
                'accuracy': test_acc,
                'loss': test_loss
            }
        
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopped!")
            break

        print(f'Test Accuracy: {100 * test_acc:.2f} %')
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test ROC AUC: {roc_auc:.4f}')
        print(f'Test F1 Score: {f1:.4f}')
        wandb.log({
            'test/epoch': epoch + 1,
            'test/accuracy': test_acc,
            'test/loss': test_loss,
            'test/roc_auc': roc_auc,
            'test/f1_score': f1
        })
        
    model.load_state_dict(best_model_wts)

    return model


def main():
    model_dir = f'saved_models/{time.strftime("%Y%m%d")}/{time.strftime("%H%M%S")}'
    os.makedirs(model_dir, exist_ok=True)

    cfg = toml.load('config.toml')
    model_cfg = cfg['model']
    train_cfg = cfg['train']
    contrast_cfg = model_cfg.get('contrast', {})

    setup_seed(model_cfg['seed'])
    
    wandb.init(
        mode="online",
        project='protac-stan',
        config=cfg,
        # group=f'run_CL_bz{train_cfg["batch_size"]}_lr{train_cfg["learning_rate"]}',
        group=f'run_CL_CLIP_trimodal_lr{train_cfg["learning_rate"]}',
    )

    wandb.run.summary['model_dir'] = model_dir

    print(cfg)
    wandb.save('model.py')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    train_loader, test_loader = PROTACLoader(
        root='data/protacdb3', name='protac_maccs', batch_size=train_cfg['batch_size'], collate_fn=collate_fn, 
        train_ratio=train_cfg['train_ratio'], seed=model_cfg['seed'])

    model = PROTAC_STAN(model_cfg)
    print(model)
    wandb.watch(model)

    model = train(
        model, train_loader, test_loader, device, 
        lr=train_cfg['learning_rate'], 
        num_epochs=train_cfg['num_epochs'],
        contrast_weight=train_cfg.get('contrast_weight', 0.0),
        contrast_temperature=train_cfg.get('contrast_temperature', 0.1),
        contrast_only_positive=train_cfg.get('contrast_only_positive', True),
        lambda_pe=contrast_cfg.get('lambda_pe', 1.0),
        lambda_po=contrast_cfg.get('lambda_po', 1.0),
        lambda_eo=contrast_cfg.get('lambda_eo', 1.0),
        lambda_pe_poi=contrast_cfg.get('lambda_pe_poi', 0.0),
        lambda_po_e3=contrast_cfg.get('lambda_po_e3', 0.0),
    )

    torch.save(model, f'{model_dir}/model.pt') # save full model (state_dict + architecture)    
    torch.save(model.state_dict(), f'{model_dir}/model_state_dict.pt') # save model state_dict only

    wandb.finish()


if __name__ == '__main__':
    main()