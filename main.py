import copy
import os
import random
import time

import numpy as np
import toml
import torch
import torch.nn as nn
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

            outputs = model(protac_data, e3_ligase_data, poi_data)
            _, predicted = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, label)
            losses.append(loss.item())
            labels.extend(label.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    loss = sum(losses)/len(losses)
    accuracy = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return accuracy, loss, roc_auc, f1


def train(model, train_loader, val_loader, test_loader, device, lr=0.001, num_epochs=10):
    """
    使用 train/val/test 三个数据集进行训练：
    - train_loader: 用于更新参数
    - val_loader:   用于早停和选择最佳模型
    - test_loader:  在训练结束后进行最终评估
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    patience = 30
    best_val_loss = float('inf')
    best_model_wts = None
    best_val_roc_auc = 0.0
    counter = 0
    
    for epoch in range(num_epochs):
        # ---------------- 训练阶段 ----------------
        model.train()
        running_loss = 0.0
        for data in train_loader:
            # for data in tqdm(train_loader):
            protac_data = data['protac'].to(device)
            e3_ligase_data = data['e3_ligase'].to(device)
            poi_data = data['poi'].to(device)
            label = data['label'].to(device)
<<<<<<< HEAD
            fingerprint = data.get('fingerprint', None)
            if fingerprint is not None:
                fingerprint = fingerprint.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(protac_data, e3_ligase_data, poi_data, fingerprint=fingerprint)
=======

            optimizer.zero_grad()

            outputs = model(protac_data, e3_ligase_data, poi_data)
>>>>>>> 27a96b2 (Apply changes to baseline branch)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch: {epoch+1}/{num_epochs}, train loss: {avg_train_loss:.3f}')
        wandb.log({
            'train/epoch': epoch + 1,
            'train/loss': avg_train_loss
        })
        
        # ---------------- 验证阶段 ----------------
        model.eval()
        if val_loader is not None:
            val_acc, val_loss, val_roc_auc, val_f1 = test(model, val_loader, device)
            
            # 以验证集 ROC AUC 作为最佳模型标准
            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"Best model updated on VAL with roc_auc={val_roc_auc:.4f}!")
                wandb.run.summary['best_val_results'] = {
                    'roc_auc': val_roc_auc,
                    'f1_score': val_f1,
                    'accuracy': val_acc,
                    'loss': val_loss
                }
            
            # 早停基于验证集 loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
            
            print(f'Val Accuracy: {100 * val_acc:.2f} %')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val ROC AUC: {val_roc_auc:.4f}')
            print(f'Val F1 Score: {val_f1:.4f}')
            wandb.log({
                'val/epoch': epoch + 1,
                'val/accuracy': val_acc,
                'val/loss': val_loss,
                'val/roc_auc': val_roc_auc,
                'val/f1_score': val_f1
            })
        else:
            # 没有验证集时，直接保存当前模型并不做早停
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        
        if counter >= patience:
            print("Early stopped on validation set!")
            break
    
    # 恢复在验证集上表现最好的模型参数
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    # ---------------- 最终测试阶段 ----------------
    if test_loader is not None:
        test_acc, test_loss, test_roc_auc, test_f1 = test(model, test_loader, device)
        print(f'Final Test Accuracy: {100 * test_acc:.2f} %')
        print(f'Final Test Loss: {test_loss:.4f}')
        print(f'Final Test ROC AUC: {test_roc_auc:.4f}')
        print(f'Final Test F1 Score: {test_f1:.4f}')
        wandb.run.summary['best_test_results'] = {
            'roc_auc': test_roc_auc,
            'f1_score': test_f1,
            'accuracy': test_acc,
            'loss': test_loss
        }
        wandb.log({
            'test/final_accuracy': test_acc,
            'test/final_loss': test_loss,
            'test/final_roc_auc': test_roc_auc,
            'test/final_f1_score': test_f1
        })
    
    return model


def main():
    model_dir = f'saved_models/{time.strftime("%Y%m%d")}/{time.strftime("%H%M%S")}'
    os.makedirs(model_dir, exist_ok=True)

    cfg = toml.load('config.toml')
    model_cfg = cfg['model']
    train_cfg = cfg['train']

    setup_seed(model_cfg['seed'])
    
    wandb.init(
        mode="online",
        project='protac-stan-TPDdb',
        config=cfg,
        group=f'run_bz{train_cfg["batch_size"]}_lr{train_cfg["learning_rate"]}',
    )

    wandb.run.summary['model_dir'] = model_dir

    print(cfg)
    wandb.save('model.py')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    train_loader, val_loader, test_loader = PROTACLoader(
        root='data/TPDdb', name='protac_maccs', batch_size=train_cfg['batch_size'], collate_fn=collate_fn, 
        train_ratio=train_cfg['train_ratio'],
        val_ratio=train_cfg.get('val_ratio', 0.1),
        seed=model_cfg['seed'])

    model = PROTAC_STAN(model_cfg)
    print(model)
    wandb.watch(model)

    model = train(
        model, train_loader, val_loader, test_loader, device, 
        lr=train_cfg['learning_rate'], 
        num_epochs=train_cfg['num_epochs'], 
    )

    torch.save(model, f'{model_dir}/model.pt') # save full model (state_dict + architecture)    
    torch.save(model.state_dict(), f'{model_dir}/model_state_dict.pt') # save model state_dict only

    wandb.finish()


if __name__ == '__main__':
    main()