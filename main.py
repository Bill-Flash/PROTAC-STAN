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


def train(model, train_loader, test_loader, device, lr=0.001, num_epochs=10):
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
        for data in train_loader:
        # for data in tqdm(train_loader):
            protac_data = data['protac'].to(device)
            e3_ligase_data = data['e3_ligase'].to(device)
            poi_data = data['poi'].to(device)
            label = data['label'].to(device)

            optimizer.zero_grad()

            outputs = model(protac_data, e3_ligase_data, poi_data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch: {epoch+1}/{num_epochs}, train loss: {running_loss/len(train_loader):.3f}')
        wandb.log({
            'train/epoch': epoch + 1,
            'train/loss': running_loss / len(train_loader)
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

    setup_seed(model_cfg['seed'])
    
    wandb.init(
        mode="online",
        project='protac-stan',
        config=cfg,
        group=f'run_bz{train_cfg["batch_size"]}_lr{train_cfg["learning_rate"]}',
    )

    wandb.run.summary['model_dir'] = model_dir

    print(cfg)
    wandb.save('model.py')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    train_loader, test_loader = PROTACLoader(root='data/protacdb3', name='protac_maccs', batch_size=train_cfg['batch_size'], collate_fn=collate_fn, train_ratio=train_cfg['train_ratio'])

    model = PROTAC_STAN(model_cfg)
    print(model)
    wandb.watch(model)

    model = train(
        model, train_loader, test_loader, device, 
        lr=train_cfg['learning_rate'], 
        num_epochs=train_cfg['num_epochs'], 
    )

    torch.save(model, f'{model_dir}/model.pt') # save full model (state_dict + architecture)    
    torch.save(model.state_dict(), f'{model_dir}/model_state_dict.pt') # save model state_dict only

    wandb.finish()


if __name__ == '__main__':
    main()