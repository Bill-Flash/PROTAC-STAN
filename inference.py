import random

import numpy as np
import pandas as pd
import toml
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader import PROTACLoader
from model import PROTAC_STAN
import argparse


def setup_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(model, test_loader, device, save_att=False): 
    model = model.to(device)
    model.eval()

    predictions = []
    labels = []
    att_maps = []
    has_labels = False

    with torch.no_grad():

        for data in test_loader:
            protac_data = data['protac'].to(device)
            e3_ligase_data = data['e3_ligase'].to(device)
            poi_data = data['poi'].to(device)
            label = data['label']
            
            # Check if labels exist
            if label is not None:
                has_labels = True
                label = label.to(device)
                labels.extend(label.cpu().numpy())

            outputs, atts = model(protac_data, e3_ligase_data, poi_data, mode='eval')
            _, predicted = torch.max(outputs.data, dim=1)
            

            predictions.extend(predicted.cpu().numpy())
            if save_att:
                att_maps.extend(atts.cpu().numpy())

    results = {
        'predictions': predictions,
        'att_maps': att_maps,
        'has_labels': has_labels
    }
    
    # Calculate metrics if labels exist
    if has_labels:
        accuracy = accuracy_score(labels, predictions)
        roc_auc = roc_auc_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        results['labels'] = labels
        results['accuracy'] = accuracy
        results['roc_auc'] = roc_auc
        results['f1'] = f1
    
    return results


def main():

    setup_seed(21332)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    cfg = toml.load('config.toml')
    model_cfg = cfg['model']

    model = PROTAC_STAN(model_cfg)
    path = 'saved_models/protac-stan.pt'
    print(f'Loading model from {path}...')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    print(model)
    
    parser = argparse.ArgumentParser(description='PROTAC-STAN Inference')
    parser.add_argument('--root', type=str, default='data/custom', help='Path to the data directory')
    parser.add_argument('--name', type=str, default='custom', help='Raw file name without extension')
    parser.add_argument('--save_att', action='store_true', help='Whether to save attention maps, might consume a lot of memory')

    args = parser.parse_args()

    root = args.root
    name = args.name
    save_att = args.save_att

    _, test_loader = PROTACLoader(root=root, name=name, batch_size=1, train_ratio=0.0)

    results = test(model, test_loader, device, save_att)
    
    predictions = results['predictions']
    has_labels = results['has_labels']
    
    if save_att:
        att_maps = results['att_maps']
        print('Saving attention maps...')
        np.save(f'{root}/{name}_att.npy', att_maps)
    
    print('\n' + '='*50)
    print('Inference Results')
    print('='*50)
    print(f'Total samples: {len(predictions)}')
    print(f'Predictions: {predictions}')
    
    if has_labels:
        labels = results['labels']
        accuracy = results['accuracy']
        roc_auc = results['roc_auc']
        f1 = results['f1']
        
        print(f'\nTrue labels: {labels}')
        print('\n' + '-'*50)
        print('Evaluation Metrics:')
        print('-'*50)
        print(f'Accuracy:  {accuracy:.4f} ({100 * accuracy:.2f}%)')
        print(f'ROC AUC:   {roc_auc:.4f}')
        print(f'F1 Score:  {f1:.4f}')
        print('='*50)
    else:
        print('\nNo labels found in dataset. Skipping metric calculation.')
        print('='*50)


if __name__ == '__main__':
    main()
