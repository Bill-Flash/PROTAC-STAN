import random

import numpy as np
import pandas as pd
import toml
import torch

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
    att_maps = []

    with torch.no_grad():

        for data in test_loader:
            protac_data = data['protac'].to(device)
            e3_ligase_data = data['e3_ligase'].to(device)
            poi_data = data['poi'].to(device)
            # label = data['label'].to(device)

            outputs, atts = model(protac_data, e3_ligase_data, poi_data, mode='eval')
            _, predicted = torch.max(outputs.data, dim=1)
            

            predictions.extend(predicted.cpu().numpy())
            if save_att:
                att_maps.extend(atts.cpu().numpy())

    results = {
        'predictions': predictions,
        'att_maps': att_maps
    }
    
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
    if save_att:
        att_maps = results['att_maps']
        print('Saving attention maps...')
        np.save(f'{root}/{name}_att.npy', att_maps)
        
    print(predictions)


if __name__ == '__main__':
    main()
