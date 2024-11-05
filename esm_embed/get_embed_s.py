import os
import os.path as osp
import pickle
import sys

import numpy as np
import torch
from torchdrug import core, data, datasets, models, tasks, utils
from torchdrug.transforms import Compose, ProteinView, TruncateProtein
from tqdm import tqdm
import argparse


class ESMSEmbedder(object):
    """
    推理EMS2-S嵌入
    模型：esm_650m_s.pth
    """
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.model = self.model.to(self.device)
    
    def load_model(self):
        print('Loading model...')
        model_dir = "./model"
        esm_model = models.EvolutionaryScaleModeling(model_dir, model="ESM-2-650M", readout="mean")

        # Load ESM-2-650M-S
        model_dict = torch.load(os.path.join(model_dir, "esm_650m_s.pth"), map_location=torch.device("cpu"))
        esm_model.load_state_dict(model_dict)
        return esm_model
    
    def get_embed(self, dataloader):
        print(f'dataset size: {len(dataloader)}')
        reprs = []
        self.model.eval()
        for batch in tqdm(dataloader, 'Embedding'):
            batch = utils.cuda(batch, device='cuda')
            graph = batch['graph']
            uniprot = batch['uniprot']
            
            print(f'Processing {uniprot}, seq length: {graph.num_residue}')
            output = self.model(graph, graph.node_feature.float())
            graph_feature = output['graph_feature']
            batch_feature = graph_feature.cpu().detach().numpy()
            embeddings = zip(uniprot, batch_feature)
            reprs.extend(embeddings)
            print(f'repr shape: {reprs[-1][1].shape}')
        print('Done!')
        return reprs
    
class PROTACTargets(data.ProteinDataset):
    """
    PROTACTargets dataset.
    Parameters:
        path (str): Path to the data `p_map.pickle`.
        transform (callable, optional): A function/transform that takes in a dictionary as input and returns a transformed version.
        verbose (int, optional): Verbosity level
        **kwargs: Additional arguments to pass to the loader
    """
    
    def __init__(self, path='data', split='test', transform=None, verbose=1, lazy=True, **kwargs):
        path = os.path.join(os.path.expanduser(path))
        self.path = path
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.lazy = lazy
        
        
        p_map = pickle.load(open(os.path.join(path, "p_map.pkl"), "rb"))
        protein_seqs = []
        for uniprot, seq in p_map.items():
            protein_seqs.append((uniprot, seq))
        
        self.data = []
        if verbose:
            protein_seqs = tqdm(protein_seqs, "Constructing proteins from sequences")
        for uniprot, seq in protein_seqs:
            protein = data.Protein.from_sequence(seq, **kwargs)
            self.data.append((uniprot, protein))

    def get_item(self, index):
        uniprot, protein = self.data[index]
        item = {'graph': protein, 'uniprot': uniprot}
        if self.transform:
            item = self.transform(item)
        return item

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: ESM-S Embedding",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


def main():
    protein_view = ProteinView(view="residue")
    truncate_protein = TruncateProtein(max_length=1022) # ESM limit
    transforms = Compose([protein_view, truncate_protein])
    
    parser = argparse.ArgumentParser(description="ESM-S Embedding")
    parser.add_argument("--root", type=str, default="../data/custom", help="Path to the protein data")
    args = parser.parse_args()

    root = args.root

    dataset = PROTACTargets(path=root, transform=transforms)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    embedder = ESMSEmbedder()
    embeddings = embedder.get_embed(dataloader)
    
    esms_map = dict(embeddings)

    with open(osp.join(root, 'esm_s_map.pkl'), 'wb') as f:
        pickle.dump(esms_map, f)

    print('Done!')


if __name__ == "__main__":
    main()