import os.path as osp

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon, TUDataset, Airports
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset


def get_dataset(path, name):
    assert name in ['Air-USA', 'Air-Brazil', 'Air-Europe', 'Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code', 'Proteins']
    name = 'dblp' if name == 'DBLP' else name
    
    if name == 'Air-USA':
        return Airports(root=path, name='USA', transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))
    if name == 'Air-Brazil':
        return Airports(root=path, name='Brazil', transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))
    if name == 'Air-Europe':
        return Airports(root=path, name='Europe', transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))
    
    if name == 'Proteins':
        return TUDataset(root=path, name='PROTEINS', transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))

    if name == 'WikiCS':
        #return WikiCS(root=path, transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))
        return WikiCS(root=path, transform=T.LargestConnectedComponents())
        
    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(path, 'OGB'), name=name, transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(path, 'Citation'), name, 
                                                           transform=T.Compose([T.LargestConnectedComponents(), T.NormalizeFeatures()]))
