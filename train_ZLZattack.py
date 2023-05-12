from torch.nn.modules.module import Module
import argparse
import os.path as osp
import time
import torch
from pGRACE.dataset import get_dataset
import numpy as np
import random
import os
from torch_geometric.utils import to_dense_adj
from utils import load_data
from torch_geometric.data import Data

class FSattack(Module):
    def __init__(self, args, data, device):
        super(FSattack, self).__init__()
        self.args = args
        self.device = device
        self.data = data.to(device)
    
    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.

        Returns
        -------
        torch.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.

        """
        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask
    
    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj
        r_inv = r_inv + 1e-5
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv
        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat
    
    def compute_gradient(self, adj, Z):
        ZLZ = self.feature_smoothing(adj, Z)
        ZLZ.backward()
        return adj.grad

    def attack(self, Z):
        perturbed_edges = []
        num_total_edges = int(self.data.num_edges / 2)
        #print('total edges', num_total_edges)
        adj_sp = torch.sparse.FloatTensor(self.data.edge_index, torch.ones(self.data.edge_index.shape[1]).to(self.device),
                                          [self.data.num_nodes, self.data.num_nodes])
        adj = adj_sp.to_dense()
        adj.requires_grad = True
        print('Begin perturbing.....')
        # save three poisoned adj when the perturbation rate reaches 1%, 5%, 10%, 15%, 20%, 25%.
        while int(len(perturbed_edges)/2) < int(args.ptb_rate * num_total_edges):
            if int(len(perturbed_edges)/2) in [int(0.05*num_total_edges), int(0.1*num_total_edges), 
                                               int(0.15*num_total_edges), int(0.2*num_total_edges)]:
                output_adj = adj.to(device)
                np.savetxt(save_dir+'/adj_FSA_'+str(np.round(int(len(perturbed_edges)/2)/num_total_edges,2))+'.txt', 
                           output_adj.cpu().data.numpy(), fmt='%d')
                print('---'+str(np.round(int(len(perturbed_edges)/2)/num_total_edges,2)*100)+'% poisoned adjacency matrix saved---')
            
            start = time.time()
            adj_grad = self.compute_gradient(adj, Z)
            "Avoid to generate isolated node!"
            singleton_mask = self.filter_potential_singletons(adj)
            adj_grad = adj_grad *  singleton_mask
            adj_grad_1d = adj_grad.view(-1)
            adj_grad_1d_abs = torch.abs(adj_grad_1d)
            values, indices = adj_grad_1d_abs.sort(descending=True)
            i = -1
            while True:
                i += 1
                index = int(indices[i])
                row = int(index / self.data.num_nodes)
                column = index % self.data.num_nodes
                if [row, column] in perturbed_edges:
                    continue
                if adj_grad_1d[index] < 0 and adj[row, column] == 1:
                    adj.data[row, column] = 0
                    adj.data[column, row] = 0
                    perturbed_edges.append([row, column])
                    perturbed_edges.append([column, row])
                    break
                elif adj_grad_1d[index] > 0 and adj[row, column] == 0:
                    adj.data[row, column] = 1
                    adj.data[column, row] = 1
                    perturbed_edges.append([row, column])
                    perturbed_edges.append([column, row])
                    break
            
            self.data.edge_index = adj.nonzero().T
            #self.data.edge_index = dense_to_sparse(adj)[0]
            end = time.time()
            print('Perturbing edges: %d/%d. Finished in %.2fs' % (len(perturbed_edges)/2, int(args.ptb_rate * num_total_edges), end-start))
        print('Number of perturbed edges: %d' % (len(perturbed_edges)/2))
        output_adj = adj.to(device)
        return output_adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--ptb_rate', type=float, default=0.2)
    args = parser.parse_args()
        
    root_dir = '/home/user/Documents/polyu/universal_defense/code/CLGA-main'
    save_dir = root_dir+'/'+str(args.dataset)
    try:
        os.makedirs(save_dir)
    except:
        pass
    seed = 666
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device(args.device)
    if args.dataset == 'cora_ml':
        adj, features, labels = load_data(dataset=args.dataset)
        data = Data(x=torch.from_numpy(np.array(features.todense())), 
                    edge_index=torch.from_numpy(np.array(adj.todense())).nonzero().T, 
                    y=torch.from_numpy(labels))
    else:
        path = osp.expanduser('dataset')
        path = osp.join(path, args.dataset)
        "Already get LCC in get_dataset func."
        dataset = get_dataset(path, args.dataset)
        data = dataset[0]

    clean_adj = to_dense_adj(data.edge_index.detach().clone())[0].to(device)

    idx_train = np.loadtxt(save_dir+'/idx_train.txt')
    idx_val = np.loadtxt(save_dir+'/idx_val.txt')
    idx_test = np.loadtxt(save_dir+'/idx_test.txt')
    train_mask = torch.zeros((data.num_nodes), dtype=torch.bool)
    val_mask = torch.zeros((data.num_nodes), dtype=torch.bool)
    test_mask = torch.zeros((data.num_nodes), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    Z = torch.from_numpy(np.loadtxt(save_dir+'/GCA_embs_0.txt')).float().to(device)
    
    model = FSattack(args, data, device).to(device)
    poisoned_adj = model.attack(Z)
    np.savetxt(save_dir+'/adj_FSA_'+str(args.ptb_rate)+'.txt', poisoned_adj.cpu().data.numpy(), fmt='%d')
    print('---20% perturbed adjacency matrix saved---')