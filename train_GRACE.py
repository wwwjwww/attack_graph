import argparse
import os.path as osp
import random
import nni
import time
import torch
import numpy as np
from torch_geometric.utils import dropout_adj, degree, to_undirected
#from simple_param.sp import SimpleParam
from GRACE.model import Encoder, GRACE, drop_feature
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset

def train():
    model.train()
    optimizer.zero_grad()

    edge_index_1 = dropout_adj(edge_index, p=param[f'drop_edge_rate_1'])[0]
    edge_index_2 = dropout_adj(edge_index, p=param[f'drop_edge_rate_2'])[0]
    x_1 = drop_feature(data.x, param['drop_edge_rate_1'])
    x_2 = drop_feature(data.x, param['drop_edge_rate_2'])
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):
    model.eval()
    z = model(data.x, data.edge_index)
    evaluator = MulticlassEvaluator()
    if args.dataset == 'Cora':
        acc = log_regression(z, data, evaluator, split='cora', num_epochs=3000)['acc']
    elif args.dataset == 'CiteSeer':
        acc = log_regression(z, data, evaluator, split='citeseer', num_epochs=3000)['acc']
    else:
        raise ValueError('Please check the split first!')

    if final:
        nni.report_final_result(acc)
    else:
        nni.report_intermediate_result(acc)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    #parser.add_argument('--param', type=str, default='local:general.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    #parser.add_argument('--save_split', action="store_true")
    #parser.add_argument('--load_split', action="store_true")
    parser.add_argument('--attack_rate', type=float, default=0.10)

    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    #sp = SimpleParam(default=default_param)
    #param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    param = default_param
    #use_nni = args.param == 'nni'
    if args.device != 'cpu':
        args.device = 'cuda'
    
    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    device = torch.device(args.device)

    root_dir = '/home/wujingwen/attack_graph/attack_graph'
    save_dir = root_dir + '/' + str(args.dataset)

    path = osp.expanduser('dataset')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]


    perturbed_adj = torch.from_numpy(np.loadtxt(save_dir+'/adj_FSA_%s.txt'%(args.attack_rate))).float().to(device)
    data.edge_index = perturbed_adj.nonzero().T

    data = data.to(device)
    edge_index = data.edge_index
    edge_sp_adj = torch.sparse.FloatTensor(edge_index,
                                           torch.ones(edge_index.shape[1]).to(device),
                                           [data.num_nodes, data.num_nodes])
    edge_adj = edge_sp_adj.to_dense().to(device)

    # generate split
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    encoder = Encoder(data.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    log = args.verbose.split(',')
    print('Begin training....')

    best_acc = 0
    for epoch in range(1, param['num_epochs'] + 1):
        start = time.time()
        loss = train()
        end = time.time()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, training time={end-start}')

    acc = test(final=True)

    if 'final' in log:
        print(f'{acc}')
    if acc > best_acc:
        best_acc = acc
    with open(f'./results_{args.dataset}_GRACE/result_acc_%s'%args.attack_rate, 'a') as f:
        f.write(str(acc))
        f.write('\n')
    print(f'best accuracy = {best_acc}')

