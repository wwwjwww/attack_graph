import argparse
import os.path as osp
import random
import nni
import time
import torch
import numpy as np
from torch_geometric.utils import dropout_adj, degree, to_undirected
#from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset

def train():
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted(data.x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(split, final=False):
    model.eval()
    z = model(data.x, data.edge_index)
    evaluator = MulticlassEvaluator()
    acc = log_regression(z, data, evaluator, split, num_epochs=3000)['acc']

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
    parser.add_argument('--attack_method', type=str, default='FSA')

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
        'num_epochs': 1000,
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


    perturbed_adj = torch.from_numpy(np.loadtxt(save_dir+'/adj_%s_%s.txt'%(args.attack_method, args.attack_rate))).float().to(device)
    data.edge_index = perturbed_adj.nonzero().T

    data = data.to(device)
    edge_index = data.edge_index
    edge_sp_adj = torch.sparse.FloatTensor(edge_index,
                                           torch.ones(edge_index.shape[1]).to(device),
                                           [data.num_nodes, data.num_nodes])
    edge_adj = edge_sp_adj.to_dense().to(device)

    encoder = Encoder(data.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        print(data.edge_index.shape)
        edge_index_ = to_undirected(data.edge_index)
        print(edge_index_.shape)
        node_deg = degree(edge_index_[1], num_nodes=data.num_nodes)
        print(node_deg.shape)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    log = args.verbose.split(',')
    print('Begin training....')

    best_acc = 0
    for epoch in range(1, param['num_epochs'] + 1):
        start = time.time()
        loss = train()
        end = time.time()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, training time={end-start}')


    idx_train = np.loadtxt(save_dir + '/idx_train.txt')
    idx_val = np.loadtxt(save_dir + '/idx_val.txt')
    idx_test = np.loadtxt(save_dir + '/idx_test.txt')
    train_mask = torch.zeros((data.num_nodes), dtype=torch.bool)
    val_mask = torch.zeros((data.num_nodes), dtype=torch.bool)
    test_mask = torch.zeros((data.num_nodes), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    split = {'train': data.train_mask,
             'valid': data.val_mask,
             'test': data.test_mask
             }

    acc = test(split, final=True)

    if 'final' in log:
        print(f'{acc}')
    if acc > best_acc:
        best_acc = acc
    with open(f'./results/results_{args.dataset}_{args.attack_method}_GCA/result_acc_%s'%(args.attack_rate), 'a') as f:
        f.write(str(acc))
        f.write('\n')
    print(f'best accuracy = {best_acc}')

