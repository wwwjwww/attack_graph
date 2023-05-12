from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model
        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, with_bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        #output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Encoder_dense(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0., with_relu=True, with_bias=True):
        super(Encoder_dense, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nout, with_bias=with_bias)
        self.dropout = dropout
        self.with_relu = with_relu
        self.ReLU = nn.PReLU()

    def forward(self, x, adj):
        if self.with_relu:
            x = self.ReLU(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    
    def get_embedding(self, x, adj):
        if self.with_relu:
            x = self.ReLU(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=False)
        x = self.gc2(x, adj)
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        #self.fc2 = torch.nn.Linear(num_proj_hidden, 7)
        self.num_hidden = num_hidden
    
# =============================================================================
#     def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
#         return self.encoder(x, adj)
# =============================================================================
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1)))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        "Space complexity: O(BN) (semi_loss: O(N^2))"
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret

    def node_level_loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        #ret = ret.mean() if mean else ret.sum()
        return ret

class SanGRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(SanGRACE, self).__init__()
        self.encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1)))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        "Space complexity: O(BN) (semi_loss: O(N^2))"
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    #def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        #h3 = self.projection(z3)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
            #l3 = self.semi_loss(h1, h3)
            #l4 = self.semi_loss(h3, h1)
            #l5 = self.semi_loss(h2, h3)
            #l6 = self.semi_loss(h3, h2)
            
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
            #l3 = self.batched_semi_loss(h1, h3, batch_size)
            #l4 = self.batched_semi_loss(h3, h1, batch_size)
            #l5 = self.batched_semi_loss(h2, h3, batch_size)
            #l6 = self.batched_semi_loss(h3, h2, batch_size)

        ret = (l1 + l2) * 0.5
        #ret = (l1 + l2) * 0.5 + eps * (l3 + l4 + l5 + l6) * 0.25
        #ret = (l3 + l4 + l5 + l6) * 0.25
        ret = ret.mean() if mean else ret.sum()
        return ret

class ClusterSanGRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(ClusterSanGRACE, self).__init__()
        self.encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.fc3 = torch.nn.Linear(num_hidden, 7)
        self.num_hidden = num_hidden
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1)))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        "Space complexity: O(BN) (semi_loss: O(N^2))"
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)
    
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, Y: torch.Tensor,
                   eps: float, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h3 = self.projection(z3)
        h3_dr = torch.relu(self.fc3(h3))

        if batch_size is None:
            #l1 = self.semi_loss(h1, h2)
            #l2 = self.semi_loss(h2, h1)
            l3 = self.semi_loss(h1, h3)
            l4 = self.semi_loss(h3, h1)
            l5 = self.semi_loss(h2, h3)
            l6 = self.semi_loss(h3, h2)
            
        else:
            #l1 = self.batched_semi_loss(h1, h2, batch_size)
            #l2 = self.batched_semi_loss(h2, h1, batch_size)
            l3 = self.batched_semi_loss(h1, h3, batch_size)
            l4 = self.batched_semi_loss(h3, h1, batch_size)
            l5 = self.batched_semi_loss(h2, h3, batch_size)
            l6 = self.batched_semi_loss(h3, h2, batch_size)

        #ret = (l1 + l2) * 0.5 + eps * (l3 + l4 + l5 + l6) * 0.25
        ret = (l3 + l4 + l5 + l6) * 0.25 + eps * F.cross_entropy(h3_dr, Y)
        ret = ret.mean() if mean else ret.sum()
        return ret

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
