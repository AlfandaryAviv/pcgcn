from torch_geometric.nn import GCNConv, GATConv, GCN2Conv
import torch
from torch_geometric.data import Data
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric.transforms as T

from pytorch_lightning.utilities.seed import seed_everything
seed_everything(seed=0)


############################# GCN ###########################
class GCN(pl.LightningModule):
    def __init__(self, num_of_features, hid_size, num_of_classes, activation=F.relu, dropout=0.6):
        super(GCN, self).__init__()
        self._layer1 = GCNConv(num_of_features, hid_size)
        self._activation = activation
        self._layer2 = GCNConv(hid_size, num_of_classes)
        self._dropout = dropout

    def forward(self, data: Data):
        x = self._layer1(data.x, data.edge_index)
        z = self._activation(x)
        z = F.dropout(z, self._dropout)
        h = self._layer2(z, data.edge_index)
        h = F.dropout(h, self._dropout)
        return z, torch.softmax(h, dim=1)


############################# GAT ###########################
class GatNet(nn.Module):
    def __init__(self, num_features, num_classes, hid_layer=10, dropout=0.3, activation="elu", heads=[8, 1]):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(num_features, hid_layer, heads=heads[0], dropout=dropout)
        self.conv2 = GATConv(hid_layer * heads[0], num_classes, heads=heads[1], concat=False, dropout=dropout)
        self.dropout = dropout
        if activation == 'elu':
            self.activation_func = F.elu
        else:
            self.activation_func = F.relu

    def forward(self, data: Data):
        x = F.dropout(data.x, p=self.dropout)
        z = self.conv1(x, data.edge_index)
        z = self.activation_func(z)
        z = F.dropout(z, p=self.dropout)
        h = self.conv2(z, data.edge_index)
        return z, torch.softmax(h, dim=1)


############################# SSP #############################
class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=False):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=False)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class SSPNet(torch.nn.Module):
    def __init__(self, num_of_features, hid_size, num_of_classes, activation=F.relu, dropout=0.6):
        super(SSPNet, self).__init__()
        self.crd = CRD(num_of_features, hid_size, dropout)
        self.cls = CLS(hid_size, num_of_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.crd(x, edge_index, data.training_inds)
        x = self.cls(z, edge_index, data.training_inds)
        return z, x


class GCNII(torch.nn.Module):
    def __init__(self,num_of_features, hid_size, num_of_classes, num_layers=64, alpha=0.1, theta=0.5,
                 shared_weights=True, dropout=0.0):
        super(GCNII, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(nn.Linear(num_of_features, hid_size))
        self.lins.append(nn.Linear(hid_size, num_of_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hid_size, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, data):
        data = T.ToSparseTensor()(data)
        x, adj_t = data.x, data.adj_t
        adj_t = gcn_norm(adj_t)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x=x, x_0=x_0, edge_index=adj_t[0])
            x = x.relu()
        z = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return z, x.log_softmax(dim=-1)
