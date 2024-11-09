import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch_geometric.nn import ChebConv
import numpy as np
from torch_geometric.nn import GCNConv

###TCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0):
        super(TemporalBlock, self).__init__()
        # 打印参数类型和值
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm1 = torch.nn.LayerNorm(30)
        self.layernorm2 = torch.nn.LayerNorm(30)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 3 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=0, alpha=0.7,
                 dropout=0.5, save_mem=True, use_bn=True, use_resi=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.conv_0 = GCNConv(in_channels, hidden_channels, cached=not save_mem)
        self.bn_0 = nn.BatchNorm1d(hidden_channels)
        self.conv_1 = GCNConv(hidden_channels, out_channels, cached=not save_mem)
        self.bn_1 = nn.BatchNorm1d(out_channels)
        self.fc1 = nn.Linear(in_channels, hidden_channels)

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_res = use_resi
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []
        x = self.conv_0(x, edge_index)
        if self.use_bn:
            x = self.bn_0(x)
        x = self.activation(x)
        layer_.append(x)
        for i in range(len(self.convs)):
            conv = self.convs[i]
            x = conv(x, edge_index)
            if self.use_res:
                x = x + layer_[0]
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            layer_.append(x)

        x = self.conv_1(x, edge_index)
        if self.use_bn:
            x = self.bn_1(x)
        x = self.activation(x)
        layer_.append(x)

        return x


class GCN_TCN(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 K: int,
                 normalization: str = "sym",
                 bias: bool = True, ):
        super(GCN_TCN, self).__init__()
        # self.gcn = GCNLayer(gcn_in_feats, gcn_hidden_feats)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.normalization = normalization
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.K = K
        self.bias = bias
        self.tcn1 = TemporalConvNet(num_inputs=12,
                                    num_channels=[128, 512, 128],
                                    kernel_size=kernel_size)
        self.conv1 = GCNConv(12, 12)
        self.conv2 = GCNConv(64, 12)
        self.gcn = GCN(12, 64, 12)
        self.fc = nn.Linear(128 * self.num_nodes, 12 * 6)
        self._batch_norm = nn.BatchNorm1d(self.num_nodes)

    def forward(self, data):
        X = self.conv1(data.x, data.edge_index)
        #X = F.relu(X)
        X = X.reshape(-1, self.num_nodes, 12)
        X = X.permute(0, 2, 1)
        T = F.relu(self.tcn1(X))
        T = torch.flatten(T, start_dim=1)
        T = T.view(-1, 30 * 128)
        out = self.fc(T)
        return out
