import numpy as np
import torch
import random
import os
from torch_geometric.data import DataLoader, Data
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
def gen_GNN_data():
    node_data = pd.read_csv("data/IEEE30_combined_data_LoadDef.csv").values
    label_data = pd.read_csv("data/all_a_array_2.csv", header=None).values

    # 节点特征处理
    new_node_data = node_data.reshape((365, 288, 20)).transpose(0, 2, 1)  # 365*20*288

    insert_positions = [1, 2, 3, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 28, 29]
    node_feat_data = np.zeros((365, 30, 288))
    node_feat_data[:, insert_positions, :] = new_node_data

    x_all = torch.from_numpy(node_feat_data).float()
    edgeindex = [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [1, 5], [3, 5], [4, 6], [5, 6], [5, 27], [5, 7], [7, 27],
                 [27, 26], [26, 29], [29, 28], [26, 28], [26, 24], [24, 25], [5, 8], [8, 10], [8, 9], [5, 9], [9, 20],
                 [20, 21], [9, 16], [15, 16], [3, 11], [11, 12], [11, 17], [11, 15], [17, 18], [18, 19], [9, 19],
                 [9, 23],
                 [11, 13], [13, 14], [14, 22], [22, 23], [23, 24]]

    edge_index = torch.from_numpy(np.array(edgeindex).transpose()).long()
    new_label_data = label_data
    y_all = torch.from_numpy(new_label_data).float()

    split = 12
    Gdata_list_train = []
    for i in range(365):
        for j in range(288 - split+1):
            if j % (split) == 0:
                x = x_all[i, :, j:j + split]
                y = y_all[i, j * 6:(j + split) * 6]
                data = Data(x=x, edge_index=edge_index, y=y)
                Gdata_list_train.append(data)
    return Gdata_list_train, split


Gdata_list_train, split = gen_GNN_data()
