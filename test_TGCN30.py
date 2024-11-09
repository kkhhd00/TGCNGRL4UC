import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.nn import GCNConv
import random
import os
import sys
import argparse
from tqdm import tqdm
from torch_geometric.data import DataLoader, Data
import pandas as pd
from torch.utils.data import random_split
from matplotlib import pyplot as plt
import matplotlib
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility


from TGT_30 import GCN_TCN
warnings.filterwarnings("ignore")

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
print("项目根目录路径：", root_path)


random.seed(30)
np.random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
node_data = pd.read_csv("data/IEEE30_combined_data_LoadDef.csv").values
label_data = pd.read_csv("data/all_a_array_2.csv", header=None).values
print('node_data', node_data.shape)
print('label_data', label_data.shape)
new_node_data = node_data.reshape((365, 288, 20)).transpose(0, 2, 1)
insert_positions = [1, 2, 3, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 28, 29]
node_feat_data = np.zeros((365, 30, 288))
node_feat_data[:, insert_positions, :] = new_node_data
x_all = torch.from_numpy(node_feat_data).float()
print('x_all', x_all.shape)
edgeindex = [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [1, 5], [3, 5], [4, 6], [5, 6], [5, 27], [5, 7], [7, 27],
             [27, 26], [26, 29], [29, 28], [26, 28], [26, 24], [24, 25], [5, 8], [8, 10], [8, 9], [5, 9], [9, 20],
             [20, 21], [9, 16], [15, 16], [3, 11], [11, 12], [11, 17], [11, 15], [17, 18], [18, 19], [9, 19], [9, 23],
             [11, 13], [13, 14], [14, 22], [22, 23], [23, 24]]
edge_index = torch.from_numpy(np.array(edgeindex).transpose()).long()

new_label_data = label_data.reshape((365, 288 * 6))
y_all = torch.from_numpy(new_label_data).float()

split = 12
Gdata_list = [Data(x=x_all[i, :, j:j + split], edge_index=edge_index, y=y_all[i, j * 6:(j + split) * 6]) for i in
              range(365) for j in range(0, 288 - split)]
train_size = int(len(Gdata_list) * 0.9)
val_size = int((len(Gdata_list) - train_size) * 0.5)
test_size = int(len(Gdata_list) - train_size - val_size)
train_dataset, val_dataset, test_dataset = random_split(Gdata_list, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

aggregate = 'cat'
lr = 1e-3
ours_weight_decay = 5e-3
weight_decay = 5e-3
epochs = 600
val_min_num = 0
in_size = 30
out_channels = 6 * split

criterion_2 = nn.L1Loss()
# criterion = torch.nn.MSELoss()
criterion = criterion_2

def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def rmse_loss(pred, target):
    return torch.sqrt(F.mse_loss(pred, target))


def relative_error(pred, target, epsilon=1e-8):
    return torch.mean(torch.abs((pred - target) / (target + epsilon)))


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    with np.errstate(divide='ignore', invalid='ignore'):
        re = np.abs((y_true - y_pred) / y_true)
        re = np.nan_to_num(re, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and Inf with 0
        re = np.mean(re)

    return mae, mse, rmse, r2, re

best_model = GCN_TCN(30, split, 128, 6 * split, 4, 10).to(device)
best_model.load_state_dict(torch.load('best_model_GCN.pt'))
best_model.eval()

test_predictions = []
test_targets = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = best_model(data)
        y = data.y.view(-1, out_channels)
        test_predictions.extend(out.cpu().numpy().tolist())
        test_targets.extend(y.cpu().numpy().tolist())

test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)
test_predictions[test_predictions < 0.1] = 0

mae, mse, rmse, r2, re = compute_metrics(test_targets, test_predictions)

results_df = pd.DataFrame({
    'MAE': [mae],
    'MSE': [mse],
    'RMSE': [rmse],
    'R2': [r2],
    'RE': [re]
})
results_df.to_csv('result_30.csv', index=False, float_format='%.6f')
test_df = pd.DataFrame({
    'Test_Predictions': [item for sublist in test_predictions for item in sublist],
    'Test_Targets': [item for sublist in test_targets for item in sublist]
})
test_df['Test_Predictions'] = test_df['Test_Predictions'].astype(float)
test_df['Test_Targets'] = test_df['Test_Targets'].astype(float)
test_df.to_csv('test_predictions.csv', index=False, float_format='%.6f')
