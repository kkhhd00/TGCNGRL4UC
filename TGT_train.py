import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from TGT_30 import GCN_TCN

warnings.filterwarnings("ignore")

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

random.seed(30)
np.random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
#torch.backends.cudnn.deterministic = True

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
# 构建边矩阵
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

aggregate = 'cat'
lr = 1e-5
ours_weight_decay = 5e-3
weight_decay = 5e-3
epochs = 600
val_min_num = 0
in_size = 30
out_channels = 6 * split
# model = GCN(in_channels=split, hidden_channels=516, out_channels=6 * split, num_heads=2).to(device)
STConv_net = GCN_TCN(30, split, 128, 6 * split, 4, 10)
model = STConv_net.to(device)

criterion_2 = nn.L1Loss()
# criterion = torch.nn.MSELoss()
criterion = criterion_2

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=7, verbose=True)


def train():
    model.train()
    total_loss = 0
    for step, data in enumerate(train_loader):
        data = data.to(device)
        # print(data.shape)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.view(-1, out_channels)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        L1loss = criterion_2(out, y)
        total_loss += L1loss.item()
    return total_loss / len(train_loader)


def validate(model_xc):
    model_xc.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            y = data.y.view(-1, out_channels)
            loss = criterion(out, y)
            L1loss = criterion_2(out, y)
            total_loss += L1loss.item()
            all_predictions.extend(out.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    return total_loss / len(val_loader), all_predictions, all_targets


val_predictions = []
val_targets = []
val_loss_list = []
train_loss_list = []

best_val_loss = float('inf')
best_epoch = 0
best_model_state_dict = None
for epoch in tqdm(range(epochs)):
    train_loss = train()
    val_loss, epoch_val_preds, epoch_val_targets = validate(model)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    if epoch > val_min_num:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state_dict = model.state_dict()
            val_predictions = epoch_val_preds
            val_targets = epoch_val_targets
            # print(f'New best model found at epoch {epoch} with validation loss {val_loss:.4f}.')
        scheduler.step(val_loss)
    print(f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

if best_model_state_dict is not None:
    torch.save(best_model_state_dict, 'best_model_GCN.pt')

print(f'Best Test Loss: {best_val_loss:.4f}')
val_df = pd.DataFrame({
    'Val_Predictions': [item for sublist in val_predictions for item in sublist],
    'Val_Targets': [item for sublist in val_targets for item in sublist]
})
# print(val_df)
val_df['Val_Predictions'] = val_df['Val_Predictions'].astype(float)
val_df['Val_Targets'] = val_df['Val_Targets'].astype(float)
# 保存到CSV文件
val_df.to_csv('prediction_GCN.csv', index=False, float_format='%.6f')

train_loss_list = train_loss_list[val_min_num:]
val_loss_list = val_loss_list[val_min_num:]

episodes_train_list = list(range(len(train_loss_list)))
episodes_val_list = list(range(len(val_loss_list)))
plt.plot(episodes_train_list, train_loss_list, label='train_loss', color='blue')
plt.plot(episodes_val_list, val_loss_list, label='val_loss', color='orange')
# plt.plot(episodes_list, return_pdemand_cost_list, label='Pdemand Cost Returns', color='red')
plt.xlabel('Episodes')
plt.ylabel('loss')
plt.legend()
plt.show()
