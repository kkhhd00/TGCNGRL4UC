import math
import torch
from TGT_30 import GCN_TCN
import gen_Data

def get_day_num(start_num: int, day):
    day_num = start_num
    print(day)
    return day_num, day
def gen_model_SL(split, device):
    num_nodes = 30
    hidden_channels = 128
    in_channels = split
    out_channels = 6 * split
    aggregate = 'cat'
    model = GCN_TCN(num_nodes, in_channels, hidden_channels, out_channels, kernel_size=4, K=1).to(device)
    return model
def pre_solve_by_SL(model, Data, day, device):
    model.eval()
    model.load_state_dict(torch.load('best_model_GCN1.pt'))
    # Data = Data.to(device)
    # print(Data.shape)
    day = math.floor(day / 288)
    day = day*24
    data = Data[day].to(device)
    out = model(data)
    out = out[0, 0:6]
    # print(out)
    return out, data

def gen_initstate(start_num: int, day, device):
    day_num, day = get_day_num(start_num, day)
    sjwl_data, split = gen_Data.gen_GNN_data()
    sjwl_model = gen_model_SL(split, device)
    SL_pri_solution, demand_data = pre_solve_by_SL(sjwl_model, sjwl_data, day_num, device)
    SL_pri_solution = (SL_pri_solution.reshape(6)).cpu().detach().numpy()
    #sum1 = sum(SL_pri_solution)
    return SL_pri_solution
