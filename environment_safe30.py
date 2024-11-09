import time

import numpy as np
import pandas as pd
import os
import json
import gym
import torch
import init_info
from gym import spaces
from dispatch_ipopt import ipopt_solve
from datetime import datetime

# from dispatch_gurobi import gurobi_solve
# from dispatch_ipopt import ipopt_solve
busload = 'data/busloads_30.csv'
ParameterData_train_demand = 'data/train_data_6gen3.csv'  # demand
Gen_units_path = 'data/kazarlis_units_6.csv'  # info
DEFAULT_NUM_GEN = 6
DEFAULT_DISPATCH_FREQ_MINS = 5
DEFAULT_EPISODE_LENGTH_HRS = 24
average_demand = pd.read_csv(os.path.join(ParameterData_train_demand))["demand"].mean()


class Env(object):
    def __init__(self, gen_info, profiles_df, busload_df,
                 mode='train', **kwargs):
        self.mode = mode
        self.gen_info = gen_info
        self.profiles_df = profiles_df
        self.busload_df = busload_df
        self.dispatch_freq_mins = kwargs.get('dispatch_freq_mins',
                                             DEFAULT_DISPATCH_FREQ_MINS)  # Dispatch frequency in minutes
        self.dispatch_resolution = self.dispatch_freq_mins / 60.
        self.num_gen = self.gen_info.shape[0]
        if self.mode == 'test':
            self.episode_length = len(self.profiles_df)
        else:
            self.episode_length = kwargs.get('episode_length_hrs', DEFAULT_EPISODE_LENGTH_HRS)
            self.episode_length = int(self.episode_length * (60 / self.dispatch_freq_mins))
        # Generator info
        self.max_output = self.gen_info['max_output'].to_numpy()
        self.min_output = self.gen_info['min_output'].to_numpy()
        self.p_max_vec = self.max_output
        self.p_min_vec = self.min_output
        self.status = self.gen_info['status'].to_numpy()
        self.a = self.gen_info['a'].to_numpy()
        self.b = self.gen_info['b'].to_numpy()
        self.c = self.gen_info['c'].to_numpy()
        self.t_min_down = self.gen_info['t_min_down'].to_numpy()
        self.t_min_up = self.gen_info['t_min_up'].to_numpy()
        self.t_max_up = self.gen_info['t_max_up'].to_numpy()
        self.RampUp = self.gen_info['RampUp'].to_numpy()
        self.RampDown = self.gen_info['RampDown'].to_numpy()
        self.min_demand = np.max(self.min_output)
        self.max_demand = np.sum(self.max_output)
        self.dispatch_tolerance = 1  # epsilon for lambda iteration.

        self.forecast = None
        self.day_cost = 0  # cost for the entire day
        self.summarize_marginal_functions()
    ### 1.******get state ******
    def _get_state(self):
        state = {'status': self.status,
                 'demand_forecast': self.episode_forecast,
                 # 'demand_errors': self.arma_demand.xs,
                 # 'wind_forecast': self.episode_wind_forecast,
                 'cost': self.fuel_cost,
                 'timestep': self.episode_timestep,
                 'power': self.power,
                 'day_cost': self.day_cost,
                 'day': self.day,
                 'ens': self.ens_amount,
                 'cons': self.cons,
                 'load': self.load
                 }
        self.state = state
        return state

    def get_current_state(self):
        self.determine_priority_orders()
        self.identify_must_ON_and_must_OFF_units()
        state = self._get_state()
        return state

    def get_next_state(self, action_vec: np.ndarray):
        self.update_gen_status(self.action_vec)
        self._update_production_capacities(action_vec)
        self.commits_vec = action_vec
        next_state_dict = self.get_current_state()
        return next_state_dict

    ### 2.*****muston/mustoff*****
    def _determine_constraints(self):
        self.must_on = np.array(
            [True if 0 < self.status[i] < self.t_min_up[i] else False for i in range(self.num_gen)])
        self.must_off = np.array(
            [True if -self.t_min_down[i] < self.status[i] < 0 else False for i in range(self.num_gen)])

    def _legalise_action(self, action):
        x = np.logical_or(np.array(action), self.must_on)
        x = x * np.logical_not(self.must_off)
        return (np.array(x, dtype=int))

    def _is_legal(self, action):
        action = np.array(action)
        illegal_on = np.any(action[self.must_on] == 0)
        illegal_off = np.any(action[self.must_off] == 1)
        if any([illegal_on, illegal_off]):
            return False
        else:
            return True

    def identify_must_ON_and_must_OFF_units(self):

        self.must_on = np.array(
            [True if 0 < self.status[i] < self.t_min_up[i] else False for i in range(self.num_gen)])
        self.must_off = np.array(
            [True if -self.t_min_down[i] < self.status[i] < 0 else False for i in range(self.num_gen)])

    def update_gen_status(self, action):
        def single_update(status, action):
            if status > 0:
                if action == 1:
                    return (status + 1)
                else:
                    return -1
            else:
                if action == 1:
                    return 1
                else:
                    return (status - 1)
        self.status = np.array([single_update(self.status[i], action[i]) for i in range(len(self.status))])

    ### 3.******生产成本*******
    def prod_cost_funs(self, loads_vec: np.ndarray):
        # 计算生产成本函数向量，不包含额外的成本项
        prod_cost_funs_vec = self.a * loads_vec ** 2 + self.b * loads_vec + self.c
        # 返回生产成本函数向量，仅在负载大于 0 时有效
        return np.where(loads_vec > 0, 1, 0) * prod_cost_funs_vec


    def _generator_fuel_costs(self, output, commitment):
        costs = 0
        for j in range(6):
            costs += (self.a[j] * output[j] ** 2 + self.b[j] * output[j] + self.c[j])
        return costs

    def calculate_lost_load_cost(self, net_demand, disp):
        diff = max(net_demand - np.sum(disp), 0)
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        ens_cost = ens_amount * 100
        return ens_cost, ens_amount

    def calculate_fuel_cost_and_dispatch(self, demand, commitment):

        demand = np.minimum(np.sum(self.max_output), demand)
        disp, Penalty, cons, correct, on_cost = self.economic_dispatch(commitment, demand, 0, 10)
        # Calculate fuel costs costs
        fuel_costs = self._generator_fuel_costs(disp, commitment)
        return fuel_costs, disp, Penalty, cons, correct, on_cost
    ### 4.*******demand/reword********
    def _get_net_demand(self):
        self.episode_timestep += 1
        self.forecast = self.episode_forecast[self.episode_timestep]
        if self.episode_timestep < 287:
            self.load = self.day_load[self.episode_timestep + 1, :]
        else:
            self.load = self.day_load[self.episode_timestep, :]
        demand_real = self.forecast
        self.demand_real = demand_real
        net_demand = demand_real
        net_demand = np.clip(net_demand, self.min_demand, self.max_demand)
        return net_demand

    def _get_reward(self):
        operating_cost = self.fuel_cost * average_demand / self.net_demand + self.Penalty + self.ens_cost + self.on_cost
        reward = -operating_cost
        self.reward = reward
        return reward

    ### 5.PL
    def summarize_marginal_functions(self):

        max_prod_cost_points_vec = np.where(self.prod_cost_funs(self.p_min_vec) > self.prod_cost_funs(self.p_max_vec),
                                            self.p_min_vec, self.p_max_vec)

        self.max_prod_costs_vec = self.prod_cost_funs(max_prod_cost_points_vec)

        self.min_prod_costs_MW_vec = self.max_prod_costs_vec / self.p_max_vec

    def determine_priority_orders(self):
        up_times_vec = np.maximum(self.t_max_up, 0.001)
        ON_costs_vec = self.min_prod_costs_MW_vec / up_times_vec
        self.ON_priorities_vec = ON_costs_vec
        self.ON_priority_idx_vec = self.ON_priorities_vec.argsort()

    ### 6.*****adjust******
    def ensure_action_legitimacy(self, demand: float, action_vec: np.ndarray):
        if self._is_legal(action_vec) is False:
            action_vec = self._legalise_action(action_vec)
        action_vec = self._check_for_future_demands(action_vec)
        if np.sum(action_vec * self.p_max_vec) < demand:
            action_vec = self._adjust_low_capacity(demand, action_vec)
        elif np.sum(action_vec * self.p_min_vec) > demand:
            action_vec = self._adjust_excess_capacity(demand, action_vec)

        return action_vec

    def _check_for_future_demands(self, action_vec: np.ndarray):
        commits_vec = np.logical_and(np.logical_not(self.must_off),
                                     np.logical_and(np.logical_not(self.must_on),
                                                    self.commits_vec == 1)) * 1

        if np.any(commits_vec):
            prev_ON_idx_vec = np.where(commits_vec == 1)[0]
            priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec if i in prev_ON_idx_vec])
            demands_vec = self.episode_forecast
            for idx in priority_idx_vec:
                max_timestep = min(self.episode_timestep + self.t_min_down[idx], self.episode_length - 1)
                max_cap = np.sum(action_vec * self.p_max_vec)
                if np.any(max_cap < demands_vec[self.episode_timestep: max_timestep]):
                    action_vec[idx] = 1
        return action_vec

    def _adjust_low_capacity(self, demand: float, action_vec: np.ndarray):
        low_action_vec = action_vec.copy()
        already_OFF_idx_vec = np.where(action_vec == 0)[0]
        must_not_OFF_idx_vec = np.where(self.must_off == False)[0]
        can_ON_idx_vec = np.intersect1d(already_OFF_idx_vec, must_not_OFF_idx_vec)
        if len(can_ON_idx_vec) > 0:
            priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec if i in can_ON_idx_vec])
            remaining_supply = demand - np.sum(action_vec * self.p_max_vec)
            for idx in priority_idx_vec:
                action_vec[idx] = 1
                remaining_supply = remaining_supply - self.p_max_vec[idx]
                if remaining_supply <= 0.0001:
                    break
        return action_vec

    def _adjust_excess_capacity(self, demand: float, action_vec: np.ndarray):
        excess_action_vec = action_vec.copy()
        already_ON_idx_vec = np.where(action_vec == 1)[0]
        must_not_ON_idx_vec = np.where(self.must_on == False)[0]
        can_OFF_idx_vec = np.intersect1d(already_ON_idx_vec, must_not_ON_idx_vec)
        if len(can_OFF_idx_vec) > 0:
            OFF_priority_idx_vec = np.array([i for i in self.ON_priority_idx_vec[::-1] if i in can_OFF_idx_vec])
            excess_supply = np.sum(action_vec * self.p_min_vec) - demand
            for idx in OFF_priority_idx_vec:
                action_vec[idx] = 0
                excess_supply -= self.p_min_vec[idx]
                if excess_supply <= 0.0001:
                    if np.sum(action_vec * self.p_max_vec) < demand:
                        action_vec[idx] = 1
                        break
                    break
        return action_vec

    ### 7.*******ED*********
    def economic_dispatch(self, action, demand, lambda_lo, lambda_hi):
        # epoch = i
        idx = np.where(np.array(action) == 1)[0]
        on_a = self.a[idx]
        on_b = self.b[idx]
        on_min = self.min_output[idx]
        on_max = self.max_output[idx]
        power = self.power[idx]
        disp = np.zeros(self.num_gen)
        rampup = self.RampUp[idx]
        rampdown = self.RampDown[idx]
        Penalty = 0
        if np.sum(on_max) < demand:
            econ = on_max
        elif np.sum(on_min) > demand:
            econ = on_min
        else:
            econ = ipopt_solve(demand, on_a, on_b, on_min, on_max, rampup, rampdown, power)

        cons = self.check_constraints(econ, power, rampup, rampdown, on_min, on_max)
        correct = self.correct_action(action, econ, on_min, idx)
        n_g = len(on_a)
        for g in range(n_g):

            if econ[g] - power[g] > rampup[g] + 0.1:
                Penalty += 100
            if econ[g] - power[g] < -rampdown[g] - 0.1:
                Penalty += 100

        for g in range(n_g):
            if econ[g] < on_min[g] - 1:
                Penalty += 500
            if econ[g] > on_max[g] + 1:
                Penalty += 500
        disp[idx] = econ

        self.power = disp
        on_cost = n_g * 0

        return disp, Penalty, cons, correct, on_cost

    ### 8.*******other********
    def onoff(self, unit):
        num_gen = len(unit)
        for i in range(num_gen):
            if unit[i] < 1:
                unit[i] = 0
            else:
                unit[i] = 1
        return unit

    def check_constraints(self, econ, power, rampup, rampdown, min_on, max_on):
        num_gen = len(econ)
        cons = 0
        cons_demand = 0
        for g in range(num_gen):
            if power[g] > 0.1 and econ[g] > 0.1:
                if econ[g] - power[g] > rampup[g] + 0.1:
                    cons += 1
                if econ[g] - power[g] < -rampdown[g] - 0.1:
                    cons += 1
                if econ[g] > max_on[g]:
                    cons_demand += 1
                if econ[g] < min_on[g]:
                    cons_demand += 1
        return cons

    def correct_action(self, action, econ, on_min, idx):
        action1 = np.array(action)
        num_gen = len(econ)
        for g in range(num_gen):
            if abs(econ[g] - on_min[g]) < 2:
                idx1 = idx[g]
                action1[idx1] = 0
        return action1

    def _transition(self, action):
        action_vec = action
        self.net_demand = self._get_net_demand()
        state = self.get_current_state()
        self.action_vec = self.ensure_action_legitimacy(self.net_demand, action_vec)
        self.action_vec = np.array(self.action_vec)
        self.fuel_cost, self.disp, self.Penalty, self.cons, correct, self.on_cost = self.calculate_fuel_cost_and_dispatch(
            self.net_demand, self.action_vec)
        self.ens_cost, self.ens_amount = self.calculate_lost_load_cost(self.net_demand, self.disp)
        self.day_cost += self.fuel_cost
        # Assign state
        state = self.get_next_state(self.action_vec)
        return state, correct

    def step(self, action):
        obs, correct = self._transition(action)
        reward = self._get_reward()
        done = self.is_terminal()
        return obs, reward, done

    def is_terminal(self):
        if self.mode == "train":
            # return (self.episode_timestep == (self.episode_length - 1)) or self.ens
            a = self.episode_timestep == (self.episode_length - 1)
            return a
        else:
            return self.episode_timestep == (self.episode_length - 1)
    def sample_day(self):

        day = np.random.choice(self.profiles_df.date, 1)
        day_profile = self.profiles_df[self.profiles_df.date == day.item()]
        formatted_dates = pd.to_datetime(day).strftime('%Y-%m-%d')
        day1 = np.array(formatted_dates)
        day0 = day[0]
        date_obj = datetime.strptime(day0, '%m/%d/%Y')
        day2 = f"{date_obj.month}/{date_obj.day}/{date_obj.year}"
        day2 = np.array(day2)
        day_load = self.busload_df[self.busload_df.date == day2.item()]
        row_number = self.profiles_df.index[self.profiles_df.date == day.item()].tolist()
        row_number = row_number[0]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        init_power = init_info.gen_initstate(row_number, day, device)
        return day, day_profile, init_power, day_load

    def _update_production_capacities(self, action_vec: np.ndarray):
        p_min_vec = self.min_output
        p_max_vec = self.max_output
        self.loads_vec = self.disp
        self.p_min_vec = np.maximum(p_min_vec, action_vec * (self.loads_vec - self.RampDown))
        self.p_max_vec = np.minimum(p_max_vec, action_vec * (self.loads_vec + self.RampUp)
                                    + np.where(action_vec == 0, 1, 0) * p_max_vec)
        self.p_max_vec = np.maximum(self.p_max_vec, self.p_min_vec + 0.1)
        if np.any(self.p_min_vec > self.p_max_vec):
            raise Exception("Min capacity > Max capacity.")

    def _update_production_capacities_init(self, action_vec: np.ndarray, loads_vec):
        p_min_vec = self.min_output
        p_max_vec = self.max_output
        self.p_min_vec = np.maximum(p_min_vec, action_vec * (loads_vec - self.RampDown))
        self.p_max_vec = np.minimum(p_max_vec, action_vec * (loads_vec + self.RampUp)
                                    + np.where(action_vec == 0, 1, 0) * p_max_vec)
        self.p_max_vec = np.maximum(self.p_max_vec, self.p_min_vec + 0.1)
        if np.any(self.p_min_vec > self.p_max_vec):
            raise Exception("Min capacity > Max capacity.")

    def reset(self):

        if self.mode == 'train':
            # Choose random day
            day, day_profile, init_power, day_load = self.sample_day()
            self.day = day
            self.episode_forecast = day_profile.demand.values
            self.day_load = day_load.iloc[:, 2:].values
        else:
            self.episode_forecast = self.profiles_df.demand.values
        self.episode_timestep = -1
        self.forecast = None
        self.net_demand = None
        self.day_cost = 0
        self.fuel_cost = 0
        # Initalise grid status and constraints
        if self.mode == "train":

            self.status = self.gen_info['status'].to_numpy()
        else:
            self.status = self.gen_info['status'].to_numpy()
        self.expected_cost = 0
        self.ens = False
        self.ens_amount = 0
        self.cons = 0
        self.load = self.day_load[self.episode_timestep + 1, :]
        self.status, action_init = check_status(init_power, self.status)
        action_init_vec = np.array(action_init)
        self.commits_vec = np.where(self.status > 0, 1, 0)
        self._determine_constraints()
        self.power = check_power(init_power)
        self._update_production_capacities_init(action_init_vec, self.power)
        state = self._get_state()
        return state

def check_status(power, status):
    num = len(power)
    action = [0] * num
    # status = [0] * num
    for i in range(num):
        if power[i] > 5:
            status[i] = 1
            action[i] = 1
        elif power[i] < 1:
            status[i] = -10
        else:
            status[i] = 0
    return status, action

def check_power(power):
    num = len(power)
    for i in range(num):
        if power[i] < 0:
            power[i] = 0
    return power


def create_gen_info(num_gen, dispatch_freq_mins):

    MIN_GENS = 5
    if num_gen < 5:
        raise ValueError("num_gen should be at least {}".format(MIN_GENS))
    script_dir = os.path.dirname(os.path.realpath(__file__))
    gen6 = pd.read_csv(os.path.join(script_dir, Gen_units_path))
    if num_gen == 5:
        gen_info = gen6[::2]
    else:
        upper_limit = int(np.floor(num_gen / 6) + 1)
        gen_info = pd.concat([gen6] * upper_limit)[:num_gen]
    gen_info = gen_info.sort_index()
    gen_info.reset_index()

    gen_info.t_min_up = gen_info.t_min_up
    gen_info.t_min_down = gen_info.t_min_down
    gen_info.status = gen_info.status
    gen_info = gen_info.astype({'t_min_down': 'int64',
                                't_min_up': 'int64',
                                'status': 'int64'})

    return gen_info


class UnitCommitmentEnv(gym.Env):
    def __init__(self, **kwargs):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        env_fn = os.path.join(script_dir, 'data/envs/6gen.json')
        params = json.load(open(env_fn))
        gen_info = create_gen_info(params.get('num_gen', DEFAULT_NUM_GEN),
                                   params.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))
        profiles_df = pd.read_csv(os.path.join(script_dir, ParameterData_train_demand))
        busload_df = pd.read_csv(os.path.join(script_dir, busload))
        # print(profiles_df)
        self.env = Env(gen_info=gen_info, profiles_df=profiles_df, busload_df=busload_df, mode='train',
                       **kwargs)

        # 定义观测空间和动作空间
        self.observation_space = spaces.Dict({
            'status': spaces.Box(low=-(np.ones(self.env.num_gen) * 10000),
                                 high=np.ones(self.env.num_gen) * 10000,
                                 dtype=np.int64),
            'load': spaces.Box(low=-(np.ones(30) * 1000),
                               high=np.ones(30) * 1000,
                               dtype=np.int64),
            'demand_forecast': spaces.Box(low=0, high=np.inf, shape=(self.env.episode_length,), dtype=np.float32),
            'cost': spaces.Box(low=0, high=np.inf, shape=(self.env.num_gen,), dtype=np.float32),
            'timestep': spaces.Discrete(self.env.episode_length),
            'power': spaces.Box(low=np.zeros(self.env.num_gen), high=np.ones(self.env.num_gen) * 1000,
                                dtype=np.float32),
            'day_cost': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            'day': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),
            'ens': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
            'cons': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64)
        })
        self.action_space = spaces.Box(low=0, high=1, shape=(self.env.num_gen,), dtype=np.int8)

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        obs, reward, done = self.env.step(action)
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass

    def close(self):
        pass
