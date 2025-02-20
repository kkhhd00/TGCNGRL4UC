import csv
import os
import random
import statistics
from math import nan

import torch
import torch.nn.functional as F
import numpy as np
import gym
import rl_utils
from tqdm import tqdm
import __init__
import matplotlib.pyplot as plt
import pandas as pd

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED environment variable for reproducibility

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim * 2)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 2)
        x = F.softmax(x, dim=1)
        return x

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPO:

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # print('probs', probs)
        action_dist = torch.distributions.Categorical(probs)
        # print('action_dist', action_dist)
        action = action_dist.sample()
        # print('action', action)
        return action.cpu().numpy()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).to(self.device)
        rewards = rewards.reshape(-1, 1)
        # print(rewards.shape)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).to(self.device)
        dones1 = 1 - dones
        dones1 = dones1.view(-1, 1)
        td_target = rewards + self.gamma * self.critic(next_states) * dones1
        cri = self.critic(next_states)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        epsilon = 1e-8
        old_log_probs = torch.log(self.actor(states).gather(1, actions) + epsilon).detach()
        old_log_probs = old_log_probs.reshape(288, 6)
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions) + epsilon)
            log_probs = log_probs.reshape(288, 6)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            sum = -torch.mean(torch.min(surr1, surr2), dim=1)
            actor_loss = torch.mean(sum)
            if actor_loss == nan:
                print(actor_loss)
            print('actor_loss', actor_loss)
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_models(self, path='ppo_checkpoint.pth'):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_models(self, path='ppo_checkpoint.pth'):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")

def get_all_state(state):
    day1 = state['day']
    day_info = {key: state[key] for key in ['day', 'status', 'power', 'ens', 'cons', 'day_cost', 'timestep'] if
                key in state}
    # print(day_info)
    insert_into_csv(day_info)

def insert_into_csv(data):
    with open('day_cost.csv', 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        csvfile.seek(0, 2)
        if csvfile.tell() == 0:
            csv_writer.writeheader()
        csv_writer.writerow(data)

###training
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
actor_lr = 1e-3
critic_lr = 1e-2
hidden_dim = 64
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
num_episodes = 600
env_name = "UnitCommitmentEnv-v0"
env = gym.make(env_name)
env.reset()
status_space = env.observation_space.spaces['status']
state_dim = 30
action_dim = env.action_space.shape[0]
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)
return_list = []
cost_list = []
date_list = []

def process_observation(state):
    obs_new_load = state['load']
    # obs_new_status = state['status']
    # obs_new=np.concatenate((obs_new_load, obs_new_status), axis=0)
    return obs_new_load

for i in tqdm(range(num_episodes), desc='Training Progress'):
    episode_return = 0
    episode_return1 = 0
    state = env.reset()
    obs = process_observation(state)
    done = False
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': []
    }
    while not done:
        obs = np.array(obs).astype(np.float32)
        action = agent.take_action(obs)
        next_state, reward, done, _ = env.step(action)
        get_all_state(next_state)
        if next_state['timestep'] == 287:
            cost_list.append(next_state['day_cost'])
            date_list.append(next_state['day'])
            print(next_state['day_cost'])
        next_obs = process_observation(next_state)
        transition_dict['states'].append(obs)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_obs)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        obs = next_obs
        episode_return += reward
    return_list.append(episode_return)
    agent.update(transition_dict)

agent.save_models('ppo_mlp.pth')
print('return', return_list)
print('cost', cost_list)
cost_list1 = cost_list[500:]
print('average cost', statistics.fmean(cost_list1))
with open('return & cost.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['day', 'return', 'cost'])
    for item1, item2, item3 in zip(date_list, return_list, cost_list):
        writer.writerow([item1, item2, item3])
fig, ax = plt.subplots()
ax.plot(pd.Series(return_list).rolling(10).mean())
ax.set_title('PPO Performance Comparison on UC')
ax.set_xlabel('Episode')
ax.set_ylabel('Average Reward')
plt.tight_layout()
plt.show()
