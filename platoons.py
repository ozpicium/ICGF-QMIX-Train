# -*- coding: utf-8 -*-
"""
Last updated time: 2021/09/22 11:50 AM
Author: Geng, Minghong
File: platoons.py
IDE: PyCharm
"""
import torch
import numpy as np
from policy.q_decom import Q_Decom
from agents import Agents
from common.replay_buffer import ReplayBuffer

'''
This file will create Platoons class that contains agents, as we want to do hierarchical control.
The platoon will be the 2nd level control, where the 1st layer will be the agents. 
'''


class Platoon:
    def __init__(self, args, evaluate=False, platoon_id=None, itr=1):
        self.args = args
        self.platoon_id = platoon_id

        # set the Hierarchical MARL algorithm for the platoon
        q_decom_policy = ['qmix', 'vdn', 'cwqmix', 'owqmix']
        if args.alg in q_decom_policy: # define the policy
            self.policy = Q_Decom(args, itr)
        else:
            raise Exception("Selected algorithm doesn't exist.")

        # record the replay buffer
        self.replay_buffer = ReplayBuffer(self.args)

        # initialize the last action taken by the team
        self.last_action = np.zeros((self.args.n_ally_agent_in_platoon, self.args.n_actions))

        # evaluate or train
        self.evaluate = evaluate

        # epsilon decay
        # self.epsilon = 0 if evaluate else self.args.epsilon
        self.epsilon = 0 if evaluate else self.args.epsilon

        # init episode buffer
        self.episode_buffer = None

    def epsilon_decay(self):
        self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon > self.args.min_epsilon else self.epsilon
        return

    def init_last_action(self):
        self.last_action = np.zeros((self.args.n_ally_agent_in_platoon, self.args.n_actions))
        return

    def choose_action(self, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False):
        # 可供选择的动作
        avail_actions = np.nonzero(avail_actions_mask)[0]
        # agent索引转为独热编码
        # onehot_agent_idx = np.zeros(self.args.n_agents)
        onehot_agent_idx = np.zeros(self.args.n_ally_agent_in_platoon)
        onehot_agent_idx[agent_idx] = 1.
        if self.args.last_action:
            # 在水平方向上平铺
            obs = np.hstack((obs, last_action))
        if self.args.reuse_network:
            obs = np.hstack((obs, onehot_agent_idx))
        hidden_state = self.policy.eval_hidden[:, agent_idx, :]
        # 转置
        obs = torch.Tensor(obs).unsqueeze(0)
        avail_actions_mask = torch.Tensor(avail_actions_mask).unsqueeze(0)
        # 是否使用 GPU
        if self.args.cuda:
            obs = obs.cuda()
            hidden_state = hidden_state.cuda()
        # 获取 Q(s, a)
        qsa, self.policy.eval_hidden[:, agent_idx, :] = self.policy.eval_rnn(obs, hidden_state)
        # 不可选的动作 q 值设为无穷小
        qsa[avail_actions_mask == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            return np.random.choice(avail_actions)
        return torch.argmax(qsa)

    def get_platoon_observation(self):
        return

    def get_platoon_action(self):
        return

    def get_max_episode_len(self, batch):
        max_len = 0
        for episode in batch['padded']:
            length = episode.shape[0] - int(episode.sum())
            if length > max_len:
                max_len = length
        return int(max_len)

    def train(self, batch, train_step, epsilon=None):
        max_episode_len = self.get_max_episode_len(batch)
        for key in batch.keys(): # 减少batch的实际大小
            batch[key] = batch[key][:, :max_episode_len]
        # self.policy.learn(batch, max_episode_len, train_step, epsilon) # origin line
        self.policy.learn(batch, max_episode_len, train_step)
        if train_step > 0 and train_step % self.args.save_model_period == 0:
            self.policy.save_model(train_step, epsilon)


