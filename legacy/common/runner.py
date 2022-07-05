# -*- coding: utf-8 -*-
"""
@Time ： 2020/08/07 10:52
@Auth ： Kunfeng Li
@File ：runner.py
@IDE ：PyCharm

"""
import os
import numpy as np
from common.replay_buffer import ReplayBuffer

# TODO: Check how does the Agents work in this file
from agents import Agents

# TODO: Create platoons and company, send action to env, and get feedback
from company import Company

import time
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch


class Runner:
    def __init__(self, env, args, itr):
        # retrieve the parameters
        # self.args = get_common_args()
        self.args = args

        # get the environment
        self.env = env

        # process ID
        self.pid = itr

        # create the company commander
        self.company = Company(args, company_id=1, itr=itr)

        # create agents
        self.agents = Agents(args, itr=itr)  # create one agent instance, looks like all agents are the same one?

        # self.replay_buffer = ReplayBuffer(self.args)

        '''
        这里，episode_reward 代表一个episode的累加奖赏，
        episodes_reward代表多个episode的累加奖赏，
        episodes_rewards代表多次评价的多个episode的累加奖赏
        '''

        # load previous training result
        if not self.args.load_result:
            self.episodes_rewards = []
            self.evaluate_itr = []
            self.win_rates = []
            self.max_win_rate = 0
        else:
            self.load_result_dir = self.args.result_dir + '/' + args.alg + '/' + args.map + '/' + str(itr)

            episodes_rewards_temp = np.load(self.load_result_dir + '/' + 'episodes_rewards.npy', encoding="latin1",
                                            allow_pickle=True).tolist()
            self.episodes_rewards = np.zeros(
                [len(episodes_rewards_temp), len(max(episodes_rewards_temp, key=lambda x: len(x)))])
            for i, j in enumerate(episodes_rewards_temp):
                self.episodes_rewards[i][0:len(j)] = j
                self.episodes_rewards[i][len(j):] = np.mean(j)
            self.episodes_rewards = self.episodes_rewards.tolist()

            self.evaluate_itr = np.load(self.load_result_dir + '/' + 'evaluate_itr.npy', encoding="latin1",
                                        allow_pickle=True).tolist()
            self.win_rates = np.load(self.load_result_dir + '/' + 'win_rates.npy', encoding="latin1",
                                     allow_pickle=True).tolist()
            self.n_trained_episodes = max(self.evaluate_itr)
            self.max_win_rate = max(self.win_rates)

        # 保存结果和模型的位置，增加计数，帮助一次运行多个实例
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.save_path = self.args.result_dir + '/' + self.args.alg + '/' + self.args.map + '/' + str(
            itr) + '/' + start_time
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        print('runner initialized')

    def generate_episode(self, episode_num, evaluate=False):  # in run(), pass in episode_num == 0
        # 为保存评价的回放做准备
        # cannot understand this line
        if self.args.replay_dir != '' and evaluate and episode_num == 0:
            self.env.close()

        self.env.reset()  # Initialized the environment | 变量初始化
        done = False
        info = None
        win = False

        # Initialize the last actions, length:  non-attack action (6), number of enemies (depends)
        if self.args.hierarchical:
            # TODO: Check is this function is useful
            # self.company.init_last_action()
            last_action = np.zeros((self.args.n_ally_platoons, self.args.n_ally_agent_in_platoon, self.args.n_actions))
        else:  # no platoon nor company
            # last_action = [np.zeros((self.args.n_ally_agent_in_platoon, self.args.n_actions))]
            last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        epsilon = 0 if evaluate else self.args.epsilon  # epsilon decay

        # method of epsilon decay
        if self.args.epsilon_anneal_scale == 'episode' or \
                (self.args.epsilon_anneal_scale == 'itr' and episode_num == 0):
            # epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon
            self.company.epsilon_decay()

        if self.args.hierarchical:  # initialize episode buffer
            episode_buffer_company = [[] for _ in range(self.args.n_ally_platoons)]
            if not evaluate:
                for platoon_id in range(self.args.n_ally_platoons):
                    episode_buffer_company[platoon_id] = \
                        {'o': np.zeros(
                            [self.args.episode_limit, self.args.n_ally_agent_in_platoon, self.args.obs_shape]),
                            's': np.zeros(
                                [self.args.episode_limit, self.args.state_shape]),
                            'a': np.zeros(
                                [self.args.episode_limit, self.args.n_ally_agent_in_platoon, 1]),
                            'onehot_a': np.zeros(
                                [self.args.episode_limit, self.args.n_ally_agent_in_platoon, self.args.n_actions]),
                            'avail_a': np.zeros(
                                [self.args.episode_limit, self.args.n_ally_agent_in_platoon, self.args.n_actions]),
                            'r': np.zeros([self.args.episode_limit, 1]),
                            'next_o': np.zeros(
                                [self.args.episode_limit, self.args.n_ally_agent_in_platoon, self.args.obs_shape]),
                            'next_s': np.zeros([self.args.episode_limit, self.args.state_shape]),
                            'next_avail_a': np.zeros(
                                [self.args.episode_limit, self.args.n_ally_agent_in_platoon, self.args.n_actions]),
                            'done': np.ones([self.args.episode_limit, 1]),
                            'padded': np.ones([self.args.episode_limit, 1])
                        }
        else:
            # self.company.init_episode_buffer()
            episode_buffer = None
            if not evaluate:
                episode_buffer = {'o': np.zeros([self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                                  's': np.zeros([self.args.episode_limit, self.args.state_shape]),
                                  'a': np.zeros([self.args.episode_limit, self.args.n_agents, 1]),
                                  'onehot_a': np.zeros(
                                      [self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                                  'avail_a': np.zeros(
                                      [self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                                  'r': np.zeros([self.args.episode_limit, 1]),
                                  'next_o': np.zeros(
                                      [self.args.episode_limit, self.args.n_agents, self.args.obs_shape]),
                                  'next_s': np.zeros([self.args.episode_limit, self.args.state_shape]),
                                  'next_avail_a': np.zeros(
                                      [self.args.episode_limit, self.args.n_agents, self.args.n_actions]),
                                  'done': np.ones([self.args.episode_limit, 1]),
                                  'padded': np.ones([self.args.episode_limit, 1])
                                  }

        if self.args.hierarchical:  # generate the episodes
            obs_company = self.env.get_obs_company()
            state_company = self.env.get_state_company()
        else:
            obs = self.env.get_obs()
            state = self.env.get_state()

        # obs_L3 = self.env.get_obs_L3()  # TODO: 2021-10-10: Observation of the Level 3 commander

        '''
        Get the available actions for each unit in the company.
        '''
        avail_actions = []  # select the actions

        # TODO: Check how to init the policy. Now the policy is initialized every episode.
        # self.agents.policy.init_hidden(1)
        if self.args.hierarchical:
            self.company.init_policy()

            company_avail_actions = [[] for _ in range(self.args.n_ally_platoons)]  # get the available actions
            for platoon_id in range(self.args.n_ally_platoons):
                for agent_id in range(self.args.n_ally_agent_in_platoon):
                    agent_avail_action = self.env.get_avail_platoon_agent_actions(agent_id, platoon_id)
                    company_avail_actions[platoon_id].append(agent_avail_action)

            # self.company.get_avail_actions()

        episode_reward = 0
        for step in range(self.args.episode_limit):
            if done:
                break
            else:
                actions = [[] for _ in range(self.args.n_ally_platoons)]
                onehot_actions = [[] for _ in range(self.args.n_ally_platoons)]
                # -----------------------------------------------------------------------
                # for agent_id in range(self.args.n_agents):
                #     # avail_action = self.env.get_avail_agent_actions(agent_id)
                #     action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                #                                        avail_actions[agent_id], epsilon, evaluate)
                #     onehot_action = np.zeros(self.args.n_actions)
                #     onehot_action[action] = 1
                #     onehot_actions.append(onehot_action)
                #     # 加入联合动作
                #     actions.append(action)
                #     # avail_actions.append(avail_action)
                #     # 记录该动作
                #     last_action[agent_id] = onehot_action
                #

                # -----------------------------------------------------------------------
                for platoon_id in range(self.args.n_ally_platoons):  # for each platoon
                    if self.args.FALCON_demo:
                        if self.env.next_movement_platoon == platoon_id:
                            for agent_id in range(self.args.n_ally_agent_in_platoon):  # for each agent
                                action = self.company.platoons[platoon_id].choose_action(  # get the action with policy
                                    obs_company[platoon_id][agent_id],
                                    last_action[platoon_id][agent_id],
                                    agent_id,
                                    company_avail_actions[platoon_id][agent_id],
                                    epsilon,  # TODO: currently using global epsilon
                                    evaluate)

                                # one-hot action
                                onehot_action = np.zeros(self.args.n_actions)
                                onehot_action[action] = 1
                                onehot_actions[platoon_id].append(onehot_action)
                                actions[platoon_id].append(action)  # add the selected action into platoon actions
                                last_action[platoon_id][agent_id] = onehot_action  # record the last action
                        else:
                            for agent_id in range(self.args.n_ally_agent_in_platoon):
                                if company_avail_actions[platoon_id][agent_id][1] == 1:
                                    action = 1
                                if company_avail_actions[platoon_id][agent_id][0] == 1:
                                    action = 0

                                # one-hot action
                                onehot_action = np.zeros(self.args.n_actions)
                                onehot_action[action] = 1
                                onehot_actions[platoon_id].append(onehot_action)
                                actions[platoon_id].append(action)  # add the selected action into platoon actions
                                last_action[platoon_id][agent_id] = onehot_action  # record the last action
                    else:
                        for agent_id in range(self.args.n_ally_agent_in_platoon):  # for each agent
                            action = self.company.platoons[platoon_id].choose_action(  # get the action with policy
                                obs_company[platoon_id][agent_id],
                                last_action[platoon_id][agent_id],
                                agent_id,
                                company_avail_actions[platoon_id][agent_id],
                                epsilon,
                                evaluate)

                            # one-hot action
                            onehot_action = np.zeros(self.args.n_actions)
                            onehot_action[action] = 1
                            onehot_actions[platoon_id].append(onehot_action)
                            actions[platoon_id].append(action)  # add the selected action into platoon actions
                            last_action[platoon_id][agent_id] = onehot_action  # record the last action
                if self.args.hierarchical:
                    reward, done, info = self.env.step_company(actions)  # perform the selected actions
                else:
                    reward, done, info = self.env.step(actions)

                if not done:  # get the information of the changed environment
                    if self.args.hierarchical:
                        next_obs_company = self.env.get_obs_company()
                        next_state_company = self.env.get_state_company()
                    else:
                        next_obs = self.env.get_obs()
                        next_state = self.env.get_state()
                else:
                    if self.args.hierarchical:
                        next_obs_company = obs_company
                        next_state = state_company
                    else:
                        next_obs = obs
                        next_state = state

                # update the available actions
                if self.args.hierarchical:
                    next_avail_actions_company = [[] for _ in range(self.args.n_ally_platoons)]
                    for platoon_id in range(self.args.n_ally_platoons):
                        for agent_id in range(self.args.n_ally_agent_in_platoon):
                            agent_avail_action = self.env.get_avail_platoon_agent_actions(agent_id, platoon_id)
                            next_avail_actions_company[platoon_id].append(agent_avail_action)
                else:
                    next_avail_actions = []
                    for agent_id in range(self.args.n_agents):
                        avail_action = self.env.get_avail_agent_actions(agent_id)
                        next_avail_actions.append(avail_action)

                if self.args.hierarchical:
                    for p_id in range(self.args.n_ally_platoons):
                        if not evaluate:
                            episode_buffer_company[p_id]['o'][step] = obs_company[p_id]
                            episode_buffer_company[p_id]['s'][step] = state_company
                            episode_buffer_company[p_id]['a'][step] = torch.reshape(torch.Tensor(actions[p_id]),(self.args.n_ally_agent_in_platoon, 1))
                            episode_buffer_company[p_id]['onehot_a'][step] = onehot_actions[p_id]
                            episode_buffer_company[p_id]['avail_a'][step] = company_avail_actions[p_id]
                            # TODO the reward for each platoon should be seperated
                            episode_buffer_company[p_id]['r'][step] = reward[p_id]
                            episode_buffer_company[p_id]['next_o'][step] = next_obs_company[p_id]
                            episode_buffer_company[p_id]['next_s'][step] = next_state_company
                            episode_buffer_company[p_id]['next_avail_a'][step] = next_avail_actions_company[p_id]
                            episode_buffer_company[p_id]['done'][step] = [done]
                            episode_buffer_company[p_id]['padded'][step] = [0.]
                else:
                    if not evaluate:
                        episode_buffer['o'][step] = obs
                        episode_buffer['s'][step] = state
                        episode_buffer['a'][step] = np.reshape(actions, [self.args.n_agents, 1])
                        episode_buffer['onehot_a'][step] = onehot_actions
                        episode_buffer['avail_a'][step] = avail_actions
                        episode_buffer['r'][step] = [reward]
                        episode_buffer['next_o'][step] = next_obs
                        episode_buffer['next_s'][step] = next_state
                        episode_buffer['next_avail_a'][step] = next_avail_actions
                        episode_buffer['done'][step] = [done]
                        episode_buffer['padded'][step] = [0.]

                # 更新变量
                episode_reward += sum(reward)

                if self.args.hierarchical:
                    obs_company = next_obs_company
                    state_company = next_state_company

                    company_avail_actions = next_avail_actions_company
                else:
                    obs = next_obs
                    state = next_state

                    avail_actions = next_avail_actions

                # Update epsilon
                if self.args.epsilon_anneal_scale == 'step':
                    epsilon = epsilon - self.args.epsilon_decay if epsilon > self.args.min_epsilon else epsilon

        if not evaluate:  # If it is training, update the epsilon
            self.args.epsilon = epsilon

        # TODO: modify here later
        # 获取对局信息
        if info.__contains__('battle_won'):
            win = True if done and info['battle_won'] else False
        if evaluate and episode_num == self.args.evaluate_num - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        # return episode_buffer, episode_reward, win
        return episode_buffer_company, episode_reward, win

    def run(self):
        train_steps = 0
        num_eval = 0
        self.max_win_rate = 0

        for itr in range(self.args.n_itr):
            print('SHUBHAM~~~~~~~~~~~~~~~~~~~~~~ episode gen itr ', itr)
            if self.args.hierarchical:
                episode_batch_company, _, _ = self.generate_episode(0)
                for episode_batch_platoon in episode_batch_company:
                    for key in episode_batch_platoon.keys():
                        episode_batch_platoon[key] = np.array([episode_batch_platoon[key]])  # transform the shape

                # If there are multiple episodes in a training process, run the rest episodes.
                for e in range(1, self.args.n_episodes):
                    episode_company, _, _ = self.generate_episode(e)
                    for episode_platoon in episode_company:
                        for key in episode_platoon.keys():
                            episode_platoon[key] = np.array([episode_platoon[key]])
                            episode_batch_platoon[key] = np.concatenate(
                                (episode_batch_platoon[key], episode_platoon[key]),
                                axis=0)

                for p_id in range(self.args.n_ally_platoons):  # store the buffer for each platoon
                    self.company.platoons[p_id].replay_buffer.store(episode_batch_company[p_id])
                # self.replay_buffer.store(episode_batch_company)

                # train
                if self.company.platoons[0].replay_buffer.size < self.args.batch_size * 12.5:
                    continue
                # TODO: [2021-11-16] disable the learning in hierarchical control
                for platoon in self.company.platoons:
                    for _ in range(self.args.train_steps):
                        # 从replay buffer(最多5000个记录)中抽取32个episode的经验
                        # sample the experience from the replay buffer
                        batch_platoon = platoon.replay_buffer.sample(self.args.batch_size)
                        platoon.train(batch_platoon, train_steps, self.args.epsilon)
                        train_steps += 1
            else:
                episode_batch, _, _ = self.generate_episode(0)
                for key in episode_batch.keys():
                    episode_batch[key] = np.array([episode_batch[key]])
                for e in range(1, self.args.n_episodes):
                    episode, _, _ = self.generate_episode(e)
                    for key in episode_batch.keys():
                        episode[key] = np.array([episode[key]])
                        episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

                self.replay_buffer.store(episode_batch)  # save the knowledge in the last 5000 episodes

                # train
                if self.replay_buffer.size < self.args.batch_size * 12.5:
                    print('replay buffer is smaller than batch size')
                    continue

                for _ in range(self.args.train_steps):
                    batch = self.replay_buffer.sample(self.args.batch_size)  # from buffer (5k) extract 32 episodes
                    self.agents.train(batch, train_steps, self.args.epsilon)
                    train_steps += 1

            if itr % self.args.evaluation_period == 0:  # 周期性评价
                num_eval += 1
                print(f'Process {self.pid}: {itr} / {self.args.n_itr}')
                win_rate, episodes_reward = self.evaluate()

                if not self.args.load_result:  # 保存测试结果
                    self.evaluate_itr.append(itr)
                else:
                    self.evaluate_itr.append(itr + self.n_trained_episodes)
                self.win_rates.append(win_rate)
                self.episodes_rewards.append(episodes_reward)

                # 表现好的模型要额外保存
                if win_rate > self.max_win_rate:
                    self.max_win_rate = win_rate
                    self.agents.policy.save_model(str(win_rate), self.args.epsilon)
                # 不时刻保存，从而减少时间花费
                if num_eval % 1 == 0:
                    self.save_results()
                    self.plot()
        # 最后把所有的都保存一下
        self.save_results()
        self.plot()
        self.env.close()

    def evaluate(self):
        """
        得到平均胜率和每次测试的累加奖赏，方便画误差阴影图
        :return:
        """
        win_number = 0
        episodes_reward = []
        for itr in range(self.args.evaluate_num):  # default is 32
            _, episode_reward, win = self.generate_episode(itr, evaluate=True)
            episodes_reward.append(episode_reward)
            if win:
                win_number += 1
        return win_number / self.args.evaluate_num, episodes_reward

    def save_results(self):
        """
        保存数据，方便后面多种算法结果画在一张图里比较
        :return:
        """
        # 如果已经有图片就删掉
        for filename in os.listdir(self.save_path):
            if filename.endswith('.npy'):
                os.remove(self.save_path + '/' + filename)
        np.save(self.save_path + '/evaluate_itr.npy', self.evaluate_itr)
        np.save(self.save_path + '/win_rates.npy', self.win_rates)
        np.save(self.save_path + '/episodes_rewards.npy', self.episodes_rewards)

    def plot(self):
        """
        定期绘图
        :return:
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        win_x = np.array(self.evaluate_itr)[:, None]
        win_y = np.array(self.win_rates)[:, None]
        plot_win = pd.DataFrame(np.concatenate((win_x, win_y), axis=1), columns=['evaluate_itr', 'win_rates'])
        sns.lineplot(x="evaluate_itr", y="win_rates", data=plot_win, ax=ax1)

        ax2 = fig.add_subplot(212)
        reward_x = np.repeat(self.evaluate_itr, self.args.evaluate_num)[:, None]
        reward_y = np.array(self.episodes_rewards).flatten()[:, None]
        plot_reward = pd.DataFrame(np.concatenate((reward_x, reward_y), axis=1),
                                   columns=['evaluate_itr', 'episodes_rewards'])
        sns.lineplot(x="evaluate_itr", y="episodes_rewards", data=plot_reward, ax=ax2,
                     ci=68, estimator=np.median)

        # 格式化成2016-03-20-11_45_39形式
        tag = self.args.alg + '-' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # 如果已经有图片就删掉
        for filename in os.listdir(self.save_path):
            if filename.endswith('.png'):
                os.remove(self.save_path + '/' + filename)
        fig.savefig(self.save_path + "/%s.png" % tag)
        plt.close()

        # plt.figure()
        # plt.cla()
        # # 平均胜率图
        # plt.subplot(2, 1, 1)
        #
        # plt.plot(self.evaluate_itr, self.win_rates, c='blue')
        #
        # plt.xlabel('itr')
        # plt.ylabel('win_rates')
        # # 累加奖赏误差阴影曲线图
        # plt.subplot(2, 1, 2)
        # avg_rewards = np.array(self.episodes_rewards).mean(axis=1)
        #
        # plt.plot(self.evaluate_itr, avg_rewards, c='blue')
        #
        # up_error = []
        # down_error = []
        # for i in range(len(self.evaluate_itr)):
        #     up_bound = np.max(self.episodes_rewards[i])
        #     down_bound = np.min(self.episodes_rewards[i])
        #     up_error.append(up_bound - avg_rewards[i])
        #     down_error.append(avg_rewards[i] - down_bound)
        # # avg_rewards = np.array(avg_rewards)
        # up_error = np.array(up_error)
        # down_error = np.array(down_error)
        #
        # plt.fill_between(self.evaluate_itr, avg_rewards - down_error, avg_rewards + up_error,
        #                  color='blue', alpha=0.2)
        # plt.xlabel('itr')
        # plt.ylabel('episode_reward')
