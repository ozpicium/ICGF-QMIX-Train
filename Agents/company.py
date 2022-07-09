# -*- coding: utf-8 -*-
"""
Last updated time: 2021/09/22 23:56 AM
Author: Geng, Minghong
File: company.py
IDE: PyCharm
"""

# from agents import Agents as Platoon
from platoons import Platoon
import numpy as np

# Updated on 2021-10-10, build the 3rd layer control.
from fusionART import *

'''
This file will create a company class that contains platoons, as we want to do hierarchical control.
The 3rd level control: 1 Company.
The 2nd level control: 3 Platoons.
The 1st level base units: 12 Unit Agents.
'''


class Company:
    def __init__(self, args, company_id, evaluate=False, itr=1):
        self.args = args
        self.company_id = company_id
        self.n_platoons = self.args.n_ally_platoons
        self.platoons = [[] for _ in range(self.args.n_ally_platoons)]

        for n in range(len(self.platoons)):
            self.platoons[n] = Platoon(args, platoon_id=n, itr=itr, evaluate=False)

        self.evaluate = evaluate

        # epsilon decay
        # self.epsilon = 0 if evaluate else self.args.epsilon
        self.epsilon = 0 if evaluate else 1

        # initial last action
        self.last_action = np.zeros((self.args.n_ally_platoons, self.args.n_ally_agent_in_platoon, self.args.n_actions))

        # location of platoons + health of platoons + enemies detected by platoons
        lengths_state = self.args.n_ally_platoons * len(self.args.map_sps) + \
                        self.args.n_ally_platoons * self.args.n_ally_agent_in_platoon + \
                        self.args.n_ally_platoons * 1

        # Commander will send the movement instruction to each platoon
        lengths_actions = len(self.args.map_sps) * self.args.n_ally_platoons

        # the reward
        lengths_reward = 1

        # init the company commander
        self.commander = FusionART(numspace=3,
                                   # TODO: Check the dimension of each channel.
                                   # The state has three elements,
                                   #    - the location of platoons (3, 11)
                                   #    - the health of platoons (3, 4)
                                   #    - the number of enemies detected by platoons (3, 1)
                                   # lengths=[4, 4, 2],
                                   lengths=[lengths_state, lengths_actions, lengths_reward],
                                   beta=[1.0, 1.0, 1.0],
                                   alpha=[0.1, 0.1, 0.1],
                                   gamma=[1.0, 1.0, 1.0],
                                   rho=[0.2, 0.2, 0.5])
        print("="*20)
        print("Company commander")
        print("  Length of state: " + str(lengths_state))
        print("  Length of actions: " + str(lengths_actions))
        print("  Length of reward: " + str(lengths_reward))
        print("=" * 20)


    def epsilon_decay(self):
        """
        The epsilon decay for the company could be different from platoons.
        The platoons use "step" as the approach. But company could use other choices.
        :return:
        """
        self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon > self.args.min_epsilon else self.epsilon
        for n in range(len(self.platoons)):
            self.platoons[n].epsilon_decay()

    def init_last_action(self):
        """
        The company get all of the last actions of each platoon.
        Check on 2021-10-26, this last_action is not used in training.
        :return:
        """
        self.last_action = np.zeros((self.args.n_ally_platoons, self.args.n_ally_agent_in_platoon, self.args.n_actions))
        for n in range(len(self.platoons)):
            self.platoons[n].init_last_action()
        return

    def choose_action(self, obs, last_action, agent_idx, avail_actions_mask, epsilon, evaluate=False):
        return

    # TODO: Check how to record the episode buffer
    def init_episode_buffer(self):
        return

    def init_policy(self):
        for n in range(len(self.platoons)):
            self.platoons[n].policy.init_hidden(1)

    # def get_avail_actions(self):
    #     for n in range(len(self.platoons)):
    #         self.platoons[n].get_avail_actions()
    #     return
    #
    # TODO the company obs is the aggregated obs of the whole company.
    # def init_platoons(self, args, itr):
    #     for n in range(len(self.platoons)):
    #         self.platoons[n] = Platoon(args, itr=itr)

    def get_company_observation(self):
        return

    # TODO the company action is to "move to certain strategic location".
    def get_company_action(self):
        return

    def record_company_last_action(self):
        return
