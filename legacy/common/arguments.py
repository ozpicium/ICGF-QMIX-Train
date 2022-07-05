# -*- coding: utf-8 -*-
"""
@Time ： 2020/7/15 17:05
@Auth ： Kunfeng Li
@File ：arguments.py
@IDE ：PyCharm

"""
import argparse
# import re
import math


def get_common_args():
    """
    得到通用参数
    :return:
    """
    parser = argparse.ArgumentParser()
    # 星际争霸环境设置
    # This is the default setting, 5m_vs_6m is used: parser.add_argument('--map', type=str,
    # default='5m_vs_6m', help='使用的地图')
    parser.add_argument('--map', type=str, default='4t_vs_0t_8SPs_RandomEnemy', help='Select the map')
    # 算法设置 ensemble
    # This is the default setting, cwqmix is used: parser.add_argument('--alg', type=str,
    # default='cwqmix', help='选择所使用的算法')
    parser.add_argument('--alg', type=str, default='qmix', help='select the algorithm')
    parser.add_argument('--last_action', type=bool, default=True, help='是否使用上一个动作帮助决策')
    parser.add_argument('--optim', type=str, default='RMS', help='optimizer')
    parser.add_argument('--reuse_network', type=bool, default=True, help='是否共享一个网络 whether share the same network')
    # 程序运行设置
    parser.add_argument('--result_dir', type=str, default='./results', help='保存模型和结果的位置')
    parser.add_argument('--model_dir', type=str, default='./model', help='这个策略模型的地址')
    # parser.add_argument('--replay_dir', type=str, default='./replay', help='where to save replays')
    parser.add_argument('--replay_dir', type=str, default='./replay',
                        help='where to save replays')
    parser.add_argument('--load_model', type=bool, default=True, help='是否加载已有模型')
    parser.add_argument('--load_result', type=bool, default=False, help='Load previous results')
    parser.add_argument('--learn', type=bool, default=True, help='是否训练模型')
    # parser.add_argument('--test', type=bool, default=False, help='whether to test the model')
    parser.add_argument('--gpu', type=str, default='0', help='使用哪个GPU，默认不使用')
    parser.add_argument('--num', type=int, default=1, help='并行执行多少个程序进程')
    # 部分参数设置
    parser.add_argument('--n_itr', type=int, default=100001, help='最大迭代次数')

    args = parser.parse_args()
    # -------------------------------------这些参数一般不会更改-------------------------------------
    # 游戏难度
    args.difficulty = '7'
    # 多少步执行动作
    # TODO 什么意思？
    args.step_mul = 8
    # 游戏版本
    # args.game_version = 'latest'
    # 随机数种子
    args.seed = 123
    # 回放的绝对路径
    args.replay_dir = './replay'
    #args.replay_dir = r'./replay/4t_vs_0t_8SPs_RandomEnemy-2022-03-18_01-00'
    args.replay_dir = './replay/4t_vs_4t_3paths_random_move'
    # 折扣因子
    args.gamma = 0.99
    # 测试的次数
    args.evaluate_num = 32

    '''
    The parameters used for hierarchical control architecture.
    It should be set to **False** if the experiment doesn't use hierarchical control architecture, 
    In a single platoon case like 4t_vs_4t_SP01, we still use the hierarchical control architecture.
    '''
    args.hierarchical = True
    args.train_on_intervals = False  # whether train on the movement between 2 SP, this is used for randomized start points
    # TODO: When record the demo for hierarchical control, set the n_ally_platoons = 3
    args.n_ally_platoons = 1
    args.n_ally_agent_in_platoon = 4

    # # update on 2021-10-10
    # args.map_sps = {
    #     "0": [48.5, 44.37],
    #     "1": [42.77, 58.51],
    #     "2": [34.62, 71.31],
    #     # "3": [28.38, 85.91],
    #     "3": [37.77, 89.07],  # the location of sp 4 in the new map 12t_1enemy_flat
    #     # "4": [47.45, 104.85],
    #     "4": [33.86, 101.59],  # the location of sp 4 in the new map 12t_1ene
    #     "5": [62.01, 111.68],
    #     "6": [75.70, 92.09],
    #     "7": [78.91, 74.61],
    #     "8": [96.28, 77.40],
    #     "9": [108.48, 90.81],
    #     "10": [110.38, 104.69]
    # }

    # TODO: Only used in 12t_1t_demo scenario.
    # args.map_sps = {
    #     "0": [48.5, 44.37],
    #     "1": [42.77, 58.51],
    #     "2": [34.62, 71.31],
    #     # "3": [28.38, 85.91],
    #     "3": [37.77, 89.07],  # the location of sp 4 in the new map 12t_1enemy_flat
    #     # "4": [47.45, 104.85],
    #     "4": [47.68, 101.87],  # the location of sp 4 in the new map 12t_1ene
    #     "5": [67.62, 107.99],
    #     "6": [75.70, 92.09],
    #     "7": [78.91, 74.61],
    #     "8": [96.28, 77.40],
    #     "9": [108.48, 90.81],
    #     "10": [110.38, 104.69]
    # }

    # TODO: 2021-12-10: Coordinates of the strategic points in the training on movement on intervals.
    # args.map_sps = {
    #     "0": [48.5, 44.37],
    #     "1": [42.77, 58.51],
    #     "2": [34.62, 71.31],
    #     "3": [37.77, 89.07],
    #     "4": [33.86, 101.59],
    #     "5": [62.01, 111.68],
    #     "6": [75.70, 92.09],
    #     "7": [78.91, 74.61],
    #     "8": [96.28, 77.40],
    #     "9": [108.48, 90.81],
    #     "10": [110.38, 104.69]
    # }

    if args.map == '4t_vs_4t_7SPs':
        args.map_sps = {"0": [28.24, 39.35],
                        "1": [48.5, 44.37],
                        "2": [34.62, 71.31],
                        "3": [33.86, 101.59],
                        "4": [62.01, 111.68],
                        "5": [78.91, 74.61],
                        "6": [110.38, 104.69]}
    if args.map in ['4t_vs_4t_8SPs', '4t_vs_4t_8SPs_weakened',
                    '4t_vs_0t_8SPs_randomized', '4t_vs_0t_8SPs',
                    '4t_vs_0t_8SPs_RandomEnemy', '4t_vs_0t_8SPs_RandomEnemy_075', '12t_vs_4t_8SPs_1QMIX']:
        args.map_sps = {"0": [28.24, 39.35],
                        "1": [48.5, 44.37],
                        "2": [34.62, 71.31],
                        "3": [33.86, 101.59],
                        "4": [62.01, 111.68],
                        "5": [78.91, 74.61],
                        "6": [104.72, 78.81],
                        "7": [110.38, 104.69]}
    
    if args.map in ['4t_vs_4t_3paths_random_move']:
        args.map_sps = {"0": [12.07, 36.92],
                        "1": [45.25, 42.38],
                        "2": [27.04, 88.21],
                        "3": [47.01, 104.75],
                        "4": [74.69, 108.51],
                        "5": [63.94, 50.89],
                        "6": [76.88, 71.08],
                        "7": [40.17, 8.29],
                        "8": [73.08, 8.54],
                        "9": [91.85, 23.38],
                        "10": [116.55, 53.57],
                        "11": [104.19, 77.89],
                        "12": [110.25, 100.89]}
        
    args.formation = True
    args.deviation = 3
    if args.formation:
        args.platoon_indi_sps = generate_individual_sp(args.n_ally_agent_in_platoon,
                                                       args.map_sps,
                                                       args.deviation)

    '''
    If the FALCON_demo is true, the three platoons in the hierarchical structure will follow a certain pattern to move.
    Set FALCON_demo = False in normal training.
    '''
    args.FALCON_demo = False

    args.obs_distance_target = True

    return args


def get_q_decom_args(args):
    """
    得到值分解算法（vdn, qmix, qtran）的参数
    :param args:
    :return:
    """
    # 网络设置
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    # 学习率
    args.lr = 5e-4
    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    # TODO: By default, the args.epsilon_decay is = (args.epsilon - args.min_epsilon) / 50000
    # args.epsilon_decay = (args.epsilon - args.min_epsilon) / 100000
    args.epsilon_decay = (args.epsilon - args.min_epsilon) / 50000
    # args.epsilon_decay = (args.epsilon - args.min_epsilon) / 800000 # The epsilon will decrease to 0.05 after
    # around 40,000 episodes.
    args.epsilon_anneal_scale = 'step'
    # 一个itr里有多少个episode
    args.n_episodes = 1
    # 一个 itr 里训练多少次
    args.train_steps = 1
    # TODO 多久评价一次，pymarl中产生200个点，即循环两百万次，每一万次开始评价，所以最好还是确定一个迭代次数后更改这个
    # TODO args.evaluation_period = 100 / args.evaluation_period = math.ceil(args.n_itr / 10.)
    # args.evaluation_period = math.ceil(args.n_itr / 100.)

    # TODO: [11-09] in normal training, evaluate the model every 500 eps.
    # TODO [21-12-27] in HCA experimrnts, evaluate the model 10 times in 1000 episodes.
    # args.evaluation_period = 500
    args.evaluation_period = math.ceil(args.n_itr / 10.)

    # 经验池，采样32个episode
    args.batch_size = 32
    # args.buffer_size = int(5e3) # 最多存储5000个buffer， 训练时从其中抽样
    # TODO: Only use size 1000 when doing development. In the actual learning, size = 5000.
    args.buffer_size = int(5e3)
    # 模型保存周期
    # TODO args.save_model_period = 5000
    # TODO args.save_model_period = math.ceil(args.n_itr / 100.)
    # args.save_model_period = 500
    # args.save_model_period = math.ceil(args.n_itr / 100.)  # todo enlarge the save model period
    args.save_model_period = math.ceil(args.n_itr / 10.)

    # target网络更新周期，episode
    args.target_update_period = 200
    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1
    # 梯度裁剪
    args.clip_norm = 10
    # maven
    return args

def generate_individual_sp(n_ally_agent_in_platoon, map_sps, deviation):
    platoon_indi_sps = [{} for _ in range(n_ally_agent_in_platoon)]

    for k, v in map_sps.items():
        for agent_id, agent_indi_sps in enumerate(platoon_indi_sps):  # set the individual target points for each agents
            if agent_id == 0:
                agent_indi_sps[k] = [v[0], v[1] + deviation]
            elif agent_id == 1:
                agent_indi_sps[k] = [v[0], v[1] - deviation]
            elif agent_id == 2:
                agent_indi_sps[k] = [v[0] + deviation, v[1]]
            elif agent_id == 3:
                agent_indi_sps[k] = [v[0] - deviation, v[1]]
    return platoon_indi_sps