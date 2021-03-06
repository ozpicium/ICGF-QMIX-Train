import argparse
# import re
import math


def get_common_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='4t_vs_4t_3paths_random_move', help='Select the map')
    parser.add_argument('--alg', type=str, default='qmix', help='select the algorithm')
    parser.add_argument('--last_action', type=bool, default=True, help='是否使用上一个动作帮助决策')
    parser.add_argument('--optim', type=str, default='RMS', help='optimizer')
    parser.add_argument('--reuse_network', type=bool, default=True, help='是否共享一个网络 whether share the same network')
    parser.add_argument('--result_dir', type=str, default='./results', help='保存模型和结果的位置')
    parser.add_argument('--model_dir', type=str, default='./model', help='这个策略模型的地址')
    # parser.add_argument('--replay_dir', type=str, default='./replay', help='where to save replays')
    parser.add_argument('--replay_dir', type=str, default='./replay',
                        help='where to save replays')
    parser.add_argument('--load_model', type=bool, default=False, help='是否加载已有模型')
    parser.add_argument('--load_result', type=bool, default=False, help='Load previous results')
    parser.add_argument('--learn', type=bool, default=True, help='是否训练模型')
    # parser.add_argument('--test', type=bool, default=False, help='whether to test the model')
    parser.add_argument('--gpu', type=str, default='0,1,2', help='使用哪个GPU，默认不使用')
    parser.add_argument('--num', type=int, default=1, help='并行执行多少个程序进程')
    parser.add_argument('--n_itr', type=int, default=300001, help='最大迭代次数')

    args = parser.parse_args()
    args.difficulty = '7'
    args.step_mul = 8
    args.seed = 123
    args.gamma = 0.99
    args.evaluate_num = 32

    '''
    The parameters used for hierarchical control architecture.
    It should be set to **False** if the experiment doesn't use hierarchical control architecture, 
    '''
    args.hierarchical = True
    args.train_on_intervals = True  # whether train on the movement between 2 SP, this is used for randomized start points

    args.n_ally_platoons = 1 #total number of platoons
    args.n_ally_agent_in_platoon = 4 #total number of agents in each platoon
    
    ## Define strategic points for the given map
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
                    '4t_vs_0t_8SPs_RandomEnemy', '4t_vs_0t_8SPs_RandomEnemy_075']:
        args.map_sps = {"0": [28.24, 39.35],
                        "1": [48.5, 44.37],
                        "2": [34.62, 71.31],
                        "3": [33.86, 101.59],
                        "4": [62.01, 111.68],
                        "5": [78.91, 74.61],
                        "6": [104.72, 78.81],
                        "7": [110.38, 104.69]}
        
    if args.map in ['4t_vs_4t_3paths_random_move', '4t_vs_4t_3paths_spawnSP1', '4t_vs_4t_3paths_spawnSP4', '4t_vs_4t_3paths_spawnSP7',
                      '4t_vs_4t_3paths_spawnSP10', '4t_vs_4t_3paths_fixed_enemy', '4t_vs_4t_3paths_dyna_enemy', '4t_vs_4t_3paths_cont_nav',
                      '4t_vs_20t_3paths', '4t_vs_12t_3paths_general']:
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

    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4
    
    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    args.epsilon_decay = (args.epsilon - args.min_epsilon) / 50000
    # args.epsilon_decay = (args.epsilon - args.min_epsilon) / 800000 # The epsilon will decrease to 0.05 after around 40,000 episodes.
    args.epsilon_anneal_scale = 'step'

    args.n_episodes = 1
    args.train_steps = 1
    args.evaluation_period = 1000
    args.batch_size = 32
    args.buffer_size = int(5e3)
    args.save_model_period = math.ceil(args.n_itr / 100.)

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
