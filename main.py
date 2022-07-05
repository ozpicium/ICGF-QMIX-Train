# -*- coding: utf-8 -*-
"""
@Time ： 2020/7/15 17:23
@Thanks： Kunfeng Li
@Author: Geng, Minghong on 2021-09-21
"""

from smac_HC.env import StarCraft2Env_HC
# argument.py creates 2 functions, get_common_args() and get_q_decom_args()
# get_common_args() defines the how we define the env.
# get_q_decom_args() defines the parameters used in algorithms like vdn, QMIX and QTRAN.
from common.arguments import get_common_args, get_q_decom_args
from common.runner import Runner
import time
from multiprocessing import Pool
import os
import torch


def main(env, arg, itr):
    # pass in the environment and arguments in Runner().
    # The "itr" is the id of process. Defined in Runner.__init__
    runner = Runner(env, arg, itr)
    # 如果训练模型
    # arguments.learn is a boolean value, and it's defined in function get_q_decom_args().
    if arguments.learn:
        runner.run()
    runner.save_results()
    runner.plot()


if __name__ == '__main__':
    start = time.time()
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('StartTime:' + start_time)

    # function get_q_decom_args() add more parameters based on the parameters generated from get_common_args(), which
    # are simple arguments.
    arguments = get_q_decom_args(get_common_args())
    print('SHUBHAM~~~~~~~~~~~ Map Name: ', arguments.map, '\n')
    arguments.replay_dir = arguments.replay_dir + '/' + arguments.map
    print('SHUBHAM~~~~~~~~~~~ Replay Dir: ', arguments.replay_dir, '\n')
    
    if arguments.load_model:
        model_dir = arguments.model_dir + '/' + arguments.alg + '/' + arguments.map + '/' + str(1)
        if os.path.exists(model_dir + '/epsilon.txt'):
            path_epsilon = model_dir + '/epsilon.txt'
            f = open(path_epsilon, 'r')
            arguments.epsilon = float(f.readline())
            f.close()
            print('Successfully load epsilon:' + str(arguments.epsilon))
        else:
            print('No epsilon in this directory.')

    # set up the GPU resource
    if arguments.gpu is not None:
        arguments.cuda = True
        if arguments.gpu == 'a':
            pass
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        arguments.cuda = False

    # Set the environment，pymarl中设置的也是环境默认参数
    # StarCraft2Env is a class. In the code below, we change 4 default parameters.
    environment = StarCraft2Env_HC(map_name=arguments.map,
                                   step_mul=arguments.step_mul,
                                   difficulty=arguments.difficulty,
                                   replay_dir=arguments.replay_dir,
                                   hierarchical=arguments.hierarchical,
                                   n_ally_platoons=arguments.n_ally_platoons,
                                   n_ally_agent_in_platoon=arguments.n_ally_agent_in_platoon,
                                   map_sps=arguments.map_sps,
                                   # formation=arguments.formation,
                                   # platoon_indi_sps = arguments.platoon_indi_sps,
                                   train_on_intervals = arguments.train_on_intervals,
                                   FALCON_demo=arguments.FALCON_demo,
                                   obs_distance_target=arguments.obs_distance_target,
                                   debug=False)

    # retrieve the information of the environment
    env_info = environment.get_env_info()

    # # number of platoons
    # arguments.n_platoons = env_info['n_platoons']
    #
    # # number of units in each platoon
    # arguments.n_units_in_platoon = env_info['n_units_in_platoon']

    # TODO The state shape of the platoon and the company need further design

    # state shape
    arguments.state_shape = env_info['state_shape']
    arguments.company_state_shape = env_info['company_state_shape']
    arguments.platoon_state_shape = env_info['platoon_state_shape']

    # observation shape
    arguments.obs_shape = env_info['obs_shape']
    arguments.company_obs_shape = env_info['company_obs_shape']
    arguments.platoon_obs_shape = env_info['platoon_obs_shape']

    # episode长度限制
    arguments.episode_limit = env_info['episode_limit']

    # number of action.
    # number of action equals to the number of none-attack actions (move to 4 directions, stop, and no-op) plus the
    # number of enemies. So, this number is 6+X.
    arguments.n_actions = env_info['n_actions']

    # agent数目
    arguments.n_agents = env_info['n_agents']

    # 进程池，数字是并行的进程数，根据资源自行调整，默认是CPU核的个数
    if arguments.num > 1:
        p = Pool(12)
        for i in range(arguments.num):
            p.apply_async(main, args=(environment, arguments, i))
        print('子进程开始...')
        p.close()
        p.join()
        print('所有子进程结束！')
    else:
        # 0是4.10,1是4.6.2；对于ensemble，1是正常每次都随机权重，0是直接平均。
        main(environment, arguments, 1)  # Start training！ Run main() function.

    duration = time.time() - start
    time_list = [0, 0, 0]
    time_list[0] = duration // 3600
    time_list[1] = (duration % 3600) // 60
    time_list[2] = round(duration % 60, 2)
    print('Time Elapsed：' + str(time_list[0]) + ' hour ' + str(time_list[1]) + 'minute' + str(time_list[2]) + 'second')
    print('StartTime：' + start_time)
    end_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('EndTime：' + end_time)
