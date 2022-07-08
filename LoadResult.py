import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from numpy import mean

# episodes_rewards_temp = np.load('results/qmix/Sandbox-4t-waypoint/1/2021-08-20_01-51-58/episodes_rewards.npy', encoding="latin1", allow_pickle=True).tolist()
#
# episodes_rewards = np.zeros([len(episodes_rewards_temp), len(max(episodes_rewards_temp,key = lambda x: len(x)))])
# for i, j in enumerate(episodes_rewards_temp):
#    episodes_rewards[i][0:len(j)] = j
#    episodes_rewards[i][len(j):] = np.mean(j)
# episodes_rewards = episodes_rewards.tolist()
# max_r = max(episodes_rewards)
#
# max_reward = []
# for i in episodes_rewards:
#     max_reward.append(max(i))
# mean_reward = []
# for i in episodes_rewards:
#     mean_reward.append(mean(i))

# episodes_rewards = np.load('results/qmix/Sandbox-4t-waypoint/1/2021-08-17-Rm20_new reward/2021-08-22_01-35-55/evaluate_itr.npy', encoding="latin1", allow_pickle=True).tolist()
evaluate_itr = np.load("results/qmix/4t_vs_0t_8SPs_RandomEnemy_075/1/2022-03-29_17-175000/evaluate_itr.npy", encoding="latin1", allow_pickle=True).tolist()
#win_rates = np.load('results/qmix/8t/1/win_rates.npy', encoding="latin1", allow_pickle=True).tolist()
#max_win_rate = max(win_rates)

#doc = open('1.txt', 'a')  # 打开一个存储文件，并依次写入
#print(episode_rewards, file=doc)
max_itr = max(evaluate_itr)

fig = plt.figure()
# ax1 = fig.add_subplot(211)
# win_x = np.array(evaluate_itr)[:, None]
# win_y = np.array(win_rates)[:, None]
# plot_win = pd.DataFrame(np.concatenate((win_x, win_y), axis=1), columns=['evaluate_itr', 'win_rates'])
# sns.lineplot(x="evaluate_itr", y="win_rates", data=plot_win, ax=ax1)

ax2 = fig.add_subplot(212)
reward_x = np.repeat(evaluate_itr, 100)[:, None]
reward_y = np.array(episodes_rewards).flatten()[:, None]
plot_reward = pd.DataFrame(np.concatenate((reward_x, reward_y), axis=1),
                           columns=['evaluate_itr', 'episodes_rewards'])
sns.lineplot(x="evaluate_itr", y="episodes_rewards", data=plot_reward, ax=ax2,
             ci=95, estimator=np.mean) #=np.median)  # 为什么用median?

# 格式化成2016-03-20-11_45_39形式
tag = 'qmix' + '-' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# 如果已经有图片就删掉
#for filename in os.listdir(self.save_path):
#    if filename.endswith('.png'):
#        os.remove(self.save_path + '/' + filename)
fig.savefig('results/')
plt.close()