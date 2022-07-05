import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time

from numpy import mean

# episodes_rewards
episodes_rewards_temp_1 = np.load('results/qmix/1p_no_enemy_flat/1/2021-11-01_8000_eps/episodes_rewards.npy', encoding="latin1", allow_pickle=True).tolist()
episodes_rewards_temp_2 = np.load('results/qmix/1p_no_enemy_flat/1/2021-11-02_12-16-28/episodes_rewards.npy', encoding="latin1", allow_pickle=True).tolist()
episodes_rewards_temp_3 = np.load('results/qmix/1p_no_enemy_flat/1/2021-11-02_21-07-40/episodes_rewards.npy', encoding="latin1", allow_pickle=True).tolist()

episodes_rewards_temp = episodes_rewards_temp_1 + episodes_rewards_temp_2 + episodes_rewards_temp_3

episodes_rewards = np.zeros([len(episodes_rewards_temp), len(max(episodes_rewards_temp,key = lambda x: len(x)))])

for i, j in enumerate(episodes_rewards_temp):
   episodes_rewards[i][0:len(j)] = j
   episodes_rewards[i][len(j):] = np.mean(j)
episodes_rewards = episodes_rewards.tolist()
max_r = max(episodes_rewards)

max_reward = []
for i in episodes_rewards:
    max_reward.append(max(i))
mean_reward = []
for i in episodes_rewards:
    mean_reward.append(mean(i))
# episodes_rewards = np.load('results/qmix/Sandbox-4t-waypoint/1/2021-08-17-Rm20_new reward/2021-08-22_01-35-55/evaluate_itr.npy', encoding="latin1", allow_pickle=True).tolist()


# evaluate_itr
evaluate_itr_1 = np.load("results/qmix/1p_no_enemy_flat/1/2021-11-01_8000_eps/evaluate_itr.npy", encoding="latin1", allow_pickle=True).tolist()

evaluate_itr_2 = np.load("results/qmix/1p_no_enemy_flat/1/2021-11-02_12-16-28/evaluate_itr.npy", encoding="latin1", allow_pickle=True).tolist()
diff_12 = (evaluate_itr_1[-1] + 500) - evaluate_itr_2[0]
for i in range(len(evaluate_itr_2)):
    evaluate_itr_2[i] += diff_12

evaluate_itr_3 = np.load("results/qmix/1p_no_enemy_flat/1/2021-11-02_21-07-40/evaluate_itr.npy", encoding="latin1", allow_pickle=True).tolist()
diff_23 = (evaluate_itr_2[-1] + 500) - evaluate_itr_3[0]
for i in range(len(evaluate_itr_3)):
    evaluate_itr_3[i] += diff_23

evaluate_itr = evaluate_itr_1 + evaluate_itr_2 + evaluate_itr_3
evaluate_itr = np.array(evaluate_itr)

# win rate
win_rates_1 = np.load('results/qmix/1p_no_enemy_flat/1/2021-11-01_8000_eps/win_rates.npy', encoding="latin1", allow_pickle=True).tolist()
win_rates_2 = np.load('results/qmix/1p_no_enemy_flat/1/2021-11-02_12-16-28/win_rates.npy', encoding="latin1", allow_pickle=True).tolist()
win_rates_3 = np.load('results/qmix/1p_no_enemy_flat/1/2021-11-02_21-07-40/win_rates.npy', encoding="latin1", allow_pickle=True).tolist()
win_rates = win_rates_1 + win_rates_2 + win_rates_3
max_win_rate = max(win_rates)

#doc = open('1.txt', 'a')  # 打开一个存储文件，并依次写入
#print(episode_rewards, file=doc)
max_itr = max(evaluate_itr)

fig = plt.figure()
ax1 = fig.add_subplot(211)
win_x = np.array(evaluate_itr)[:, None]
win_y = np.array(win_rates)[:, None]
plot_win = pd.DataFrame(np.concatenate((win_x, win_y), axis=1), columns=['evaluate_itr', 'win_rates'])
sns.lineplot(x="evaluate_itr", y="win_rates", data=plot_win, ax=ax1)

ax2 = fig.add_subplot(212)
reward_x = np.repeat(evaluate_itr, 100)[:, None]
reward_y = np.array(episodes_rewards).flatten()[:, None]
plot_reward = pd.DataFrame(np.concatenate((reward_x, reward_y), axis=1),
                           columns=['evaluate_itr', 'avg episodes_rewards'])
sns.lineplot(x="evaluate_itr", y="avg episodes_rewards", data=plot_reward, ax=ax2,
             ci=95, estimator=np.mean) #=np.median)  # 为什么用median?

# 格式化成2016-03-20-11_45_39形式
tag = 'qmix' + '-' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# 如果已经有图片就删掉
#for filename in os.listdir(self.save_path):
#    if filename.endswith('.png'):
#        os.remove(self.save_path + '/' + filename)
fig.savefig(r"D:\Documents\02 DSO\HierarchicalControl\results\qmix\1p_no_enemy_flat\1\results")
plt.close()