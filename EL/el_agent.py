import numpy as np
import matplotlib.pyplot as plt


class ELAgent():

    def __init__(self, epsilon, verbose=False):
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []
        self.verbose = verbose

    def policy(self, s, actions):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(actions))
        else:
            if s in self.Q and sum(self.Q[s]) != 0:
                return np.argmax(self.Q[s])
            else:
                return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def print_reward_log(self, episode, stride=50):
        rewards = self.reward_log[max(episode-stride, 0):episode]
        mean = np.round(np.mean(rewards), 3)
        std = np.round(np.std(rewards), 3)
        print("At Episode {} average reward is {} (+/-{}).".format(episode, mean, std))

    def plot_reward_log(self, stride=50):
        indices = list(range(0, len(self.reward_log), stride))
        means = []
        stds = []
        for i in indices:
            rewards = self.reward_log[i:(i + stride)]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
        means = np.array(means)
        stds = np.array(stds)

        _, ax = plt.subplots(1,1)
        ax.title.set_text("Reward History")
        ax.grid()
        ax.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
        ax.plot(indices, means, "o-", color="g", label="Rewards for each {} episode".format(stride))
        ax.legend(loc="best")

        return ax
