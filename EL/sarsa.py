from collections import defaultdict
import gymnasium as gym
from el_agent import ELAgent
from frozen_lake_util import plot_q_value


class SARSAAgent(ELAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        for e in range(episode_count):
            s, _ = env.reset()
            done = False
            a = self.policy(s, actions)
            while not done:
                n_state, reward, done, _, _ = env.step(a)

                n_action = self.policy(n_state, actions) # On-policy
                gain = reward + gamma * self.Q[n_state][n_action]
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                a = n_action
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0 and self.verbose:
                self.print_reward_log(episode=e)


if __name__ == "__main__":
    gym.register(
        id="FrozenLakeEasy-v0",
        entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
        kwargs={"is_slippery": False}
    )

    agent = SARSAAgent(epsilon=0.1, verbose=True)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=500)

    import matplotlib.pyplot as plt
    plot_q_value(agent.Q)
    plt.show()
    agent.plot_reward_log()
    plt.show()
