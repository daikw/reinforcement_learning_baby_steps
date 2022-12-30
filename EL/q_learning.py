from collections import defaultdict
import gymnasium as gym
from el_agent import ELAgent
from frozen_lake_util import plot_q_value


class QLearningAgent(ELAgent):

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
            while not done:
                a = self.policy(s, actions)
                n_state, reward, done, _, _ = env.step(a)

                # Feedback experience to Q-table, after each action was `step`ped
                #   - Value-based learning: `max(self.Q[n_state])` 
                gain = reward + gamma * max(self.Q[n_state])
                self.Q[s][a] += learning_rate * (gain - self.Q[s][a])
                s = n_state
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.print_reward_log(episode=e)


if __name__ == "__main__":
    gym.register(
        id="FrozenLakeEasy-v0",
        entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
        kwargs={"is_slippery": False}
    )

    agent = QLearningAgent(epsilon=0.1, verbose=True)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=500)

    import matplotlib.pyplot as plt
    plot_q_value(agent.Q)
    plt.show()
    agent.plot_reward_log()
    plt.show()
