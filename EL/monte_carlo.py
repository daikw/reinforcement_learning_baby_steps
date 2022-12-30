import math
import gymnasium as gym

from collections import defaultdict
from el_agent import ELAgent
from frozen_lake_util import plot_q_value

class MonteCarloAgent(ELAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, env, episode_count=1000, gamma=0.9, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s, _ = env.reset()
            done = False
            # Play until the end of episode.
            experience = []
            while not done:
                a = self.policy(s, actions)
                n_state, reward, done, _, _  = env.step(a)
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
            else:
                self.log(reward)

            # Evaluate each state, action.
            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]

                # Calculate discounted future reward of s.
                G = sum([
                    math.pow(gamma, t) * x_t["reward"]
                    for t, x_t in enumerate(experience[i:])
                ])

                N[s][a] += 1  # count of s, a pair
                alpha = 1 / N[s][a]
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0 and self.verbose:
                self.print_reward_log(e)


if __name__ == "__main__":
    gym.register(
        id="FrozenLakeEasy-v0",
        entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
        kwargs={"is_slippery": False}
    )

    agent = MonteCarloAgent(epsilon=0.01, verbose=True)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=500)

    import matplotlib.pyplot as plt
    plot_q_value(agent.Q)
    plt.show()
    agent.plot_reward_log()
    plt.show()
