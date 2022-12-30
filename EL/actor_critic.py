import numpy as np
import gymnasium as gym
from el_agent import ELAgent
from frozen_lake_util import plot_q_value


class Actor(ELAgent):

    def __init__(self, env):
        super().__init__(epsilon=-1)
        nrow = env.observation_space.n
        ncol = env.action_space.n
        self.actions = list(range(env.action_space.n))
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s):
        a = np.random.choice(self.actions, 1, p=self.softmax(self.Q[s]))
        return a[0]


class Critic():

    def __init__(self, env):
        states = env.observation_space.n
        self.V = np.zeros(states)


class ActorCritic():

    def __init__(self, actor_class, critic_class, verbose=False):
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.verbose = verbose

    def train(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, report_interval=50):
        actor = self.actor_class(env)
        critic = self.critic_class(env)

        actor.init_log()
        for e in range(episode_count):
            s, _ = env.reset()
            done = False
            while not done:
                a = actor.policy(s)
                n_state, reward, done, _, _ = env.step(a)

                gain = reward + gamma * critic.V[n_state]
                estimated = critic.V[s]
                td = gain - estimated
                actor.Q[s][a] += learning_rate * td
                critic.V[s] += learning_rate * td
                s = n_state

            else:
                actor.log(reward)

            if e != 0 and e % report_interval == 0 and self.verbose:
                actor.print_reward_log(episode=e)

        return actor, critic


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gym.register(
        id="FrozenLakeEasy-v0",
        entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",
        kwargs={"is_slippery": False}
    )

    import ipdb; ipdb.set_trace()
    trainer = ActorCritic(Actor, Critic)
    env = gym.make("FrozenLakeEasy-v0")
    actor, critic = trainer.train(env, episode_count=3000, learning_rate=0.1)
    plot_q_value(actor.Q)
    plt.show()
    actor.plot_reward_log()
    plt.show()
