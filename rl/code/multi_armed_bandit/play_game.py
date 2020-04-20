import numpy as np
import matplotlib.pyplot as plt


class KB_Game:
    def __init__(self, *args, **kwargs):
        self.rewards = np.array([0.0, 0.0, 0.0])
        self.action_counts = np.array([0, 0, 0])
        self.actions = [0, 1, 2]
        self.counts = 0
        self.counts_history = []
        self.current_cumulative_reward = 0.0
        self.cumulative_rewards_history = []
        self.current_action = 1
        self.current_reward = 0.0

    def step(self, action):
        reward = 0
        if action == 0:
            reward = np.random.normal(1, 1)
        if action == 1:
            reward = np.random.normal(2, 1)
        if action == 2:
            reward = np.random.normal(1.5, 1)

        return reward

    def choose_action(self, policy, **kwargs):
        action = 0
        if policy == 'e_greedy':
            if np.random.random() < kwargs['epsilon']:
                action = np.random.randint(0, 3)
            else:
                action = np.argmax(self.rewards)

        if policy == 'ucb':
            c_ratio = kwargs['c_ratio']
            if 0 in self.action_counts:
                action = np.where(self.action_counts == 0)[0][0]
            else:
                values = self.rewards + c_ratio * np.sqrt(np.log(self.counts) / self.action_counts)
                action = np.argmax(values)

        if policy == 'boltzmann':
            tau = kwargs['temperature']
            p = np.exp(self.rewards / tau) / (np.sum(np.exp(self.rewards / tau)))
            action = np.random.choice(self.actions, p=p.ravel())

        return action

    def train(self, play_times, policy, **kwargs):
        for i in range(play_times):
            action = 0
            if policy == 'e_greedy':
                action = self.choose_action(policy, epsilon=kwargs['epsilon'])
            if policy == 'ucb':
                action = self.choose_action(policy, c_ratio=kwargs['c_ratio'])
            if policy == 'boltzmann':
                action = self.choose_action(policy, temperature=kwargs['temperature'])

            self.current_action = action
            self.current_reward = self.step(self.current_action)
            self.counts += 1
            self.rewards[self.current_action] = self.rewards[self.current_action] + \
                                                (self.current_reward - self.rewards[self.current_action]) \
                                                / (self.action_counts[self.current_action] + 1)

            self.action_counts[self.current_action] += 1
            self.current_cumulative_reward += self.current_reward
            self.cumulative_rewards_history.append(self.current_cumulative_reward)
            self.counts_history.append(i)

    def reset(self):
        self.rewards = np.array([0.0, 0.0, 0.0])
        self.action_counts = np.array([0, 0, 0])
        self.actions = [0, 1, 2]
        self.counts = 0
        self.counts_history = []
        self.current_cumulative_reward = 0.0
        self.cumulative_rewards_history = []
        self.current_action = 1
        self.current_reward = 0.0


if __name__ == '__main__':
    np.random.seed(0)
    k_gamble = KB_Game()
    play_times = 2000

    plt.figure(1)

    k_gamble.train(play_times=play_times, policy='e_greedy', epsilon=0.05)
    plt.plot(k_gamble.counts_history, k_gamble.cumulative_rewards_history, 'r', label='e_greedy', linestyle='-.')
    k_gamble.reset()
    k_gamble.train(play_times=play_times, policy='boltzmann', temperature=1)
    plt.plot(k_gamble.counts_history, k_gamble.cumulative_rewards_history, 'g', label='boltzmann', linestyle='--')
    k_gamble.reset()
    k_gamble.train(play_times=play_times, policy='ucb', c_ratio=0.5)
    plt.plot(k_gamble.counts_history, k_gamble.cumulative_rewards_history, 'b', label='ucb', linestyle='-')

    plt.legend()
    plt.xlabel('n', fontsize=18)
    plt.ylabel('total rewards', fontsize=18)
    plt.show()
