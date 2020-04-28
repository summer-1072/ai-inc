import random
import numpy as np
from rl.env.mc_bird_env import MCBirdEnv
from rl.tools.iter_evaluate import mse
from rl.tools.iter_evaluate import show


class MC:
    def __init__(self, mcbirdenv):
        self.mcbirdenv = mcbirdenv
        self.states = mcbirdenv.states
        self.values = mcbirdenv.values
        self.actions = mcbirdenv.actions
        self.q_values = mcbirdenv.q_values
        self.gamma = mcbirdenv.gamma

        self.n = np.zeros((100, 4))
        self.important_weights = 0.001 * np.ones((100, 4))

    def epsilon_greedy_policy(self, state, epsilon):
        max_index = self.q_values[state, :].argmax()
        if random.random() < 1 - epsilon:
            return 1 - epsilon, self.actions[max_index]
        else:
            random_index = random.randint(0, len(self.actions) - 1)
            while random_index == max_index:
                random_index = random.randint(0, len(self.actions) - 1)

            return epsilon, self.actions[random_index]

    def iter_important_weight(self, trajectory):
        important_weight = 1
        for step in trajectory:
            explore_state = step['state']
            explore_pro = step['pro']
            explore_action = step['action']

            max_index = self.q_values[explore_state, :].argmax()
            obejct_action = self.actions[max_index]
            if explore_action == obejct_action:
                obejct_pro = 0.999
            else:
                obejct_pro = 0.001

            important_weight *= obejct_pro / explore_pro

        return important_weight

    def find_action_index(self, action):
        for i in range(len(self.actions)):
            if action == self.actions[i]:
                return i

    def first_visit(self, trajectory, policy_type):
        done_states = set([])
        for i in range(len(trajectory)):
            state = trajectory[i]['state']
            action = trajectory[i]['action']
            action_index = self.find_action_index(action)
            if state in done_states:
                continue
            else:
                done_states.add(state)
                g = 0
                for j in range(len(trajectory[i:])):
                    reward = trajectory[i:][j]['reward']
                    g += reward * pow(self.gamma, j)

                q_value = self.q_values[state, action_index]

                if policy_type == 'on policy':
                    self.n[state, action_index] += 1
                    q_value = q_value + (1 / self.n[state, action_index]) * (g - q_value)
                    self.q_values[state, action_index] = q_value
                elif policy_type == 'off policy':
                    important_weight = self.iter_important_weight(trajectory[i:])
                    self.important_weights[state, action_index] += important_weight
                    q_value = q_value + (important_weight / self.important_weights[state, action_index]) * (g - q_value)
                    self.q_values[state, action_index] = q_value
                else:
                    raise Exception("please choose correct policy")

    def every_visit(self, trajectory, policy_type):
        for i in range(len(trajectory)):
            state = trajectory[i]['state']
            action = trajectory[i]['action']
            action_index = self.find_action_index(action)
            g = 0
            for j in range(len(trajectory[i:])):
                reward = trajectory[i:][j]['reward']
                g += reward * pow(self.gamma, j)

            q_value = self.q_values[state, action_index]
            if policy_type == 'on policy':
                self.n[state, action_index] += 1
                q_value = q_value + (1 / self.n[state, action_index]) * (g - q_value)
                self.q_values[state, action_index] = q_value
            elif policy_type == 'off policy':
                important_weight = self.iter_important_weight(trajectory[i:])
                self.important_weights[state, action_index] += important_weight
                q_value = q_value + (important_weight / self.important_weights[state, action_index]) * (g - q_value)
                self.q_values[state, action_index] = q_value
            else:
                raise Exception("please choose correct policy")

    def iter_policy(self, num_iters, visit_type, policy_type, epsilon=0.3):
        iter_history = []
        last_q_values = None
        for i in range(num_iters):
            trajectory = []
            state = self.mcbirdenv.reset()
            is_over = False

            while is_over == False:
                pro, action = self.epsilon_greedy_policy(state, epsilon)
                next_state, reward, is_over = self.mcbirdenv.transform(state, action)
                trajectory.append({'state': state, 'pro': pro, 'action': action, 'reward': reward})
                state = next_state

            if visit_type == 'first visit':
                self.first_visit(trajectory, policy_type)
            elif visit_type == 'every visit':
                self.every_visit(trajectory, policy_type)
            else:
                raise Exception("please choose correct visit_type")

            if i % 10 == 0:
                if last_q_values is None:
                    last_q_values = self.q_values.copy()
                else:
                    diff = mse(last_q_values, self.q_values)
                    iter_history.append(diff)
                    last_q_values = self.q_values.copy()

                    print('policy iter num %s, diff is %s' % (i, diff))

        return iter_history

    def find_policy(self):
        policy = {}
        for state in self.states:
            max_index = self.q_values[state, :].argmax()
            policy[state] = self.actions[max_index]

        return policy


if __name__ == "__main__":
    mcbirdenv = MCBirdEnv()
    mc = MC(mcbirdenv)
    iter_history = mc.iter_policy(50000, 'every visit', 'on policy')
    show('iter_history', iter_history)

    policy = mc.find_policy()
    flag = True
    state = 0
    path = [state]
    mcbirdenv.q_values = np.around(mc.q_values, 2)

    while flag:
        action = policy[state]
        next_state, reward, is_over = mcbirdenv.transform(state, action)

        if is_over:
            flag = False

        state = next_state
        path.append(state)

    print('path:', path)
    mcbirdenv.render(path)
