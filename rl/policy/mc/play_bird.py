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
        self.n = mcbirdenv.n
        self.gamma = mcbirdenv.gamma

    def epsilon_greedy_policy(self, q_values, state, epsilon):
        max_index = q_values[state, :].argmax()
        if random.random() < 1 - epsilon:
            return self.actions[max_index]
        else:
            random_index = random.randint(0, len(self.actions) - 1)
            while random_index == max_index:
                random_index = random.randint(0, len(self.actions) - 1)

            return self.actions[random_index]

    def find_action_index(self, action):
        for i in range(len(self.actions)):
            if action == self.actions[i]:
                return i

    def first_visit(self, trajectory, step_num):
        done_states = set([])
        for i in range(step_num):
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

                self.n[state, action_index] += 1
                q_value = self.q_values[state, action_index]
                q_value = q_value + (1 / self.n[state, action_index]) * (g - q_value)
                self.q_values[state, action_index] = q_value

    def every_visit(self, trajectory, step_num):
        for i in range(step_num):
            state = trajectory[i]['state']
            action = trajectory[i]['action']
            action_index = self.find_action_index(action)
            g = 0
            for j in range(len(trajectory[i:])):
                reward = trajectory[i:][j]['reward']
                g += reward * pow(self.gamma, j)

            self.n[state, action_index] += 1
            q_value = self.q_values[state, action_index]
            q_value = q_value + (1 / self.n[state, action_index]) * (g - q_value)
            self.q_values[state, action_index] = q_value

    def on_policy(self, num_iters, visit_type):
        iter_history = []
        last_q_values = None
        for i in range(num_iters):
            trajectory = []
            state = self.mcbirdenv.reset()
            is_over = False
            step_num = 0
            epsilon = 0.3

            while is_over == False:
                action = self.epsilon_greedy_policy(self.q_values, state, epsilon)
                next_state, reward, is_over = self.mcbirdenv.transform(state, action)
                trajectory.append({'state': state, 'action': action, 'reward': reward})
                state = next_state
                step_num += 1

            if visit_type == 'first':
                self.first_visit(trajectory, step_num)
            elif visit_type == 'every':
                self.every_visit(trajectory, step_num)
            else:
                raise Exception("please choose correct visit_type")

            if i % 10 == 0:
                if last_q_values is None:
                    last_q_values = self.q_values.copy()
                else:
                    diff = mse(last_q_values, self.q_values)
                    iter_history.append(diff)
                    last_q_values = self.q_values.copy()

        return iter_history

    def off_policy(self):
        

        return 1

    def find_policy(self):
        policy = {}
        for state in self.states:
            max_index = self.q_values[state, :].argmax()
            policy[state] = self.actions[max_index]

        return policy


if __name__ == "__main__":
    mcbirdenv = MCBirdEnv()
    mc = MC(mcbirdenv)
    iter_history = mc.on_policy(500000, 'every')
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
