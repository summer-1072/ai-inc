import random
import time
from reinforcement_learning.dynamic_programming.bird.code.bird_env import BirdEnv


class DP_Policy_Iter:
    def __init__(self, bird_env, init_policy='random'):
        self.bird_env = bird_env
        self.states = bird_env.states
        self.actions = bird_env.actions
        self.values = bird_env.values
        self.gamma = bird_env.gamma
        self.pi = {}

        for state in self.states:
            flag1 = bird_env.collision_detection(bird_env.state_to_position(state))
            flag2 = bird_env.destination_detection(bird_env.state_to_position(state))

            if flag1 or flag2: continue

            if init_policy == 'random':
                self.pi[state] = [{'action': self.actions[int(random.random() * len(self.actions))], 'pro': 1}]

            elif init_policy == 'balance':
                self.pi[state] = [{'action': action, 'pro': 1 / len(self.actions)} for action in self.actions]

    def policy_evaluate(self):
        flag = True
        i = 0
        while flag:
            delta = 0
            i += 1
            for state in self.states:
                position = self.bird_env.state_to_position(state)
                flag1 = self.bird_env.collision_detection(position)
                flag2 = self.bird_env.destination_detection(position)
                if flag1 or flag2:
                    continue

                new_value = 0
                step_actions = self.pi[state]
                for step_action in step_actions:
                    action = step_action['action']
                    pro = step_action['pro']
                    new_state, reward, is_over = self.bird_env.transform(state, action)
                    new_value += pro * (reward + self.gamma * self.values[new_state])

                delta += abs(new_value - self.values[state])
                self.values[state] = new_value

            if delta < 1e-6:
                flag = False

        print('policy evaluate iter num: ', i)

    def policy_improve(self):
        for state in self.states:
            flag1 = self.bird_env.collision_detection(self.bird_env.state_to_position(state))
            flag2 = self.bird_env.destination_detection(self.bird_env.state_to_position(state))
            if flag1 or flag2: continue

            best_action = None
            value = None
            for action in self.actions:
                new_state, reward, is_over = self.bird_env.transform(state, action)
                new_value = reward + self.gamma * self.values[new_state]

                if (value is None) or (value < new_value):
                    best_action = action
                    value = new_value

            self.pi[state] = [{'action': best_action, 'pro': 1}]

    def policy_iterate(self):
        flag = True
        i = 0
        while flag:
            i += 1
            self.policy_evaluate()
            pi_old = self.pi.copy()
            self.policy_improve()

            if pi_old == self.pi:
                flag = False

        print('policy improve iter num: ', i)


if __name__ == "__main__":
    bird_env = BirdEnv()
    policy_value = DP_Policy_Iter(bird_env, init_policy='balance')
    policy_value.policy_iterate()

    flag = 1
    state = 0
    path = []
    bird_env.values = [round(x, 3) for x in policy_value.values]


    step_num = 0
    while flag:
        action = policy_value.pi[state]
        bird_env.current_position = bird_env.state_to_position(state)
        bird_env.render(path)
        time.sleep(0.5)
        step_num += 1

        new_state, reward, is_over = bird_env.transform(state, action[0]['action'])

        if is_over:
            flag = 0

        state = new_state
        bird_env.current_position = bird_env.state_to_position(state)
        path.append(state)
        bird_env.render(path)

    while True:
        bird_env.render(path)


