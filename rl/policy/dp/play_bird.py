import random
from rl.env.bird_env import BirdEnv


class DP_Iter:
    def __init__(self, birdenv, init_policy='random'):
        self.birdenv = birdenv
        self.states = birdenv.states
        self.values = birdenv.values
        self.actions = birdenv.actions
        self.gamma = birdenv.gamma
        self.pi = {}

        for state in self.states:
            flag1 = birdenv.collision_detection(birdenv.state_to_position(state))
            flag2 = birdenv.destination_detection(birdenv.state_to_position(state))
            if flag1 or flag2: continue

            if init_policy == 'random':
                self.pi[state] = [{'action': self.actions[int(random.random() * len(self.actions))], 'pro': 1}]

            elif init_policy == 'balance':
                self.pi[state] = [{'action': action, 'pro': 1 / len(self.actions)} for action in self.actions]

    # 策略评估
    def policy_evaluate(self):
        flag = True
        i = 0
        while flag:
            delta = 0
            i += 1
            for state in self.states:
                flag1 = self.birdenv.collision_detection(birdenv.state_to_position(state))
                flag2 = self.birdenv.destination_detection(birdenv.state_to_position(state))
                if flag1 or flag2: continue

                new_value = 0
                step_actions = self.pi[state]
                for step_action in step_actions:
                    action = step_action['action']
                    pro = step_action['pro']
                    next_state, reward, is_over = self.birdenv.transform(state, action)
                    new_value += pro * (reward + self.gamma * self.values[next_state])

                delta += abs(new_value - self.values[state])
                self.values[state] = new_value

            if delta < 1e-6:
                flag = False

        print('policy evaluate iter num: ', i)

    # 策略改进
    def policy_improve(self):
        for state in self.states:
            flag1 = self.birdenv.collision_detection(self.birdenv.state_to_position(state))
            flag2 = self.birdenv.destination_detection(self.birdenv.state_to_position(state))
            if flag1 or flag2: continue

            best_action = None
            value = None
            for action in self.actions:
                next_state, reward, is_over = self.birdenv.transform(state, action)
                new_value = reward + self.gamma * self.values[next_state]

                if (value is None) or (value < new_value):
                    best_action = action
                    value = new_value

            self.pi[state] = [{'action': best_action, 'pro': 1}]

    # 策略迭代
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

    # 值迭代
    def value_iterate(self):
        flag = True
        i = 0
        while flag:
            i += 1
            for state in self.states:
                flag1 = self.birdenv.collision_detection(self.birdenv.state_to_position(state))
                flag2 = self.birdenv.destination_detection(self.birdenv.state_to_position(state))
                if flag1 or flag2: continue

                new_value = 0
                step_actions = self.pi[state]
                for step_action in step_actions:
                    action = step_action['action']
                    pro = step_action['pro']
                    next_state, reward, is_over = self.birdenv.transform(state, action)
                    new_value += pro * (reward + self.gamma * self.values[next_state])

                self.values[state] = new_value

            delta = 0
            for state in self.states:
                flag1 = self.birdenv.collision_detection(self.birdenv.state_to_position(state))
                flag2 = self.birdenv.destination_detection(self.birdenv.state_to_position(state))
                if flag1 or flag2: continue

                best_action = None
                value = None
                for action in self.actions:
                    next_state, reward, is_over = self.birdenv.transform(state, action)
                    new_value = reward + self.gamma * self.values[next_state]

                    if (value is None) or (value < new_value):
                        best_action = action
                        value = new_value

                delta += abs(value - self.values[state])
                self.pi[state] = [{'action': best_action, 'pro': 1}]

            if delta < 1e-6:
                flag = False

        print('value iter num: ', i)


if __name__ == "__main__":
    birdenv = BirdEnv()
    policy_value = DP_Iter(birdenv, init_policy='balance')

    # 通过策略迭代改进策略
    # policy_value.policy_iterate()

    # 通过值迭代改进策略
    policy_value.value_iterate()

    flag = True
    state = 0
    path = [state]
    birdenv.values = [round(x, 3) for x in policy_value.values]

    while flag:
        action = policy_value.pi[state]
        next_state, reward, is_over = birdenv.transform(state, action[0]['action'])

        if is_over:
            flag = False

        state = next_state
        path.append(state)

    print('path:', path)
    birdenv.render(path)
