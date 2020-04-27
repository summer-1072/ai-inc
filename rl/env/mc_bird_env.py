import time
import numpy as np
import pygame
from rl.env.bird_env import BirdEnv


class MCBirdEnv(BirdEnv):
    def init_bird(self):
        self.states = [i for i in range(100)]
        self.values = [0 for i in range(100)]
        self.actions = ['e', 's', 'w', 'n']
        self.q_values = np.zeros((100, 4))
        self.n = np.zeros((100, 4))
        self.gamma = 0.9

    def transform(self, state, action):
        current_position = self.state_to_position(state)
        next_position = [0, 0]
        is_collision = self.collision_detection(current_position)
        is_destination = self.destination_detection(current_position)

        if is_collision or is_destination:
            return state, 0, True

        if action == 'e':
            next_position[0] = current_position[0] + 120
            next_position[1] = current_position[1]
        if action == 's':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] + 90
        if action == 'w':
            next_position[0] = current_position[0] - 120
            next_position[1] = current_position[1]
        if action == 'n':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] - 90

        if self.collision_detection(next_position):
            return self.position_to_state(current_position), -10, True

        if self.destination_detection(next_position):
            return self.position_to_state(next_position), 100, True

        return self.position_to_state(next_position), -1, False

    def render(self, path):
        self.init_render()
        # 绘制值函数
        for i in range(100):
            x = int(i / 10)
            y = i % 10
            surface = self.textFont.render(str(self.q_values[i, 0]), True, (0, 0, 0))
            self.screen.blit(surface, (120 * y + 85, 90 * x + 45))
            surface = self.textFont.render(str(self.q_values[i, 1]), True, (0, 0, 0))
            self.screen.blit(surface, (120 * y + 50, 90 * x + 75))
            surface = self.textFont.render(str(self.q_values[i, 2]), True, (0, 0, 0))
            self.screen.blit(surface, (120 * y + 10, 90 * x + 45))
            surface = self.textFont.render(str(self.q_values[i, 3]), True, (0, 0, 0))
            self.screen.blit(surface, (120 * y + 50, 90 * x + 5))

        for i in range(len(path)):
            # 绘制鸟
            state = path[i]
            self.current_position = self.state_to_position(state)
            self.screen.blit(self.bird, self.current_position)

            # 绘制红框、路径编号
            pygame.draw.rect(self.screen, [255, 0, 0], [self.current_position[0], self.current_position[1], 120, 90], 2)

            surface = self.textFont.render(str(i), True, (255, 0, 0))
            self.screen.blit(surface, (self.current_position[0] + 5, self.current_position[1] + 5))

            # 绘制值函数
            x = int(path[i] / 10)
            y = path[i] % 10
            surface = self.textFont.render(str(self.q_values[path[i], 0]), True, (0, 0, 0))
            self.screen.blit(surface, (120 * y + 85, 90 * x + 45))
            surface = self.textFont.render(str(self.q_values[path[i], 1]), True, (0, 0, 0))
            self.screen.blit(surface, (120 * y + 50, 90 * x + 75))
            surface = self.textFont.render(str(self.q_values[path[i], 2]), True, (0, 0, 0))
            self.screen.blit(surface, (120 * y + 10, 90 * x + 45))
            surface = self.textFont.render(str(self.q_values[path[i], 3]), True, (0, 0, 0))
            self.screen.blit(surface, (120 * y + 50, 90 * x + 5))

            # 清理上一步
            if i >= 1:
                last_state = path[i - 1]
                last_position = self.state_to_position(last_state)

                # 绿矩形和红框填充
                pygame.draw.rect(self.screen, [0, 180, 0], [last_position[0], last_position[1], 120, 90], 0)
                pygame.draw.rect(self.screen, [255, 0, 0], [last_position[0], last_position[1], 120, 90], 2)

                # 绘制上一步编号
                surface = self.textFont.render(str(i - 1), True, (255, 0, 0))
                self.screen.blit(surface, (last_position[0] + 5, last_position[1] + 5))

                # 绘制上一步值函数
                x = int(last_state / 10)
                y = last_state % 10
                surface = self.textFont.render(str(self.q_values[path[i - 1], 0]), True, (0, 0, 0))
                self.screen.blit(surface, (120 * y + 85, 90 * x + 45))
                surface = self.textFont.render(str(self.q_values[path[i - 1], 1]), True, (0, 0, 0))
                self.screen.blit(surface, (120 * y + 50, 90 * x + 75))
                surface = self.textFont.render(str(self.q_values[path[i - 1], 2]), True, (0, 0, 0))
                self.screen.blit(surface, (120 * y + 10, 90 * x + 45))
                surface = self.textFont.render(str(self.q_values[path[i - 1], 3]), True, (0, 0, 0))
                self.screen.blit(surface, (120 * y + 50, 90 * x + 5))

            time.sleep(0.5)
            pygame.time.Clock().tick(30)
            pygame.display.update()
            self.game_over()

        while True:
            self.game_over()
