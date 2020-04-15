import pygame
import random
from pygame.locals import *
import time


class BirdEnv:
    def __init__(self):
        self.states = [i for i in range(100)]
        self.values = [0 for i in range(100)]
        self.actions = ['e', 's', 'w', 'n']
        self.gamma = 0.8

        self.screen_size = (1200, 900)
        self.current_position = [0, 0]
        self.destination_position = [1080, 0]
        self.limit_distance_x = 120
        self.limit_distance_y = 90
        self.obstacle_size = [120, 90]

        self.obstacle1_x = []
        self.obstacle1_y = []

        self.obstacle2_x = []
        self.obstacle2_y = []

        for i in range(8):
            self.obstacle1_x.append(360)
            if i <= 3:
                self.obstacle1_y.append(90 * i)
            else:
                self.obstacle1_y.append(90 * (i + 2))

        for i in range(8):
            self.obstacle2_x.append(720)
            if i <= 4:
                self.obstacle2_y.append(90 * i)
            else:
                self.obstacle2_y.append(90 * (i + 2))

    def collision_detection(self, position):
        flag, flag1, flag2 = True, True, True

        dx1, dy1 = [], []
        for i in range(8):
            diff_x = abs(self.obstacle1_x[i] - position[0])
            diff_y = abs(self.obstacle1_y[i] - position[1])
            dx1.append(diff_x)
            dy1.append(diff_y)

        min_diff_x = min(dx1)
        min_diff_y = min(dy1)
        if (min_diff_x >= self.limit_distance_x) or (min_diff_y >= self.limit_distance_y):
            flag1 = False

        dx2, dy2 = [], []
        for i in range(8):
            diff_x = abs(self.obstacle2_x[i] - position[0])
            diff_y = abs(self.obstacle2_y[i] - position[1])
            dx2.append(diff_x)
            dy2.append(diff_y)

        min_diff_x = min(dx2)
        min_diff_y = min(dy2)
        if (min_diff_x >= self.limit_distance_x) or (min_diff_y >= self.limit_distance_y):
            flag2 = False

        flag = flag1 or flag2

        if (position[0] < 0 or position[0] > 1080) or (position[1] < 0 or position[1] > 810):
            flag = True

        return flag

    def destination_detection(self, position):
        flag = False
        if abs(position[0] - self.destination_position[0]) < 0.1 and abs(
                position[1] - self.destination_position[1]) < 0.1:
            flag = True

        return flag

    def state_to_position(self, state):
        x = int(state / 10)
        y = state % 10
        position = [120 * y, 90 * x]

        return position

    def position_to_state(self, position):
        y = position[0] / 120
        x = position[1] / 90
        state = int(10 * x + y)

        return state

    def reset(self):
        state = 0
        is_collision, is_destination = True, True
        while is_collision or is_destination:
            state = self.states[int(random.random() * len(self.states))]
            position = self.state_to_position(state)
            is_collision = self.collision_detection(position)
            is_destination = self.destination_detection(position)

        return state

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
            return self.position_to_state(current_position), -1, True

        if self.destination_detection(next_position):
            return self.position_to_state(next_position), 1, True

        return self.position_to_state(next_position), 0, False

    def game_over(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

    def render(self, path):
        pygame.init()
        pygame.display.set_caption('Bird')
        screen = pygame.display.set_mode(self.screen_size)

        bird = pygame.image.load("../pic/bird.jpg")
        bird = pygame.transform.scale(bird, (120, 90))

        wall = pygame.image.load("../pic/wall.jpg")
        wall = pygame.transform.scale(wall, (120, 90))

        destination = pygame.image.load("../pic/destination.jpg")
        destination = pygame.transform.scale(destination, (120, 90))

        textFont = pygame.font.SysFont('times', 20)

        # 绘制网格
        screen.fill((0, 180, 0))
        for i in range(11):
            pygame.draw.lines(screen, (255, 255, 255), True, ((120 * i, 0), (120 * i, 900)), 1)
            pygame.draw.lines(screen, (255, 255, 255), True, ((0, 90 * i), (1200, 90 * i)), 1)

        # 绘制障碍物
        for i in range(8):
            screen.blit(wall, (self.obstacle1_x[i], self.obstacle1_y[i]))
            screen.blit(wall, (self.obstacle2_x[i], self.obstacle2_y[i]))

        # 绘制终点
        screen.blit(destination, self.destination_position)

        # 绘制值函数
        for i in range(100):
            x = int(i / 10)
            y = i % 10
            surface = textFont.render(str(self.values[i]), True, (0, 0, 0))
            screen.blit(surface, (120 * y + 5, 90 * x + 75))

        for i in range(len(path)):
            # 绘制鸟
            state = path[i]
            self.current_position = self.state_to_position(state)
            screen.blit(bird, self.current_position)

            # 绘制红框、路径编号
            pygame.draw.rect(screen, [255, 0, 0], [self.current_position[0], self.current_position[1], 120, 90], 2)

            surface = textFont.render(str(i), True, (255, 0, 0))
            screen.blit(surface, (self.current_position[0] + 5, self.current_position[1] + 5))

            # 绘制值函数
            surface = textFont.render(str(self.values[path[i]]), True, (0, 0, 0))
            x = int(path[i] / 10)
            y = path[i] % 10
            screen.blit(surface, (120 * y + 5, 90 * x + 75))

            # 清理上一步
            if i >= 1:
                last_state = path[i - 1]
                last_position = self.state_to_position(last_state)

                # 绿矩形和红框填充
                pygame.draw.rect(screen, [0, 180, 0], [last_position[0], last_position[1], 120, 90], 0)
                pygame.draw.rect(screen, [255, 0, 0], [last_position[0], last_position[1], 120, 90], 2)

                # 绘制上一步编号
                surface = textFont.render(str(i - 1), True, (255, 0, 0))
                screen.blit(surface, (last_position[0] + 5, last_position[1] + 5))

                # 绘制上一步值函数
                surface = textFont.render(str(self.values[last_state]), True, (0, 0, 0))
                x = int(last_state / 10)
                y = last_state % 10
                screen.blit(surface, (120 * y + 5, 90 * x + 75))

            time.sleep(0.5)
            pygame.time.Clock().tick(30)
            pygame.display.update()
            self.game_over()

        while True:
            self.game_over()
