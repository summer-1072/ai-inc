import pygame
import random
import numpy as np
from pygame.locals import *


class MazeEnv:
    def __init__(self):
        self.states = np.arange(0, 100, 1).reshape([10, 10])
        self.values = np.zeros(100).reshape([10, 10])
        self.actions = ['e', 's', 'w', 'n']
        self.gamma = 0.8

        self.screen = None
        self.bird = None
        self.wall = None
        self.destination = None
        self.textFont = None
        self.FPSClock = pygame.time.Clock()
        self.screen_size = (1200, 900)

        self.origin_position = [0, 0]
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

        flag = flag1 and flag2

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
        is_collision, is_destination = True, True
        while is_collision or is_destination:
            state = self.states[int(random.random() * len(self.states))]
            state_position = self.state_to_position(state)
            is_collision = self.collision_detection(state_position)
            is_destination = self.destination_detection(state_position)

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
            return self.position_to_state(next_position), -1, True

        if self.destination_detection(next_position):
            return self.position_to_state(next_position), 1, True

        return self.position_to_state(next_position), 0, False

    def game_over(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption('Maze')
            self.screen = pygame.display.set_mode(self.screen_size)

            bird = pygame.image.load("../pic/bird.jpg")
            self.bird = pygame.transform.scale(bird, (120, 90))

            wall = pygame.image.load("../pic/wall.jpg")
            self.wall = pygame.transform.scale(wall, (120, 90))

            destination = pygame.image.load("../pic/destination.jpg")
            self.destination = pygame.transform.scale(destination, (120, 90))

            self.textFont = pygame.font.SysFont('times', 20)

            # 绘制网格
            self.screen.fill((0, 180, 0))
            for i in range(11):
                pygame.draw.lines(self.screen, (255, 255, 255), True, ((120 * i, 0), (120 * i, 900)), 1)

            for i in range(11):
                pygame.draw.lines(self.screen, (255, 255, 255), True, ((0, 90 * i), (1200, 90 * i)), 1)

            # 绘制障碍物
            for i in range(8):
                self.screen.blit(self.wall, (self.obstacle1_x[i], self.obstacle1_y[i]))
                self.screen.blit(self.wall, (self.obstacle2_x[i], self.obstacle2_y[i]))

            # 绘制终点
            self.screen.blit(self.destination, self.destination_position)

        # 绘制鸟
        self.screen.blit(self.bird, self.current_position)

        # 绘制值函数
        for i in range(10):
            for j in range(10):
                surface = self.textFont.render(str(self.values[i][j]), True, (0, 0, 0))
                self.screen.blit(surface, (120 * i + 5, 90 * j + 2))

        pygame.display.update()
        self.game_over()
        self.FPSClock.tick(30)


if __name__ == "__main__":
    maze = MazeEnv()
    maze.render()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()