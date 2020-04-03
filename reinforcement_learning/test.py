import pygame
from pygame.locals import *

pygame.init()
pygame.display.set_caption('GAME ON!')
screen = pygame.display.set_mode((600, 500))

pos_x = 100
pos_y = 100

velx = 1
vely = 1

bird = pygame.image.load("./mdp/pic/bird.jpg")
width, height = bird.get_size()
pic = pygame.transform.scale(bird, (120, 90))


while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()

    screen.fill((0, 180, 0))
    pos_x += velx
    pos_y += vely
    position = pos_x, pos_y, 50, 50

    if pos_x > 480 or pos_x < 0:
        velx = -velx
    if pos_y > 410 or pos_y < 0:
        vely = -vely

    screen.blit(pic, position)

    # pygame.draw.rect(screen, (255, 0, 0), position, 0)
    pygame.draw.lines(screen, (255, 255, 255), True, ((120 , 0), (120 , 900)), 1)

    pygame.draw.lines(screen, (255, 255, 255), True, ((0, 90), (1200, 90)), 1)

    pygame.display.update()
    pygame.time.delay(1)
