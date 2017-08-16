import numpy as np
import pygame
from skimage import io
from skimage import transform

pygame.font.init()

font = pygame.font.Font('./NotoSansHans-Regular.otf', 44)

with open('./data/hanzi.txt', 'r') as file:
    for char in map(str.strip, file.readlines()):
        try:
            img = pygame.Surface((48, 48))
            
            text = font.render(char, False, (255, 255, 255))

            img.blit(text, (2,2))

            arr = pygame.surfarray.array3d(img)
            arr = transform.rotate(arr, -90)

            io.imsave('./data/images/' + char + '.jpg', arr)
        except:
            pass