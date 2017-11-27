import torch
from torch.autograd import Variable
from torch import nn
from skimage import color
from skimage import transform
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import utils
import torch
import vae
import pickle
import pygame
import sys, os
import time

np.random.seed(47)

with open('./data/latent_dataset.pickle', 'rb') as handle:
    dataset = pickle.load(handle)

def get_closest(z):
    closest = None
    distance = np.inf

    for k, v in enumerate(dataset['latent']):
        aux = np.linalg.norm(z - v)

        if aux < distance:
            closest = k
            distance = aux

    return closest

model = vae.VariationalAutoEncoder()

model.load_state_dict(torch.load('./weights/epoch.20.th'))

if __name__ == '__main__':
    pygame.init()
    
    mouse = pygame.mouse
    fpsClock = pygame.time.Clock()
    
    width = 192 * 2
    height = 192
    
    window = pygame.display.set_mode((width, height))

    canvas = pygame.Surface((192, 192))
    result = pygame.Surface((192, 192))
    
    pygame.display.set_caption('Paintme')
    
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    
    canvas.fill(BLACK)
    result.fill(BLACK)
    
    while True:
        left_pressed, middle_pressed, right_pressed = mouse.get_pressed()
    
        if left_pressed:
            pygame.draw.circle(canvas, WHITE, (pygame.mouse.get_pos()),7)
        elif right_pressed:
            canvas.fill(BLACK)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        window.fill(BLACK)
    
        window.blit(canvas, (0, 0))        
        window.blit(result, (192, 0))
        
        s_canvas = pygame.surfarray.array2d(canvas)
 
        if left_pressed:
            s_canvas = (s_canvas + s_canvas.min()) / s_canvas.max()
            s_canvas = transform.resize(s_canvas, (48, 48))
            s_canvas = s_canvas[np.newaxis, np.newaxis, ...]
    
            X = torch.FloatTensor(s_canvas)
            X = Variable(X)

            mu, sig = model.encoder(X)
    
            z = model.sample_latent(mu, sig)
            z = z.data.numpy()
    
            closest = get_closest(z)
            
            print(dataset['original'][closest])
            
            s_result = io.imread(dataset['original'][closest])
            s_result = transform.rescale(s_result, 4)
            s_result = transform.rotate(s_result, 90)
            s_result = pygame.surfarray.make_surface(s_result * 255)

            result.blit(s_result, (0, 0))

        pygame.display.update()


