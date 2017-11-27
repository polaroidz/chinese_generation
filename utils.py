from torch.autograd import Variable
from skimage import io
from skimage import color
import numpy as np
import torch
import glob

dataset = glob.glob('./data/images/*.jpg')

img_size = 48

def batch_generator(batch_size, nb_batches):
    batch_count = 0
    
    while True:
        pos = batch_count * batch_size
        batch = dataset[pos:pos+batch_size]
        
        X = np.zeros((batch_size, 1, img_size, img_size), dtype=np.float32)
        
        for k, path in enumerate(batch):
            im = io.imread(path)
            im = color.rgb2gray(im)
            
            X[k] = im[np.newaxis, ...]
    
        X = torch.from_numpy(X)
        X = Variable(X)
            
        yield X, batch

        batch_count += 1
        
        if batch_count > nb_batches:
            batch_count = 0
