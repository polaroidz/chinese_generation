import matplotlib.pyplot as plt
import numpy as np
import utils
import torch
import vae
import pickle

np.random.seed(47)

batch_size = 12
nb_batches = 600

if __name__ == '__main__':
    model = vae.VariationalAutoEncoder()
    
    model.load_state_dict(torch.load('./weights/epoch.20.th'))
    
    ds = {
       'original': [],
       'latent': []
    }
    
    gen = utils.batch_generator(batch_size, nb_batches)
    
    for i in range(nb_batches):
        X, path = next(gen)
        mu, sig = model.encoder(X)
        z = model.sample_latent(mu, sig)
        z = z.data.numpy()
    
        print('processing batch: ' + str(i))
    
        for ii in range(batch_size):
            ds['original'].append(path[ii])
            ds['latent'].append(z[ii])

    with open('./data/latent_dataset.pickle', 'wb') as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
