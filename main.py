import numpy as np
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import cv2 as cv
from tqdm import tqdm
import os
from sampler import *

def train_Unet_DDPM(epochs, lr, MODEL_PATH, data_path, betas = (0.9, 0.99), chkpt = 100):

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    training_images = torch.tensor(np.load(data_path))                                  # images are normalized from 0 to 1

    model = Unet(                                                                       # Standard UNet model.  
        dim = training_images.shape[-2],
        dim_mults = (1, 2),
        flash_attn = True, 
        channels=2
    )

    diffusion = GaussianDiffusion(                                                      # Using Gaussian Distribution for implementing DDPM training paradigm. 
        model,
        image_size = (training_images.shape[-2], training_images.shape[-1]),
        timesteps = 1000    # number of steps
    )

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)                    # Using Adam optimizer to optimize UNet model. 

    history = []
    for epoch in tqdm(range(1, epochs+1)):
        optim.zero_grad()
        loss = diffusion(training_images)
        loss.backward()
        optim.step()
        
        if epoch%chkpt==0:
            history.append(loss.item())
            torch.save({                                                                # Saving checkpoints for UNet model. 
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss},  
                os.path.join(MODEL_PATH, f'unet_{epoch}.pt'))       

    np.save(os.path.join(MODEL_PATH, 'history.npy'), np.array(history))                 # Saving loss history for training. 
    print(F"Model trained. \nCheckpoints and history saved at: {MODEL_PATH}")

def sample_Unet_DDPM(model_path, output_path, image_size = 64, batch_size = 1):         # Generate Motion from trained UNet model.
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    model = Unet(
        dim = image_size,
        dim_mults = (1, 2),
        flash_attn = True, 
        channels=2
    )
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    diffusion = GaussianDiffusion(model, image_size=(image_size, 20), timesteps=1000)
    output = diffusion.sample(batch_size = batch_size)                                  # Generating sample images. 
    output = output.reshape(batch_size, image_size, 40)
    meta_data, cols = get_meta_data()
    for n, image in enumerate(output):
        data = {col: [] for col in cols}
        for row in image:
            for i, col in enumerate(cols):
                x = float(row[i].detach().numpy())
                y = float(row[i+20].detach().numpy())
                x = meta_data[col+'_x']['diff']*x + meta_data[col+'_x']['min']
                y = meta_data[col+'_y']['diff']*y + meta_data[col+'_y']['min']
                data[col].append([x, y])
        with open(os.path.join(output_path, f'sample_{n}.json'), 'w+') as file:
            json.dump(data, file)                                                       # Saving .json file of generated motion. 
    


if __name__=='__main__':
    train_Unet_DDPM(1000, 8e-6, 'model_trial', 'dataset/images.npy', chkpt=100)           # Example of training UNet. 