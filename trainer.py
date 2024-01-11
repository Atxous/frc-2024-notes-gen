import torch
from torch.optim import NAdam
from torch.utils.data import DataLoader
from model.unet import UNet
from model.diffusion_model import DiffusionModel
from sinusoidal_embedding import sinusoidal_embedding
from diffusion_schedules import offset_cosine_diffusion_schedule
from data import build_dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

EPOCHS = 50 # iterations
IMG_SIZE = (128, 128) # size of training imgs
CHANNELS = 128 # hyperparameter for conv layers channels
EMBEDDING_DIM = 64 # must be half of channels
EMBEDDING_FN = sinusoidal_embedding # embedding for noise
NUM_BLOCKS = 4 # down/upsampling blocks in u-net
NUM_RESIDUE = 3 # num of residue blocks at the bottom of u-net
DIFFUSION_SCHEDULE = offset_cosine_diffusion_schedule # adding noise scheduler
SAMPLES = 10 # generate n samples every 5 epochs
LOAD_MODEL = False # load existing parameters
ROOT = "" # data file

assert EMBEDDING_DIM + CHANNELS // 2 == CHANNELS, "embedding_dim must be half of channels"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
DATA = build_dataset(ROOT, 12, IMG_SIZE, device)

model = DiffusionModel(UNet(CHANNELS, IMG_SIZE, EMBEDDING_FN, EMBEDDING_DIM, NUM_BLOCKS, NUM_RESIDUE), DIFFUSION_SCHEDULE).to(device)
optimizer = NAdam(model.parameters(), lr = 0.001)
loss_fn = torch.nn.L1Loss()
if LOAD_MODEL:
    model.load_state_dict(torch.load("trained_model/diffusion_model.pt"))

for epoch in range(EPOCHS):
    with tqdm(DATA, position = 0, leave = True, unit = " batches") as tepoch: 
        for batch in tepoch:
            batch = batch.to(device)
            
            images = model.normalize_batch(batch) # from -1 to 1
            noise = torch.randn(batch.shape).to(device) # noise of mean 0 and std 1
            diffusion_times = torch.rand(batch.shape[0], 1, 1, 1).to(device)
            noise_rates, signal_rates = model.diffusion_schedule(diffusion_times)
            noisy_imgs = images * signal_rates + noise * noise_rates # add noise to images based on iteration
            
            pred_noises, pred_imgs = model(noisy_imgs, noise_rates, signal_rates, True) 
            loss = loss_fn(pred_noises, noise)
            loss.backward()
            optimizer.step()
            
            for weight, ema_weight in zip(model.cv_model.parameters(), model.ema_model.parameters()):
                ema_weight.data.mul_(0.999).add_(0.001 * weight.data) # exponential moving average often produces better imgs

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss.item())
            tepoch.update()
        torch.save(model.state_dict(), f"diffusion{epoch+3}.pt")
        if SAMPLES != 0 and epoch % 5 == 0:
            #showing n generated images
            generated_imgs = model.generate_imgs(SAMPLES, IMG_SIZE[0], IMG_SIZE[1], 20)
            rows = int(SAMPLES ** 0.5)
            cols = int(SAMPLES ** 0.5) + 1
            fig = plt.figure(figsize=(rows, cols))
            for i in range(SAMPLES):
                fig.add_subplot(rows, cols, i+1)
                plt.imshow(generated_imgs[i].cpu().permute(1, 2, 0))
            #save generated img
            plt.savefig(f"generated_imgs/epoch{epoch}.png")
    
        

    
    
        