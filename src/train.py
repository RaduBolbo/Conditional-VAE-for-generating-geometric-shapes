# **** TO DO:
# 0. MOST IMPORTANT; add a method to be able to perform inference without the envcoder -> and add a sample method
# 1. Add Batch Norm
# 2. for the embedder add msking
# 3. adjust lr
# 4. adjust network size
# 5. Can I get rid of the linear layers?

import torch
from torch.utils.data import random_split
from dataset import ShapeDataset, custom_collate_fn
from networks.cvae import CVAE
import numpy as np
from loss import loss_function
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)

device  ='cuda'
model = CVAE()
num_epochs = 10
beta = 5
batch_size = 32

dataset = ShapeDataset('dataset', 'dataset.json')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True)


model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    for batch in tqdm(train_dataloader):
        images, padded_conditioning_vectors, lengths = batch
        
        images = torch.tensor(np.transpose(images, (0, 3, 1, 2)).contiguous(), dtype=torch.float32).to(device)
        padded_conditioning_vectors = torch.tensor(padded_conditioning_vectors, dtype=torch.float32).to(device)
        lengths = lengths.to(device)
        
        reconstructed_image, m, log_v = model(images, padded_conditioning_vectors, lengths)
        
        loss, recon_loss, kl_loss = loss_function(reconstructed_image, images, m, log_v, beta)
        epoch_loss += loss
        epoch_recon_loss += recon_loss
        epoch_kl_loss += kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"TRAIN Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss.item()/(len(dataset)*batch_size)}, Recon Loss: {epoch_recon_loss.item()/(len(dataset)*batch_size)}, KL Loss: {epoch_kl_loss.item()/(len(dataset)*batch_size)}")
    

    torch.save(model.state_dict(), f'weights/model_weights_epoch_{epoch}.pth')

    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    model.eval()
    for batch in tqdm(val_dataloader):
        images, padded_conditioning_vectors, lengths = batch
        
        images = torch.tensor(np.transpose(images, (0, 3, 1, 2)).contiguous(), dtype=torch.float32).to(device)
        padded_conditioning_vectors = torch.tensor(padded_conditioning_vectors, dtype=torch.float32).to(device)
        lengths = lengths.to(device)
        
        reconstructed_image, m, log_v = model(images, padded_conditioning_vectors, lengths)
        
        loss, recon_loss, kl_loss = loss_function(reconstructed_image, images, m, log_v, beta)
        epoch_loss += loss
        epoch_recon_loss += recon_loss
        epoch_kl_loss += kl_loss

    print(f"EVAL Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss.item()/(len(dataset)*batch_size)}, Recon Loss: {epoch_recon_loss.item()/(len(dataset)*batch_size)}, KL Loss: {epoch_kl_loss.item()/(len(dataset)*batch_size)}")
    
