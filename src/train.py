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
from networks.cvae_bn_one_small_smallerlatent_deeper import CVAE
#from networks.cvae_bn_one_small import CVAE
#from networks.cvae_bn_one_old import CVAE
import numpy as np
from loss import loss_function
from tqdm import tqdm
import torchvision.models as models
vgg11 = models.vgg11(pretrained=True).features
for param in vgg11.parameters():
    param.requires_grad = False
vgg11.to('cuda')
vgg11.eval()

seed = 42
torch.manual_seed(seed)


device  ='cuda'
model = CVAE(latent_dim=12) # latent_dim was originally 32 (bad results), then lowered to 16 (better results) and then to 8 (?)
# LOAD STATE DICT IF NEEDED
#model.load_state_dict(torch.load('weights/model_weights_epoch_0_almost_works_small_PRELU_beta_1.pth'))
model.load_state_dict(torch.load('weights/model_weights_epoch_42.pth'))
num_epochs = 300
#beta = 1
#beta = 0.1
beta = 1
#lr = 0.000001
#lr = 0.001 # THIS WAS ALMOST WORKING
lr = 0.001
batch_size = 32

dataset = ShapeDataset('dataset_one', 'dataset_one.json')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True)


model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    torch.cuda.empty_cache()
    for batch in tqdm(train_dataloader):
        images, padded_conditioning_vectors, lengths = batch
        
        images = torch.tensor(np.transpose(images, (0, 3, 1, 2)).contiguous(), dtype=torch.float32).to(device)
        padded_conditioning_vectors = torch.tensor(padded_conditioning_vectors, dtype=torch.float32).to(device)
        lengths = lengths.to(device)
        #print(padded_conditioning_vectors)
        
        reconstructed_image, m, log_v = model(images, padded_conditioning_vectors, lengths)
        
        loss, recon_loss, kl_loss = loss_function(reconstructed_image, images, m, log_v, beta, vgg11, algo='ssim')
        epoch_loss += loss
        epoch_recon_loss += recon_loss
        epoch_kl_loss += kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"TRAIN Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss.item()/(len(dataset)*batch_size)}, Recon Loss: {epoch_recon_loss.item()/(len(dataset)*batch_size)}, KL Loss: {epoch_kl_loss.item()/(len(dataset)*batch_size)}")
    
    # if epoch % 5 == 0:
    #     torch.save(model.state_dict(), f'weights/model_weights_epoch_{epoch}.pth')
    torch.save(model.state_dict(), f'weights/model_weights_epoch_{epoch}.pth')

    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        for batch in tqdm(val_dataloader):
            images, padded_conditioning_vectors, lengths = batch
            
            images = torch.tensor(np.transpose(images, (0, 3, 1, 2)).contiguous(), dtype=torch.float32).to(device)
            padded_conditioning_vectors = torch.tensor(padded_conditioning_vectors, dtype=torch.float32).to(device)
            lengths = lengths.to(device)
            
            reconstructed_image, m, log_v = model(images, padded_conditioning_vectors, lengths)
            
            loss, recon_loss, kl_loss = loss_function(reconstructed_image, images, m, log_v, beta, vgg11, algo='ssim')
            epoch_loss += loss
            epoch_recon_loss += recon_loss
            epoch_kl_loss += kl_loss

    print(f"EVAL Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss.item()/(len(dataset)*batch_size)}, Recon Loss: {epoch_recon_loss.item()/(len(dataset)*batch_size)}, KL Loss: {epoch_kl_loss.item()/(len(dataset)*batch_size)}")
    
