import torch.nn.functional as F
import torch

def kl_divergence(m, log_v):
    kl = -0.5 * torch.sum(1 + log_v - m.pow(2) - log_v.exp())
    return kl

def loss_function(reconstructed_image, images, m, log_v, beta):
    recon_loss = F.mse_loss(reconstructed_image, images, reduction='sum') / (reconstructed_image.shape[2] * reconstructed_image.shape[3])
    kl_loss = kl_divergence(m, log_v)
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss