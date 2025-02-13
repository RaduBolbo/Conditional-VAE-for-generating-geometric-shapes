import torch.nn.functional as F
import torch
import torch.nn as nn
from pytorch_msssim import ssim

def kl_divergence(m, log_v):
    kl = -0.5 * torch.sum(1 + log_v - m.pow(2) - log_v.exp())
    return kl

def extract_vgg_features(image, vgg, selected_layers=[4, 9, 18, 27]):
    features = []
    for i, layer in enumerate(vgg):
        image = layer(image)
        if i in selected_layers:
            features.append(image)
    return features

def compute_vgg_loss(generated_image, target_image, vgg, selected_layers=[4, 9, 11]):
    gen_features = extract_vgg_features(generated_image, vgg, selected_layers)
    target_features = extract_vgg_features(target_image, vgg, selected_layers)
    
    loss = 0
    for gen_feat, target_feat in zip(gen_features, target_features):
        loss += torch.nn.functional.l1_loss(gen_feat, target_feat)
    
    return loss

class DiceScoreLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super(DiceScoreLoss, self).__init__()
        self.threshold = threshold

    def dice(self, reconstructed, target_bin, reconstructed_bin, dice_scores):
        for i in range(reconstructed.shape[1]):  # Iterate over the RGB channels
            reconstructed_channel = reconstructed_bin[:, i, :, :]
            target_channel = target_bin[:, i, :, :]

            intersection = (reconstructed_channel * target_channel).sum(dim=(1, 2))
            union = reconstructed_channel.sum(dim=(1, 2)) + target_channel.sum(dim=(1, 2))
            
            dice_score = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice_score)

        avg_dice_score = torch.mean(torch.stack(dice_scores), dim=0)
        return avg_dice_score, dice_scores

    def forward(self, reconstructed, target):
        dice_scores = []

        reconstructed_bin = (reconstructed > self.threshold).float()
        target_bin = (target > self.threshold).float()
        avg_dice_score, dice_scores = self.dice(reconstructed, target_bin, reconstructed_bin, dice_scores)

        reconstructed_bin = (reconstructed < self.threshold).float()
        target_bin = (target < self.threshold).float()
        avg_dice_score, dice_scores = self.dice(reconstructed, target_bin, reconstructed_bin, dice_scores)
        
        dice_loss = 1 - avg_dice_score.mean()

        return dice_loss

def loss_function(reconstructed_image, original_image, m, log_v, beta, vgg, algo='ssim'):

    '''
    # if algo == 'mse':
    #     recon_loss = F.mse_loss(reconstructed_image, original_image, reduction='mean')
    # elif algo == 'ssim':
    #     recon_loss = 1 - ssim(reconstructed_image, original_image, data_range=1.0, size_average=True)
    # else:
    #     raise ValueError(f"Unknown reconstruction loss function: {algo}. Use 'mse' or 'ssim'.")
    #mse_loss = F.mse_loss(reconstructed_image, original_image, reduction='mean')
    #mse_loss = F.mse_loss(reconstructed_image, original_image, reduction='mean')
    #print('mse_loss: ', mse_loss)
    l1_loss = F.l1_loss(reconstructed_image, original_image, reduction='mean')
    #print('l1_loss: ', l1_loss)
    ssim_loss = 1 - ssim(reconstructed_image, original_image, data_range=1.0, size_average=True)
    #print('ssim_loss: ', ssim_loss)
    vgg_loss = compute_vgg_loss(reconstructed_image, original_image, vgg)
    #print('vgg_loss: ', vgg_loss)
    #recon_loss = l1_loss + ssim_loss + vgg_loss

    # dice score_component:
    dice_loss_computer = DiceScoreLoss(0.5)
    dice_loss = dice_loss_computer(reconstructed_image, original_image)
    #print('dice_loss: ', dice_loss)

    recon_loss = l1_loss + ssim_loss + vgg_loss + dice_loss
    '''
    criterion = nn.CrossEntropyLoss()
    ce_loss = criterion(reconstructed_image, original_image)

    final_layer = F.sigmoid
    reconstructed_image_sigmoid = final_layer(reconstructed_image)
    l1_loss = F.l1_loss(reconstructed_image_sigmoid, original_image, reduction='mean')
    ssim_loss = 1 - ssim(reconstructed_image, original_image, data_range=1.0, size_average=True)
    dice_loss_computer = DiceScoreLoss(0.5)
    #dice_loss = dice_loss_computer(reconstructed_image, original_image)
    #print('ssim_loss: ', ssim_loss)
    #print(recon_loss)

    kl_loss = -0.5 * torch.mean(1 + log_v - m.pow(2) - log_v.exp())
    recon_loss = ce_loss + l1_loss + ssim_loss
    # print(ce_loss)
    # print(l1_loss)
    # print(ssim_loss)
    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss