import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.condition_embeddings import ShapeEmbeddingRNN
from dataset import ShapeDataset, custom_collate_fn


class Encoder(nn.Module):
    def __init__(self, condition_embedding_size=6, latent_dim=32):
        super().__init__()
        self.condition_embedding_size = condition_embedding_size
        self.latent_dim = latent_dim

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'), 
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'), 
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),
        )
        self.linear_encoder = nn.Sequential(
            #nn.Linear(256 * 8 * 8 + condition_embedding_size, 512), # ****  de ce nu sunt bune dimensiunile?
            nn.Linear(4102, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
        )

        self.fc_m = nn.Linear(64, latent_dim)  # Mean
        self.fc_log_v = nn.Linear(64, latent_dim)  # Log variance

    def forward(self, image, condition):
        # 1) Pass the image through the conv encoder
        x = self.conv_encoder(image)

        # 2) Concatenate the image feature with the condition
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = torch.cat([x, condition], dim=-1)

        # 3) Pass the image data + conditioning through the linear encoding to mak the data to a distribution
        x = self.linear_encoder(x)
        m = self.fc_m(x)
        log_v = self.fc_log_v(x)

        return m, log_v


class Decoder(nn.Module):
    def __init__(self, condition_embedding_size=6, latent_dim=32):
        super().__init__()
        self.condition_embedding_size = condition_embedding_size
        self.latent_dim = latent_dim
        self.final_layer = F.sigmoid

        self.conv_decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), # 8x8 -> 16x16
        nn.BatchNorm2d(64),
        nn.PReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 16x16 -> 32x32
        nn.BatchNorm2d(32),
        nn.PReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),   # 32x32 -> 64x64
        nn.BatchNorm2d(32),
        nn.PReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),    # 64x64 -> 128x128
        nn.BatchNorm2d(3),
        nn.PReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.linear_decoder = nn.Sequential(
            #nn.Linear(latent_dim + condition_embedding_size, 512), # **** why it doesn tworK?
            nn.Linear(38, 128), # **** why it doesn tworK?
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.PReLU(),
        )

    def forward(self, z, condition):
        # 1) concatenate condition and the samples
        z = torch.cat([z, condition], dim=1)

        # 2) Pass the latent space through the linear decoder
        x = self.linear_decoder(z)

        # 3) reshape the lienar features to the propper 2D image shape
        x = x.view(x.size(0), 128, 8, 8)

        # 4) Pass the latent space ythrough the decoder
        x = self.conv_decoder(x)

        # 5) Pass the data trhough the final layer to obtain a normalized image
        #x = self.final_layer(x)

        return x
        

class CVAE(nn.Module):

    def __init__(self, shape_vocab_size=4, color_vocab_size=4, condition_embedding_size=6, hidden_size=16, latent_dim=32, device='cuda'):
        super().__init__()

        #self.conditional_embedder = ShapeEmbeddingRNN(shape_vocab_size, color_vocab_size, condition_embedding_size, hidden_size)
        #self.conditional_embedder = ShapeEmbeddingRNN(hidden_size=16)

        self.encoder = Encoder(condition_embedding_size, latent_dim)
        self.decoder = Decoder(condition_embedding_size, latent_dim)
        self.device = device

    def one_hot_embeddings(self, tensor):
        one_hot_mapping = {
            0: [0, 0, 0],
            1: [1, 0, 0],
            2: [0, 1, 0],
            3: [0, 0, 1]
        }
        
        transformed_batch = []
        for element in tensor:
            first_part = one_hot_mapping[int(element[0])]
            second_part = one_hot_mapping[int(element[1])]
            
            transformed_element = first_part + second_part
            transformed_batch.append(transformed_element)
        
        return torch.tensor(transformed_batch, dtype=torch.float32)

    def forward(self, image, conditioning_vector, lengths):

        # 1) Get the conditioning vector
        #conditioning_embedding = self.conditional_embedder(conditioning_vector)
        #conditioning_embedding = torch.zeros(conditioning_vector.shape)
        conditioning_embedding = self.one_hot_embeddings(conditioning_vector).to('cuda')
        #print(conditioning_embedding)

        # 1) Pass the image through the encoder and obtain mean and variance
        m, log_v = self.encoder(image, conditioning_embedding)
        
        # 2)
        std = torch.exp(0.5 * log_v) # obtain the standard deviation
        eps = torch.rand_like(std) # here is the random component for sampling
        z = m + eps * std # sample from distribution N(m, std)

        # 3) Pass data through decoder
        reconstructed_image = self.decoder(z, conditioning_embedding)

        return reconstructed_image, m, log_v
    
    def decode(self, conditioning_vector):

        z = torch.randn(1, 32).to(self.device)

        conditioning_embedding = self.one_hot_embeddings(conditioning_vector).to('cuda')

        reconstructed_image = self.decoder(z, conditioning_embedding.view(1, 6))

        return reconstructed_image


if __name__ == '__main__':          
    model = CVAE()
    dataset = ShapeDataset('dataset', 'dataset.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)

    for images, padded_conditioning_vectors in dataloader:
        reconstructed_image, conditioning_vector = model(images, padded_conditioning_vectors)
        reconstructed_image = model()
        