import torch
import cv2
#from networks.cvae_bn_one_old import CVAE
#from networks.cvae_bn_one_small import CVAE
#from networks.cvae_bn_one_small_smallerlatent import CVAE
from networks.cvae_bn_one_small_smallerlatent_deeper import CVAE
import numpy as np
import torch.nn.functional as F

seed = 48 # to try: 42, 43, 48 (INTERESANT la triunghi + confuzie hexagon-cerc)
# bad seeds: 41, 44 (e interesant -> fragmentari), 45, 47
torch.manual_seed(seed)

#padded_conditioning_vectors = [0, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
#padded_conditioning_vectors = [0, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
#padded_conditioning_vectors = [0, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
#padded_conditioning_vectors = [2, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1]
#padded_conditioning_vectors = [0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

# seed to try: 
#padded_conditioning_vectors = [[0, 0]] # red square
#padded_conditioning_vectors = [[0, 1]] # blue square
#padded_conditioning_vectors = [[0, 2]] # green square
#padded_conditioning_vectors = [[0, 3]] # yellow square

# seed to try: 
#padded_conditioning_vectors = [[0, 1]] # blue square
#padded_conditioning_vectors = [[1, 1]] # blue circle
#padded_conditioning_vectors = [[2, 1]] # blue triangle
#padded_conditioning_vectors = [[3, 1]] # blue hexagon


device = 'cuda'

model = CVAE(latent_dim=12).to(device)
# ********* MODEL LINK: https://drive.google.com/file/d/1L0Td_hLvjQYyhvLemoI8zFnXgyTY1632/view?usp=sharing
model.load_state_dict(torch.load('weights/model_weights_epoch_0_KL_anealing_CE+L1+SIM.pth'))
model.eval()


padded_conditioning_vectors = torch.tensor(padded_conditioning_vectors, dtype=torch.float32).to(device)
#generated_image = model.decode(padded_conditioning_vectors)
generated_image = model.decode(padded_conditioning_vectors)
generated_image = generated_image.squeeze(0)

# torch.set_printoptions(threshold=torch.inf)  # Sets the threshold to infinite so all values are printed
# print(generated_image)

print(generated_image.shape)
#img = cv2.cvtColor(np.uint8(np.transpose(generated_image.cpu().detach().numpy(), (1, 2, 0))), cv2.COLOR_RGB2BGR)
final_layer = F.sigmoid
generated_image = final_layer(generated_image)
img = cv2.cvtColor(np.uint8(255*np.transpose(generated_image.cpu().detach().numpy(), (1, 2, 0))), cv2.COLOR_RGB2BGR)
#img = cv2.cvtColor(np.uint8(255*generated_image.cpu().detach().numpy()), cv2.COLOR_BGR2RGB)
#img = np.uint8(255*generated_image.cpu().detach().numpy())
cv2.imshow('Image from Dataset', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('generated_image.png', img)
