import torch
import cv2
from networks.cvae_bn import CVAE
import numpy as np

#padded_conditioning_vectors = [0, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
#padded_conditioning_vectors = [0, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
#padded_conditioning_vectors = [0, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
#padded_conditioning_vectors = [2, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1]
padded_conditioning_vectors = [0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


device = 'cuda'

model = CVAE().to(device)
model.load_state_dict(torch.load('weights/model_weights_epoch_0.pth'))
model.eval()


padded_conditioning_vectors = torch.tensor(padded_conditioning_vectors, dtype=torch.float32).to(device)
generated_image = model.decode(padded_conditioning_vectors)
generated_image = generated_image.squeeze(0)

# torch.set_printoptions(threshold=torch.inf)  # Sets the threshold to infinite so all values are printed
# print(generated_image)

print(generated_image.shape)
#img = cv2.cvtColor(np.uint8(np.transpose(generated_image.cpu().detach().numpy(), (1, 2, 0))), cv2.COLOR_RGB2BGR)
img = cv2.cvtColor(np.uint8(255*np.transpose(generated_image.cpu().detach().numpy(), (1, 2, 0))), cv2.COLOR_RGB2BGR)
#img = cv2.cvtColor(np.uint8(255*generated_image.cpu().detach().numpy()), cv2.COLOR_BGR2RGB)
#img = np.uint8(255*generated_image.cpu().detach().numpy())
cv2.imshow('Image from Dataset', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('generated_image.png', img)
