import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class ShapeDataset(Dataset):
    def __init__(self, images_path, json_path, transform=None):
        self.images_path = images_path
        self.transform = transform

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_metadata = self.data[idx]
        
        img_path = os.path.join(self.images_path, img_metadata['image_id'])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        conditioning_vector = tuple(
            (shape['shape_id'], shape['color_id']) for shape in img_metadata['shapes']
        )

        return np.array(img), conditioning_vector

if __name__ == '__main__':          
    dataset = ShapeDataset('dataset', 'dataset.json')

    for i in range(10):
        img, conditioning_vector = dataset[i]
        print(f'index = {i}')
        print(type(img), img.shape)
        print(conditioning_vector)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image from Dataset', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
