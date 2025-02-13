import numpy as np
import random
import json
import cv2
import os
from PIL import Image, ImageDraw

# Set a seed for reproducibility
random.seed(42)
np.random.seed(42)

generated_dataset_path = '/home/radu-bolborici/fac/IA3/lecture_IA3/lecture_project_IA3/Conditional-VAE-for-generating-geometric-shapes/dataset_one'
json_path = '/home/radu-bolborici/fac/IA3/lecture_IA3/lecture_project_IA3/Conditional-VAE-for-generating-geometric-shapes/dataset_one.json'

# Define constants
IMG_SIZE = 128
#MAX_SHAPES = 6
MAX_SHAPES = 1
#NUM_EXAMPLES = 100
NUM_EXAMPLES = 100000
min_size, max_size = 20, 80

# Shape and color identifiers
SHAPE_IDS = {'square': 0, 'circle': 1, 'triangle': 2, 'hexagon': 3}
COLOR_IDS = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3}

# Colors in RGB
COLORS = {
    0: (255, 0, 0),  # Red
    1: (0, 0, 255),  # Blue
    2: (0, 255, 0),  # Green
    3: (255, 255, 0) # Yellow
}

# Directory to save the images and JSON
os.makedirs('generated_dataset', exist_ok=True)

# Function to draw different shapes
def draw_shape(draw, shape_type, color, position, size):
    if shape_type == 'square':
        x1, y1 = position
        x2, y2 = x1 + size, y1 + size
        draw.rectangle([x1, y1, x2, y2], fill=color)
    elif shape_type == 'circle':
        x1, y1 = position
        x2, y2 = x1 + size, y1 + size
        draw.ellipse([x1, y1, x2, y2], fill=color)
    elif shape_type == 'triangle':
        x1, y1 = position
        x2, y2 = x1 + size, y1 + size
        draw.polygon([x1, y2, (x1+x2)//2, y1, x2, y2], fill=color)
    elif shape_type == 'hexagon':
        x1, y1 = position
        r = size // 2
        x_center, y_center = x1 + r, y1 + r
        hexagon = [(x_center + r*np.cos(theta), y_center + r*np.sin(theta)) for theta in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
        draw.polygon(hexagon, fill=color)

dataset_info = []

for img_id in range(NUM_EXAMPLES):
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    num_shapes = random.randint(1, MAX_SHAPES)
    shape_data = []
    for _ in range(num_shapes):
        shape = random.choice(list(SHAPE_IDS.keys()))
        color = random.choice(list(COLOR_IDS.keys()))

        size = random.randint(min_size, max_size)
        position = (random.randint(0, IMG_SIZE - size), random.randint(0, IMG_SIZE - size))

        draw_shape(draw, shape, COLORS[COLOR_IDS[color]], position, size)

        shape_data.append({
            'shape_id': SHAPE_IDS[shape],
            'color_id': COLOR_IDS[color],
            'position': position,
            'size': size
        })

    img_name = f"{img_id}.png"
    img.save(f'{generated_dataset_path}/{img_name}')

    dataset_info.append({
        'image_id': img_name,
        'shapes': shape_data
    })

with open(json_path, 'w') as f:
    json.dump(dataset_info, f, indent=4)

