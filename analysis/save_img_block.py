from ipfml import processing, utils
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

data_folder = "../fichiersSVD_light"

scene   = 'Cuisine01'
mean_svd_values = []
indices = ["00050", "00300", "01200"]
id_block = 10

def get_block_image(image_path):
    image = Image.open(image_path)
    blocks = processing.divide_in_blocks(image, (200, 200))
    return blocks[id_block]

for index in indices:
    path = os.path.join(data_folder, scene + '/cuisine01_' + index + '.png')
    img_block = get_block_image(path)
    img_block.save(scene + '_' + str(index) + '_' + str(id_block) + '.png')


