from ipfml import image_processing
from PIL import Image
import numpy as np
from ipfml import metrics
from skimage import color

import cv2

nb_bits = 3
max_nb_bits = 8

low_bits_svd_values = []

def open_and_display(path, i):

    img = Image.open(path)

    block_used = np.array(img)

    low_bits_block = image_processing.rgb_to_LAB_L_bits(block_used, (i + 1, i + nb_bits + 1))

    low_bits_svd = metrics.get_SVD_s(low_bits_block)

    low_bits_svd = [b / low_bits_svd[0] for b in low_bits_svd]

    low_bits_svd_values[i].append(low_bits_svd)

path_noisy = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_00050.png'
path_threshold = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_00400.png'
path_ref = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_01200.png'

path_list = [path_noisy, path_threshold, path_ref]


for i in range(0, max_nb_bits - nb_bits + 1):

    low_bits_svd_values.append([])
    for p in path_list:
        open_and_display(p, i)

import matplotlib.pyplot as plt

# SVD
# make a little extra space between the subplots

fig=plt.figure(figsize=(8, 8))

for id, l in enumerate(low_bits_svd_values):

     fig.add_subplot(3, 3, (id + 1))
     plt.plot(l[0], label='Noisy')
     plt.plot(l[1], label='Threshold')
     plt.plot(l[2], label='Reference')
     plt.title('Low ' + str(nb_bits) + ' bits SVD shifted by ' + str(id))
     plt.ylabel('Low ' + str(nb_bits) + ' bits SVD values')
     plt.xlabel('Vector features')
     plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
     plt.ylim(0, 0.1)

plt.show()
