from ipfml import image_processing
from PIL import Image
import numpy as np
from ipfml import metrics
from skimage import color

import cv2

low_bits_2_svd_values = []

low_bits_3_svd_values = []

low_bits_4_svd_values = []

low_bits_5_svd_values = []

low_bits_6_svd_values = []

low_bits_7_svd_values = []

def open_and_display(path):
    img = Image.open(path)

    block_used = np.array(img)

    # computation of low bits parts 2 bits
    low_bits_block = image_processing.rgb_to_LAB_L_low_bits(block_used, 3)

    low_bits_svd = metrics.get_SVD_s(low_bits_block)

    low_bits_svd = [b / low_bits_svd[0] for b in low_bits_svd]

    low_bits_2_svd_values.append(low_bits_svd)

    # computation of low bits parts 3 bits
    low_bits_block = image_processing.rgb_to_LAB_L_low_bits(block_used, 7)

    low_bits_svd = metrics.get_SVD_s(low_bits_block)

    low_bits_svd = [b / low_bits_svd[0] for b in low_bits_svd]

    low_bits_3_svd_values.append(low_bits_svd)

    # computation of low bits parts 4 bits
    low_bits_block = image_processing.rgb_to_LAB_L_low_bits(block_used)

    low_bits_svd = metrics.get_SVD_s(low_bits_block)

    low_bits_svd = [b / low_bits_svd[0] for b in low_bits_svd]

    low_bits_4_svd_values.append(low_bits_svd)

    # computation of low bits parts 5 bits
    low_bits_block = image_processing.rgb_to_LAB_L_low_bits(block_used, 31)

    low_bits_svd = metrics.get_SVD_s(low_bits_block)

    low_bits_svd = [b / low_bits_svd[0] for b in low_bits_svd]

    low_bits_5_svd_values.append(low_bits_svd)

    # computation of low bits parts 6 bits
    low_bits_block = image_processing.rgb_to_LAB_L_low_bits(block_used, 63)

    low_bits_svd = metrics.get_SVD_s(low_bits_block)

    low_bits_svd = [b / low_bits_svd[0] for b in low_bits_svd]

    low_bits_6_svd_values.append(low_bits_svd)

    # computation of low bits parts 7 bits
    low_bits_block = image_processing.rgb_to_LAB_L_low_bits(block_used, 127)

    low_bits_svd = metrics.get_SVD_s(low_bits_block)

    low_bits_svd = [b / low_bits_svd[0] for b in low_bits_svd]

    low_bits_7_svd_values.append(low_bits_svd)

path_noisy = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_00050.png'
path_threshold = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_00400.png'
path_ref = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_01200.png'

path_list = [path_noisy, path_threshold, path_ref]

for p in path_list:
    open_and_display(p)

import matplotlib.pyplot as plt

# SVD
# make a little extra space between the subplots

plt.plot(low_bits_2_svd_values[0], label='Noisy')
plt.plot(low_bits_2_svd_values[1], label='Threshold')
plt.plot(low_bits_2_svd_values[2], label='Reference')
plt.ylabel('Low 2 bits SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()

plt.plot(low_bits_3_svd_values[0], label='Noisy')
plt.plot(low_bits_3_svd_values[1], label='Threshold')
plt.plot(low_bits_3_svd_values[2], label='Reference')
plt.ylabel('Low 3 bits SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()

plt.plot(low_bits_4_svd_values[0], label='Noisy')
plt.plot(low_bits_4_svd_values[1], label='Threshold')
plt.plot(low_bits_4_svd_values[2], label='Reference')
plt.ylabel('Low 4 bits SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()

plt.plot(low_bits_5_svd_values[0], label='Noisy')
plt.plot(low_bits_5_svd_values[1], label='Threshold')
plt.plot(low_bits_5_svd_values[2], label='Reference')
plt.ylabel('Low 5 bits SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()

plt.plot(low_bits_6_svd_values[0], label='Noisy')
plt.plot(low_bits_6_svd_values[1], label='Threshold')
plt.plot(low_bits_6_svd_values[2], label='Reference')
plt.ylabel('Low 6 bits SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()

plt.plot(low_bits_7_svd_values[0], label='Noisy')
plt.plot(low_bits_7_svd_values[1], label='Threshold')
plt.plot(low_bits_7_svd_values[2], label='Reference')
plt.ylabel('Low 7 bits SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()
