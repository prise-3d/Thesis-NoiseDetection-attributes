from ipfml import image_processing
from PIL import Image
import numpy as np
from ipfml import metrics
from skimage import color

import cv2

low_bits_svd_values_norm = []
low_bits_svd_values_norm_together = []
low_bits_svd_values = []

mscn_svd_values_norm = []
mscn_svd_values_norm_together = []
mscn_svd_values = []

lab_svd_values_norm = []
lab_svd_values_norm_together = []
lab_svd_values = []

def open_and_display(path):
    img = Image.open(path)

    blocks = image_processing.divide_in_blocks(img, (200, 200), False)

    block_used = blocks[11]

    img_mscn = image_processing.rgb_to_mscn(block_used)

    #img_mscn_norm = image_processing.normalize_2D_arr(img_mscn)

    #print(img_mscn)
    img_output = img_mscn.astype('uint8')

    print('-------------------------')

    # MSCN part computation
    mscn_s = metrics.get_SVD_s(img_output)

    mscn_svd_values.append(mscn_s)
    mscn_svd_values_norm.append(image_processing.normalize_arr(mscn_s))

    mscn_min_val = 10000000
    mscn_max_val = 0

     # check for each block of image
    for block in blocks:

        current_img_mscn = image_processing.rgb_to_mscn(block)

        current_img_output = img_mscn.astype('uint8')

        # MSCN part computation
        current_mscn_s = metrics.get_SVD_s(img_output)

        current_min = current_mscn_s.min()
        current_max = current_mscn_s.max()

        if current_min < mscn_min_val:
            mscn_min_val = current_min

        if current_max > mscn_max_val:
            mscn_max_val = current_max

    mscn_svd_values_norm_together.append(image_processing.normalize_arr_with_range(mscn_s, mscn_min_val, mscn_max_val))

    # LAB part computation
    path_block_img = '/tmp/lab_img.png'

    img_used_pil = Image.fromarray(block_used.astype('uint8'), 'RGB')
    img_used_pil.save(path_block_img)

    #img_used_pil.show()

    lab_s = image_processing.get_LAB_L_SVD_s(Image.open(path_block_img))

    lab_svd_values.append(lab_s)
    lab_svd_values_norm.append(image_processing.normalize_arr(lab_s))

    lab_min_val = 10000000
    lab_max_val = 0

    # check for each block of image
    for block in blocks:

        current_img_used_pil = Image.fromarray(block.astype('uint8'), 'RGB')
        current_img_used_pil.save(path_block_img)

        current_lab_s = image_processing.get_LAB_L_SVD_s(Image.open(path_block_img))

        current_min = current_lab_s.min()
        current_max = current_lab_s.max()

        if current_min < lab_min_val:
            lab_min_val = current_min

        if current_max > lab_max_val:
            lab_max_val = current_max

    lab_svd_values_norm_together.append(image_processing.normalize_arr_with_range(lab_s, lab_min_val, lab_max_val))

    # computation of low bits parts
    low_bits_block = image_processing.rgb_to_grey_low_bits(block_used)

    low_bits_svd = metrics.get_SVD_s(low_bits_block)

    low_bits_svd_values.append(low_bits_svd)
    low_bits_svd_values_norm.append(image_processing.normalize_arr(low_bits_svd))

    low_bits_min_val = 10000000
    low_bits_max_val = 0


        # check for each block of image
    for block in blocks:

        current_grey_block = np.array(color.rgb2gray(block)*255, 'uint8')
        current_low_bit_block = current_grey_block & 15
        current_low_bits_svd = metrics.get_SVD_s(current_low_bit_block)

        current_min = current_low_bits_svd.min()
        current_max = current_low_bits_svd.max()

        if current_min < low_bits_min_val:
            low_bits_min_val = current_min

        if current_max > low_bits_max_val:
            low_bits_max_val = current_max

    low_bits_svd_values_norm_together.append(image_processing.normalize_arr_with_range(low_bits_svd, low_bits_min_val, low_bits_max_val))

    # Other MSCN
    img_grey = np.array(color.rgb2gray(np.asarray(block_used))*255, 'uint8')


    img_mscn_in_grey = np.array(image_processing.normalize_2D_arr(image_processing.calculate_mscn_coefficients(img_grey, 7))*255, 'uint8')
    svd_s_values = metrics.get_SVD_s(img_mscn_in_grey)
    #print(svd_s_values[0:10])

    img_mscn_pil = Image.fromarray(img_mscn_in_grey.astype('uint8'), 'L')
    #img_mscn_pil.show()




#path_noisy = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Appart1opt02/appartAopt_00020.png'
#path_threshold = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Appart1opt02/appartAopt_00300.png'
#path_ref = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Appart1opt02/appartAopt_00900.png'

path_noisy = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_00050.png'
path_threshold = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_00400.png'
path_ref = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_01200.png'


path_list = [path_noisy, path_threshold, path_ref]

for p in path_list:
    open_and_display(p)

import matplotlib.pyplot as plt

# SVD
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.5)

ax1.plot(lab_svd_values[0], label='Noisy')
ax1.plot(lab_svd_values[1], label='Threshold')
ax1.plot(lab_svd_values[2], label='Reference')
ax1.set_ylabel('LAB SVD comparisons')
ax1.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)

ax2.plot(mscn_svd_values[0], label='Noisy')
ax2.plot(mscn_svd_values[1], label='Threshold')
ax2.plot(mscn_svd_values[2], label='Reference')
ax2.set_ylabel('MSCN SVD comparisons')
ax2.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)

ax3.plot(low_bits_svd_values[0], label='Noisy')
ax3.plot(low_bits_svd_values[1], label='Threshold')
ax3.plot(low_bits_svd_values[2], label='Reference')
ax3.set_ylabel('Low bits SVD comparisons')
ax3.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)

plt.show()

# SVDN

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.5)

ax1.plot(lab_svd_values_norm[0], label='Noisy')
ax1.plot(lab_svd_values_norm[1], label='Threshold')
ax1.plot(lab_svd_values_norm[2], label='Reference')
ax1.set_ylabel('LAB SVDN comparisons')
ax1.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)

ax2.plot(mscn_svd_values_norm[0], label='Noisy')
ax2.plot(mscn_svd_values_norm[1], label='Threshold')
ax2.plot(mscn_svd_values_norm[2], label='Reference')
ax2.set_ylabel('MSCN SVDN comparisons')
ax2.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)

ax3.plot(low_bits_svd_values_norm[0], label='Noisy')
ax3.plot(low_bits_svd_values_norm[1], label='Threshold')
ax3.plot(low_bits_svd_values_norm[2], label='Reference')
ax3.set_ylabel('Low bits SVD comparisons')
ax3.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)

plt.show()

# SVDNE
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.5)

ax1.plot(lab_svd_values_norm_together[0], label='Noisy')
ax1.plot(lab_svd_values_norm_together[1], label='Threshold')
ax1.plot(lab_svd_values_norm_together[2], label='Reference')
ax1.set_ylabel('LAB SVDNE comparisons')
ax1.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)

ax2.plot(mscn_svd_values_norm_together[0], label='Noisy')
ax2.plot(mscn_svd_values_norm_together[1], label='Threshold')
ax2.plot(mscn_svd_values_norm_together[2], label='Reference')
ax2.set_ylabel('MSCN SVDNE comparisons')
ax2.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)

ax3.plot(low_bits_svd_values_norm_together[0], label='Noisy')
ax3.plot(low_bits_svd_values_norm_together[1], label='Threshold')
ax3.plot(low_bits_svd_values_norm_together[2], label='Reference')
ax3.set_ylabel('Low bits SVD comparisons')
ax3.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.show()


#print(mscn_svd_values[0][0:3])
#print(mscn_svd_values[1][0:3])
#print(mscn_svd_values[2][0:3])

