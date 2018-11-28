from ipfml import image_processing
from PIL import Image
import numpy as np
from ipfml import metrics
from skimage import color

import cv2

low_bits_2_svd_values = []

low_bits_3_svd_values = []

low_bits_4_svd_values = []

mscn_revisited_svd_values = []

mscn_svd_values = []

lab_svd_values = []

def open_and_display(path):
    img = Image.open(path)

    block_used = np.array(img)

    img_mscn = image_processing.rgb_to_mscn(block_used)

    #img_mscn_norm = image_processing.normalize_2D_arr(img_mscn)

    #print(img_mscn)
    img_output = img_mscn.astype('uint8')

    print('-------------------------')

    # MSCN part computation
    mscn_s = metrics.get_SVD_s(img_output)

    mscn_s = [m / mscn_s[0] for m in mscn_s]

    mscn_revisited_svd_values.append(mscn_s)

    # LAB part computation
    path_block_img = '/tmp/lab_img.png'

    img_used_pil = Image.fromarray(block_used.astype('uint8'), 'RGB')
    img_used_pil.save(path_block_img)

    lab_s = image_processing.get_LAB_L_SVD_s(Image.open(path_block_img))

    lab_s = [l / lab_s[0] for l in lab_s]
    lab_svd_values.append(lab_s)

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

    # Other MSCN
    img_grey = np.array(color.rgb2gray(np.asarray(block_used))*255, 'uint8')


    img_mscn_in_grey = np.array(image_processing.normalize_2D_arr(image_processing.calculate_mscn_coefficients(img_grey, 7))*255, 'uint8')
    svd_s_values = metrics.get_SVD_s(img_mscn_in_grey)

    svd_s_values = [s / svd_s_values[0] for s in svd_s_values]
    mscn_svd_values.append(svd_s_values)




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
# make a little extra space between the subplots

plt.plot(lab_svd_values[0], label='Noisy')
plt.plot(lab_svd_values[1], label='Threshold')
plt.plot(lab_svd_values[2], label='Reference')
plt.ylabel('LAB SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()

plt.plot(mscn_svd_values[0], label='Noisy')
plt.plot(mscn_svd_values[1], label='Threshold')
plt.plot(mscn_svd_values[2], label='Reference')
plt.ylabel('MSCN SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()

plt.plot(mscn_revisited_svd_values[0], label='Noisy')
plt.plot(mscn_revisited_svd_values[1], label='Threshold')
plt.plot(mscn_revisited_svd_values[2], label='Reference')
plt.ylabel('Revisited MSCN SVD')
plt.xlabel('Vector features')
plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2)
plt.ylim(0, 0.1)
plt.show()

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
