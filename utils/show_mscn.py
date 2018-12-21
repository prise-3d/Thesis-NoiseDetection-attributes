from ipfml import image_processing
from PIL import Image
import numpy as np
from ipfml import metrics
from skimage import color

import cv2

path_noisy = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_00050.png'
path_threshold = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_00400.png'
path_ref = '/home/jbuisine/Documents/Thesis/Development/NoiseDetection_In_SynthesisImages/fichiersSVD_light/Cuisine01/cuisine01_01200.png'

path_list = [path_noisy, path_threshold, path_ref]
labels = ['noisy', 'threshold', 'reference']

for id, p in enumerate(path_list):

    img = Image.open(p)
    img.show()

    # Revisited MSCN
    current_img_mscn = image_processing.rgb_to_mscn(img)
    current_img_output = current_img_mscn.astype('uint8')
    img_mscn_pil = Image.fromarray(current_img_output.astype('uint8'), 'L')
    img_mscn_pil.show()
    img_mscn_pil.save('/home/jbuisine/Downloads/' + labels[id] + '_revisited.png')


    # MSCN
    img_grey = np.array(color.rgb2gray(np.asarray(img))*255, 'uint8')

    img_mscn_in_grey = np.array(image_processing.normalize_2D_arr(image_processing.calculate_mscn_coefficients(img_grey, 7))*255, 'uint8')

    img_mscn_pil = Image.fromarray(img_mscn_in_grey.astype('uint8'), 'L')
    img_mscn_pil.show()
    img_mscn_pil.save('/home/jbuisine/Downloads/' + labels[id] + '_mscn.png')

