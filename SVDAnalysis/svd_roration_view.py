from ipfml import processing, utils
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

data_folder = "../fichiersSVD_light"

def get_svd_mean_and_image_rotations(img_path):

    print("Extract features from... " + img_path)
    img = np.asarray(Image.open(img_path))
    width, height, dim = img.shape

    img_mean = np.empty([width, height, 3])
    rotations = []
    svd_data_rotation = []

    for i in range(4):
        rotations.append(processing.rotate_image(img, (i+1)*90, pil=False))
        svd_data_rotation.append(processing.get_LAB_L_SVD_s(rotations[i]))

    mean_image = processing.fusion_images(rotations, pil=False)
    mean_data = processing.get_LAB_L_SVD_s(mean_image)

    # getting max and min information from min_max_filename
    with open(data_folder + "/lab_min_max_values", 'r') as f:
        min_val = float(f.readline())
        max_val = float(f.readline())

    mean_data = utils.normalize_arr_with_range(mean_data, min_val, max_val)

    return [utils.normalize_arr_with_range(data, min_val, max_val) for data in svd_data_rotation], mean_data

scene   = 'Appart1opt02'
indices = ["00020", "00080", "00150", "00300", "00500", "00700", "00900"]

for index in indices:
    path = os.path.join(data_folder, scene + '/appartAopt_' + index + '.png')
    svd_data, mean_svd_data = get_svd_mean_and_image_rotations(path)

    plt.title("SVD information of rotations and merged image from " + scene + " scene", fontsize=22)

    plt.ylabel('Singular values', fontsize=18)
    plt.xlabel('Vector features', fontsize=18)

    for id, data in enumerate(svd_data):

        p_label = "appartAopt_" + index + "_" + str((id +1) * 90)
        plt.plot(data, label=p_label)

    mean_label = "appartAopt_" + index + "_mean"
    plt.plot(mean_svd_data, label=mean_label)

    plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.2, fontsize=16)

    plt.ylim(0, 0.01)
    plt.show()

