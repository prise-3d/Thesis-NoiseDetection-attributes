# main imports
import sys, os, argparse
import numpy as np
import random
import time
import json

# image processing imports
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt

from ipfml.processing import compression, transform

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

# variables and parameters
zone_folder         = cfg.zone_folder
min_max_filename    = cfg.min_max_filename_extension

# define all scenes values
scenes_list         = cfg.scenes_names
scenes_indices      = cfg.scenes_indices
path                = cfg.dataset_path
zones               = cfg.zones_indices
seuil_expe_filename = cfg.seuil_expe_filename

max_nb_bits = 8

def display_data_scenes(nb_bits, p_scene):
    """
    @brief Method display shifted values for specific scene
    @param nb_bits, number of bits expected
    @param p_scene, scene we want to show values
    @return nothing
    """

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    # go ahead each scenes
    for folder_scene in scenes:

        if p_scene == folder_scene:
            print(folder_scene)
            scene_path = os.path.join(path, folder_scene)

            # construct each zones folder name
            zones_folder = []

            # get zones list info
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str

                current_zone = "zone"+index_str
                zones_folder.append(current_zone)

            threshold_info = []

            for zone_folder in zones_folder:

                zone_path = os.path.join(scene_path, zone_folder)

                # get threshold information
                path_seuil = os.path.join(zone_path, seuil_expe_filename)

                # open treshold path and get this information
                with open(path_seuil, "r") as seuil_file:
                    seuil_learned = int(seuil_file.readline().strip())
                    threshold_info.append(seuil_learned)

            # compute mean threshold values
            mean_threshold = sum(threshold_info) / float(len(threshold_info))

            print(mean_threshold, "mean threshold found")
            threshold_image_found = False

            # get all images of folder
            scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])

            start_image_path = scene_images[0]
            end_image_path   = scene_images[-1]

            start_quality_image = dt.get_scene_image_quality(scene_images[0])
            end_quality_image   = dt.get_scene_image_quality(scene_images[-1])

            # for each images
            for img_path in scene_images:
                current_quality_image = dt.get_scene_image_quality(img_path)

                if mean_threshold < int(current_quality_image) and not threshold_image_found:

                    threshold_image_found = True
                    threshold_image_path = img_path

                    threshold_image = dt.get_scene_image_quality(img_path)

            # all indexes of picture to plot
            images_path = [start_image_path, threshold_image_path, end_image_path]

            low_bits_svd_values = []

            for i in range(0, max_nb_bits - nb_bits + 1):

                low_bits_svd_values.append([])

                for img_path in images_path:

                    current_img = Image.open(img_path)

                    block_used = np.array(current_img)

                    low_bits_block = transform.rgb_to_LAB_L_bits(block_used, (i + 1, i + nb_bits + 1))
                    low_bits_svd = compression.get_SVD_s(low_bits_block)
                    low_bits_svd = [b / low_bits_svd[0] for b in low_bits_svd]
                    low_bits_svd_values[i].append(low_bits_svd)


            fig=plt.figure(figsize=(8, 8))
            fig.suptitle("Lab SVD " + str(nb_bits) +  " bits values shifted for " + p_scene + " scene", fontsize=20)

            for id, data in enumerate(low_bits_svd_values):
                fig.add_subplot(3, 3, (id + 1))
                plt.plot(data[0], label='Noisy_' + start_quality_image)
                plt.plot(data[1], label='Threshold_' + threshold_image)
                plt.plot(data[2], label='Reference_' + end_quality_image)
                plt.ylabel('Lab SVD ' + str(nb_bits) + ' bits values shifted by ' + str(id), fontsize=14)
                plt.xlabel('Vector features', fontsize=16)
                plt.legend(bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0.2, fontsize=14)
                plt.ylim(0, 0.1)
            plt.show()

def main():

    parser = argparse.ArgumentParser(description="Display curves of shifted bits influence of L canal on specific scene")

    parser.add_argument('--bits', type=str, help='Number of bits to display')
    parser.add_argument('--scene', type=str, help="scene index to use", choices=scenes_indices)

    args = parser.parse_args()

    p_bits  = args.bits
    p_scene = scenes_list[scenes_indices.index(args.scene)]

    display_data_scenes(p_bits, p_scene)

if __name__== "__main__":
    main()
