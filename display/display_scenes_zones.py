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

from data_attributes import get_svd_data

from ipfml.processing import segmentation, transform, compression
from ipfml import utils

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
norm_choices        = cfg.normalization_choices
path                = cfg.dataset_path
zones               = cfg.zones_indices
seuil_expe_filename = cfg.seuil_expe_filename

features_choices      = cfg.features_choices_labels


def display_data_scenes(data_type, p_scene, p_kind):
    """
    @brief Method which displays data from scene
    @param data_type,  feature choice
    @param scene, scene choice
    @param mode, normalization choice
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

            zones_images_data = []
            threshold_info = []

            # get all images of folder
            scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])

            start_image_path = scene_images[0]
            end_image_path   = scene_images[-1]

            start_quality_image = dt.get_scene_image_quality(scene_images[0])
            end_quality_image   = dt.get_scene_image_quality(scene_images[-1])

            for id_zone, zone_folder in enumerate(zones_folder):

                zone_path = os.path.join(scene_path, zone_folder)

                # get threshold information
                path_seuil = os.path.join(zone_path, seuil_expe_filename)

                # open treshold path and get this information
                with open(path_seuil, "r") as seuil_file:
                    threshold_learned = int(seuil_file.readline().strip())

                threshold_image_found = False

                for img_path in scene_images:
                    current_quality_image = dt.get_scene_image_quality(img_path)

                    if threshold_learned < int(current_quality_image) and not threshold_image_found:

                        threshold_image_found = True
                        threshold_image_path = img_path

                        threshold_image = dt.get_scene_image_postfix(img_path)
                        threshold_info.append(threshold_image)

                # all indexes of picture to plot
                images_path = [start_image_path, threshold_image_path, end_image_path]
                images_data = []

                for img_path in images_path:

                    current_img = Image.open(img_path)
                    img_blocks = segmentation.divide_in_blocks(current_img, (200, 200))

                    # getting expected block id
                    block = img_blocks[id_zone]

                    data = get_svd_data(data_type, block)

                    ##################
                    # Data mode part #
                    ##################

                    # modify data depending mode

                    if p_kind == 'svdn':
                        data = utils.normalize_arr(data)

                    if p_kind == 'svdne':
                        path_min_max = os.path.join(path, data_type + min_max_filename)

                        with open(path_min_max, 'r') as f:
                            min_val = float(f.readline())
                            max_val = float(f.readline())

                        data = utils.normalize_arr_with_range(data, min_val, max_val)

                    # append of data
                    images_data.append(data)

                zones_images_data.append(images_data)

            fig=plt.figure(figsize=(8, 8))
            fig.suptitle(data_type + " values for " + p_scene + " scene (normalization : " + p_kind + ")", fontsize=20)

            for id, data in enumerate(zones_images_data):
                fig.add_subplot(4, 4, (id + 1))
                plt.plot(data[0], label='Noisy_' + start_quality_image)
                plt.plot(data[1], label='Threshold_' + threshold_info[id])
                plt.plot(data[2], label='Reference_' + end_quality_image)
                plt.ylabel(data_type + ' SVD, ZONE_' + str(id + 1), fontsize=18)
                plt.xlabel('Vector features', fontsize=18)
                plt.legend(bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0.2, fontsize=18)
                plt.ylim(0, 0.1)
            plt.show()

def main():

    parser = argparse.ArgumentParser(description="Display zones curves of feature on scene ")

    parser.add_argument('--feature', type=str, help='feature data choice', choices=features_choices)
    parser.add_argument('--scene', type=str, help='scene index to use', choices=scenes_indices)
    parser.add_argument('--kind', type=str, help='Kind of normalization level wished', choices=norm_choices)

    args = parser.parse_args()

    p_feature = args.feature
    p_kind   = args.kind
    p_scene  = scenes_list[scenes_indices.index(args.scene)]

    display_data_scenes(p_feature, p_scene, p_kind)

if __name__== "__main__":
    main()
