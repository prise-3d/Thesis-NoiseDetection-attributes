#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 21:02:42 2018

@author: jbuisine
"""

from __future__ import print_function
import sys, os, getopt
import numpy as np
import random
import time
import json

from PIL import Image
from ipfml import processing
from ipfml import metrics
from skimage import color
import matplotlib.pyplot as plt

from modules.utils import config as cfg

config_filename     = cfg.config_filename
zone_folder         = cfg.zone_folder
min_max_filename    = cfg.min_max_filename_extension

# define all scenes values
scenes_list         = cfg.scenes_names
scenes_indexes      = cfg.scenes_indices
choices             = cfg.normalization_choices
path                = cfg.dataset_path
zones               = cfg.zones_indices
seuil_expe_filename = cfg.seuil_expe_filename

metric_choices      = cfg.metric_choices_labels

max_nb_bits = 8

def display_data_scenes(p_scene, p_bits, p_shifted):
    """
    @brief Method which generates all .csv files from scenes photos
    @param p_scene, scene we want to show values
    @param nb_bits, number of bits expected
    @param p_shifted, number of bits expected to be shifted
    @return nothing
    """

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        if p_scene == folder_scene:
            print(folder_scene)
            scene_path = os.path.join(path, folder_scene)

            config_file_path = os.path.join(scene_path, config_filename)

            with open(config_file_path, "r") as config_file:
                last_image_name = config_file.readline().strip()
                prefix_image_name = config_file.readline().strip()
                start_index_image = config_file.readline().strip()
                end_index_image = config_file.readline().strip()
                step_counter = int(config_file.readline().strip())

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

            for id_zone, zone_folder in enumerate(zones_folder):

                zone_path = os.path.join(scene_path, zone_folder)

                current_counter_index = int(start_index_image)
                end_counter_index = int(end_index_image)

                # get threshold information
                path_seuil = os.path.join(zone_path, seuil_expe_filename)

                # open treshold path and get this information
                with open(path_seuil, "r") as seuil_file:
                    seuil_learned = int(seuil_file.readline().strip())

                threshold_image_found = False
                while(current_counter_index <= end_counter_index and not threshold_image_found):

                    if seuil_learned < int(current_counter_index):
                        current_counter_index_str = str(current_counter_index)

                        while len(start_index_image) > len(current_counter_index_str):
                            current_counter_index_str = "0" + current_counter_index_str

                        threshold_image_found = True
                        threshold_image_zone = current_counter_index_str
                        threshold_info.append(threshold_image_zone)

                    current_counter_index += step_counter

                # all indexes of picture to plot
                images_indexes = [start_index_image, threshold_image_zone, end_index_image]
                images_data = []

                print(images_indexes)

                for index in images_indexes:

                    img_path = os.path.join(scene_path, prefix_image_name + index + ".png")

                    current_img = Image.open(img_path)
                    img_blocks = processing.divide_in_blocks(current_img, (200, 200))

                    # getting expected block id
                    block = img_blocks[id_zone]

                    # get data from mode
                    # Here you can add the way you compute data
                    low_bits_block = processing.rgb_to_LAB_L_bits(block, (p_shifted + 1, p_shifted + p_bits + 1))
                    data = metrics.get_SVD_s(low_bits_block)

                    ##################
                    # Data mode part #
                    ##################

                    # modify data depending mode
                    data = processing.normalize_arr(data)
                    images_data.append(data)

                zones_images_data.append(images_data)

            fig=plt.figure(figsize=(8, 8))
            fig.suptitle('Lab SVD ' + str(p_bits) + ' bits shifted by ' + str(p_shifted) + " for " + p_scene + " scene", fontsize=20)

            for id, data in enumerate(zones_images_data):
                fig.add_subplot(4, 4, (id + 1))
                plt.plot(data[0], label='Noisy_' + start_index_image)
                plt.plot(data[1], label='Threshold_' + threshold_info[id])
                plt.plot(data[2], label='Reference_' + end_index_image)
                plt.ylabel('Lab SVD ' + str(p_bits) + ' bits shifted by ' + str(p_shifted) + ', ZONE_' + str(id + 1), fontsize=14)
                plt.xlabel('Vector features', fontsize=16)
                plt.legend(bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0.2, fontsize=14)
                plt.ylim(0, 0.1)
            plt.show()

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python generate_all_data.py --scene A --bits 3 --shifted 3')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:b:s", ["help=", "scene=", "bits=", "shifted="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python generate_all_data.py --scene A --bits 3 --shifted 3')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python generate_all_data.py --scene A --bits 3 --shifted 3')
            sys.exit()
        elif o in ("-b", "--bits"):
            p_bits = int(a)

        elif o in ("-s", "--scene"):
            p_scene = a

            if p_scene not in scenes_indexes:
                assert False, "Invalid metric choice"
            else:
                p_scene = scenes_list[scenes_indexes.index(p_scene)]
        elif o in ("-f", "--shifted"):
            p_shifted = int(a)
        else:
            assert False, "unhandled option"

    if p_bits + p_shifted > max_nb_bits:
        assert False, "Invalid parameters, cannot have bits greater than 8 after shift move"

    display_data_scenes(p_scene, p_bits, p_shifted)

if __name__== "__main__":
    main()
