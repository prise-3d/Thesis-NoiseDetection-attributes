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
from ipfml import image_processing
from ipfml import metrics
from skimage import color
import matplotlib.pyplot as plt

config_filename   = "config"
zone_folder       = "zone"
min_max_filename  = "_min_max_values"

# define all scenes values
scenes_list = ['Appart1opt02', 'Bureau1', 'Cendrier', 'Cuisine01', 'EchecsBas', 'PNDVuePlongeante', 'SdbCentre', 'SdbDroite', 'Selles']
scenes_indexes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
choices = ['svd', 'svdn', 'svdne']
path = '../fichiersSVD_light'
zones = np.arange(16)
seuil_expe_filename = 'seuilExpe'

metric_choices = ['lab', 'mscn', 'mscn_revisited', 'low_bits_2', 'low_bits_3', 'low_bits_4', 'low_bits_5', 'low_bits_6', 'low_bits_4_shifted_2']

def display_data_scenes(data_type, p_scene, p_kind):
    """
    @brief Method which generates all .csv files from scenes photos
    @param path - path of scenes folder information
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
                    img_blocks = image_processing.divide_in_blocks(current_img, (200, 200))

                    # getting expected block id
                    block = img_blocks[id_zone]

                    # get data from mode
                    # Here you can add the way you compute data
                    if data_type == 'lab':

                        block_file_path = '/tmp/lab_img.png'
                        block.save(block_file_path)
                        data = image_processing.get_LAB_L_SVD_s(Image.open(block_file_path))

                    if data_type == 'mscn_revisited':

                        img_mscn_revisited = image_processing.rgb_to_mscn(block)

                        # save tmp as img
                        img_output = Image.fromarray(img_mscn_revisited.astype('uint8'), 'L')
                        mscn_revisited_file_path = '/tmp/mscn_revisited_img.png'
                        img_output.save(mscn_revisited_file_path)
                        img_block = Image.open(mscn_revisited_file_path)

                        # extract from temp image
                        data = metrics.get_SVD_s(img_block)

                    if data_type == 'mscn':

                        img_gray = np.array(color.rgb2gray(np.asarray(block))*255, 'uint8')
                        img_mscn = image_processing.calculate_mscn_coefficients(img_gray, 7)
                        img_mscn_norm = image_processing.normalize_2D_arr(img_mscn)

                        img_mscn_gray = np.array(img_mscn_norm*255, 'uint8')

                        data = metrics.get_SVD_s(img_mscn_gray)

                    if data_type == 'low_bits_6':

                        low_bits_6 = image_processing.rgb_to_LAB_L_low_bits(block, 63)

                        # extract from temp image
                        data = metrics.get_SVD_s(low_bits_6)


                    if data_type == 'low_bits_5':

                        low_bits_5 = image_processing.rgb_to_LAB_L_low_bits(block, 31)

                        # extract from temp image
                        data = metrics.get_SVD_s(low_bits_5)


                    if data_type == 'low_bits_4':

                        low_bits_4 = image_processing.rgb_to_LAB_L_low_bits(block)

                        # extract from temp image
                        data = metrics.get_SVD_s(low_bits_4)

                    if data_type == 'low_bits_3':

                        low_bits_3 = image_processing.rgb_to_LAB_L_low_bits(block, 7)

                        # extract from temp image
                        data = metrics.get_SVD_s(low_bits_3)

                    if data_type == 'low_bits_2':

                        low_bits_2 = image_processing.rgb_to_LAB_L_low_bits(block, 3)

                        # extract from temp image
                        data = metrics.get_SVD_s(low_bits_2)

                    ##################
                    # Data mode part #
                    ##################

                    # modify data depending mode

                    if p_kind == 'svdn':
                        data = image_processing.normalize_arr(data)

                    if p_kind == 'svdne':
                        path_min_max = os.path.join(path, data_type + min_max_filename)

                        with open(path_min_max, 'r') as f:
                            min_val = float(f.readline())
                            max_val = float(f.readline())

                        data = image_processing.normalize_arr_with_range(data, min_val, max_val)

                    # append of data
                    images_data.append(data)

                zones_images_data.append(images_data)

            fig=plt.figure(figsize=(8, 8))
            fig.suptitle(data_type + " values for " + p_scene + " scene (normalization : " + p_kind + ")", fontsize=20)

            for id, data in enumerate(zones_images_data):
                fig.add_subplot(4, 4, (id + 1))
                plt.plot(data[0], label='Noisy_' + start_index_image)
                plt.plot(data[1], label='Threshold_' + threshold_info[id])
                plt.plot(data[2], label='Reference_' + end_index_image)
                plt.ylabel(data_type + ' SVD, ZONE_' + str(id + 1), fontsize=18)
                plt.xlabel('Vector features', fontsize=18)
                plt.legend(bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0.2, fontsize=18)
                plt.ylim(0, 0.1)
            plt.show()

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python generate_all_data.py --metric all --scene A --kind svdn')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:s:k", ["help=", "metric=", "scene=", "kind="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python generate_all_data.py --metric all --scene A --kind svdn')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python generate_all_data.py --metric all --scene A --kind svdn')
            sys.exit()
        elif o in ("-m", "--metric"):
            p_metric = a

            if p_metric != 'all' and p_metric not in metric_choices:
                assert False, "Invalid metric choice"
        elif o in ("-s", "--scene"):
            p_scene = a

            if p_scene not in scenes_indexes:
                assert False, "Invalid metric choice"
            else:
                p_scene = scenes_list[scenes_indexes.index(p_scene)]
        elif o in ("-k", "--kind"):
            p_kind = a

            if p_kind not in choices:
                assert False, "Invalid metric choice"
        else:
            assert False, "unhandled option"


    display_data_scenes(p_metric, p_scene, p_kind)

if __name__== "__main__":
    main()
