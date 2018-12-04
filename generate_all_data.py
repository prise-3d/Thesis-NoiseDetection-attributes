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

config_filename   = "config"
zone_folder       = "zone"
min_max_filename  = "_min_max_values"
generic_output_file_svd = '_random.csv'
output_data_folder = 'data'

# define all scenes values
scenes = ['Appart1opt02', 'Bureau1', 'Cendrier', 'Cuisine01', 'EchecsBas', 'PNDVuePlongeante', 'SdbCentre', 'SdbDroite', 'Selles']
scenes_indexes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
choices = ['svd', 'svdn', 'svdne']
path = './fichiersSVD_light'
zones = np.arange(16)
seuil_expe_filename = 'seuilExpe'

metric_choices = ['lab', 'mscn', 'mscn_revisited', 'low_bits_2', 'low_bits_3', 'low_bits_4', 'low_bits_5', 'low_bits_6','low_bits_4_shifted_2']

def generate_data_svd(data_type, mode):
    """
    @brief Method which generates all .csv files from scenes photos
    @param path - path of scenes folder information
    @return nothing
    """

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    # keep in memory min and max data found from data_type
    min_val_found = 100000000000
    max_val_found = 0

    data_min_max_filename = os.path.join(path, data_type + min_max_filename)

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        print(folder_scene)
        scene_path = os.path.join(path, folder_scene)

        config_file_path = os.path.join(scene_path, config_filename)

        with open(config_file_path, "r") as config_file:
            last_image_name = config_file.readline().strip()
            prefix_image_name = config_file.readline().strip()
            start_index_image = config_file.readline().strip()
            end_index_image = config_file.readline().strip()
            step_counter = int(config_file.readline().strip())

        # getting output filename
        output_svd_filename = data_type + "_" + mode + generic_output_file_svd

        # construct each zones folder name
        zones_folder = []
        svd_output_files = []

        # get zones list info
        for index in zones:
            index_str = str(index)
            if len(index_str) < 2:
                index_str = "0" + index_str

            current_zone = "zone"+index_str
            zones_folder.append(current_zone)

            zone_path = os.path.join(scene_path, current_zone)
            svd_file_path = os.path.join(zone_path, output_svd_filename)

            # add writer into list
            svd_output_files.append(open(svd_file_path, 'w'))


        current_counter_index = int(start_index_image)
        end_counter_index = int(end_index_image)


        while(current_counter_index <= end_counter_index):

            current_counter_index_str = str(current_counter_index)

            while len(start_index_image) > len(current_counter_index_str):
                current_counter_index_str = "0" + current_counter_index_str

            img_path = os.path.join(scene_path, prefix_image_name + current_counter_index_str + ".png")

            current_img = Image.open(img_path)
            img_blocks = image_processing.divide_in_blocks(current_img, (200, 200))

            for id_block, block in enumerate(img_blocks):

                ###########################
                # Metric computation part #
                ###########################

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

                if data_type == 'low_bits_4_shifted_2':

                    data = metrics.get_SVD_s(image_processing.rgb_to_LAB_L_bits(block, (3, 6)))


                ##################
                # Data mode part #
                ##################

                # modify data depending mode
                if mode == 'svdne':

                    # getting max and min information from min_max_filename
                    with open(data_min_max_filename, 'r') as f:
                        min_val = float(f.readline())
                        max_val = float(f.readline())

                    data = image_processing.normalize_arr_with_range(data, min_val, max_val)

                if mode == 'svdn':
                    data = image_processing.normalize_arr(data)

                # save min and max found from dataset in order to normalize data using whole data known
                if mode == 'svd':

                    current_min = data.min()
                    current_max = data.max()

                    if current_min < min_val_found:
                        min_val_found = current_min

                    if current_max > max_val_found:
                        max_val_found = current_max

                # now write data into current writer
                current_file = svd_output_files[id_block]

                # add of index
                current_file.write(current_counter_index_str + ';')

                for val in data:
                    current_file.write(str(val) + ";")

                current_file.write('\n')

            start_index_image_int = int(start_index_image)
            print(data_type + "_" + mode + "_" + folder_scene + " - " + "{0:.2f}".format((current_counter_index - start_index_image_int) / (end_counter_index - start_index_image_int)* 100.) + "%")
            current_counter_index += step_counter

        for f in svd_output_files:
            f.close()

    # save current information about min file found
    if mode == 'svd':
        with open(data_min_max_filename, 'w') as f:
            f.write(str(min_val_found) + '\n')
            f.write(str(max_val_found) + '\n')

    print("End of data generation")


def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python generate_all_data.py --metric all')
        print('python generate_all_data.py --metric lab')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm", ["help=", "metric="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python generate_all_data.py --metric all')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python generate_all_data.py --metric all')
            sys.exit()
        elif o in ("-m", "--metric"):
            p_metric = a

            if p_metric != 'all' and p_metric not in metric_choices:
                assert False, "Invalid metric choice"
        else:
            assert False, "unhandled option"

    # generate all or specific metric data
    if p_metric == 'all':
        for m in metric_choices:
            generate_data_svd(m, 'svd')
            generate_data_svd(m, 'svdn')
            generate_data_svd(m, 'svdne')
    else:
        generate_data_svd(p_metric, 'svd')
        generate_data_svd(p_metric, 'svdn')
        generate_data_svd(p_metric, 'svdne')

if __name__== "__main__":
    main()
