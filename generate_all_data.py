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
                
                # get data from mode 
                if data_type == 'lab':

                    block_file_path = '/tmp/lab_img.png'
                    block.save(block_file_path)
                    data = image_processing.get_LAB_L_SVD_s(Image.open(block_file_path))
                
                if data_type == 'mscn':

                    img_mscn = image_processing.rgb_to_mscn(block)

                    # save tmp as img
                    img_output = Image.fromarray(img_mscn.astype('uint8'), 'L')
                    mscn_file_path = '/tmp/mscn_img.png'
                    img_output.save(mscn_file_path)
                    img_block = Image.open(mscn_file_path)

                    # extract from temp image
                    data = metrics.get_SVD_s(img_block)

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

    # all mscn data
    generate_data_svd('mscn', 'svd')
    generate_data_svd('mscn', 'svdn')
    generate_data_svd('mscn', 'svdne')

    # all lab data
    generate_data_svd('lab', 'svd')
    generate_data_svd('lab', 'svdn')
    generate_data_svd('lab', 'svdne')

if __name__== "__main__":
    main()
