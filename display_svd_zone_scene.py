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
from ipfml import processing, metrics, utils
from skimage import color

import matplotlib.pyplot as plt
from modules.utils.data import get_svd_data

from modules.utils import config as cfg

# getting configuration information
config_filename     = cfg.config_filename
zone_folder         = cfg.zone_folder
min_max_filename    = cfg.min_max_filename_extension

# define all scenes values
scenes_list         = cfg.scenes_names
scenes_indices      = cfg.scenes_indices
choices             = cfg.normalization_choices
path                = cfg.dataset_path
zones               = cfg.zones_indices
seuil_expe_filename = cfg.seuil_expe_filename

metric_choices      = cfg.metric_choices_labels

max_nb_bits = 8

def display_svd_values(p_scene, p_interval, p_indices, p_zone, p_metric, p_mode, p_step, p_norm, p_ylim):
    """
    @brief Method which gives information about svd curves from zone of picture
    @param p_scene, scene expected to show svd values
    @param p_interval, interval [begin, end] of svd data to display
    @param p_interval, interval [begin, end] of samples or minutes from render generation engine
    @param p_zone, zone's identifier of picture
    @param p_metric, metric computed to show
    @param p_mode, normalization's mode
    @param p_step, step of images indices
    @param p_norm, normalization or not of selected svd data
    @param p_ylim, ylim choice to better display of data
    @return nothing
    """

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    begin_data, end_data = p_interval
    begin_index, end_index = p_indices

    data_min_max_filename = os.path.join(path, p_metric + min_max_filename)

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        if p_scene == folder_scene:
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
            images_indices = []

            zone_folder = zones_folder[p_zone]

            zone_path = os.path.join(scene_path, zone_folder)

            current_counter_index = int(start_index_image)
            end_counter_index = int(end_index_image)

            # get threshold information
            path_seuil = os.path.join(zone_path, seuil_expe_filename)

            # open treshold path and get this information
            with open(path_seuil, "r") as seuil_file:
                seuil_learned = int(seuil_file.readline().strip())

            threshold_image_found = False

            while(current_counter_index <= end_counter_index):

                current_counter_index_str = str(current_counter_index)

                while len(start_index_image) > len(current_counter_index_str):
                    current_counter_index_str = "0" + current_counter_index_str

                if current_counter_index % p_step == 0:
                    if current_counter_index >= begin_index and current_counter_index <= end_index:
                        images_indices.append(current_counter_index_str)

                    if seuil_learned < int(current_counter_index) and not threshold_image_found:

                        threshold_image_found = True
                        threshold_image_zone = current_counter_index_str

                current_counter_index += step_counter

            # all indices of picture to plot
            print(images_indices)

            for index in images_indices:

                img_path = os.path.join(scene_path, prefix_image_name + str(index) + ".png")

                current_img = Image.open(img_path)
                img_blocks = processing.divide_in_blocks(current_img, (200, 200))

                # getting expected block id
                block = img_blocks[p_zone]

                # get data from mode
                # Here you can add the way you compute data
                data = get_svd_data(p_metric, block)

                if p_norm:
                    data = data[begin_data:end_data]

                ##################
                # Data mode part #
                ##################

                if p_mode == 'svdne':

                    # getting max and min information from min_max_filename
                    with open(data_min_max_filename, 'r') as f:
                        min_val = float(f.readline())
                        max_val = float(f.readline())

                    data = utils.normalize_arr_with_range(data, min_val, max_val)

                if p_mode == 'svdn':
                    data = utils.normalize_arr(data)

                if not p_norm:
                    zones_images_data.append(data[begin_data:end_data])
                else:
                    zones_images_data.append(data)

            plt.title(p_scene + ' scene interval information SVD['+ str(begin_data) +', '+ str(end_data) +'], from scenes indices [' + str(begin_index) + ', '+ str(end_index) + ']' + p_metric + ' metric, ' + p_mode + ', with step of ' + str(p_step) + ', svd norm ' + str(p_norm), fontsize=20)
            plt.ylabel('Image samples or time (minutes) generation', fontsize=14)
            plt.xlabel('Vector features', fontsize=16)

            for id, data in enumerate(zones_images_data):

                p_label = p_scene + "_" + images_indices[id]

                if images_indices[id] == threshold_image_zone:
                    plt.plot(data, label=p_label, lw=4, color='red')
                else:
                    plt.plot(data, label=p_label)

            plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.2, fontsize=14)

            start_ylim, end_ylim = p_ylim
            plt.ylim(start_ylim, end_ylim)

            plt.show()

def main():


    # by default p_step value is 10 to enable all photos
    p_step = 10
    p_norm = 0
    p_ylim = (0, 1)

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python display_svd_zone_scene.py --scene A --interval "0,200" --indices "0, 900" --zone 3 --metric lab --mode svdne --step 50 --norm 0 --ylim "0, 0.1"')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:i:i:z:l:m:s:n:y", ["help=", "scene=", "interval=", "indices=", "zone=", "metric=", "mode=", "step=", "norm=", "ylim="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python display_svd_zone_scene.py --scene A --interval "0,200" --indices "0, 900" --zone 3 --metric lab --mode svdne --step 50 --norm 0 --ylim "0, 0.1"')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python display_svd_zone_scene.py --scene A --interval "0,200" --indices "0, 900" --zone 3 --metric lab --mode svdne --step 50 --norm 0 --ylim "0, 0.1"')
            sys.exit()
        elif o in ("-s", "--scene"):
            p_scene = a

            if p_scene not in scenes_indices:
                assert False, "Invalid scene choice"
            else:
                p_scene = scenes_list[scenes_indices.index(p_scene)]
        elif o in ("-i", "--interval"):
            p_interval = list(map(int, a.split(',')))

        elif o in ("-i", "--indices"):
            p_indices = list(map(int, a.split(',')))

        elif o in ("-z", "--zone"):
            p_zone = int(a)

        elif o in ("-m", "--metric"):
            p_metric = a

            if p_metric not in metric_choices:
                assert False, "Invalid metric choice"

        elif o in ("-m", "--mode"):
            p_mode = a

            if p_mode not in choices:
                assert False, "Invalid normalization choice, expected ['svd', 'svdn', 'svdne']"

        elif o in ("-s", "--step"):
            p_step = int(a)

        elif o in ("-n", "--norm"):
            p_norm = int(a)

        elif o in ("-y", "--ylim"):
            p_ylim = list(map(float, a.split(',')))

        else:
            assert False, "unhandled option"

    # TODO: if p_norm find custom min max values
    display_svd_values(p_scene, p_interval, p_indices, p_zone, p_metric, p_mode, p_step, p_norm, p_ylim)

if __name__== "__main__":
    main()
