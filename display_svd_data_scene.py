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
import ipfml.iqa.fr as fr_iqa

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

max_nb_bits         = 8
display_error       = False


def display_svd_values(p_scene, p_interval, p_indices, p_metric, p_mode, p_step, p_norm, p_ylim):
    """
    @brief Method which gives information about svd curves from zone of picture
    @param p_scene, scene expected to show svd values
    @param p_interval, interval [begin, end] of svd data to display
    @param p_interval, interval [begin, end] of samples or minutes from render generation engine
    @param p_metric, metric computed to show
    @param p_mode, normalization's mode
    @param p_norm, normalization or not of selected svd data
    @param p_ylim, ylim choice to better display of data
    @return nothing
    """

    max_value_svd = 0
    min_value_svd = sys.maxsize

    image_indices = []

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

            images_data = []
            images_indices = []

            threshold_learned_zones = []

            for id, zone_folder in enumerate(zones_folder):

                # get threshold information

                zone_path = os.path.join(scene_path, zone_folder)
                path_seuil = os.path.join(zone_path, seuil_expe_filename)

                # open treshold path and get this information
                with open(path_seuil, "r") as seuil_file:
                    threshold_learned = int(seuil_file.readline().strip())
                    threshold_learned_zones.append(threshold_learned)

            current_counter_index = int(start_index_image)
            end_counter_index = int(end_index_image)

            threshold_mean = np.mean(np.asarray(threshold_learned_zones))
            threshold_image_found = False

            file_path = os.path.join(scene_path, prefix_image_name + "{}.png")

            svd_data = []

            while(current_counter_index <= end_counter_index):

                current_counter_index_str = str(current_counter_index)

                while len(start_index_image) > len(current_counter_index_str):
                    current_counter_index_str = "0" + current_counter_index_str

                image_path = file_path.format(str(current_counter_index_str))
                img = Image.open(image_path)

                svd_values = get_svd_data(p_metric, img)

                if p_norm:
                    svd_values = svd_values[begin_data:end_data]

                # update min max values
                min_value = svd_values.min()
                max_value = svd_values.max()

                if min_value < min_value_svd:
                    min_value_svd = min_value

                if max_value > min_value_svd:
                    max_value_svd = max_value

                # keep in memory used data
                if current_counter_index % p_step == 0:
                    if current_counter_index >= begin_index and current_counter_index <= end_index:
                        images_indices.append(current_counter_index_str)
                        svd_data.append(svd_values)

                    if threshold_mean < int(current_counter_index) and not threshold_image_found:

                        threshold_image_found = True
                        threshold_image_zone = current_counter_index_str

                current_counter_index += step_counter
                print('%.2f%%' % (current_counter_index / end_counter_index * 100))
                sys.stdout.write("\033[F")


            # all indices of picture to plot
            print(images_indices)

            for id, data in enumerate(svd_data):

                current_data = data

                if not p_norm:
                    current_data = current_data[begin_data:end_data]

                if p_mode == 'svdn':
                    current_data = utils.normalize_arr(current_data)

                if p_mode == 'svdne':
                    current_data = utils.normalize_arr_with_range(current_data, min_value_svd, max_value_svd)

                images_data.append(current_data)


            # display all data using matplotlib (configure plt)
            fig = plt.figure(figsize=(30, 22))

            plt.rc('xtick', labelsize=22)
            plt.rc('ytick', labelsize=22)

            plt.title(p_scene + ' scene interval information SVD['+ str(begin_data) +', '+ str(end_data) +'], from scenes indices [' + str(begin_index) + ', '+ str(end_index) + '], ' + p_metric + ' metric, ' + p_mode + ', with step of ' + str(p_step) + ', svd norm ' + str(p_norm), fontsize=24)
            plt.ylabel('Component values', fontsize=24)
            plt.xlabel('Vector features', fontsize=24)

            for id, data in enumerate(images_data):

                p_label = p_scene + '_' + str(images_indices[id])

                if images_indices[id] == threshold_image_zone:
                    plt.plot(data, label=p_label + " (threshold mean)", lw=4, color='red')
                else:
                    plt.plot(data, label=p_label)

            plt.legend(bbox_to_anchor=(0.65, 0.98), loc=2, borderaxespad=0.2, fontsize=22)

            start_ylim, end_ylim = p_ylim
            plt.ylim(start_ylim, end_ylim)

            plot_name = p_scene + '_' + p_metric + '_' + str(p_step) + '_' + p_mode + '_' + str(p_norm) + '.png'
            plt.savefig(plot_name)

def main():


    # by default p_step value is 10 to enable all photos
    p_step = 10
    p_ylim = (0, 1)

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python display_svd_data_scene.py --scene A --interval "0,800" --indices "0, 900" --metric lab --mode svdne --step 50 --norm 0 --ylim "0, 0.1"')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:i:i:z:l:m:s:n:e:y", ["help=", "scene=", "interval=", "indices=", "metric=", "mode=", "step=", "norm=", "error=", "ylim="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python display_svd_data_scene.py --scene A --interval "0,800" --indices "0, 900" --metric lab --mode svdne --step 50 --norm 0 --ylim "0, 0.1"')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python display_svd_data_scene.py --scene A --interval "0,800" --indices "0, 900" --metric lab --mode svdne --step 50 --norm 0 --ylim "0, 0.1"')
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

    display_svd_values(p_scene, p_interval, p_indices, p_metric, p_mode, p_step, p_norm, p_ylim)

if __name__== "__main__":
    main()
