#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 21:02:42 2018

@author: jbuisine
"""

from __future__ import print_function
import sys, os, argparse

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

max_nb_bits = 8

integral_area_choices = ['trapz', 'simps']

def get_area_under_curve(p_area, p_data):

    noise_method = None
    function_name = 'integral_area_' + p_area

    try:
        area_method = getattr(utils, function_name)
    except AttributeError:
        raise NotImplementedError("Error `{}` not implement `{}`".format(utils.__name__, function_name))

    return area_method(p_data, dx=800)


def display_svd_values(p_scene, p_interval, p_indices, p_metric, p_mode, p_step, p_norm, p_area, p_ylim):
    """
    @brief Method which gives information about svd curves from zone of picture
    @param p_scene, scene expected to show svd values
    @param p_interval, interval [begin, end] of svd data to display
    @param p_interval, interval [begin, end] of samples or minutes from render generation engine
    @param p_metric, metric computed to show
    @param p_mode, normalization's mode
    @param p_norm, normalization or not of selected svd data
    @param p_area, area method name to compute area under curve
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

            previous_data = []
            area_data = []

            for id, data in enumerate(svd_data):

                current_data = data

                if not p_norm:
                    current_data = current_data[begin_data:end_data]

                if p_mode == 'svdn':
                    current_data = utils.normalize_arr(current_data)

                if p_mode == 'svdne':
                    current_data = utils.normalize_arr_with_range(current_data, min_value_svd, max_value_svd)

                images_data.append(current_data)

                # not use this script for 'sub_blocks_stats'
                current_area = get_area_under_curve(p_area, current_data)
                area_data.append(current_area)

            # display all data using matplotlib (configure plt)
            gridsize = (3, 2)

            # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(30, 22))
            fig = plt.figure(figsize=(30, 22))
            ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
            ax2 = plt.subplot2grid(gridsize, (2, 0), colspan=2)


            ax1.set_title(p_scene + ' scene interval information SVD['+ str(begin_data) +', '+ str(end_data) +'], from scenes indices [' + str(begin_index) + ', '+ str(end_index) + ']' + p_metric + ' metric, ' + p_mode + ', with step of ' + str(p_step) + ', svd norm ' + str(p_norm), fontsize=20)
            ax1.set_ylabel('Image samples or time (minutes) generation', fontsize=14)
            ax1.set_xlabel('Vector features', fontsize=16)

            for id, data in enumerate(images_data):

                p_label = p_scene + '_' + str(images_indices[id]) + " | " + p_area + ": " + str(area_data[id])

                if images_indices[id] == threshold_image_zone:
                    ax1.plot(data, label=p_label, lw=4, color='red')
                    threshold_id = id
                else:
                    ax1.plot(data, label=p_label)

            ax1.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2, fontsize=14)

            start_ylim, end_ylim = p_ylim
            ax1.set_ylim(start_ylim, end_ylim)

            ax2.set_title(p_area + " information for whole step images")
            ax2.set_ylabel(p_area + ' area values')
            ax2.set_xlabel('Number of samples per pixels or times')
            ax2.set_xticks(range(len(images_indices)))
            ax2.set_xticklabels(list(map(int, images_indices)))
            ax2.plot([threshold_id, threshold_id], [np.min(area_data), np.max(area_data)], 'k-', lw=2, color='red')
            ax2.plot(area_data)

            plt.show()

def main():

    parser = argparse.ArgumentParser(description="Display area under curve data on scene")

    parser.add_argument('--scene', type=str, help='scene index to use', choices=cfg.scenes_indices)
    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--indices', type=str, help='Samples interval to display', default='"0, 900"')
    parser.add_argument('--metric', type=str, help='Metric data choice', choices=metric_choices)
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=cfg.normalization_choices)
    parser.add_argument('--step', type=int, help='Each step samples to display', default=10)
    parser.add_argument('--norm', type=int, help='If values will be normalized or not', choices=[0, 1])
    parser.add_argument('--area', type=int, help='Way of computing area under curve', choices=integral_area_choices)
    parser.add_argument('--ylim', type=str, help='ylim interval to use', default='"0, 1"')

    args = parser.parse_args()

    p_scene    = scenes_list[scenes_indices.index(args.scene)]
    p_indices  = list(map(int, args.indices.split(',')))
    p_interval = list(map(int, args.interval.split(',')))
    p_metric   = args.metric
    p_mode     = args.mode
    p_step     = args.step
    p_norm     = args.norm
    p_area     = args.area
    p_ylim     = list(map(int, args.ylim.split(',')))

    display_svd_values(p_scene, p_interval, p_indices, p_metric, p_mode, p_step, p_norm, p_area, p_ylim)

if __name__== "__main__":
    main()
