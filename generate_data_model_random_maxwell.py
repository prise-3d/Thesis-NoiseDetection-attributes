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
from ipfml import image_processing, metrics

config_filename   = "config"
zone_folder       = "zone"
min_max_filename  = "_min_max_values"
generic_output_file_svd = '_random.csv'
output_data_folder = 'data'

# define all scenes values, here only use Maxwell scenes
scenes_list = ['Appart1opt02', 'Cuisine01', 'SdbCentre', 'SdbDroite']
scenes_indexes = ['A', 'D', 'G', 'H']
choices = ['svd', 'svdn', 'svdne']
path = './fichiersSVD_light'
zones = np.arange(16)
seuil_expe_filename = 'seuilExpe'

min_value_interval = sys.maxsize
max_value_interval = 0

def construct_new_line(path_seuil, interval, line, norm, sep, index):
    begin, end = interval

    line_data = line.split(';')
    seuil = line_data[0]
    metrics = line_data[begin+1:end+1]

    metrics = [float(m) for m in metrics]

    if norm:
        metrics = image_processing.normalize_arr_with_range(metrics, min_value_interval, max_value_interval)

    with open(path_seuil, "r") as seuil_file:
        seuil_learned = int(seuil_file.readline().strip())

    if seuil_learned > int(seuil):
        line = '1'
    else:
        line = '0'

    for idx, val in enumerate(metrics):
        if index:
            line += " " + str(idx + 1)
        line += sep
        line += str(val)
    line += '\n'

    return line

def get_min_max_value_interval(_filename, _interval, _choice, _metric, _scenes = scenes_list, _nb_zones = 4, _percent = 1):

    global min_value_interval, max_value_interval

    scenes = os.listdir(path)

    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    for id_scene, folder_scene in enumerate(scenes):

        # only take care of maxwell scenes
        if folder_scene in scenes_list:

            scene_path = os.path.join(path, folder_scene)

            zones_folder = []
            # create zones list
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str
                zones_folder.append("zone"+index_str)

            # shuffle list of zones (=> randomly choose zones)
            random.shuffle(zones_folder)

            for id_zone, zone_folder in enumerate(zones_folder):
                zone_path = os.path.join(scene_path, zone_folder)
                data_filename = _metric + "_" + _choice + generic_output_file_svd
                data_file_path = os.path.join(zone_path, data_filename)

                # getting number of line and read randomly lines
                f = open(data_file_path)
                lines = f.readlines()

                counter = 0
                # check if user select current scene and zone to be part of training data set
                for line in lines:


                    begin, end = _interval

                    line_data = line.split(';')
                    metrics = line_data[begin+1:end+1]
                    metrics = [float(m) for m in metrics]

                    min_value = min(metrics)
                    max_value = max(metrics)

                    if min_value < min_value_interval:
                        min_value_interval = min_value

                    if max_value > max_value_interval:
                        max_value_interval = max_value

                    counter += 1


def generate_data_model(_filename, _interval, _choice, _metric, _scenes = scenes_list, _nb_zones = 4, _percent = 1, _norm = False, _sep=':', _index=True):

    output_train_filename = _filename + ".train"
    output_test_filename = _filename + ".test"

    if not '/' in output_train_filename:
        raise Exception("Please select filename with directory path to save data. Example : data/dataset")

    # create path if not exists
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    train_file = open(output_train_filename, 'w')
    test_file = open(output_test_filename, 'w')

    scenes = os.listdir(path)

    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    for id_scene, folder_scene in enumerate(scenes):

        # only take care of maxwell scenes
        if folder_scene in scenes_list:

            scene_path = os.path.join(path, folder_scene)

            zones_folder = []
            # create zones list
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str
                zones_folder.append("zone"+index_str)

            # shuffle list of zones (=> randomly choose zones)
            random.shuffle(zones_folder)

            for id_zone, zone_folder in enumerate(zones_folder):
                zone_path = os.path.join(scene_path, zone_folder)
                data_filename = _metric + "_" + _choice + generic_output_file_svd
                data_file_path = os.path.join(zone_path, data_filename)

                # getting number of line and read randomly lines
                f = open(data_file_path)
                lines = f.readlines()

                num_lines = len(lines)

                lines_indexes = np.arange(num_lines)
                random.shuffle(lines_indexes)

                path_seuil = os.path.join(zone_path, seuil_expe_filename)

                counter = 0
                # check if user select current scene and zone to be part of training data set
                for index in lines_indexes:
                    line = construct_new_line(path_seuil, _interval, lines[index], _norm, _sep, _index)

                    percent = counter / num_lines

                    if id_zone < _nb_zones and folder_scene in _scenes and percent <= _percent:
                        train_file.write(line)
                    else:
                        test_file.write(line)

                    counter += 1

                f.close()

    train_file.close()
    test_file.close()


def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python generate_data_model_random.py --output xxxx --interval 0,20  --kind svdne --metric lab --scenes "A, B, D" --nb_zones 5 --percent 0.7 --norm 1 --sep : --rowindex 1')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:i:k:s:n:p:r", ["help=", "output=", "interval=", "kind=", "metric=","scenes=", "nb_zones=", "percent=", "norm=", "sep=", "rowindex="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python generate_data_model_random.py --output xxxx --interval 0,20  --kind svdne --metric lab --scenes "A, B, D" --nb_zones 5 --percent 0.7 --norm 1 --sep : --rowindex 1')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python generate_data_model_random.py --output xxxx --interval 0,20  --kind svdne --metric lab --scenes "A, B, D" --nb_zones 5 --percent 0.7 --norm 1 --sep : --rowindex 1')
            sys.exit()
        elif o in ("-o", "--output"):
            p_filename = a
        elif o in ("-i", "--interval"):
            p_interval = list(map(int, a.split(',')))
        elif o in ("-k", "--kind"):
            p_kind = a
        elif o in ("-m", "--metric"):
            p_metric = a
        elif o in ("-s", "--scenes"):
            p_scenes = a.split(',')
        elif o in ("-n", "--nb_zones"):
            p_nb_zones = int(a)
        elif o in ("-n", "--norm"):
            if int(a) == 1:
                p_norm = True
            else:
                p_norm = False
        elif o in ("-p", "--percent"):
            p_percent = float(a)
        elif o in ("-s", "--sep"):
            p_sep = a
        elif o in ("-r", "--rowindex"):
            if int(a) == 1:
                p_rowindex = True
            else:
                p_rowindex = False
        else:
            assert False, "unhandled option"

    # getting scenes from indexes user selection
    scenes_selected = []

    for scene_id in p_scenes:
        index = scenes_indexes.index(scene_id.strip())
        scenes_selected.append(scenes_list[index])

    # find min max value if necessary to renormalize data
    if p_norm:
        get_min_max_value_interval(p_filename, p_interval, p_kind, p_metric, scenes_selected, p_nb_zones, p_percent)

    # create database using img folder (generate first time only)
    generate_data_model(p_filename, p_interval, p_kind, p_metric, scenes_selected, p_nb_zones, p_percent, p_norm, p_sep, p_rowindex)

if __name__== "__main__":
    main()
