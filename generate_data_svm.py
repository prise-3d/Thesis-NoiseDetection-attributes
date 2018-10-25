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

config_filename   = "config"
zone_folder       = "zone"
min_max_filename  = "min_max_values"
output_file_svd   = "SVD_LAB_test_im6.csv"
output_file_svdn  = "SVDN_LAB_test_im6.csv"
output_file_svdne = "SVDNE_LAB_test_im6.csv"

# define all scenes values
scenes = ['Appart1opt02', 'Bureau1', 'Cendrier', 'Cuisine01', 'EchecsBas', 'PNDVuePlongeante', 'SdbCentre', 'SdbDroite', 'Selles']
scenes_indexes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
choices = ['svd', 'svdn', 'svdne']
path = './fichiersSVD_light'
zones = np.arange(16)
file_choice = [output_file_svd, output_file_svdn, output_file_svdne]
seuil_expe_filename = 'seuilExpe'

def generate_data_svd_lab():
    """
    @brief Method which generates all .csv files from scenes photos
    @param path - path of scenes folder information
    @return nothing
    """

    # TODO :
    # - parcourir chaque dossier de scene
    scenes = os.listdir(path)

    for folder_scene in scenes:

        folder_path = path + "/" + folder_scene

        with open(folder_path + "/" + config_filename, "r") as config_file:
            last_image_name = config_file.readline().strip()
            prefix_image_name = config_file.readline().strip()
            start_index_image = config_file.readline().strip()
            end_index_image = config_file.readline().strip()
            step_counter = int(config_file.readline().strip())


        current_counter_index = int(start_index_image)
        end_counter_index = int(start_index_image)

        print(current_counter_index)
        while(current_counter_index <= end_index_image):
            print(current_counter_index)
            current_counter_index += step_counter

    # - récupérer les informations des fichiers de configurations
    # - création des fichiers de sortie SVD, SVDE, SVDNE

def construct_new_line(path_seuil, interval, line, sep, index):
    begin, end = interval

    line_data = line.split(';')
    seuil = line_data[0]
    metrics = line_data[begin+1:end+1]

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
        line += val
    line += '\n'

    return line

def generate_data_svm(_filename, _interval, _choice, _scenes = scenes, _zones = zones, _percent = 1, _sep=':', _index=True):

    output_train_filename = _filename + ".train"
    output_test_filename = _filename + ".test"

    if not '/' in output_train_filename:
        raise Exception("Please select filename with directory path to save data. Example : data/dataset")

    # create path if not exists
    output_folder = output_train_filename.split('/')[0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_file = open(output_train_filename, 'w')
    test_file = open(output_test_filename, 'w')

    scenes = os.listdir(path)

    if min_max_filename in scenes:
        scenes.remove(min_max_filename)

    for id_scene, folder_scene in enumerate(scenes):
        scene_path = path + "/" + folder_scene

        zones_folder = []
        # create zones list
        for index in zones:
            index_str = str(index)
            if len(index_str) < 2:
                index_str = "0" + index_str
            zones_folder.append("zone"+index_str)

        for id_zone, zone_folder in enumerate(zones_folder):
            zone_path = scene_path + "/" + zone_folder
            data_filename = file_choice[choices.index(_choice)]
            data_file_path = zone_path + "/" + data_filename

             # getting number of line and read randomly lines
            f = open(data_file_path)
            lines = f.readlines()

            num_lines = len(lines)

            lines_indexes = np.arange(num_lines)
            random.shuffle(lines_indexes)

            path_seuil = zone_path + "/" + seuil_expe_filename

            counter = 0
            # check if user select current scene and zone to be part of training data set
            for index in lines_indexes:
                line = construct_new_line(path_seuil, _interval, lines[index], _sep, _index)

                percent = counter / num_lines
                
                if id_zone in _zones and folder_scene in _scenes and percent <= _percent:
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
        print('python generate_data_svm.py --output xxxx --interval 0,20  --kind svdne --scenes "A, B, D" --zones "1, 2, 3" --percent 0.7 --sep ":" --rowindex "1"')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:i:k:s:z:p:r", ["help=", "output=", "interval=", "kind=", "scenes=", "zones=", "percent=", "sep=", "rowindex="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python generate_data_svm.py --output xxxx --interval 0,20  --kind svdne --scenes "A, B, D" --zones "1, 2, 3" --percent 0.7 --sep ":" --rowindex "1"')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python generate_data_svm.py --output xxxx --interval 0,20  --kind svdne --scenes "A, B, D" --zones "1, 2, 3" --percent 0.7 --sep ":" --rowindex "1"')
            sys.exit()
        elif o in ("-o", "--output"):
            p_filename = a
        elif o in ("-i", "--interval"):
            p_interval = list(map(int, a.split(',')))
        elif o in ("-k", "--kind"):
            p_kind = a
        elif o in ("-s", "--scenes"):
            p_scenes = a.split(',')
        elif o in ("-z", "--zones"):
            if ',' in a:
                p_zones = list(map(int, a.split(',')))
            else:
                p_zones = [a.strip()]
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
        scenes_selected.append(scenes[index])

    # create database using img folder (generate first time only)
    generate_data_svm(p_filename, p_interval, p_kind, scenes_selected, p_zones, p_percent, p_sep, p_rowindex)

if __name__== "__main__":
    main()