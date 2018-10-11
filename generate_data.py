#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 21:02:42 2018

@author: jbuisine
"""

from __future__ import print_function
import sys, os
import numpy as np
import random
import time

config_filename   = "config"
zone_folder       = "zone"
min_max_filename  = "min_max_values"
output_file_svd   = "SVD_LAB_test_im6.csv"
output_file_svdn  = "SVDN_LAB_test_im6.csv"
output_file_svdne = "SVDNE_LAB_test_im6.csv" 

# define all scenes values
scenes = ['Appart1opt02', 'Bureau1', 'Cendrier', 'EchecsBas', 'PNDVuePlongeante', 'SdbCentre', 'SdbDroite', 'Selles']
choices = ['svd', 'svdn', 'svdne']
path = './data'
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

def construct_new_line(path_seuil, interval, line, sep):
    begin, end = interval

    line_data = line.split(';')
    seuil = line_data[0]
    metrics = line_data[begin+1:end]

    with open(path_seuil, "r") as seuil_file:
        seuil_learned = int(seuil_file.readline().strip())
       
    if seuil_learned > int(seuil):
        line = '0'
    else:
        line = '1'

    for idx, val in enumerate(metrics):
        line += " " + str(idx + 1) + ":" + val
    line += '\n'
    
    return line

def generate_data_svm(_filename, _interval, _choice, _scenes = scenes, _zones = zones, _percent = 1, _sep=':'):

    output_train_filename = _filename + ".train"
    output_test_filename = _filename + ".test"

    train_file = open(output_train_filename, 'w')
    test_file = open(output_test_filename, 'w')

    scenes = os.listdir(path)

    for id_scene, folder_scene in enumerate(scenes):
        scene_path = path + "/" + folder_scene
        
        print("Current path scene : " + scene_path)
        zones_folder = []
        # create zones list
        for index in zones:
            index_str = str(index)
            if len(index_str) < 2:
                index_str = "0" + index_str
            zones_folder.append("zone"+index_str)

        for id_zone, zone_folder in enumerate(zones_folder):
            print(zone_folder)
            zone_path = scene_path + "/" + zone_folder
            data_filename = file_choice[choices.index(_choice)]
            data_file_path = zone_path + "/" + data_filename
            print(data_file_path)

            print(id_zone in _zones)

             # getting number of line and read randomly lines
            f = open(data_file_path)       
            lines = f.readlines()
            #num_lines = sum(1 for line in open(data_file_path))
            
            num_lines = len(lines)

            lines_indexes = np.arange(num_lines)
            random.shuffle(lines_indexes)

            path_seuil = zone_path + "/" + seuil_expe_filename

            # check if user select current scene and zone to be part of training data set
            for index in lines_indexes:
                line = construct_new_line(path_seuil, _interval, lines[index], _sep)

                if id_zone in _zones and folder_scene in _scenes:
                    train_file.write(line)
                else:
                    test_file.write(line)

            f.close()

    train_file.close()
    test_file.close()
                


def main():

    # create database using img folder (generate first time only)
    generate_data_svm('test', [20, 100], 'svdne', _scenes=['Appart1opt02', 'Bureau1', 'Cendrier', 'EchecsBas', 'PNDVuePlongeante', 'SdbCentre'], _zones=[2, 3, 7, 8, 9, 10, 15, 0])

if __name__== "__main__":
    main()
