# main imports
import sys, os, argparse
import numpy as np
import pandas as pd
import subprocess
import random

# image processing imports
from PIL import Image

from ipfml import utils

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt
from data_attributes import get_image_features


# getting configuration information
learned_folder          = cfg.learned_zones_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes variables
all_scenes_list         = cfg.scenes_names
all_scenes_indices      = cfg.scenes_indices

renderer_choices        = cfg.renderer_choices
normalization_choices   = cfg.normalization_choices
path                    = cfg.dataset_path
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

features_choices        = cfg.features_choices_labels
output_data_folder      = cfg.output_data_folder
custom_min_max_folder   = cfg.min_max_custom_folder
min_max_ext             = cfg.min_max_filename_extension

generic_output_file_svd = '_random.csv'

min_value_interval      = sys.maxsize
max_value_interval      = 0


def construct_new_line(path_seuil, indices, line, choice, norm):

    # increase indices values by one to avoid label
    f = lambda x : x + 1
    indices = f(indices)

    line_data = np.array(line.split(';'))
    seuil = line_data[0]
    features = line_data[indices]
    features = features.astype('float32')

    # TODO : check if it's always necessary to do that (loss of information for svd)
    if norm:
        if choice == 'svdne':
            features = utils.normalize_arr_with_range(features, min_value_interval, max_value_interval)
        if choice == 'svdn':
            features = utils.normalize_arr(features)

    with open(path_seuil, "r") as seuil_file:
        seuil_learned = int(seuil_file.readline().strip())

    if seuil_learned > int(seuil):
        line = '1'
    else:
        line = '0'

    for val in features:
        line += ';'
        line += str(val)
    line += '\n'

    return line

def get_min_max_value_interval(_scenes_list, _indices, _feature):

    global min_value_interval, max_value_interval

    # increase indices values by one to avoid label
    f = lambda x : x + 1
    _indices = f(_indices)

    scenes = os.listdir(path)

    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    for folder_scene in scenes:

        # only take care of maxwell scenes
        if folder_scene in _scenes_list:

            scene_path = os.path.join(path, folder_scene)

            zones_folder = []
            # create zones list
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str
                zones_folder.append("zone"+index_str)

            for zone_folder in zones_folder:

                zone_path = os.path.join(scene_path, zone_folder)

                # if custom normalization choices then we use svd values not already normalized
                data_filename = _feature + "_svd"+ generic_output_file_svd

                data_file_path = os.path.join(zone_path, data_filename)

                # getting number of line and read randomly lines
                f = open(data_file_path)
                lines = f.readlines()

                # check if user select current scene and zone to be part of training data set
                for line in lines:

                    line_data = np.array(line.split(';'))

                    features = line_data[[_indices]]
                    features = [float(m) for m in features]

                    min_value = min(features)
                    max_value = max(features)

                    if min_value < min_value_interval:
                        min_value_interval = min_value

                    if max_value > max_value_interval:
                        max_value_interval = max_value


def generate_data_model(_scenes_list, _filename, _interval, _choice, _feature, _scenes, _nb_zones = 4, _percent = 1, _random=0, _step=1, _custom = False):

    output_train_filename = _filename + ".train"
    output_test_filename = _filename + ".test"

    if not '/' in output_train_filename:
        raise Exception("Please select filename with directory path to save data. Example : data/dataset")

    # create path if not exists
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    train_file_data = []
    test_file_data  = []

    for folder_scene in _scenes_list:

        scene_path = os.path.join(path, folder_scene)

        zones_indices = zones

        # shuffle list of zones (=> randomly choose zones)
        # only in random mode
        if _random:
            random.shuffle(zones_indices)

        # store zones learned
        learned_zones_indices = zones_indices[:_nb_zones]

        # write into file
        folder_learned_path = os.path.join(learned_folder, _filename.split('/')[1])

        if not os.path.exists(folder_learned_path):
            os.makedirs(folder_learned_path)

        file_learned_path = os.path.join(folder_learned_path, folder_scene + '.csv')

        with open(file_learned_path, 'w') as f:
            for i in learned_zones_indices:
                f.write(str(i) + ';')

        for id_zone, index_folder in enumerate(zones_indices):

            index_str = str(index_folder)
            if len(index_str) < 2:
                index_str = "0" + index_str
            current_zone_folder = "zone" + index_str

            zone_path = os.path.join(scene_path, current_zone_folder)

            # if custom normalization choices then we use svd values not already normalized
            if _custom:
                data_filename = _feature + "_svd"+ generic_output_file_svd
            else:
                data_filename = _feature + "_" + _choice + generic_output_file_svd

            data_file_path = os.path.join(zone_path, data_filename)

            # getting number of line and read randomly lines
            f = open(data_file_path)
            lines = f.readlines()

            num_lines = len(lines)

            # randomly shuffle image
            if _random:
                random.shuffle(lines)

            path_seuil = os.path.join(zone_path, seuil_expe_filename)

            counter = 0
            # check if user select current scene and zone to be part of training data set
            for data in lines:

                percent = counter / num_lines
                image_index = int(data.split(';')[0])

                if image_index % _step == 0:
                    line = construct_new_line(path_seuil, _interval, data, _choice, _custom)

                    if id_zone < _nb_zones and folder_scene in _scenes and percent <= _percent:
                        train_file_data.append(line)
                    else:
                        test_file_data.append(line)

                counter += 1

            f.close()

    train_file = open(output_train_filename, 'w')
    test_file = open(output_test_filename, 'w')

    for line in train_file_data:
        train_file.write(line)

    for line in test_file_data:
        test_file.write(line)

    train_file.close()
    test_file.close()


def main():

    # getting all params
    parser = argparse.ArgumentParser(description="Generate data for model using correlation matrix information from data")

    parser.add_argument('--output', type=str, help='output file name desired (.train and .test)')
    parser.add_argument('--n', type=int, help='Number of features wanted')
    parser.add_argument('--highest', type=int, help='Specify if highest or lowest values are wishes', choices=[0, 1])
    parser.add_argument('--label', type=int, help='Specify if label correlation is used or not', choices=[0, 1])
    parser.add_argument('--kind', type=str, help='Kind of normalization level wished', choices=normalization_choices)
    parser.add_argument('--feature', type=str, help='feature data choice', choices=features_choices)
    parser.add_argument('--scenes', type=str, help='List of scenes to use for training data')
    parser.add_argument('--nb_zones', type=int, help='Number of zones to use for training data set')
    parser.add_argument('--random', type=int, help='Data will be randomly filled or not', choices=[0, 1])
    parser.add_argument('--percent', type=float, help='Percent of data use for train and test dataset (by default 1)')
    parser.add_argument('--step', type=int, help='Photo step to keep for build datasets', default=1)
    parser.add_argument('--renderer', type=str, help='Renderer choice in order to limit scenes used', choices=renderer_choices, default='all')
    parser.add_argument('--custom', type=str, help='Name of custom min max file if use of renormalization of data', default=False)

    args = parser.parse_args()

    p_filename = args.output
    p_n        = args.n
    p_highest  = args.highest
    p_label    = args.label
    p_kind     = args.kind
    p_feature  = args.feature
    p_scenes   = args.scenes.split(',')
    p_nb_zones = args.nb_zones
    p_random   = args.random
    p_percent  = args.percent
    p_step     = args.step
    p_renderer = args.renderer
    p_custom   = args.custom

    # list all possibles choices of renderer
    scenes_list = dt.get_renderer_scenes_names(p_renderer)
    scenes_indices = dt.get_renderer_scenes_indices(p_renderer)

    # getting scenes from indexes user selection
    scenes_selected = []

    for scene_id in p_scenes:
        index = scenes_indices.index(scene_id.strip())
        scenes_selected.append(scenes_list[index])

    # Get indices to keep from correlation information
    # compute temp data file to get correlation information
    temp_filename = 'temp'
    temp_filename_path = os.path.join(cfg.output_data_folder, temp_filename)

    cmd = ['python', 'generate_data_model_random.py',
            '--output', temp_filename_path,
            '--interval', '0, 200',
            '--kind', p_kind,
            '--feature', p_feature,
            '--scenes', args.scenes,
            '--nb_zones', str(16),
            '--random', str(int(p_random)),
            '--percent', str(p_percent),
            '--step', str(p_step),
            '--each', str(1),
            '--renderer', p_renderer,
            '--custom', temp_filename + min_max_ext]

    subprocess.Popen(cmd).wait()

    temp_data_file_path = temp_filename_path + '.train'
    df = pd.read_csv(temp_data_file_path, sep=';', header=None)

    indices = []

    # compute correlation matrix from whole data scenes of renderer (using or not label column)
    if p_label:

        # compute pearson correlation between features and label
        corr = df.corr()

        features_corr = []

        for id_row, row in enumerate(corr):
            for id_col, val in enumerate(corr[row]):
                if id_col == 0 and id_row != 0:
                    features_corr.append(abs(val))

    else:
        df = df.drop(df.columns[[0]], axis=1)

        # compute pearson correlation between features using only features
        corr = df[1:200].corr()

        features_corr = []

        for id_row, row in enumerate(corr):
            correlation_score = 0
            for id_col, val in enumerate(corr[row]):
                if id_col != id_row:
                    correlation_score += abs(val)

            features_corr.append(correlation_score)

    # find `n` min or max indices to keep
    if p_highest:
        indices = utils.get_indices_of_highest_values(features_corr, p_n)
    else:
        indices = utils.get_indices_of_lowest_values(features_corr, p_n)

    indices = np.sort(indices)

    # save indices found
    if not os.path.exists(cfg.correlation_indices_folder):
        os.makedirs(cfg.correlation_indices_folder)

    indices_file_path = os.path.join(cfg.correlation_indices_folder, p_filename.replace(cfg.output_data_folder + '/', '') + '.csv')

    with open(indices_file_path, 'w') as f:
        for i in indices:
            f.write(str(i) + ';')

    # find min max value if necessary to renormalize data from `n` indices found
    if p_custom:
        get_min_max_value_interval(scenes_list, indices, p_feature)

        # write new file to save
        if not os.path.exists(custom_min_max_folder):
            os.makedirs(custom_min_max_folder)

        min_max_current_filename = p_filename.replace(cfg.output_data_folder + '/', '').replace('deep_keras_', '') + min_max_filename
        min_max_filename_path = os.path.join(custom_min_max_folder, min_max_current_filename)

        print(min_max_filename_path)
        with open(min_max_filename_path, 'w') as f:
            f.write(str(min_value_interval) + '\n')
            f.write(str(max_value_interval) + '\n')

    # create database using img folder (generate first time only)
    generate_data_model(scenes_list, p_filename, indices, p_kind, p_feature, scenes_selected, p_nb_zones, p_percent, p_random, p_step, p_custom)

if __name__== "__main__":
    main()
