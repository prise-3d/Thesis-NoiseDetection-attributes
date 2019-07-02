# main imports
import sys, os, argparse
import numpy as np

# image processing imports
from PIL import Image
import matplotlib.pyplot as plt

from ipfml.processing import segmentation
import ipfml.iqa.fr as fr_iqa
from ipfml import utils

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt
from data_attributes import get_svd_data

# getting configuration information
zone_folder         = cfg.zone_folder
min_max_filename    = cfg.min_max_filename_extension

# define all scenes values
scenes_list         = cfg.scenes_names
scenes_indices      = cfg.scenes_indices
choices             = cfg.normalization_choices
path                = cfg.dataset_path
zones               = cfg.zones_indices
seuil_expe_filename = cfg.seuil_expe_filename

features_choices    = cfg.features_choices_labels

generic_output_file_svd = '_random.csv'

max_nb_bits = 8
min_value_interval = sys.maxsize
max_value_interval = 0

def get_min_max_value_interval(_scene, _interval, _feature):

    global min_value_interval, max_value_interval

    scenes = os.listdir(path)

    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    for folder_scene in scenes:

        # only take care of current scene
        if folder_scene == _scene:

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
                data_filename = _feature + "_svd" + generic_output_file_svd
                data_file_path = os.path.join(zone_path, data_filename)

                # getting number of line and read randomly lines
                f = open(data_file_path)
                lines = f.readlines()

                # check if user select current scene and zone to be part of training data set
                for line in lines:

                    begin, end = _interval

                    line_data = line.split(';')
                    features = line_data[begin+1:end+1]
                    features = [float(m) for m in features]

                    min_value = min(features)
                    max_value = max(features)

                    if min_value < min_value_interval:
                        min_value_interval = min_value

                    if max_value > max_value_interval:
                        max_value_interval = max_value


def display_svd_values(p_scene, p_interval, p_indices, p_zone, p_feature, p_mode, p_step, p_norm, p_ylim):
    """
    @brief Method which gives information about svd curves from zone of picture
    @param p_scene, scene expected to show svd values
    @param p_interval, interval [begin, end] of svd data to display
    @param p_interval, interval [begin, end] of samples or minutes from render generation engine
    @param p_zone, zone's identifier of picture
    @param p_feature, feature computed to show
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

    data_min_max_filename = os.path.join(path, p_feature + min_max_filename)

    # go ahead each scenes
    for folder_scene in scenes:

        if p_scene == folder_scene:
            scene_path = os.path.join(path, folder_scene)

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
            images_path = []

            zone_folder = zones_folder[p_zone]

            zone_path = os.path.join(scene_path, zone_folder)

            # get threshold information
            path_seuil = os.path.join(zone_path, seuil_expe_filename)

            # open treshold path and get this information
            with open(path_seuil, "r") as seuil_file:
                seuil_learned = int(seuil_file.readline().strip())

            threshold_image_found = False

            # get all images of folder
            scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])

            # for each images
            for img_path in scene_images:
                    
                current_quality_image = dt.get_scene_image_quality(img_path)

                if current_quality_image % p_step == 0:
                    if current_quality_image >= begin_index and current_quality_image <= end_index:
                        images_path.append(dt.get_scene_image_postfix(img_path))

                    if seuil_learned < current_quality_image and not threshold_image_found:

                        threshold_image_found = True
                        threshold_image_zone = dt.get_scene_image_postfix(img_path)


            for img_path in images_path:

                current_img = Image.open(img_path)
                img_blocks = segmentation.divide_in_blocks(current_img, (200, 200))

                # getting expected block id
                block = img_blocks[p_zone]

                # get data from mode
                # Here you can add the way you compute data
                data = get_svd_data(p_feature, block)

                # TODO : improve part of this code to get correct min / max values
                if p_norm:
                    data = data[begin_data:end_data]

                ##################
                # Data mode part #
                ##################

                if p_mode == 'svdne':

                    # getting max and min information from min_max_filename
                    if not p_norm:
                        with open(data_min_max_filename, 'r') as f:
                            min_val = float(f.readline())
                            max_val = float(f.readline())
                    else:
                        min_val = min_value_interval
                        max_val = max_value_interval

                    data = utils.normalize_arr_with_range(data, min_val, max_val)

                if p_mode == 'svdn':
                    data = utils.normalize_arr(data)

                if not p_norm:
                    zones_images_data.append(data[begin_data:end_data])
                else:
                    zones_images_data.append(data)

            plt.title(p_scene + ' scene interval information SVD['+ str(begin_data) +', '+ str(end_data) +'], from scenes indices [' + str(begin_index) + ', '+ str(end_index) + ']' + p_feature + ' feature, ' + p_mode + ', with step of ' + str(p_step) + ', svd norm ' + str(p_norm), fontsize=20)
            plt.ylabel('Image samples or time (minutes) generation', fontsize=14)
            plt.xlabel('Vector features', fontsize=16)

            for id, data in enumerate(zones_images_data):

                p_label = p_scene + "_" + images_path[id]

                if images_path[id] == threshold_image_zone:
                    plt.plot(data, label=p_label, lw=4, color='red')
                else:
                    plt.plot(data, label=p_label)

            plt.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.2, fontsize=14)

            start_ylim, end_ylim = p_ylim
            plt.ylim(start_ylim, end_ylim)

            plt.show()

def main():

    parser = argparse.ArgumentParser(description="Display SVD data of scene zone")

    parser.add_argument('--scene', type=str, help='scene index to use', choices=cfg.scenes_indices)
    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--indices', type=str, help='Samples interval to display', default='"0, 900"')
    parser.add_argument('--zone', type=int, help='Zone to display', choices=list(range(0, 16)))
    parser.add_argument('--feature', type=str, help='feature data choice', choices=features_choices)
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=cfg.normalization_choices)
    parser.add_argument('--step', type=int, help='Each step samples to display', default=10)
    parser.add_argument('--norm', type=int, help='If values will be normalized or not', choices=[0, 1])
    parser.add_argument('--ylim', type=str, help='ylim interval to use', default='"0, 1"')

    args = parser.parse_args()

    p_scene    = scenes_list[scenes_indices.index(args.scene)]
    p_indices  = list(map(int, args.indices.split(',')))
    p_interval = list(map(int, args.interval.split(',')))
    p_zone     = args.zone
    p_feature   = args.feature
    p_mode     = args.mode
    p_step     = args.step
    p_norm     = args.norm
    p_ylim     = list(map(int, args.ylim.split(',')))

    display_svd_values(p_scene, p_interval, p_indices, p_zone, p_feature, p_mode, p_step, p_norm, p_ylim)

if __name__== "__main__":
    main()
