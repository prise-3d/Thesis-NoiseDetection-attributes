# main imports
import sys, os, argparse
import numpy as np

# image processing imports
from PIL import Image
import matplotlib.pyplot as plt

import ipfml.iqa.fr as fr_iqa
from ipfml import utils

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt
from data_attributes import get_image_features

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

max_nb_bits = 8

integral_area_choices = ['trapz', 'simps']

def get_area_under_curve(p_area, p_data):

    function_name = 'integral_area_' + p_area

    try:
        area_method = getattr(utils, function_name)
    except AttributeError:
        raise NotImplementedError("Error `{}` not implement `{}`".format(utils.__name__, function_name))

    return area_method(p_data, dx=800)


def display_svd_values(p_interval, p_indices, p_metric, p_mode, p_step, p_norm, p_area, p_ylim):
    """
    @brief Method which gives information about svd curves from zone of picture
    @param p_interval, interval [begin, end] of svd data to display
    @param p_indices, indices to display
    @param p_feature, feature computed to show
    @param p_mode, normalization's mode
    @param p_norm, normalization or not of selected svd data
    @param p_area, area method name to compute area under curve
    @param p_ylim, ylim choice to better display of data
    @return nothing
    """

    image_indices = []

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    begin_data, end_data = p_interval
    begin_index, end_index = p_indices

    # Store all informations about scenes
    scenes_area_data = []
    scenes_images_indices = []
    scenes_threshold_mean = []

    # go ahead each scenes
    for folder_scene in scenes:

        max_value_svd = 0
        min_value_svd = sys.maxsize

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

        # store data information for current scene
        images_data = []
        images_indices = []
        threshold_learned_zones = []

        # get all images of folder
        scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
        number_scene_image = len(scene_images)

        for id, zone_folder in enumerate(zones_folder):

            # get threshold information
            zone_path = os.path.join(scene_path, zone_folder)
            path_seuil = os.path.join(zone_path, seuil_expe_filename)

            # open treshold path and get this information
            with open(path_seuil, "r") as seuil_file:
                threshold_learned = int(seuil_file.readline().strip())
                threshold_learned_zones.append(threshold_learned)

        threshold_mean = np.mean(np.asarray(threshold_learned_zones))
        threshold_image_found = False
        scenes_threshold_mean.append(int(threshold_mean / p_step))

        svd_data = []

        # for each images
        for id_img, img_path in enumerate(scene_images):
            
            current_quality_image = dt.get_scene_image_quality(img_path)

            img = Image.open(img_path)

            svd_values = get_image_features(p_metric, img)

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
            if current_quality_image % p_step == 0:
                if current_quality_image >= begin_index and current_quality_image <= end_index:
                    images_indices.append(dt.get_scene_image_postfix(img_path))
                    svd_data.append(svd_values)

                if threshold_mean < current_quality_image and not threshold_image_found:

                    threshold_image_found = True

            print('%.2f%%' % ((id_img + 1) / number_scene_image * 100))
            sys.stdout.write("\033[F")


            # all indices of picture to plot
        print("Scene %s : %s" % (folder_scene, images_indices))

        scenes_images_indices.append(image_indices)

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

        scenes_area_data.append(area_data)

    # display all data using matplotlib (configure plt)
    plt.title('Scenes area interval information SVD['+ str(begin_data) +', '+ str(end_data) +'], from scenes indices [' + str(begin_index) + ', '+ str(end_index) + ']' + p_metric + ' metric, ' + p_mode + ', with step of ' + str(p_step) + ', svd norm ' + str(p_norm), fontsize=20)
    plt.ylabel('Image samples or time (minutes) generation', fontsize=14)
    plt.xlabel('Vector features', fontsize=16)

    plt.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2, fontsize=14)

    for id, area_data in enumerate(scenes_area_data):

        threshold_id = 0
        scene_name = scenes[id]
        image_indices = scenes_images_indices[id]

        p_label = scene_name + '_' + str(images_indices[id])

        threshold_id = scenes_threshold_mean[id]

        print(p_label)

        plt.plot(area_data, label=p_label)
        #ax2.set_xticks(range(len(images_indices)))
        #ax2.set_xticklabels(list(map(int, images_indices)))
        if threshold_id != 0:
            print("Plot threshold ", threshold_id)
            plt.plot([threshold_id, threshold_id], [np.min(area_data), np.max(area_data)], 'k-', lw=2, color='red')


    start_ylim, end_ylim = p_ylim
    plt.ylim(start_ylim, end_ylim)

    plt.show()

def main():

    parser = argparse.ArgumentParser(description="Display area under curve on scene")

    #parser.add_argument('--scene', type=str, help='scene index to use', choices=cfg.scenes_indices)
    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--indices', type=str, help='Samples interval to display', default='"0, 900"')
    parser.add_argument('--feature', type=str, help='Metric data choice', choices=features_choices)
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=cfg.normalization_choices)
    parser.add_argument('--step', type=int, help='Each step samples to display', default=10)
    parser.add_argument('--norm', type=int, help='If values will be normalized or not', choices=[0, 1])
    parser.add_argument('--area', type=int, help='Way of computing area under curve', choices=integral_area_choices)
    parser.add_argument('--ylim', type=str, help='ylim interval to use', default='"0, 1"')

    args = parser.parse_args()

    #p_scene    = scenes_list[scenes_indices.index(args.scene)]
    p_indices  = list(map(int, args.indices.split(',')))
    p_interval = list(map(int, args.interval.split(',')))
    p_feature  = args.feature
    p_mode     = args.mode
    p_step     = args.step
    p_norm     = args.norm
    p_area     = args.area
    p_ylim     = list(map(int, args.ylim.split(',')))

    display_svd_values(p_interval, p_indices, p_feature, p_mode, p_step, p_norm, p_area, p_ylim)

if __name__== "__main__":
    main()
