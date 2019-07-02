# main imports
import sys, os, argparse
import numpy as np

# image processing imports
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt

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

max_nb_bits         = 8
display_error       = False

error_data_choices  = ['mae', 'mse', 'ssim', 'psnr']


def get_error_distance(p_error, y_true, y_test):

    function_name = p_error

    try:
        error_method = getattr(fr_iqa, function_name)
    except AttributeError:
        raise NotImplementedError("Error `{}` not implement `{}`".format(fr_iqa.__name__, function_name))

    return error_method(y_true, y_test)


def display_svd_values(p_scene, p_interval, p_indices, p_feature, p_mode, p_step, p_norm, p_error, p_ylim):
    """
    @brief Method which gives information about svd curves from zone of picture
    @param p_scene, scene expected to show svd values
    @param p_interval, interval [begin, end] of svd data to display
    @param p_interval, interval [begin, end] of samples or minutes from render generation engine
    @param p_feature, feature computed to show
    @param p_mode, normalization's mode
    @param p_norm, normalization or not of selected svd data
    @param p_error, error feature used to display
    @param p_ylim, ylim choice to better display of data
    @return nothing
    """

    max_value_svd = 0
    min_value_svd = sys.maxsize

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    begin_data, end_data = p_interval
    begin_index, end_index = p_indices

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

            images_data = []
            images_path = []

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

            svd_data = []
           
            # for each images
            for id_img, img_path in enumerate(scene_images):
                
                current_quality_image = dt.get_scene_image_quality(img_path)

                img = Image.open(img_path)

                svd_values = get_svd_data(p_feature, img)

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
                        images_path.append(img_path)
                        svd_data.append(svd_values)

                    if threshold_mean < current_quality_image and not threshold_image_found:

                        threshold_image_found = True
                        threshold_image_zone = dt.get_scene_image_postfix(img_path)

                print('%.2f%%' % ((id_img + 1) / number_scene_image * 100))
                sys.stdout.write("\033[F")

            previous_data = []
            error_data = [0.]

            for id, data in enumerate(svd_data):

                current_data = data

                if not p_norm:
                    current_data = current_data[begin_data:end_data]

                if p_mode == 'svdn':
                    current_data = utils.normalize_arr(current_data)

                if p_mode == 'svdne':
                    current_data = utils.normalize_arr_with_range(current_data, min_value_svd, max_value_svd)

                images_data.append(current_data)

                # use of whole image data for computation of ssim or psnr
                if p_error == 'ssim' or p_error == 'psnr':
                    current_data = np.asarray(Image.open(images_path[id]))

                if len(previous_data) > 0:

                    current_error = get_error_distance(p_error, previous_data, current_data)
                    error_data.append(current_error)

                if len(previous_data) == 0:
                    previous_data = current_data

            # display all data using matplotlib (configure plt)
            gridsize = (3, 2)

            # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(30, 22))
            fig = plt.figure(figsize=(30, 22))
            ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
            ax2 = plt.subplot2grid(gridsize, (2, 0), colspan=2)


            ax1.set_title(p_scene + ' scene interval information SVD['+ str(begin_data) +', '+ str(end_data) +'], from scenes indices [' + str(begin_index) + ', '+ str(end_index) + '], ' + p_feature + ' feature, ' + p_mode + ', with step of ' + str(p_step) + ', svd norm ' + str(p_norm), fontsize=20)
            ax1.set_ylabel('Image samples or time (minutes) generation', fontsize=14)
            ax1.set_xlabel('Vector features', fontsize=16)

            for id, data in enumerate(images_data):
                
                current_quality_image = dt.get_scene_image_quality(images_path[id])
                current_quality_postfix = dt.get_scene_image_postfix(images_path[id])

                if display_error:
                    p_label = p_scene + '_' + current_quality_postfix + " | " + p_error + ": " + str(error_data[id])
                else:
                    p_label = p_scene + '_' + current_quality_postfix

                if current_quality_image == threshold_image_zone:
                    ax1.plot(data, label=p_label + " (threshold mean)", lw=4, color='red')
                else:
                    ax1.plot(data, label=p_label)

            ax1.legend(bbox_to_anchor=(0.7, 1), loc=2, borderaxespad=0.2, fontsize=14)

            start_ylim, end_ylim = p_ylim
            ax1.set_ylim(start_ylim, end_ylim)

            ax2.set_title(p_error + " information for whole step images")
            ax2.set_ylabel(p_error + ' error')
            ax2.set_xlabel('Number of samples per pixels or times')
            ax2.set_xticks(range(len(current_quality_image)))
            ax2.set_xticklabels(list(map(dt.get_scene_image_quality, current_quality_image)))
            ax2.plot(error_data)

            plot_name = p_scene + '_' + p_feature + '_' + str(p_step) + '_' + p_mode + '_' + str(p_norm) + '.png'
            plt.savefig(plot_name)

def main():

    parser = argparse.ArgumentParser(description="Display evolution of error on scene")

    parser.add_argument('--scene', type=str, help='scene index to use', choices=cfg.scenes_indices)
    parser.add_argument('--interval', type=str, help='Interval value to keep from svd', default='"0, 200"')
    parser.add_argument('--indices', type=str, help='Samples interval to display', default='"0, 900"')
    parser.add_argument('--feature', type=str, help='feature data choice', choices=features_choices)
    parser.add_argument('--mode', type=str, help='Kind of normalization level wished', choices=cfg.normalization_choices)
    parser.add_argument('--step', type=int, help='Each step samples to display', default=10)
    parser.add_argument('--norm', type=int, help='If values will be normalized or not', choices=[0, 1])
    parser.add_argument('--error', type=int, help='Way of computing error', choices=error_data_choices)
    parser.add_argument('--ylim', type=str, help='ylim interval to use', default='"0, 1"')

    args = parser.parse_args()

    p_scene    = scenes_list[scenes_indices.index(args.scene)]
    p_indices  = list(map(int, args.indices.split(',')))
    p_interval = list(map(int, args.interval.split(',')))
    p_feature   = args.feature
    p_mode     = args.mode
    p_step     = args.step
    p_norm     = args.norm
    p_error    = args.error
    p_ylim     = list(map(int, args.ylim.split(',')))

    display_svd_values(p_scene, p_interval, p_indices, p_feature, p_mode, p_step, p_norm, p_error, p_ylim)

if __name__== "__main__":
    main()
