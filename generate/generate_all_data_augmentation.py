# main imports
import sys, os, argparse
import numpy as np
import random
import time
import json

# image processing imports
from PIL import Image

from ipfml.processing import transform, segmentation
from ipfml import utils

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt
from data_attributes import get_image_features


# getting configuration information
zone_folder             = cfg.zone_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes values
scenes_list             = cfg.scenes_names
scenes_indexes          = cfg.scenes_indices
choices                 = cfg.normalization_choices
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

features_choices        = cfg.features_choices_labels
output_data_folder      = cfg.output_data_folder

data_augmented_filename = cfg.data_augmented_filename
generic_output_file_svd = '_random.csv'

def generate_data_svd(data_type, mode, path):
    """
    @brief Method which generates all .csv files from scenes
    @param data_type,  feature choice
    @param mode, normalization choice
    @param path, data augmented path
    @return nothing
    """

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename and generic_output_file_svd not in s]

    # keep in memory min and max data found from data_type
    min_val_found = sys.maxsize
    max_val_found = 0

    data_min_max_filename = os.path.join(path, data_type + min_max_filename)
    data_filename = os.path.join(path, data_augmented_filename)

    # getting output filename
    output_svd_filename = data_type + "_" + mode + generic_output_file_svd

    current_file = open(os.path.join(path, output_svd_filename), 'w')

    with open(data_filename, 'r') as f:

        lines = f.readlines()
        number_of_images = len(lines)

        for index, line in enumerate(lines):        
            
            data = line.split(';')

            scene_name = data[0]
            number_of_samples = data[2]
            label_img = data[3]
            img_path = data[4].replace('\n', '')

            block = Image.open(os.path.join(path, img_path))
         
            ###########################
            # feature computation part #
            ###########################

            data = get_image_features(data_type, block)

            ##################
            # Data mode part #
            ##################

            # modify data depending mode
            if mode == 'svdne':

                # getting max and min information from min_max_filename
                with open(data_min_max_filename, 'r') as f:
                    min_val = float(f.readline())
                    max_val = float(f.readline())

                data = utils.normalize_arr_with_range(data, min_val, max_val)

            if mode == 'svdn':
                data = utils.normalize_arr(data)

            # save min and max found from dataset in order to normalize data using whole data known
            if mode == 'svd':

                current_min = data.min()
                current_max = data.max()

                if current_min < min_val_found:
                    min_val_found = current_min

                if current_max > max_val_found:
                    max_val_found = current_max

            # add of index
            current_file.write(scene_name + ';' + number_of_samples + ';' + label_img + ';')

            for val in data:
                current_file.write(str(val) + ";")
                
            print(data_type + "_" + mode + "_" + scene_name + " - " + "{0:.2f}".format((index + 1) / number_of_images * 100.) + "%")
            sys.stdout.write("\033[F")

            current_file.write('\n')

        print('\n')

    # save current information about min file found
    if mode == 'svd':
        with open(data_min_max_filename, 'w') as f:
            f.write(str(min_val_found) + '\n')
            f.write(str(max_val_found) + '\n')

    print("%s_%s : end of data generation\n" % (data_type, mode))


def main():

    parser = argparse.ArgumentParser(description="Compute and prepare data of feature of all scenes (keep in memory min and max value found)")

    parser.add_argument('--feature', type=str, 
                                    help="feature choice in order to compute data (use 'all' if all features are needed)")
    parser.add_argument('--folder', type=str, help="folder which contains the whole dataset")

    args = parser.parse_args()

    p_feature = args.feature
    p_folder  = args.folder

    # generate all or specific feature data
    if p_feature == 'all':
        for m in features_choices:
            generate_data_svd(m, 'svd', p_folder)
            generate_data_svd(m, 'svdn', p_folder)
            generate_data_svd(m, 'svdne', p_folder)
    else:

        if p_feature not in features_choices:
            raise ValueError('Unknown feature choice : ', features_choices)
            
        generate_data_svd(p_feature, 'svd', p_folder)
        generate_data_svd(p_feature, 'svdn', p_folder)
        generate_data_svd(p_feature, 'svdne', p_folder)

if __name__== "__main__":
    main()
