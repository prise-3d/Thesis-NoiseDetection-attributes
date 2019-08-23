# main imports
import sys, os, argparse
import numpy as np
import time
import random
import math

# image processing imports
from PIL import Image

from ipfml.processing import transform, segmentation
from ipfml import utils

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt


# getting configuration information
zone_folder             = cfg.zone_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes values
scenes_list             = cfg.scenes_names
scenes_indexes          = cfg.scenes_indices
choices                 = cfg.normalization_choices
path                    = cfg.dataset_path
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

features_choices        = cfg.features_choices_labels
output_data_folder      = cfg.output_data_folder


image_scene_size        = (800, 800)
image_zone_size         = (200, 200)
possible_point_zone     = tuple(np.asarray(image_scene_size) - np.array(image_zone_size))
data_augmented_filename = cfg.data_augmented_filename

def main():
    
    parser = argparse.ArgumentParser(description="Compute and prepare data augmentation of scenes")

    parser.add_argument('--output', type=str, help="output folder expected", required=True)
    parser.add_argument('--number', type=int, help="number of images for each sample of scene", required=True)
    parser.add_argument('--rotation', type=bool, help="", required=True, default=False)

    args = parser.parse_args()

    p_output   = args.output
    p_number   = args.number
    p_rotation = args.rotation

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    # getting image zone size and usefull information
    zone_width, zone_height = image_zone_size
    scene_width, scene_height = image_scene_size
    nb_x_parts = math.floor(scene_width / zone_width)

    output_dataset_filename_path = os.path.join(p_output, data_augmented_filename)

    # go ahead each scenes
    for folder_scene in scenes:

        scene_path = os.path.join(path, folder_scene)

        # build output scene path
        output_scene_path = os.path.join(p_output, folder_scene)

        if not os.path.exists(output_scene_path):
            os.makedirs(output_scene_path)

        # construct each zones folder name
        zones_folder = []
        zones_threshold = []

        # get zones list info
        for index in zones:
            index_str = str(index)
            if len(index_str) < 2:
                index_str = "0" + index_str

            current_zone = "zone"+index_str
            zones_folder.append(current_zone)

            zone_path = os.path.join(scene_path, current_zone)

            with open(os.path.join(zone_path, seuil_expe_filename)) as f:
                zones_threshold.append(int(f.readline()))

        possible_x, possible_y = possible_point_zone

        # get all images of folder
        scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
        number_scene_image = len(scene_images)

        for id_img, img_path in enumerate(scene_images):
            
            current_img = Image.open(img_path)
            img = np.array(current_img)

            for generation in range(p_number):
                p_x, p_y = (random.randrange(possible_x), random.randrange(possible_y))

                # extract random zone into scene image
                extracted_img = img[p_y:(p_y + zone_height), p_x:(p_x + zone_width)]
                
                extracted_img.shape

                pil_extracted_img = Image.fromarray(extracted_img)

                # coordinate of specific zone, hence use threshold of zone
                if p_x % zone_width == 0 and p_y % zone_height == 0:
                    
                    zone_index = math.floor(p_x / zone_width) + math.floor(p_y / zone_height) * nb_x_parts

                    final_threshold = int(zones_threshold[zone_index])
                else:
                    # get zone identifiers of this new zones (from endpoints)
                    p_top_left = (p_x, p_y)
                    p_top_right = (p_x + zone_width, p_y)
                    p_bottom_right = (p_x + zone_width, p_y + zone_height)
                    p_bottom_left = (p_x, p_y + zone_height)

                    points = [p_top_left, p_top_right, p_bottom_right, p_bottom_left]

                    p_zones_indices = []
                    # for each points get threshold information
                    for p in points:
                        x, y = p

                        zone_index = math.floor(x / zone_width) + math.floor(y / zone_height) * nb_x_parts
                        p_zones_indices.append(zone_index)

                    p_thresholds = np.array(zones_threshold)[p_zones_indices]

                    # get proportions of pixels of img into each zone
                    overlaps = []

                    p_x_max = p_x + zone_width
                    p_y_max = p_y + zone_height

                    for index, zone_index in enumerate(p_zones_indices):
                        x_zone = (zone_index % nb_x_parts) * zone_width
                        y_zone = (math.floor(zone_index / nb_x_parts)) * zone_height

                        x_max_zone = x_zone + zone_width
                        y_max_zone = y_zone + zone_height

                        # computation of overlap
                        # x_overlap = max(0, min(rect1.right, rect2.right) - max(rect1.left, rect2.left))
                        # y_overlap = max(0, min(rect1.bottom, rect2.bottom) - max(rect1.top, rect2.top))
                        x_overlap = max(0, min(x_max_zone, p_x_max) - max(x_zone, p_x))
                        y_overlap = max(0, min(y_max_zone, p_y_max) - max(y_zone, p_y))

                        overlapArea = x_overlap * y_overlap
                        overlaps.append(overlapArea)

                    overlapSum = sum(overlaps)
                    proportions = [item / overlapSum for item in overlaps]
                    
                    final_threshold = 0

                    for index, proportion in enumerate(proportions):
                        final_threshold += proportion * p_thresholds[index]
                    
                    final_threshold = int(final_threshold)

                # save image into new scene folder
                current_image_postfix = dt.get_scene_image_postfix(img_path)

                # prepare output img name
                label_img = (int(current_image_postfix) < final_threshold)
                extracted_image_name = dt.get_scene_image_prefix(img_path) + '_' + str(generation) + '_x' + str(p_x) + '_y' + str(p_y) + '_label' + str(int(label_img))

                # if wished add of rotations images with same final threshold (increase data)
                # write new line into global .csv ('threshold', 'filepath')
                if p_rotation:

                    # do rotations and save
                    rotations = [0, 90, 180, 270]

                    for rotation in rotations:

                        rotated_img_name = extracted_image_name +  'rot' + str(rotation) + '_' + current_image_postfix + cfg.scene_image_extension
                        rotated_img_path = os.path.join(output_scene_path, rotated_img_name)
                        rotated_img = pil_extracted_img.rotate(rotation)
                        rotated_img.save(os.path.join(rotated_img_path))

                        csv_line = folder_scene + ';' + str(final_threshold) + ';' + str(int(current_image_postfix)) + ';' + str(int(label_img)) + ';' + rotated_img_path + '\n'

                        with open(output_dataset_filename_path, 'a') as f:
                            f.write(csv_line)

                else:
                    extracted_image_name += current_image_postfix + cfg.scene_image_extension
                    extracted_image_path = os.path.join(output_scene_path, extracted_image_name)
                    
                    pil_extracted_img.save(extracted_image_path)

                    csv_line = folder_scene + ';' + str(final_threshold) + ';' + str(int(current_image_postfix)) + ';' + str(int(label_img)) + ';' + extracted_image_path + '\n'

                    with open(output_dataset_filename_path, 'a') as f:
                        f.write(csv_line)

                print(folder_scene + " - " + "{0:.2f}".format(((id_img * p_number + generation) + 1) / (p_number * number_scene_image) * 100.) + "%")
                sys.stdout.write("\033[F")


if __name__== "__main__":
    main()
