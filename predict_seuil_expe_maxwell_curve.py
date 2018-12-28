from sklearn.externals import joblib

import numpy as np

from ipfml import image_processing
from PIL import Image

import sys, os, getopt
import subprocess
import time

current_dirpath = os.getcwd()

config_filename   = "config"
scenes_path = './fichiersSVD_light'
min_max_filename = '_min_max_values'
threshold_expe_filename = 'seuilExpe'
tmp_filename = '/tmp/__model__img_to_predict.png'

maxwell_scenes = ['Appart1opt02', 'Cuisine01', 'SdbCentre', 'SdbDroite']

threshold_map_folder = "threshold_map"
simulation_curves_zones = "simulation_curves_zones_"

zones = np.arange(16)

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python predict_seuil_expe_maxwell.py --interval "0,20" --model path/to/xxxx.joblib --mode svdn --metric lab --limit_detection xx')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:m:o:l", ["help=", "interval=", "model=", "mode=", "metric=", "limit_detection="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python predict_seuil_expe_maxwell.py --interval "xx,xx" --model path/to/xxxx.joblib --mode svdn --metric lab --limit_detection xx')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python predict_seuil_expe_maxwell.py --interval "xx,xx" --model path/to/xxxx.joblib --mode svdn --metric lab --limit_detection xx')
            sys.exit()
        elif o in ("-t", "--interval"):
            p_interval = a
        elif o in ("-m", "--model"):
            p_model_file = a
        elif o in ("-o", "--mode"):
            p_mode = a

            if p_mode != 'svdn' and p_mode != 'svdne' and p_mode != 'svd':
                assert False, "Mode not recognized"

        elif o in ("-m", "--metric"):
            p_metric = a
        elif o in ("-l", "--limit_detection"):
            p_limit = int(a)
        else:
            assert False, "unhandled option"

    scenes = os.listdir(scenes_path)

    if min_max_filename in scenes:
        scenes.remove(min_max_filename)

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        # only take in consideration maxwell scenes
        if folder_scene in maxwell_scenes:

            print(folder_scene)

            scene_path = os.path.join(scenes_path, folder_scene)

            config_path = os.path.join(scene_path, config_filename)

            with open(config_path, "r") as config_file:
                last_image_name = config_file.readline().strip()
                prefix_image_name = config_file.readline().strip()
                start_index_image = config_file.readline().strip()
                end_index_image = config_file.readline().strip()
                step_counter = int(config_file.readline().strip())

            threshold_expes = []
            threshold_expes_found = []
            block_predictions_str = []

            # get zones list info
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str
                zone_folder = "zone"+index_str

                threshold_path_file = os.path.join(os.path.join(scene_path, zone_folder), threshold_expe_filename)

                with open(threshold_path_file) as f:
                    threshold = int(f.readline())
                    threshold_expes.append(threshold)

                    # Initialize default data to get detected model threshold found
                    threshold_expes_found.append(int(end_index_image)) # by default use max

                block_predictions_str.append(index_str + ";" + p_model_file + ";" + str(threshold) + ";" + str(start_index_image) + ";" + str(step_counter))

            current_counter_index = int(start_index_image)
            end_counter_index = int(end_index_image)

            print(current_counter_index)

            while(current_counter_index <= end_counter_index):

                current_counter_index_str = str(current_counter_index)

                while len(start_index_image) > len(current_counter_index_str):
                    current_counter_index_str = "0" + current_counter_index_str

                img_path = os.path.join(scene_path, prefix_image_name + current_counter_index_str + ".png")

                current_img = Image.open(img_path)
                img_blocks = image_processing.divide_in_blocks(current_img, (200, 200))

                for id_block, block in enumerate(img_blocks):

                    # check only if necessary for this scene (not already detected)
                    #if not threshold_expes_detected[id_block]:

                        tmp_file_path = tmp_filename.replace('__model__',  p_model_file.split('/')[-1].replace('.joblib', '_'))
                        block.save(tmp_file_path)

                        python_cmd = "python metrics_predictions/predict_noisy_image_svd_" + p_metric + ".py --image " + tmp_file_path + \
                                        " --interval '" + p_interval + \
                                        "' --model " + p_model_file  + \
                                        " --mode " + p_mode

                        ## call command ##
                        p = subprocess.Popen(python_cmd, stdout=subprocess.PIPE, shell=True)

                        (output, err) = p.communicate()

                        ## Wait for result ##
                        p_status = p.wait()

                        prediction = int(output)

                        # save here in specific file of block all the predictions done
                        block_predictions_str[id_block] = block_predictions_str[id_block] + ";" + str(prediction)

                        print(str(id_block) + " : " + str(current_counter_index) + "/" + str(threshold_expes[id_block]) + " => " + str(prediction))

                current_counter_index += step_counter
                print("------------------------")
                print("Scene " + str(id_scene + 1) + "/" + str(len(scenes)))
                print("------------------------")

            # end of scene => display of results

            # construct path using model name for saving threshold map folder
            model_treshold_path = os.path.join(threshold_map_folder, p_model_file.split('/')[-1].replace('.joblib', ''))

            # create threshold model path if necessary
            if not os.path.exists(model_treshold_path):
                os.makedirs(model_treshold_path)

            map_filename = os.path.join(model_treshold_path, simulation_curves_zones + folder_scene)
            f_map = open(map_filename, 'w')

            for line in block_predictions_str:
                f_map.write(line + '\n')
            f_map.close()

            print("Scene " + str(id_scene + 1) + "/" + str(len(maxwell_scenes)) + " Done..")
            print("------------------------")

            print("Model predictions are saved into %s" map_filename)
            time.sleep(10)


if __name__== "__main__":
    main()
