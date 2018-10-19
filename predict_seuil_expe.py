from sklearn.externals import joblib

import numpy as np

from ipfml import image_processing
from PIL import Image

import sys, os, getopt
import subprocess

config_filename   = "config"
scenes_path = './fichiersSVD_light'
min_max_filename = 'min_max_values'
seuil_expe_filename = 'seuilExpe'
tmp_filename = '/tmp/img_to_predict.png'
zones = np.arange(16)

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python predict_noisy_image.py --interval "0,20" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:m:o", ["help=", "interval=", "model=", "mode="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python predict_noisy_image.py --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python predict_noisy_image.py --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
            sys.exit()
        elif o in ("-t", "--interval"):
            p_interval = a
        elif o in ("-m", "--model"):
            p_model_file = a
        elif o in ("-o", "--mode"):
            p_mode = a

            if p_mode != 'svdn' and p_mode != 'svdne':
                assert False, "Mode not recognized"
        else:
            assert False, "unhandled option"

    scenes = os.listdir(scenes_path)
    
    if min_max_filename in scenes:
        scenes.remove(min_max_filename)

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):
    
        print(folder_scene)
        scene_path = scenes_path + "/" + folder_scene
        with open(scene_path + "/" + config_filename, "r") as config_file:
            last_image_name = config_file.readline().strip()
            prefix_image_name = config_file.readline().strip()
            start_index_image = config_file.readline().strip()
            end_index_image = config_file.readline().strip()
            step_counter = int(config_file.readline().strip())

        seuil_expes = []
        seuil_expes_detected = []
        seuil_expes_counter = []
        seuil_expes_found = []

        # get zones list info
        for index in zones:
            index_str = str(index)
            if len(index_str) < 2:
                index_str = "0" + index_str
            zone_folder = "zone"+index_str

            with open(scene_path + "/" + zone_folder + "/" + seuil_expe_filename) as f:
                seuil_expes.append(int(f.readline()))

                # Initialize default data to get detected model seuil found
                seuil_expes_detected.append(False)
                seuil_expes_counter.append(0)
                seuil_expes_found.append(0)

        for seuil in seuil_expes:
            print(seuil)

        current_counter_index = int(start_index_image)
        end_counter_index = int(end_index_image)

        print(current_counter_index)
        while(current_counter_index <= end_counter_index):
            
            current_counter_index_str = str(current_counter_index)

            while len(start_index_image) > len(current_counter_index_str):
                current_counter_index_str = "0" + current_counter_index_str

            img_path = scene_path + "/" + prefix_image_name + current_counter_index_str + ".png"

            print(img_path)

            current_img = Image.open(img_path)
            img_blocks = image_processing.divide_in_blocks(current_img, (200, 200))

            for id_block, block in enumerate(img_blocks):
                block.save(tmp_filename)

                python_cmd = "python predict_noisy_image_sdv_lab.py --image " + tmp_filename + \
                                   " --interval '" + p_interval + \
                                   "' --model " + p_model_file  + \
                                   " --mode " + p_mode
                ## call command ##
                p = subprocess.Popen(python_cmd, stdout=subprocess.PIPE, shell=True)
                
                (output, err) = p.communicate()
                
                ## Wait for result ##
                p_status = p.wait()

                prediction = int(output)



                print(str(current_counter_index) + "/" + str(seuil_expes[id_block]) + " => " + str(prediction))

            current_counter_index += step_counter
        
        # end of scene => display of results




if __name__== "__main__":
    main()