from sklearn.externals import joblib

import numpy as np

from ipfml import image_processing
from PIL import Image

import sys, os, getopt
import subprocess
import time

current_dirpath = os.getcwd()

threshold_map_folder = "threshold_map"
threshold_map_file_prefix = "treshold_map_"

markdowns_folder = "models_info"

zones = np.arange(16)

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python save_model_result_in_md.py --interval "0,20" --model path/to/xxxx.joblib --mode ["svd", "svdn", "svdne"] --metric ["lab", "mscn"]')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:m:o:l", ["help=", "interval=", "model=", "mode=", "metric="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python save_model_result_in_md.py --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svd", "svdn", "svdne"] --metric ["lab", "mscn"]')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python save_model_result_in_md.py --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svd", "svdn", "svdne"] --metric ["lab", "mscn"]')
            sys.exit()
        elif o in ("-t", "--interval"):
            p_interval = list(map(int, a.split(',')))
        elif o in ("-m", "--model"):
            p_model_file = a
        elif o in ("-o", "--mode"):
            p_mode = a

            if p_mode != 'svdn' and p_mode != 'svdne' and p_mode != 'svd':
                assert False, "Mode not recognized"
        elif o in ("-c", "--metric"):
            p_metric = a
        else:
            assert False, "unhandled option"

    
    # call model and get global result in scenes

    begin, end = p_interval

    bash_cmd = "bash testModelByScene_maxwell.sh '" + str(begin) + "' '" + str(end) + "' '" + p_model_file + "' '" + p_mode + "' '" + p_metric + "'" 
    print(bash_cmd)
     
    ## call command ##
    p = subprocess.Popen(bash_cmd, stdout=subprocess.PIPE, shell=True)
    
    (output, err) = p.communicate()
    
    ## Wait for result ##
    p_status = p.wait()

    if not os.path.exists(markdowns_folder):
        os.makedirs(markdowns_folder)
        
    # get model name to construct model
    md_model_path = os.path.join(markdowns_folder, p_model_file.split('/')[-1].replace('.joblib', '.md'))

    with open(md_model_path, 'w') as f:
        f.write(output.decode("utf-8"))

        # read each threshold_map information if exists
        model_map_info_path = os.path.join(threshold_map_folder, p_model_file.replace('saved_models/', ''))

        if not os.path.exists(model_map_info_path):
            f.write('\n\n No threshold map information')
        else:
            maps_files = os.listdir(model_map_info_path)

            # get all map information
            for t_map_file in maps_files:
                
                file_path = os.path.join(model_map_info_path, t_map_file)
                with open(file_path, 'r') as map_file:

                    title_scene =  t_map_file.replace(threshold_map_file_prefix, '')
                    f.write('\n\n## ' + title_scene + '\n')
                    content = map_file.readlines()

                    # getting each map line information
                    for line in content:
                        f.write(line)

        f.close()

if __name__== "__main__":
    main()