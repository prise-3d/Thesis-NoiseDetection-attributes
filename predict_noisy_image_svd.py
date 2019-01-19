from sklearn.externals import joblib

import numpy as np

from ipfml import processing, utils
from PIL import Image

import sys, os, getopt

from modules.utils import config as cfg
from modules.utils import data as dt

path                  = cfg.dataset_path
min_max_ext           = cfg.min_max_filename_extension
metric_choices        = cfg.metric_choices_labels
normalization_choices = cfg.normalization_choices

custom_min_max_folder = cfg.min_max_custom_folder

def main():

    p_custom = False

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python predict_noisy_image_svd.py --image path/to/xxxx --interval "0,20" --model path/to/xxxx.joblib --metric lab --mode ["svdn", "svdne"] --custom min_max_file')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:t:m:m:o:c", ["help=", "image=", "interval=", "model=", "metric=", "mode=", "custom="])
    except getopt.GetoptError:
        # print help information and exit
        print('python predict_noisy_image_svd_lab.py --image path/to/xxxx --interval "xx,xx" --model path/to/xxxx.joblib --metric lab --mode ["svdn", "svdne"] --custom min_max_file')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python predict_noisy_image_svd_lab.py --image path/to/xxxx --interval "xx,xx" --model path/to/xxxx.joblib --metric lab --mode ["svdn", "svdne"] --custom min_max_file')
            sys.exit()
        elif o in ("-i", "--image"):
            p_img_file = os.path.join(os.path.dirname(__file__), a)
        elif o in ("-t", "--interval"):
            p_interval = list(map(int, a.split(',')))
        elif o in ("-m", "--model"):
            p_model_file = os.path.join(os.path.dirname(__file__), a)
        elif o in ("-m", "--metric"):
            p_metric = a

            if not p_metric in metric_choices:
                assert False, "Unknow metric choice"
        elif o in ("-o", "--mode"):
            p_mode = a

            if not p_mode in normalization_choices:
                assert False, "Mode of normalization not recognized"
        elif o in ("-c", "--custom"):
            p_custom = a

        else:
            assert False, "unhandled option"

    # load of model file
    model = joblib.load(p_model_file)

    # load image
    img = Image.open(p_img_file)

    data = dt.get_svd_data(p_metric, img)

    # get interval values
    begin, end = p_interval

    # check if custom min max file is used
    if p_custom:

        test_data = data[begin:end]

        if p_mode == 'svdne':

            # set min_max_filename if custom use
            min_max_file_path = custom_min_max_folder + '/' +  p_custom

            # need to read min_max_file
            file_path = os.path.join(os.path.dirname(__file__), min_max_file_path)
            with open(file_path, 'r') as f:
                min_val = float(f.readline().replace('\n', ''))
                max_val = float(f.readline().replace('\n', ''))

            test_data = utils.normalize_arr_with_range(test_data, min_val, max_val)

        if p_mode == 'svdn':
            test_data = utils.normalize_arr(test_data)

    else:

        # check mode to normalize data
        if p_mode == 'svdne':

            # set min_max_filename if custom use
            min_max_file_path = path + '/' + p_metric + min_max_ext

            # need to read min_max_file
            file_path = os.path.join(os.path.dirname(__file__), min_max_file_path)
            with open(file_path, 'r') as f:
                min_val = float(f.readline().replace('\n', ''))
                max_val = float(f.readline().replace('\n', ''))

            l_values = utils.normalize_arr_with_range(data, min_val, max_val)

        elif p_mode == 'svdn':
            l_values = utils.normalize_arr(data)
        else:
            l_values = data

        test_data = l_values[begin:end]


    # get prediction of model
    prediction = model.predict([test_data])[0]

    # output expected from others scripts
    print(prediction)

if __name__== "__main__":
    main()
