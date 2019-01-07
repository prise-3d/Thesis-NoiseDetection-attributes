from sklearn.externals import joblib

import numpy as np

from ipfml import processing
from PIL import Image

import sys, os, getopt

from modules.utils import config as cfg
from modules.utils import data_type as dt

path                  = cfg.dataset_path
min_max_ext           = cfg.min_max_filename_extension
metric_choices       = cfg.metric_choices_labels
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
        elif o in ("-m", "--custom"):
            p_custom = a

        else:
            assert False, "unhandled option"

    # load of model file
    model = joblib.load(p_model_file)

    # load image
    img = Image.open(p_img_file)

    data = dt.get_svd_data(p_metric, img)

    # check mode to normalize data
    if p_mode == 'svdne':

        # set min_max_filename if custom use
        if p_custom:
            min_max_filename = custom_min_max_folder + '/' +  p_custom
        else:
            min_max_file_path = path + '/' + p_metric + min_max_ext

        # need to read min_max_file
        file_path = os.path.join(os.path.join(os.path.dirname(__file__),'../'), min_max_file_path)
        with open(file_path, 'r') as f:
            min = float(f.readline().replace('\n', ''))
            max = float(f.readline().replace('\n', ''))

        l_values = processing.normalize_arr_with_range(data, min, max)

    elif p_mode == 'svdn':
        l_values = processing.normalize_arr(data)
    else:
        l_values = data


    # get interval values
    begin, end = p_interval
    test_data = l_values[begin:end]

    # get prediction of model
    prediction = model.predict([test_data])[0]

    # output expected from others scripts
    print(prediction)

if __name__== "__main__":
    main()
