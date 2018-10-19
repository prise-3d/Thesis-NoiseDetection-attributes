from sklearn.externals import joblib

import numpy as np

from ipfml import image_processing
from PIL import Image

import sys, os, getopt

min_max_file_path = './../fichiersSVD_light/min_max_values'

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python predict_noisy_image.py --image path/to/xxxx --interval "0,20" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:t:m:o", ["help=", "image=", "interval=", "model=", "mode="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python predict_noisy_image.py --image path/to/xxxx --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python predict_noisy_image.py --image path/to/xxxx --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
            sys.exit()
        elif o in ("-i", "--image"):
            p_img_file = a
        elif o in ("-t", "--interval"):
            p_interval = list(map(int, a.split(',')))
        elif o in ("-m", "--model"):
            p_model_file = a
        elif o in ("-o", "--mode"):
            p_mode = a

            if p_mode != 'svdn' and p_mode != 'svdne':
                assert False, "Mode not recognized"
        else:
            assert False, "unhandled option"

    # load of model file
    model = joblib.load(p_model_file) 

    # load image

    img = Image.open(p_img_file)
    LAB_L = image_processing.get_LAB_L_SVD_s(img)

    # check mode to normalize data
    if p_mode == 'svdne':
        
        # need to read min_max_file
        with open(min_max_file_path, 'r') as f:
            min = float(f.readline().replace('\n', ''))
            max = float(f.readline().replace('\n', ''))

        l_values_normalized = image_processing.normalize_arr_with_range(LAB_L, min, max)

    else:
        l_values_normalized = image_processing.normalize_arr(LAB_L)

    
    # get interval values
    begin, end = p_interval
    test_data = l_values_normalized[begin:end]

    # get prediction of model
    prediction = model.predict([test_data])[0]

    print(prediction)


if __name__== "__main__":
    main()