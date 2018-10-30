from sklearn.externals import joblib

import numpy as np

from ipfml import image_processing
from ipfml import metrics
from PIL import Image

import sys, os, getopt

min_max_file_path = 'fichiersSVD_light/low_bits_2_min_max_values'

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python predict_noisy_image_svd_lab.py --image path/to/xxxx --interval "0,20" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:t:m:o", ["help=", "image=", "interval=", "model=", "mode="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python predict_noisy_image_svd_lab.py --image path/to/xxxx --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python predict_noisy_image_svd_lab.py --image path/to/xxxx --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svdn", "svdne"]')
            sys.exit()
        elif o in ("-i", "--image"):
            p_img_file = os.path.join(os.path.join(os.path.dirname(__file__),'../'), a)
        elif o in ("-t", "--interval"):
            p_interval = list(map(int, a.split(',')))
        elif o in ("-m", "--model"):
            p_model_file = os.path.join(os.path.join(os.path.dirname(__file__),'../'), a)
        elif o in ("-o", "--mode"):
            p_mode = a

            if p_mode != 'svdn' and p_mode != 'svdne' and p_mode != 'svd':
                assert False, "Mode not recognized"
        else:
            assert False, "unhandled option"

    # load of model file
    model = joblib.load(p_model_file) 

    # load image
    img = Image.open(p_img_file)
    low_bits_2_values = metrics.get_SVD_s(image_processing.rgb_to_grey_low_bits(img, 3))

    # check mode to normalize data
    if p_mode == 'svdne':
        
        # need to read min_max_file
        file_path = os.path.join(os.path.join(os.path.dirname(__file__),'../'), min_max_file_path)
        with open(file_path, 'r') as f:
            min = float(f.readline().replace('\n', ''))
            max = float(f.readline().replace('\n', ''))

        l_values = image_processing.normalize_arr_with_range(low_bits_2_values, min, max)

    elif p_mode == 'svdn':
        l_values = image_processing.normalize_arr(low_bits_2_values)
    else:
        l_values = low_bits_2_values

    
    # get interval values
    begin, end = p_interval
    test_data = l_values[begin:end]

    # get prediction of model
    prediction = model.predict([test_data])[0]

    print(prediction)


if __name__== "__main__":
    main()