from ipfml import image_processing, metrics
from PIL import Image

import numpy as np


def get_svd_data(data_type, block):
    """
    Method which returns the data type expected
    """

    if data_type == 'lab':

        block_file_path = '/tmp/lab_img.png'
        block.save(block_file_path)
        data = image_processing.get_LAB_L_SVD_s(Image.open(block_file_path))

    if data_type == 'mscn_revisited':

        img_mscn_revisited = image_processing.rgb_to_mscn(block)

        # save tmp as img
        img_output = Image.fromarray(img_mscn_revisited.astype('uint8'), 'L')
        mscn_revisited_file_path = '/tmp/mscn_revisited_img.png'
        img_output.save(mscn_revisited_file_path)
        img_block = Image.open(mscn_revisited_file_path)

        # extract from temp image
        data = metrics.get_SVD_s(img_block)

    if data_type == 'mscn':

        img_gray = np.array(color.rgb2gray(np.asarray(block))*255, 'uint8')
        img_mscn = image_processing.calculate_mscn_coefficients(img_gray, 7)
        img_mscn_norm = image_processing.normalize_2D_arr(img_mscn)

        img_mscn_gray = np.array(img_mscn_norm*255, 'uint8')

        data = metrics.get_SVD_s(img_mscn_gray)

    if data_type == 'low_bits_6':

        low_bits_6 = image_processing.rgb_to_LAB_L_low_bits(block, 63)

        # extract from temp image
        data = metrics.get_SVD_s(low_bits_6)

    if data_type == 'low_bits_5':

        low_bits_5 = image_processing.rgb_to_LAB_L_low_bits(block, 31)

        # extract from temp image
        data = metrics.get_SVD_s(low_bits_5)


    if data_type == 'low_bits_4':

        low_bits_4 = image_processing.rgb_to_LAB_L_low_bits(block)

        # extract from temp image
        data = metrics.get_SVD_s(low_bits_4)

    if data_type == 'low_bits_3':

        low_bits_3 = image_processing.rgb_to_LAB_L_low_bits(block, 7)

        # extract from temp image
        data = metrics.get_SVD_s(low_bits_3)

    if data_type == 'low_bits_2':

        low_bits_2 = image_processing.rgb_to_LAB_L_low_bits(block, 3)

        # extract from temp image
        data = metrics.get_SVD_s(low_bits_2)

    if data_type == 'low_bits_4_shifted_2':

        data = metrics.get_SVD_s(image_processing.rgb_to_LAB_L_bits(block, (3, 6)))

    return data


