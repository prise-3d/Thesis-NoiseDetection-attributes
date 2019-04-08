from ipfml import processing, metrics, utils
from modules.utils.config import *

from PIL import Image
from skimage import color
from sklearn.decomposition import FastICA

import numpy as np


_scenes_names_prefix   = '_scenes_names'
_scenes_indices_prefix = '_scenes_indices'

# store all variables from current module context
context_vars = vars()


def get_svd_data(data_type, block):
    """
    Method which returns the data type expected
    """

    if data_type == 'lab':

        block_file_path = '/tmp/lab_img.png'
        block.save(block_file_path)
        data = processing.get_LAB_L_SVD_s(Image.open(block_file_path))

    if data_type == 'mscn':

        img_mscn_revisited = processing.rgb_to_mscn(block)

        # save tmp as img
        img_output = Image.fromarray(img_mscn_revisited.astype('uint8'), 'L')
        mscn_revisited_file_path = '/tmp/mscn_revisited_img.png'
        img_output.save(mscn_revisited_file_path)
        img_block = Image.open(mscn_revisited_file_path)

        # extract from temp image
        data = metrics.get_SVD_s(img_block)

    """if data_type == 'mscn':

        img_gray = np.array(color.rgb2gray(np.asarray(block))*255, 'uint8')
        img_mscn = processing.calculate_mscn_coefficients(img_gray, 7)
        img_mscn_norm = processing.normalize_2D_arr(img_mscn)

        img_mscn_gray = np.array(img_mscn_norm*255, 'uint8')

        data = metrics.get_SVD_s(img_mscn_gray)
    """

    if data_type == 'low_bits_6':

        low_bits_6 = processing.rgb_to_LAB_L_low_bits(block, 6)
        data = metrics.get_SVD_s(low_bits_6)

    if data_type == 'low_bits_5':

        low_bits_5 = processing.rgb_to_LAB_L_low_bits(block, 5)
        data = metrics.get_SVD_s(low_bits_5)

    if data_type == 'low_bits_4':

        low_bits_4 = processing.rgb_to_LAB_L_low_bits(block, 4)
        data = metrics.get_SVD_s(low_bits_4)

    if data_type == 'low_bits_3':

        low_bits_3 = processing.rgb_to_LAB_L_low_bits(block, 3)
        data = metrics.get_SVD_s(low_bits_3)

    if data_type == 'low_bits_2':

        low_bits_2 = processing.rgb_to_LAB_L_low_bits(block, 2)
        data = metrics.get_SVD_s(low_bits_2)

    if data_type == 'low_bits_4_shifted_2':

        data = metrics.get_SVD_s(processing.rgb_to_LAB_L_bits(block, (3, 6)))

    if data_type == 'sub_blocks_stats':

        block = np.asarray(block)
        width, height, _= block.shape
        sub_width, sub_height = int(width / 4), int(height / 4)

        sub_blocks = processing.divide_in_blocks(block, (sub_width, sub_height))

        data = []

        for sub_b in sub_blocks:

            # by default use the whole lab L canal
            l_svd_data = np.array(processing.get_LAB_L_SVD_s(sub_b))

            # get information we want from svd
            data.append(np.mean(l_svd_data))
            data.append(np.median(l_svd_data))
            data.append(np.percentile(l_svd_data, 25))
            data.append(np.percentile(l_svd_data, 75))
            data.append(np.var(l_svd_data))

            area_under_curve = utils.integral_area_trapz(l_svd_data, dx=100)
            data.append(area_under_curve)

        # convert into numpy array after computing all stats
        data = np.asarray(data)

    if data_type == 'sub_blocks_stats_reduced':

        block = np.asarray(block)
        width, height, _= block.shape
        sub_width, sub_height = int(width / 4), int(height / 4)

        sub_blocks = processing.divide_in_blocks(block, (sub_width, sub_height))

        data = []

        for sub_b in sub_blocks:

            # by default use the whole lab L canal
            l_svd_data = np.array(processing.get_LAB_L_SVD_s(sub_b))

            # get information we want from svd
            data.append(np.mean(l_svd_data))
            data.append(np.median(l_svd_data))
            data.append(np.percentile(l_svd_data, 25))
            data.append(np.percentile(l_svd_data, 75))
            data.append(np.var(l_svd_data))

        # convert into numpy array after computing all stats
        data = np.asarray(data)

    if data_type == 'sub_blocks_area':

        block = np.asarray(block)
        width, height, _= block.shape
        sub_width, sub_height = int(width / 8), int(height / 8)

        sub_blocks = processing.divide_in_blocks(block, (sub_width, sub_height))

        data = []

        for sub_b in sub_blocks:

            # by default use the whole lab L canal
            l_svd_data = np.array(processing.get_LAB_L_SVD_s(sub_b))

            area_under_curve = utils.integral_area_trapz(l_svd_data, dx=50)
            data.append(area_under_curve)

        # convert into numpy array after computing all stats
        data = np.asarray(data)

    if data_type == 'sub_blocks_area_normed':

        block = np.asarray(block)
        width, height, _= block.shape
        sub_width, sub_height = int(width / 8), int(height / 8)

        sub_blocks = processing.divide_in_blocks(block, (sub_width, sub_height))

        data = []

        for sub_b in sub_blocks:

            # by default use the whole lab L canal
            l_svd_data = np.array(processing.get_LAB_L_SVD_s(sub_b))
            l_svd_data = utils.normalize_arr(l_svd_data)

            area_under_curve = utils.integral_area_trapz(l_svd_data, dx=50)
            data.append(area_under_curve)

        # convert into numpy array after computing all stats
        data = np.asarray(data)

    if data_type == 'mscn_var_4':

        data = _get_mscn_variance(block, (100, 100))

    if data_type == 'mscn_var_16':

        data = _get_mscn_variance(block, (50, 50))

    if data_type == 'mscn_var_64':

        data = _get_mscn_variance(block, (25, 25))

    if data_type == 'mscn_var_16_max':

        data = _get_mscn_variance(block, (50, 50))
        data = np.asarray(data)
        size = int(len(data) / 4)
        indices = data.argsort()[-size:][::-1]
        data = data[indices]

    if data_type == 'mscn_var_64_max':

        data = _get_mscn_variance(block, (25, 25))
        data = np.asarray(data)
        size = int(len(data) / 4)
        indices = data.argsort()[-size:][::-1]
        data = data[indices]

    if data_type == 'ica_diff':
        current_image = metrics.get_LAB_L(block)

        ica = FastICA(n_components=50)
        ica.fit(current_image)

        image_ica = ica.fit_transform(current_image)
        image_restored = ica.inverse_transform(image_ica)

        final_image = utils.normalize_2D_arr(image_restored)
        final_image = np.array(final_image * 255, 'uint8')

        sv_values = utils.normalize_arr(metrics.get_SVD_s(current_image))
        ica_sv_values = utils.normalize_arr(metrics.get_SVD_s(final_image))

        data = abs(np.array(sv_values) - np.array(ica_sv_values))

    return data

def _get_mscn_variance(block, sub_block_size=(50, 50)):

    blocks = processing.divide_in_blocks(block, sub_block_size)

    data = []

    for block in blocks:
        mscn_coefficients = processing.get_mscn_coefficients(block)
        flat_coeff = mscn_coefficients.flatten()
        data.append(np.var(flat_coeff))

    return np.sort(data)


def get_renderer_scenes_indices(renderer_name):

    if renderer_name not in renderer_choices:
        raise ValueError("Unknown renderer name")

    if renderer_name == 'all':
        return scenes_indices
    else:
        return context_vars[renderer_name + _scenes_indices_prefix]

def get_renderer_scenes_names(renderer_name):

    if renderer_name not in renderer_choices:
        raise ValueError("Unknown renderer name")

    if renderer_name == 'all':
        return scenes_names
    else:
        return context_vars[renderer_name + _scenes_names_prefix]

