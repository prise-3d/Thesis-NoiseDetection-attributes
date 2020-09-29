# main imports
import numpy as np
import sys

# image transform imports
from PIL import Image
from skimage import color
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import svd as lin_svd
from scipy.signal import medfilt2d, wiener, cwt
import pywt
import cv2

from ipfml.processing import transform, compression, segmentation
from ipfml.filters import convolution, kernels
from ipfml import utils


# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt


def get_image_features(data_type, block):
    """
    Method which returns the data type expected
    """

    if data_type == 'lab':

        block_file_path = '/tmp/lab_img.png'
        block.save(block_file_path)
        data = transform.get_LAB_L_SVD_s(Image.open(block_file_path))

    if data_type == 'mscn':

        img_mscn_revisited = transform.rgb_to_mscn(block)

        # save tmp as img
        img_output = Image.fromarray(img_mscn_revisited.astype('uint8'), 'L')
        mscn_revisited_file_path = '/tmp/mscn_revisited_img.png'
        img_output.save(mscn_revisited_file_path)
        img_block = Image.open(mscn_revisited_file_path)

        # extract from temp image
        data = compression.get_SVD_s(img_block)

    """if data_type == 'mscn':

        img_gray = np.array(color.rgb2gray(np.asarray(block))*255, 'uint8')
        img_mscn = transform.calculate_mscn_coefficients(img_gray, 7)
        img_mscn_norm = transform.normalize_2D_arr(img_mscn)

        img_mscn_gray = np.array(img_mscn_norm*255, 'uint8')

        data = compression.get_SVD_s(img_mscn_gray)
    """

    if data_type == 'low_bits_6':

        low_bits_6 = transform.rgb_to_LAB_L_low_bits(block, 6)
        data = compression.get_SVD_s(low_bits_6)

    if data_type == 'low_bits_5':

        low_bits_5 = transform.rgb_to_LAB_L_low_bits(block, 5)
        data = compression.get_SVD_s(low_bits_5)

    if data_type == 'low_bits_4':

        low_bits_4 = transform.rgb_to_LAB_L_low_bits(block, 4)
        data = compression.get_SVD_s(low_bits_4)

    if data_type == 'low_bits_3':

        low_bits_3 = transform.rgb_to_LAB_L_low_bits(block, 3)
        data = compression.get_SVD_s(low_bits_3)

    if data_type == 'low_bits_2':

        low_bits_2 = transform.rgb_to_LAB_L_low_bits(block, 2)
        data = compression.get_SVD_s(low_bits_2)

    if data_type == 'low_bits_4_shifted_2':

        data = compression.get_SVD_s(transform.rgb_to_LAB_L_bits(block, (3, 6)))

    if data_type == 'sub_blocks_stats':

        block = np.asarray(block)
        width, height, _= block.shape
        sub_width, sub_height = int(width / 4), int(height / 4)

        sub_blocks = segmentation.divide_in_blocks(block, (sub_width, sub_height))

        data = []

        for sub_b in sub_blocks:

            # by default use the whole lab L canal
            l_svd_data = np.array(transform.get_LAB_L_SVD_s(sub_b))

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

        sub_blocks = segmentation.divide_in_blocks(block, (sub_width, sub_height))

        data = []

        for sub_b in sub_blocks:

            # by default use the whole lab L canal
            l_svd_data = np.array(transform.get_LAB_L_SVD_s(sub_b))

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

        sub_blocks = segmentation.divide_in_blocks(block, (sub_width, sub_height))

        data = []

        for sub_b in sub_blocks:

            # by default use the whole lab L canal
            l_svd_data = np.array(transform.get_LAB_L_SVD_s(sub_b))

            area_under_curve = utils.integral_area_trapz(l_svd_data, dx=50)
            data.append(area_under_curve)

        # convert into numpy array after computing all stats
        data = np.asarray(data)

    if data_type == 'sub_blocks_area_normed':

        block = np.asarray(block)
        width, height, _= block.shape
        sub_width, sub_height = int(width / 8), int(height / 8)

        sub_blocks = segmentation.divide_in_blocks(block, (sub_width, sub_height))

        data = []

        for sub_b in sub_blocks:

            # by default use the whole lab L canal
            l_svd_data = np.array(transform.get_LAB_L_SVD_s(sub_b))
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
        current_image = transform.get_LAB_L(block)

        ica = FastICA(n_components=50)
        ica.fit(current_image)

        image_ica = ica.fit_transform(current_image)
        image_restored = ica.inverse_transform(image_ica)

        final_image = utils.normalize_2D_arr(image_restored)
        final_image = np.array(final_image * 255, 'uint8')

        sv_values = utils.normalize_arr(compression.get_SVD_s(current_image))
        ica_sv_values = utils.normalize_arr(compression.get_SVD_s(final_image))

        data = abs(np.array(sv_values) - np.array(ica_sv_values))

    if data_type == 'svd_trunc_diff':

        current_image = transform.get_LAB_L(block)

        svd = TruncatedSVD(n_components=30, n_iter=100, random_state=42)
        transformed_image = svd.fit_transform(current_image)
        restored_image = svd.inverse_transform(transformed_image)

        reduced_image = (current_image - restored_image)

        U, s, V = compression.get_SVD(reduced_image)
        data = s

    if data_type == 'ipca_diff':

        current_image = transform.get_LAB_L(block)

        transformer = IncrementalPCA(n_components=20, batch_size=25)
        transformed_image = transformer.fit_transform(current_image)
        restored_image = transformer.inverse_transform(transformed_image)

        reduced_image = (current_image - restored_image)

        U, s, V = compression.get_SVD(reduced_image)
        data = s

    if data_type == 'svd_reconstruct':

        reconstructed_interval = (90, 200)
        begin, end = reconstructed_interval

        lab_img = transform.get_LAB_L(block)
        lab_img = np.array(lab_img, 'uint8')

        U, s, V = lin_svd(lab_img, full_matrices=True)

        smat = np.zeros((end-begin, end-begin), dtype=complex)
        smat[:, :] = np.diag(s[begin:end])
        output_img = np.dot(U[:, begin:end],  np.dot(smat, V[begin:end, :]))

        output_img = np.array(output_img, 'uint8')

        data = compression.get_SVD_s(output_img)

    if 'sv_std_filters' in data_type:

        # convert into lab by default to apply filters
        lab_img = transform.get_LAB_L(block)
        arr = np.array(lab_img)
        images = []
        
        # Apply list of filter on arr
        images.append(medfilt2d(arr, [3, 3]))
        images.append(medfilt2d(arr, [5, 5]))
        images.append(wiener(arr, [3, 3]))
        images.append(wiener(arr, [5, 5]))
        
        # By default computation of current block image
        s_arr = compression.get_SVD_s(arr)
        sv_vector = [s_arr]

        # for each new image apply SVD and get SV 
        for img in images:
            s = compression.get_SVD_s(img)
            sv_vector.append(s)
            
        sv_array = np.array(sv_vector)
        
        _, length = sv_array.shape
        
        sv_std = []
        
        # normalize each SV vectors and compute standard deviation for each sub vectors
        for i in range(length):
            sv_array[:, i] = utils.normalize_arr(sv_array[:, i])
            sv_std.append(np.std(sv_array[:, i]))
        
        indices = []

        if 'lowest' in data_type:
            indices = utils.get_indices_of_lowest_values(sv_std, 200)

        if 'highest' in data_type:
            indices = utils.get_indices_of_highest_values(sv_std, 200)

        # data are arranged following std trend computed
        data = s_arr[indices]

    # with the use of wavelet
    if 'wave_sv_std_filters' in data_type:

        # convert into lab by default to apply filters
        lab_img = transform.get_LAB_L(block)
        arr = np.array(lab_img)
        images = []
        
        # Apply list of filter on arr
        images.append(medfilt2d(arr, [3, 3]))
        
        # By default computation of current block image
        s_arr = compression.get_SVD_s(arr)
        sv_vector = [s_arr]

        # for each new image apply SVD and get SV 
        for img in images:
            s = compression.get_SVD_s(img)
            sv_vector.append(s)
            
        sv_array = np.array(sv_vector)
        
        _, length = sv_array.shape
        
        sv_std = []
        
        # normalize each SV vectors and compute standard deviation for each sub vectors
        for i in range(length):
            sv_array[:, i] = utils.normalize_arr(sv_array[:, i])
            sv_std.append(np.std(sv_array[:, i]))
        
        indices = []

        if 'lowest' in data_type:
            indices = utils.get_indices_of_lowest_values(sv_std, 200)

        if 'highest' in data_type:
            indices = utils.get_indices_of_highest_values(sv_std, 200)

        # data are arranged following std trend computed
        data = s_arr[indices]

    # with the use of wavelet
    if 'sv_std_filters_full' in data_type:

        # convert into lab by default to apply filters
        lab_img = transform.get_LAB_L(block)
        arr = np.array(lab_img)
        images = []
        
        # Apply list of filter on arr
        kernel = np.ones((3,3),np.float32)/9
        images.append(cv2.filter2D(arr,-1,kernel))

        kernel = np.ones((5,5),np.float32)/25
        images.append(cv2.filter2D(arr,-1,kernel))

        images.append(cv2.GaussianBlur(arr, (3, 3), 0.5))

        images.append(cv2.GaussianBlur(arr, (3, 3), 1))

        images.append(cv2.GaussianBlur(arr, (3, 3), 1.5))

        images.append(cv2.GaussianBlur(arr, (5, 5), 0.5))

        images.append(cv2.GaussianBlur(arr, (5, 5), 1))

        images.append(cv2.GaussianBlur(arr, (5, 5), 1.5))

        images.append(medfilt2d(arr, [3, 3]))

        images.append(medfilt2d(arr, [5, 5]))

        images.append(wiener(arr, [3, 3]))

        images.append(wiener(arr, [5, 5]))

        wave = w2d(arr, 'db1', 2)
        images.append(np.array(wave, 'float64'))
        
        # By default computation of current block image
        s_arr = compression.get_SVD_s(arr)
        sv_vector = [s_arr]

        # for each new image apply SVD and get SV 
        for img in images:
            s = compression.get_SVD_s(img)
            sv_vector.append(s)
            
        sv_array = np.array(sv_vector)
        
        _, length = sv_array.shape
        
        sv_std = []
        
        # normalize each SV vectors and compute standard deviation for each sub vectors
        for i in range(length):
            sv_array[:, i] = utils.normalize_arr(sv_array[:, i])
            sv_std.append(np.std(sv_array[:, i]))
        
        indices = []

        if 'lowest' in data_type:
            indices = utils.get_indices_of_lowest_values(sv_std, 200)

        if 'highest' in data_type:
            indices = utils.get_indices_of_highest_values(sv_std, 200)

        # data are arranged following std trend computed
        data = s_arr[indices]

    if 'sv_entropy_std_filters' in data_type:

        lab_img = transform.get_LAB_L(block)
        arr = np.array(lab_img)

        images = []

        kernel = np.ones((3,3),np.float32)/9
        images.append(cv2.filter2D(arr,-1,kernel))

        kernel = np.ones((5,5),np.float32)/25
        images.append(cv2.filter2D(arr,-1,kernel))

        images.append(cv2.GaussianBlur(arr, (3, 3), 0.5))

        images.append(cv2.GaussianBlur(arr, (3, 3), 1))

        images.append(cv2.GaussianBlur(arr, (3, 3), 1.5))

        images.append(cv2.GaussianBlur(arr, (5, 5), 0.5))

        images.append(cv2.GaussianBlur(arr, (5, 5), 1))

        images.append(cv2.GaussianBlur(arr, (5, 5), 1.5))

        images.append(medfilt2d(arr, [3, 3]))

        images.append(medfilt2d(arr, [5, 5]))

        images.append(wiener(arr, [3, 3]))

        images.append(wiener(arr, [5, 5]))

        wave = w2d(arr, 'db1', 2)
        images.append(np.array(wave, 'float64'))

        sv_vector = []
        sv_entropy_list = []
        
        # for each new image apply SVD and get SV 
        for img in images:
            s = compression.get_SVD_s(img)
            sv_vector.append(s)

            sv_entropy = [utils.get_entropy_contribution_of_i(s, id_sv) for id_sv, sv in enumerate(s)]
            sv_entropy_list.append(sv_entropy)
        
        sv_std = []
        
        sv_array = np.array(sv_vector)
        _, length = sv_array.shape
        
        # normalize each SV vectors and compute standard deviation for each sub vectors
        for i in range(length):
            sv_array[:, i] = utils.normalize_arr(sv_array[:, i])
            sv_std.append(np.std(sv_array[:, i]))
        
        indices = []

        if 'lowest' in data_type:
            indices = utils.get_indices_of_lowest_values(sv_std, 200)

        if 'highest' in data_type:
            indices = utils.get_indices_of_highest_values(sv_std, 200)

        # data are arranged following std trend computed
        s_arr = compression.get_SVD_s(arr)
        data = s_arr[indices]

    if 'convolutional_kernels' in data_type:

        sub_zones = segmentation.divide_in_blocks(block, (20, 20))

        data = []

        diff_std_list_3 = []
        diff_std_list_5 = []
        diff_mean_list_3 = []
        diff_mean_list_5 = []

        plane_std_list_3 = []
        plane_std_list_5 = []
        plane_mean_list_3 = []
        plane_mean_list_5 = []

        plane_max_std_list_3 = []
        plane_max_std_list_5 = []
        plane_max_mean_list_3 = []
        plane_max_mean_list_5 = []

        for sub_zone in sub_zones:
            l_img = transform.get_LAB_L(sub_zone)
            normed_l_img = utils.normalize_2D_arr(l_img)

            # bilateral with window of size (3, 3)
            normed_diff = convolution.convolution2D(normed_l_img, kernels.min_bilateral_diff, (3, 3))
            std_diff = np.std(normed_diff)
            mean_diff = np.mean(normed_diff)

            diff_std_list_3.append(std_diff)
            diff_mean_list_3.append(mean_diff)

            # bilateral with window of size (5, 5)
            normed_diff = convolution.convolution2D(normed_l_img, kernels.min_bilateral_diff, (5, 5))
            std_diff = np.std(normed_diff)
            mean_diff = np.mean(normed_diff)

            diff_std_list_5.append(std_diff)
            diff_mean_list_5.append(mean_diff)

            # plane mean with window of size (3, 3)
            normed_plane_mean = convolution.convolution2D(normed_l_img, kernels.plane_mean, (3, 3))
            std_plane_mean = np.std(normed_plane_mean)
            mean_plane_mean = np.mean(normed_plane_mean)

            plane_std_list_3.append(std_plane_mean)
            plane_mean_list_3.append(mean_plane_mean)

            # plane mean with window of size (5, 5)
            normed_plane_mean = convolution.convolution2D(normed_l_img, kernels.plane_mean, (5, 5))
            std_plane_mean = np.std(normed_plane_mean)
            mean_plane_mean = np.mean(normed_plane_mean)

            plane_std_list_5.append(std_plane_mean)
            plane_mean_list_5.append(mean_plane_mean)

            # plane max error with window of size (3, 3)
            normed_plane_max = convolution.convolution2D(normed_l_img, kernels.plane_max_error, (3, 3))
            std_plane_max = np.std(normed_plane_max)
            mean_plane_max = np.mean(normed_plane_max)

            plane_max_std_list_3.append(std_plane_max)
            plane_max_mean_list_3.append(mean_plane_max)

            # plane max error with window of size (5, 5)
            normed_plane_max = convolution.convolution2D(normed_l_img, kernels.plane_max_error, (5, 5))
            std_plane_max = np.std(normed_plane_max)
            mean_plane_max = np.mean(normed_plane_max)

            plane_max_std_list_5.append(std_plane_max)
            plane_max_mean_list_5.append(mean_plane_max)

        diff_std_list_3 = np.array(diff_std_list_3)
        diff_std_list_5 = np.array(diff_std_list_5)

        diff_mean_list_3 = np.array(diff_mean_list_3)
        diff_mean_list_5 = np.array(diff_mean_list_5)

        plane_std_list_3 = np.array(plane_std_list_3)
        plane_std_list_5 = np.array(plane_std_list_5)

        plane_mean_list_3 = np.array(plane_mean_list_3)
        plane_mean_list_5 = np.array(plane_mean_list_5)

        plane_max_std_list_3 = np.array(plane_max_std_list_3)
        plane_max_std_list_5 = np.array(plane_max_std_list_5)

        plane_max_mean_list_3 = np.array(plane_max_mean_list_3)
        plane_max_mean_list_5 = np.array(plane_max_mean_list_5)

        if 'std_max_blocks' in data_type:

            data.append(np.std(diff_std_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.std(diff_mean_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.std(diff_std_list_5[0:int(len(sub_zones)/5)]))
            data.append(np.std(diff_mean_list_5[0:int(len(sub_zones)/5)]))

            data.append(np.std(plane_std_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.std(plane_mean_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.std(plane_std_list_5[0:int(len(sub_zones)/5)]))
            data.append(np.std(plane_mean_list_5[0:int(len(sub_zones)/5)]))

            data.append(np.std(plane_max_std_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.std(plane_max_mean_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.std(plane_max_std_list_5[0:int(len(sub_zones)/5)]))
            data.append(np.std(plane_max_mean_list_5[0:int(len(sub_zones)/5)]))

        if 'mean_max_blocks' in data_type:

            data.append(np.mean(diff_std_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.mean(diff_mean_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.mean(diff_std_list_5[0:int(len(sub_zones)/5)]))
            data.append(np.mean(diff_mean_list_5[0:int(len(sub_zones)/5)]))

            data.append(np.mean(plane_std_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.mean(plane_mean_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.mean(plane_std_list_5[0:int(len(sub_zones)/5)]))
            data.append(np.mean(plane_mean_list_5[0:int(len(sub_zones)/5)]))
            
            data.append(np.mean(plane_max_std_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.mean(plane_max_mean_list_3[0:int(len(sub_zones)/5)]))
            data.append(np.mean(plane_max_std_list_5[0:int(len(sub_zones)/5)]))
            data.append(np.mean(plane_max_mean_list_5[0:int(len(sub_zones)/5)]))

        if 'std_normed' in data_type:

            data.append(np.std(diff_std_list_3))
            data.append(np.std(diff_mean_list_3))
            data.append(np.std(diff_std_list_5))
            data.append(np.std(diff_mean_list_5))

            data.append(np.std(plane_std_list_3))
            data.append(np.std(plane_mean_list_3))
            data.append(np.std(plane_std_list_5))
            data.append(np.std(plane_mean_list_5))
            
            data.append(np.std(plane_max_std_list_3))
            data.append(np.std(plane_max_mean_list_3))
            data.append(np.std(plane_max_std_list_5))
            data.append(np.std(plane_max_mean_list_5))

        if 'mean_normed' in data_type:

            data.append(np.mean(diff_std_list_3))
            data.append(np.mean(diff_mean_list_3))
            data.append(np.mean(diff_std_list_5))
            data.append(np.mean(diff_mean_list_5))

            data.append(np.mean(plane_std_list_3))
            data.append(np.mean(plane_mean_list_3))
            data.append(np.mean(plane_std_list_5))
            data.append(np.mean(plane_mean_list_5))
            
            data.append(np.mean(plane_max_std_list_3))
            data.append(np.mean(plane_max_mean_list_3))
            data.append(np.mean(plane_max_std_list_5))
            data.append(np.mean(plane_max_mean_list_5))

        data = np.array(data)

    if data_type == 'convolutional_kernel_stats_svd':

        l_img = transform.get_LAB_L(block)
        normed_l_img = utils.normalize_2D_arr(l_img)

        # bilateral with window of size (5, 5)
        normed_diff = convolution.convolution2D(normed_l_img, kernels.min_bilateral_diff, (5, 5))

        # getting sigma vector from SVD compression
        s = compression.get_SVD_s(normed_diff)

        data = s

    if data_type == 'svd_entropy':
        l_img = transform.get_LAB_L(block)

        blocks = segmentation.divide_in_blocks(l_img, (20, 20))

        values = []
        for b in blocks:
            sv = compression.get_SVD_s(b)
            values.append(utils.get_entropy(sv))
        data = np.array(values)

    if data_type == 'svd_entropy_20':
        l_img = transform.get_LAB_L(block)

        blocks = segmentation.divide_in_blocks(l_img, (20, 20))

        values = []
        for b in blocks:
            sv = compression.get_SVD_s(b)
            values.append(utils.get_entropy(sv))
        data = np.array(values)

    if data_type == 'svd_entropy_noise_20':
        l_img = transform.get_LAB_L(block)

        blocks = segmentation.divide_in_blocks(l_img, (20, 20))

        values = []
        for b in blocks:
            sv = compression.get_SVD_s(b)
            sv_size = len(sv)
            values.append(utils.get_entropy(sv[int(sv_size / 4):]))
        data = np.array(values)
        
    return data


def w2d(arr, mode='haar', level=1):
    #convert to float   
    imArray = arr
    np.divide(imArray, 255)

    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def _get_mscn_variance(block, sub_block_size=(50, 50)):

    blocks = segmentation.divide_in_blocks(block, sub_block_size)

    data = []

    for block in blocks:
        mscn_coefficients = transform.get_mscn_coefficients(block)
        flat_coeff = mscn_coefficients.flatten()
        data.append(np.var(flat_coeff))

    return np.sort(data)

