import numpy as np

config_filename                 = "config"
zone_folder                     = "zone"
min_max_filename_extension      = "_min_max_values"
output_data_folder              = 'data'
dataset_path                    = 'fichiersSVD_light'
seuil_expe_filename             = 'seuilExpe'
threshold_map_folder            = 'threshold_map'
models_information_folder       = 'models_info'
saved_models_folder             = 'saved_models'
csv_model_comparisons_filename  = "models_comparisons.csv"

models_names_list               = ["svm_model","ensemble_model","ensemble_model_v2"]


# define all scenes values
scenes_names                    = ['Appart1opt02', 'Bureau1', 'Cendrier', 'Cuisine01', 'EchecsBas', 'PNDVuePlongeante', 'SdbCentre', 'SdbDroite', 'Selles']
scenes_indices                  = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

maxwell_scenes_names            = ['Appart1opt02', 'Cuisine01', 'SdbCentre', 'SdbDroite']
maxwell_scenes_indices          = ['A', 'D', 'G', 'H']

normalization_choices           = ['svd', 'svdn', 'svdne']
zones_indices                   = np.arange(16)

metric_choices_labels           = ['lab', 'mscn', 'mscn_revisited', 'low_bits_2', 'low_bits_3', 'low_bits_4', 'low_bits_5', 'low_bits_6','low_bits_4_shifted_2']
