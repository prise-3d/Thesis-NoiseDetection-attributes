#! bin/bash

# file which contains model names we want to use for simulation
simulate_models="simulate_models_keras.csv"

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

start_index=0
features_size=( ["sub_blocks_stats"]=24 ["sub_blocks_stats_reduced"]=20 ["sub_blocks_area"]=16 ["sub_blocks_area_normed"]=20)

for feature in {"sub_blocks_stats","sub_blocks_stats_reduced","sub_blocks_area","sub_blocks_area_normed"}; do
    for nb_zones in {4,6,8,10,12}; do

        for mode in {"svd","svdn","svdne"}; do

            end_index=${features_size[${feature}]}
            FILENAME="data/deep_keras_N${end_index}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}"
            MODEL_NAME="deep_keras_N${end_index}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}"

            CUSTOM_MIN_MAX_FILENAME="N${size}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}_min_max"

            if grep -xq "${MODEL_NAME}" "${simulate_models}"; then
                echo "Run simulation for model ${MODEL_NAME}"

                # by default regenerate model
                python generate/generate_data_model_random.py --output ${FILENAME} --interval "${start_index},${end_index}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 40 --random 1 --custom ${CUSTOM_MIN_MAX_FILENAME}

                python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

                python prediction/predict_seuil_expe_maxwell_curve.py --interval "${start_index},${end_index}" --model "saved_models/${MODEL_NAME}.json" --mode "${mode}" --feature ${feature} --limit_detection '2' --custom ${CUSTOM_MIN_MAX_FILENAME}

                python others/save_model_result_in_md_maxwell.py --interval "${start_index},${end_index}" --model "saved_models/${MODEL_NAME}.json" --mode "${mode}" --feature ${feature}

            fi
        done
    done
done
