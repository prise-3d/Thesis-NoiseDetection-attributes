#! bin/bash

# file which contains model names we want to use for simulation
simulate_models="simulate_models.csv"

start_index=0
size=24

# selection of four scenes (only maxwell)
scenes="A, D, G, H"
metric="lab"

for label in {"0","1"}; do
    for highest in {"0","1"}; do
        for nb_zones in {4,6,8,10,12}; do
            for size in {5,10,15,20,25,30,35,40}; do
                for mode in {"svd","svdn","svdne"}; do
                    for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do

                            FILENAME="data/${model}_N${size}_B${start_index}_E${size}_nb_zones_${nb_zones}_${metric}_${mode}_corr_L${label}_H${highest}"
                            MODEL_NAME="${model}_N${size}_B${start_index}_E${size}_nb_zones_${nb_zones}_${metric}_${mode}_corr_L${label}_H${highest}"
                            CUSTOM_MIN_MAX_FILENAME="${model}_N${size}_B${start_index}_E${size}_nb_zones_${nb_zones}_${metric}_${mode}_corr_L${label}_H${highest}_min_max_values"

                            echo ${MODEL_NAME}

                        if grep -xq "${MODEL_NAME}" "${simulate_models}"; then
                            echo "Run simulation for model ${MODEL_NAME}"

                            python generate/generate/generate_data_model_corr_random.py --output ${FILENAME} --n ${size} --highest ${highest} --label ${label} --kind ${mode} --metric ${metric} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 10 --random 1 --custom 1

                            python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

                            python prediction/predict_seuil_expe_maxwell_curve.py --interval "${start_index},${size}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --metric ${metric} --limit_detection '2' --custom ${CUSTOM_MIN_MAX_FILENAME}

                            python others/save_model_result_in_md_maxwell.py --interval "${start_index},${size}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --metric ${metric}

                        fi
                    done
                done
            done
        done
    done
done
