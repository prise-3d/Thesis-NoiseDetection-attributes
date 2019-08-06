#! bin/bash

# file which contains model names we want to use for simulation
simulate_models="simulate_models.csv"

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

feature="sub_blocks_stats_reduced"
start_index=0
end_index=24
number=24

for nb_zones in {4,6,8,10,12}; do

    for mode in {"svd","svdn","svdne"}; do
        for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do

            FILENAME="data/${model}_N${number}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}"
            MODEL_NAME="${model}_N${number}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}"

            if grep -xq "${MODEL_NAME}" "${simulate_models}"; then
                echo "Run simulation for model ${MODEL_NAME}"

                # by default regenerate model
                python generate/generate_data_model_random.py --output ${FILENAME} --interval "${start_index},${end_index}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --random 1

                python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

                python prediction/predict_seuil_expe_maxwell_curve.py --interval "${start_index},${end_index}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature} --limit_detection '2'

                python others/save_model_result_in_md_maxwell.py --interval "${start_index},${end_index}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature}

            fi
        done
    done
done
