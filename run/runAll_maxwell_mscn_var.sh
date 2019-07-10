#! bin/bash

# erase "results/models_comparisons.csv" file and write new header
file_path='results/models_comparisons.csv'

erased=$1

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
    touch ${file_path}

    # add of header
    echo 'model_name; vector_size; start_index; end; nb_zones; feature; mode; tran_size; val_size; test_size; train_pct_size; val_pct_size; test_pct_size; train_acc; val_acc; test_acc; all_acc; F1_train; recall_train; roc_auc_train; F1_val; recall_val; roc_auc_val; F1_test; recall_test; roc_auc_test; F1_all; recall_all; roc_auc_all;' >> ${file_path}

fi

start_index=0
end_index=4

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

declare -A features_size
features_size=( ["mscn_var_4"]=4 ["mscn_var_16"]=16 ["mscn_var_64"]=64 ["mscn_var_16_max"]=4 ["mscn_var_64_max"]=16)

for nb_zones in {4,6,8,10,12}; do

    for mode in {"svd","svdn","svdne"}; do
        for feature in {"mscn_var_4","mscn_var_16","mscn_var_64","mscn_var_16_max","mscn_var_64_max"}; do
            for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do

                end_index=${features_size[${feature}]}

                FILENAME="data/${model}_N${end_index}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}"
                MODEL_NAME="${model}_N${end_index}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}"

                echo $FILENAME

                # only compute if necessary (perhaps server will fall.. Just in case)
                if grep -q "${MODEL_NAME}" "${file_path}"; then

                    echo "${MODEL_NAME} results already generated..."
                else
                    python generate/generate_data_model_random.py --output ${FILENAME} --interval "${start_index},${end_index}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 10 --random 1
                    python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

                    python others/save_model_result_in_md_maxwell.py --interval "${start_index},${end_index}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature}
                fi
            done
        done
    done
done


