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
    echo 'model_name; vector_size; start; end; nb_zones; metric; mode; tran_size; val_size; test_size; train_pct_size; val_pct_size; test_pct_size; train_acc; val_acc; test_acc; all_acc; F1_train; recall_train; roc_auc_train; F1_val; recall_val; roc_auc_val; F1_test; recall_test; roc_auc_test; F1_all; recall_all; roc_auc_all;' >> ${file_path}


fi

start_index=0
end_index=24

# selection of four scenes (only maxwell)
scenes="A, D, G, H"
metric="lab"

for label in {"0","1"}; do
    for highest in {"0","1"}; do
        for nb_zones in {4,6,8,10,12}; do
            for size in {5,10,15,20,25,30,35,40}; do
                for mode in {"svd","svdn","svdne"}; do

                    FILENAME="data/deep_keras_N${size}_B${start_index}_E${size}_nb_zones_${nb_zones}_${metric}_${mode}_corr_L${label}_H${highest}"
                    MODEL_NAME="deep_keras_N${size}_B${start_index}_E${size}_nb_zones_${nb_zones}_${metric}_${mode}_corr_L${label}_H${highest}"

                    echo $FILENAME

                    # only compute if necessary (perhaps server will fall.. Just in case)
                    if grep -q "${MODEL_NAME}" "${file_path}"; then

                        echo "${MODEL_NAME} results already generated..."
                    else
                        python generate/generate_data_model_corr_random.py --output ${FILENAME} --n ${size} --highest ${highest} --label ${label} --kind ${mode} --metric ${metric} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 10 --random 1 --custom 1
                        python deep_network_keras_svd.py --data ${FILENAME} --output ${MODEL_NAME} --size ${size}

                        # use of interval but it is not really an interval..
                        python others/save_model_result_in_md_maxwell.py --interval "${start_index},${size}" --model "saved_models/${MODEL_NAME}.json" --mode "${mode}" --metric ${metric}
                    fi
                done
            done
        done
    done
done

