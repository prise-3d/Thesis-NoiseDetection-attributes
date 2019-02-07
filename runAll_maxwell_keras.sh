#! bin/bash

# erase "models_info/models_comparisons.csv" file and write new header
file_path='models_info/models_comparisons.csv'

erased=$1

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
    touch ${file_path}

    # add of header
    echo 'model_name; vector_size; start_index; end; nb_zones; metric; mode; tran_size; val_size; test_size; train_pct_size; val_pct_size; test_pct_size; train_acc; val_acc; test_acc; all_acc; F1_train; recall_train; roc_auc_train; F1_test; recall_test; roc_auc_test; F1_all; recall_all; roc_auc_all;' >> ${file_path}

fi

start_index=0
end_index=24

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

metrics_size=( ["sub_blocks_stats"]=24 ["sub_blocks_stats_reduced"]=20 ["sub_blocks_area"]=16 ["sub_blocks_area_normed"]=20)

for metric in {"sub_blocks_stats","sub_blocks_stats_reduced","sub_blocks_area","sub_blocks_area_normed"}; do
    for nb_zones in {4,6,8,10,12}; do

        for mode in {"svd","svdn","svdne"}; do

            end_index=${metrics_size[${metric}]}
            FILENAME="data/data_maxwell_N${end_index}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${metric}_${mode}"
            MODEL_NAME="deep_keras_N${end_index}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${metric}_${mode}"

            echo $FILENAME

            # only compute if necessary (perhaps server will fall.. Just in case)
            if grep -q "${MODEL_NAME}" "${file_path}"; then

                echo "${MODEL_NAME} results already generated..."
            else
                python generate_data_model_random.py --output ${FILENAME} --interval "${start_index},${end_index}" --kind ${mode} --metric ${metric} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 10 --random 1
                python deep_network_keras_svd.py --data ${FILENAME} --output ${MODEL_NAME} --size ${end_index}

                python save_model_result_in_md_maxwell.py --interval "${start_index},${end_index}" --model "saved_models/${MODEL_NAME}.json" --mode "${mode}" --metric ${metric}
            fi
        done
    done
done

