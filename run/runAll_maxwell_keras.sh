#! bin/bash

# erase "results/models_comparisons.csv" file and write new header
file_path='results/models_comparisons.csv'
list="all, center, split"

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need argument from [${list}]"
    exit 1
fi

if [[ "$1" =~ ^(all|center|split)$ ]]; then
    echo "$1 is in the list"
else
    echo "$1 is not in the list"
fi

data=$1
erased=$2

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p results
    touch ${file_path}

    # add of header
    echo 'model_name; vector_size; start; end; nb_zones; feature; mode; tran_size; val_size; test_size; train_pct_size; val_pct_size; test_pct_size; train_acc; val_acc; test_acc; all_acc; F1_train; recall_train; roc_auc_train; F1_val; recall_val; roc_auc_val; F1_test; recall_test; roc_auc_test; F1_all; recall_all; roc_auc_all;' >> ${file_path}


fi

start_index=0
end_index=24

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

declare -A features_size
features_size=( ["sub_blocks_stats"]="24" ["sub_blocks_stats_reduced"]="20" ["sub_blocks_area"]="16" ["sub_blocks_area_normed"]="20")

for feature in {"sub_blocks_stats","sub_blocks_stats_reduced","sub_blocks_area","sub_blocks_area_normed"}; do
    for nb_zones in {4,6,8,10,12}; do

        for mode in {"svd","svdn","svdne"}; do

            end_index=${features_size[${feature}]}

            FILENAME="data/deep_keras_N${end_index}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}"
            MODEL_NAME="deep_keras_N${end_index}_B${start_index}_E${end_index}_nb_zones_${nb_zones}_${feature}_${mode}"

            echo $FILENAME

            # only compute if necessary (perhaps server will fall.. Just in case)
            if grep -q "${MODEL_NAME}" "${file_path}"; then

                echo "${MODEL_NAME} results already generated..."
            else
                python generate/generate_data_model_random_${data}.py --output ${FILENAME} --interval "${start_index},${end_index}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 10 --random 1
                python deep_network_keras_svd.py --data ${FILENAME} --output ${MODEL_NAME} --size ${end_index}

                python others/save_model_result_in_md_maxwell.py --interval "${start_index},${end_index}" --model "saved_models/${MODEL_NAME}.json" --mode "${mode}" --feature ${feature}
            fi
        done
    done
done

