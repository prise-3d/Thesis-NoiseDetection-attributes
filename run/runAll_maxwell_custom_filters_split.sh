#! bin/bash

# erase "results/models_comparisons.csv" file and write new header
file_path='results/models_comparisons.csv'

erased=$1

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p results
    touch ${file_path}

    # add of header
    echo 'model_name; vector_size; start; end; nb_zones; metric; mode; train_size; val_size; test_size; train_pct_size; val_pct_size; test_pct_size; train_acc; val_acc; test_acc; all_acc; F1_train; recall_train; roc_auc_train; F1_val; recall_val; roc_auc_val; F1_test; recall_test; roc_auc_test; F1_all; recall_all; roc_auc_all;' >> ${file_path}

fi

for size in {"4","8","16","26","32","40","60","80"}; do

#    for metric in {"highest_sv_std_filters","lowest_sv_std_filters","highest_wave_sv_std_filters","lowest_sv_std_filters","highest_sv_std_filters_full","lowest_sv_std_filters_full","highest_sv_entropy_std_filters","lowest_sv_entropy_std_filters"}; do
    for metric in {"highest_sv_entropy_std_filters","lowest_sv_entropy_std_filters"}; do
        bash data_processing/generateAndTrain_maxwell_custom_filters_split.sh ${size} ${metric} &
    done
done
