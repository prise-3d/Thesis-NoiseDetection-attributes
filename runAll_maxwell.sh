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
    echo 'model_name; vector_size; start; end; nb_zones; metric; mode; tran_size; val_size; test_size; train_pct_size; val_pct_size; test_pct_size; train_acc; val_acc; test_acc; all_acc; F1_train; F1_val; F1_test; F1_all' >> ${file_path}

fi

for size in {"4","8","16","26","32","40"}; do

    for metric in {"lab","mscn","mscn_revisited","low_bits_2","low_bits_3","low_bits_4"}; do
        bash generateAndTrain_maxwell.sh ${size} ${metric}
    done
done
