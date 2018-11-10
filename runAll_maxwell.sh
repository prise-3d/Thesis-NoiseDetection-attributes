#! bin/bash

# erase "models_info/models_comparisons.csv" file and write new header
file_path='models_info/models_comparisons.csv'
rm ${file_path}

erased=$1

if [ "$erased" == "Y" ]; then
    echo "Previous data file erased..."
    mkdir -p models_info
    touch ${file_path}
fi

# add of header
echo 'model_name; vector_size; start; end; nb_zones; metric; mode; train_size; val_size; test_size; train_acc; val_acc; test_acc; mean_acc; F1_train; F1_val; F1_test; F1_mean' >> ${file_path}

for size in {"4","8","16","26","32","40"}; do

    for metric in {"lab","mscn","mscn_revisited","low_bits_2","low_bits_3","low_bits_4"}; do
        bash generateAndTrain_maxwell.sh ${size} ${metric}
    done
done
