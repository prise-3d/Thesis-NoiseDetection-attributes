#! bin/bash

# erase "models_info/models_comparisons.csv" file and write new header
file_path='models_info/models_comparisons.csv'
rm ${file_path}
mkdir -p models_info
touch ${file_path}

# add of header
echo 'model_name; vector_size; start; end; nb_zones; metric; mode; train; val; test; F1_train; F1_val; F1_test' >> ${file_path}

for size in {"4","8","16","26","32","40"}; do

    for metric in {"lab","mscn","mscn_revisited","low_bits_2","low_bits_3","low_bits_4"}; do
        bash generateAndTrain_maxwell.sh ${size} ${metric} &
    done
done
