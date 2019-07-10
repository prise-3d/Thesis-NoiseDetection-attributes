#! bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need of vector size"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No argument supplied"
    echo "Need of feature information"
    exit 1
fi

result_filename="results/models_comparisons.csv"
VECTOR_SIZE=200
size=$1
feature=$2

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

for nb_zones in {4,6,8,10,12}; do
    for mode in {"svd","svdn","svdne"}; do
        for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do

            FILENAME="data/${model}_N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}"
            MODEL_NAME="${model}_N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}"
            CUSTOM_MIN_MAX_FILENAME="N${size}_B0_E${size}_nb_zones_${nb_zones}_${feature}_${mode}_min_max"

            echo $FILENAME

            # only compute if necessary (perhaps server will fall.. Just in case)
            if grep -q "${MODEL_NAME}" "${result_filename}"; then

                echo "${MODEL_NAME} results already generated..."
            else
                python generate/generate_data_model_random_split.py --output ${FILENAME} --interval "0,${size}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 40 --random 1 --custom ${CUSTOM_MIN_MAX_FILENAME}
                python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

                python others/save_model_result_in_md_maxwell.py --interval "0,${size}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature}
            fi
        done
    done
done

