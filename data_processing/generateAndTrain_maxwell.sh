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

if [ -z "$3" ]
  then
    echo "No argument supplied"
    echo "Need of kind of data to use"
    exit 1
fi

result_filename="results/models_comparisons.csv"
VECTOR_SIZE=200

size=$1
feature=$2
data=$3

# selection of four scenes (only maxwell)
scenes="A, D, G, H"

half=$(($size/2))
start=-$half
for counter in {0..4}; do
    end=$(($start+$size))

    if [ "$end" -gt "$VECTOR_SIZE" ]; then
        start=$(($VECTOR_SIZE-$size))
        end=$(($VECTOR_SIZE))
    fi

    if [ "$start" -lt "0" ]; then
        start=$((0))
        end=$(($size))
    fi

    for nb_zones in {4,6,8,10,12}; do

        echo $start $end

        for mode in {"svd","svdn","svdne"}; do
            for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do

                FILENAME="data/${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}"
                MODEL_NAME="${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}"

                echo $FILENAME

                # only compute if necessary (perhaps server will fall.. Just in case)
                if grep -q "${MODEL_NAME}" "${result_filename}"; then

                    echo "${MODEL_NAME} results already generated..."
                else
                    python generate/generate_data_model_random_${data}.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --renderer "maxwell" --step 40 --random 1 --percent 1
                    python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

                    #python prediction/predict_seuil_expe_maxwell.py --interval "${start},${end}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature} --limit_detection '2'
                    python others/save_model_result_in_md_maxwell.py --interval "${start},${end}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature}
                fi
            done
        done
    done

    if [ "$counter" -eq "0" ]; then
        start=$(($start+50-$half))
    else
        start=$(($start+50))
    fi

done
