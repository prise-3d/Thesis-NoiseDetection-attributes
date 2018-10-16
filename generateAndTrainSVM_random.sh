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
    echo "Need of model output name"
    exit 1
fi

VECTOR_SIZE=$1
INPUT_MODEL_NAME=$2

# selection of six scenes
scenes="A, B, C, D, E, F, G, H, I"

for size in {"4","8","16","26","32","40"}; do

  start=0
  for counter in {0..4}; do
    end=$(($start+$size))

    if [ "$end" -gt "$VECTOR_SIZE" ]; then
        start=$(($VECTOR_SIZE-$size))
        end=$(($VECTOR_SIZE))
    fi

    for nb_zones in {2,3,4,5,6,7,8,9,10}; do

        for mode in {"svd","svdn","svdne"}; do

            FILENAME="data_svm/data_${mode}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_random"
            MODEL_NAME="saved_models/${INPUT_MODEL_NAME}_${mode}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}"

            echo $FILENAME
            python generate_data_svm_random.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --sep ';' --rowindex '0'
            python svm_model_train.py --data ${FILENAME}.train --output ${MODEL_NAME} &

        done
    done
if [ -z "$2" ]
  then
    echo "No argument supplied"
    echo "Need of model output name"
    exit 1
fi

VECTOR_SIZE=$1
INPUT_MODEL_NAME=$2
    start=$(($start+50))
  done

done
