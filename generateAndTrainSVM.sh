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
scenes="A, B, C, D, E, G"

for size in {"4","8","16","26","32","40"}; do

  start=0
  for counter in {0..4}; do
    end=$(($start+$size))

    if [ "$end" -gt "$VECTOR_SIZE" ]; then
        start=$(($VECTOR_SIZE-$size))
        end=$(($VECTOR_SIZE))
    fi

    for zones in {"1, 3, 7, 9","0, 2, 7, 8, 9","2, 6, 8, 10, 13, 15","1, 2, 4, 7, 9, 10, 13, 15"}; do

        zones_str="${zones//, /-}"

        for metric in {"lab","mscn"}; do

            for mode in {"svd","svdn","svdne"}; do

                FILENAME="data/data_${mode}_${metric}_N${size}_B${start}_E${end}_zones${zones_str}"
                MODEL_NAME="saved_models/${INPUT_MODEL_NAME}_${mode}_${metric}_N${size}_B${start}_E${end}_zones_${zones_str}"

                echo $FILENAME
                python generate_data_model.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --metric ${metric} --scenes "${scenes}" --zones "${zones}" --percent 1 --sep ';' --rowindex '0'
                python svm_model_train.py --data ${FILENAME}.train --output ${MODEL_NAME} &

            done
        done
    done

    start=$(($start+50))
  done

done
