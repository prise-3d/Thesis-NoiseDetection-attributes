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

        for metric in {"lab","mscn"}; do
    
            for mode in {"svd","svdn","svdne"}; do

                FILENAME="data/data_${mode}_${metric}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_random"
                MODEL_NAME="${INPUT_MODEL_NAME}_${mode}_${metric}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}"

                echo $FILENAME
                python generate_data_model_random.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --metric ${metric} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --sep ';' --rowindex '0'
                python ensemble_model_train.py --data ${FILENAME}.train --output ${MODEL_NAME}
                bash testModelByScene.sh "${start}" "${end}" "./saved_models/${MODEL_NAME}.joblib" "${mode}" "${metric}" >> "./saved_models/${MODEL_NAME}.tex"

            done
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
