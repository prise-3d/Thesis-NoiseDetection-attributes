#! bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need of vector size"
    exit 1
fi

VECTOR_SIZE=200
size=$1

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

for nb_zones in {6,8,10,12,16}; do

    echo $start $end

    for metric in {"lab","mscn"}; do
        for mode in {"svd","svdn","svdne"}; do
            for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do

                FILENAME="data/data_maxwell_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${metric}_${mode}"
                MODEL_NAME="${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${metric}_${mode}"

                echo $FILENAME
                python generate_data_model_random.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --metric ${metric} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --sep ';' --rowindex '0'
                python models/${model}_train.py --data ${FILENAME}.train --output ${MODEL_NAME}
                python predict_seuil_expe.py --interval "${start}, ${end}" --model "./saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --metric ${metric} --limit_detection '2'
                python save_model_result_in_md.py --interval "${start}, ${end}" --model "./saved_models/${MODEL_NAME}.joblib" --mode "${mode}"
            done
        done
    done
done

if [ "$counter" -eq "0" ]; then
    start=$(($start+50-$half))
else 
    start=$(($start+50))
fi

done