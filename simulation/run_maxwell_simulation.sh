#! bin/bash

# file which contains model names we want to use for simulation
simulate_models="simulate_models.csv"

# selection of four scenes (only maxwell)
scenes="A, D, G, H"
VECTOR_SIZE=200

for size in {"4","8","16","26","32","40"}; do
    for feature in {"lab","mscn","mscn_revisited","low_bits_2","low_bits_3","low_bits_4","low_bits_5","low_bits_6","low_bits_4_shifted_2"}; do

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

             for nb_zones in {4,6,8,10,12,14}; do

                 for mode in {"svd","svdn","svdne"}; do
                     for model in {"svm_model","ensemble_model","ensemble_model_v2"}; do

                        FILENAME="data/${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}"
                        MODEL_NAME="${model}_N${size}_B${start}_E${end}_nb_zones_${nb_zones}_${feature}_${mode}"

                        if grep -xq "${MODEL_NAME}" "${simulate_models}"; then
                            echo "Run simulation for model ${MODEL_NAME}"

                            # by default regenerate model
                            python generate/generate_data_model_random.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --feature ${feature} --scenes "${scenes}" --nb_zones "${nb_zones}" --percent 1 --renderer "maxwell" --step 40 --random 1

                            python train_model.py --data ${FILENAME} --output ${MODEL_NAME} --choice ${model}

                            python prediction/predict_seuil_expe_maxwell_curve.py --interval "${start},${end}" --model "saved_models/${MODEL_NAME}.joblib" --mode "${mode}" --feature ${feature} --limit_detection '2'

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
    done
done
