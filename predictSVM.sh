#! bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need of vector size"
    exit 1
fi

VECTOR_SIZE=$1
# selection of six scenes
scenes="Appart1opt02, Bureau1, Cendrier, PNDVuePlongeante, SdbDroite, Selles"

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

        for mode in {"svd","svdn","svdne"}; do


            MODEL_FILENAME="data_svm/data_${mode}_N${size}_B${start}_E${end}_zones${zones_str}.train.model"
            TEST_FILENAME="data_svm/data_${mode}_N${size}_B${start}_E${end}_zones${zones_str}.test"

             ./prediction.sh ${TEST_FILENAME} ${MODEL_FILENAME} &

        done
    done

    start=$(($start+50))
  done

done
