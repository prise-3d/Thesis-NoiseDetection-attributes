#! bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need of vector size"
    exit 1
fi

VECTOR_SIZE=$1
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

        for mode in {"svd","svdn","svdne"}; do

            FILENAME="data_svm/data_${mode}_N${size}_B${start}_E${end}_zones${zones_str}"

            echo $FILENAME
            python generate_data_svm.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --scenes "${scenes}" --zones "${zones}" --percent 1 --sep : --rowindex 1
            ./apprentissage.sh -log2c -20,20,1 -log2g -20,20,1 ${FILENAME}.train &

        done
    done

    start=$(($start+50))
  done

done
