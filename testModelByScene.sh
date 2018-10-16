#! bin/bash

if [ -z "$1" ]
  then
    echo "No first argument supplied"
    echo "Need of vector size"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No second argument supplied"
    echo "Need of model input"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No third argument supplied"
    echo "Need of separator char : ':', ';'"
    exit 1
fi

if [ -z "$4" ]
  then
    echo "No fourth argument supplied"
    echo "Need of index row indication : 0 or 1"
    exit 1
fi

VECTOR_SIZE=$1
INPUT_MODEL=$2
INPUT_SEP=$3
INPUT_ROW=$4


for size in {"4","8","16","26","32","40"}; do

  start=0
  for counter in {0..4}; do
    end=$(($start+$size))

    if [ "$end" -gt "$VECTOR_SIZE" ]; then
        start=$(($VECTOR_SIZE-$size))
        end=$(($VECTOR_SIZE))
    fi

  
    zones = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"
    zones_str="${zones//, /-}"

    for scene in {"A","B","C","D","E","F","G","H","I"}; do

        for mode in {"svd","svdn","svdne"}; do
            FILENAME="data_svm/data_${mode}_N${size}_B${start}_E${end}_scene${scene}"

            echo $FILENAME
            python generate_data_svm.py --output ${FILENAME} --interval "${start},${end}" --kind ${mode} --scenes "${scene}" --zones "${zones}" --percent 1 --sep "${INPUT_SEP}" --rowindex "${INPUT_ROW}"
            python prediction.py --data "$FILENAME.train" --model ${INPUT_MODEL} --output "${INPUT_MODEL}.prediction"

        done
    done

    start=$(($start+50))
  done

done
