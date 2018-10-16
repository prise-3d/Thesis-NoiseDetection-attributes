#! bin/bash

if [ -z "$1" ]
  then
    echo "No first argument supplied"
    echo "Need of begin vector index"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No second argument supplied"
    echo "Need of end vector index"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No third argument supplied"
    echo "Need of model input"
    exit 1
fi

if [ -z "$4" ]
  then
    echo "No fourth argument supplied"
    echo "Need of mode file : 'svd', 'svdn', svdne"
    exit 1
fi

INPUT_BEGIN=$1
INPUT_END=$2
INPUT_MODEL=$3
INPUT_MODE=$4

zones="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"

for scene in {"A","B","C","D","E","F","G","H","I"}; do

  FILENAME="data_svm/data_${mode}_B${INPUT_BEGIN}_E${INPUT_END}_scene${scene}"

  python generate_data_svm.py --output ${FILENAME} --interval "${INPUT_BEGIN},${INPUT_END}" --kind ${INPUT_MODE} --scenes "${scene}" --zones "${zones}" --percent 1 --sep ";" --rowindex "0"

  python prediction.py --data "$FILENAME.train" --model ${INPUT_MODEL} --output "${INPUT_MODEL}_Scene${scene}_mode_${INPUT_MODE}.prediction"

done