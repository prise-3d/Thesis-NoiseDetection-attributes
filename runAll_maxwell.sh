#! bin/bash

for size in {"4","8","16","26","32","40"}; do

    for metric in {"lab","mscn","low_bits_4","low_bits_2"}; do
        bash generateAndTrain_maxwell.sh ${size} ${metric}
    done
done
