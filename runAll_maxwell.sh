#! bin/bash

for size in {"4","8","16","26","32","40"}; do
    bash generateAndTrain_maxwell.sh ${size} &
done
