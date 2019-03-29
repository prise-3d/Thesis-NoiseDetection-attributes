#! bin/bash

for metric in {"lab","mscn","low_bits_2","low_bits_3","low_bits_4","low_bits_5","low_bits_6","low_bits_4_shifted_2"}; do
    for scene in {"A","D","G","H"}; do
        python display_svd_data_scene.py --scene ${scene} --interval "0,800" --indices "0, 2000" --metric ${metric} --mode svdne --step 100 --norm 1 --ylim "0, 0.01"
    done
done
