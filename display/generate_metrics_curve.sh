#! /bin/bash

for feature in {"lab","mscn","low_bits_2","low_bits_3","low_bits_4","low_bits_5","low_bits_6","low_bits_4_shifted_2"}; do

    python display/display/display_svd_data_scene.py --scene D --interval "0, 800" --indices "0, 1200" --feature ${feature} --mode svdne --step 100 --norm 1 --error mse --ylim "0, 0.1"

done


