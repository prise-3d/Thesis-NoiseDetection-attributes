import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os, sys

label_freq = 6
folder_path = "Curve_simulations"

data_files = os.listdir(folder_path)

scene_names = [f.split('_')[3] for f in data_files]

for id, f in enumerate(data_files):

    print(scene_names[id])
    path_file = os.path.join(folder_path, f)

    df = pd.read_csv(path_file, header=None, sep=";")


    fig=plt.figure(figsize=(8, 8))
    fig.suptitle("Detection simulation for " + scene_names[id] + " scene", fontsize=20)

    for index, row in df.iterrows():

        row = np.asarray(row)

        threshold = row[2]
        start_index = row[3]
        step_value = row[4]

        counter_index = 0

        current_value = start_index

        while(current_value < threshold):
            counter_index += 1
            current_value += step_value

        fig.add_subplot(4, 4, (index + 1))
        plt.plot(row[5:])

        # draw vertical line from (70,100) to (70, 250)
        plt.plot([counter_index, counter_index], [-2, 2], 'k-', lw=2, color='red')
        plt.ylabel('Not noisy / Noisy', fontsize=18)
        plt.xlabel('Time in minutes / Samples per pixel', fontsize=16)

        x_labels = [id * step_value + start_index for id, val in enumerate(row[5:]) if id % label_freq == 0]

        x = [v for v in np.arange(0, len(row[5:])+1) if v % label_freq == 0]

        plt.xticks(x, x_labels, rotation=45)
        plt.ylim(-1, 2)

    plt.show()
