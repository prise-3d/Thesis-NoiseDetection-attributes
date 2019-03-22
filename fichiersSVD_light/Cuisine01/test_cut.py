from PIL import Image
import numpy as np

image_path_noisy = 'cuisine01_00400.png'
image_path_ref = 'cuisine01_01200.png'

image_noisy = np.asarray(Image.open(image_path_noisy))
image_ref = np.asarray(Image.open(image_path_ref))

first_part = image_noisy[:, 0:400]
second_part = image_ref[:, 400:800]

final_image = Image.fromarray(np.concatenate((first_part, second_part), axis=1))

final_image.show()

