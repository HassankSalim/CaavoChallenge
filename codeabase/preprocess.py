import pandas as pd
import numpy as np
from opencv_utils import resize, save, showImg, read


best_width = 209
best_height = 162

df = pd.read_csv('../dataset/best_dataset.csv')

filenames = df['path']
counter = 0

for filename in filenames:
	img = read(filename, 1)
	resized_img = resize(img, best_width, best_height)
	save(filename, resized_img)