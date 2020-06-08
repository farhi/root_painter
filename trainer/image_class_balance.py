from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import os
import math
import im_utils
import numpy as np


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.figure(figsize=(12*0.8, 8*0.8))

plt.title('Image 1 class balance')
data_dir = os.path.join(Path.home(), 'datasets', 'Thoracic_OAR', '1')
annot, _ = im_utils.load_image(os.path.join(data_dir, 'label.nii.gz'))

class_names = ['background', 'lung1', 'lung2', 'heart',
               'esophagus', 'trachia', 'spine']

names = []
vals = []
for i, name in enumerate(class_names):
    names.append(name)
    vals.append(np.sum(annot==i))


rects = plt.bar(list(range(len(vals))), vals, color=(plt.rcParams['axes.prop_cycle'].by_key()['color']))
autolabel(rects)
plt.ylabel('number of voxels')
plt.xlabel('class')
plt.xticks(range(len(names)), names)
plt.savefig('image_class_balance.png')

plt.figure(figsize=(12*0.8, 8*0.8))
plt.title('Image 1 forground class balance')
data_dir = os.path.join(Path.home(), 'datasets', 'Thoracic_OAR', '1')
annot, _ = im_utils.load_image(os.path.join(data_dir, 'label.nii.gz'))

class_names = ['background', 'lung1', 'lung2', 'heart',
               'esophagus', 'trachia', 'spine']

names = []
vals = []
for i, name in enumerate(class_names):
    if name != 'background':
        names.append(name)
        vals.append(np.sum(annot==i))


rects = plt.bar(list(range(len(vals))), vals, color=(plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]))
autolabel(rects)
plt.ylabel('number of voxels')
plt.xlabel('class')
plt.xticks(range(len(names)), names)
plt.savefig('image_class_balance_foreground.png')
