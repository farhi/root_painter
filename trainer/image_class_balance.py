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


plt.figure(figsize=(12*0.8, 8*0.8))
plt.ylabel('number of esophagus voxels')
plt.xlabel('Image (ordered by esophagus size)')
vals = []
names = []
for i in range(1, 51):
    data_dir = os.path.join(Path.home(), 'datasets', 'Thoracic_OAR', str(i))
    annot, _ = im_utils.load_image(os.path.join(data_dir, 'label.nii.gz'))
    vals.append(np.sum(annot==4))
    names.append(str(i))


plt.grid()
vals, names = zip(*sorted(zip(vals, names)))
rects = plt.bar(list(range(len(vals))), vals, color=([plt.rcParams['axes.prop_cycle'].by_key()['color'][4]] * 50))
#autolabel(rects)
#plt.xticks(range(len(names)), names)
plt.savefig('esophagus_sizes.png')

### as %
plt.figure(figsize=(12*0.8, 8*0.8))
plt.ylabel('Esophagus % of image')
plt.xlabel('Image (ordered by esophagus size)')
vals = []
names = []
for i in range(1, 51):
    data_dir = os.path.join(Path.home(), 'datasets', 'Thoracic_OAR', str(i))
    annot, _ = im_utils.load_image(os.path.join(data_dir, 'label.nii.gz'))
    vals.append(100 * ((np.sum(annot==4)) / annot.size))
    names.append(str(i))


plt.grid()
vals, names = zip(*sorted(zip(vals, names)))
rects = plt.bar(list(range(len(vals))), vals, color=([plt.rcParams['axes.prop_cycle'].by_key()['color'][4]] * 50))
#autolabel(rects)
#plt.xticks(range(len(names)), names)
plt.savefig('esophagus_percent.png')


### Heart % of image.
plt.figure(figsize=(12*0.8, 8*0.8))
plt.ylabel('Heart % of image')
plt.xlabel('Image (ordered by heart size)')
vals = []
names = []
for i in range(1, 51):
    data_dir = os.path.join(Path.home(), 'datasets', 'Thoracic_OAR', str(i))
    annot, _ = im_utils.load_image(os.path.join(data_dir, 'label.nii.gz'))
    vals.append(100 * ((np.sum(annot==3)) / annot.size))
    names.append(str(i))

plt.grid()
vals, names = zip(*sorted(zip(vals, names)))
rects = plt.bar(list(range(len(vals))), vals, color=([plt.rcParams['axes.prop_cycle'].by_key()['color'][3]] * 50))
#autolabel(rects)
#plt.xticks(range(len(names)), names)
plt.savefig('heart_percent.png')

