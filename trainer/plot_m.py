from collections import defaultdict
import matplotlib.pyplot as plt
import os
import math


csv_files = [
    'metric_log_cross_entropy.csv',
    'metric_log_dice.csv',
    'metric_log_combined.csv'
]

names = ['cross entropy', 'dice', 'combined']

#plt.figure(figsize=(16, 40))
fig, axs = plt.subplots(4, figsize=(14, 14))
fig.suptitle('Fitting a single patch from struct seg. Comparing loss functions')

counts = defaultdict(int)


f1s_list = []

for i, f in enumerate(csv_files):
    lines = open(f).readlines()
    print('line length before', len(lines))
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if len(l)] 
    print('line length after', len(lines))
    parts_list = [l.split(',') for l in lines]
    
    f1s = defaultdict(list)
    for row in parts_list:
        t, name, tp, fp, tn, fn, prec, rec, f1 = row
        if name == 'lungs1':
            name = 'lung1'
        f1 = float(f1)
        if math.isnan(f1):
            f1s[name].append(0)
        else:
            f1s[name].append(float(f1))
        print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)
        counts[name] = int(tp) + int(fn)
        print(name, counts[name], counts[name] <= 677448)
        
    axs[i].set_title(names[i])
    axs[i].set_ylabel('dice')
    for key in f1s:
        axs[i].plot(f1s[key], label=key)
    axs[i].grid()
    if i == 2:
        axs[i].legend()
    axs[i].set_ylim(0, 1)
    axs[i].set_xlim(0, 1050)
    axs[i].set_xticks(range(0, 1050, 50))
    f1s_list.append(f1s)


for j, f1s in enumerate(f1s_list):
    mean_f1s = []
    for i in range(len(f1s['heart'])):
        s = 0
        t = 0
        for key in f1s:
            t += 1
            f1 = float(f1s[key][i])
            if math.isnan(f1):
                s += 0 
            else:
                print('not nan:', f1)
                s += f1
        mean_f1s.append(s/t)
    axs[3].plot(mean_f1s, label=names[j])
    
axs[3].set_title('Mean dice for each step')
axs[3].set_ylabel('dice')
axs[3].set_ylim(0, 1)
axs[3].grid()
axs[3].set_xlim(0, 1050)
axs[3].set_xlabel('update step')
axs[3].set_xticks(range(0, 1050, 50))
axs[3].legend()

plt.savefig('loss_compare.png')


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.figure()
plt.title('Class Balance in Patch')
names = []
vals = []
for key in counts:
    names.append(key)
    vals.append(int(counts[key]))
plt.ylim(0, 350000) 

rects = plt.bar(list(range(len(vals))), vals, color=(plt.rcParams['axes.prop_cycle'].by_key()['color']))
autolabel(rects)
plt.xticks(range(len(names)), names)
plt.savefig('patch_class_balance.png')
