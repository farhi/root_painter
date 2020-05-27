"""
Computing metrics on output segmentations for root images

Copyright (C) 2019, 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# pylint: disable=C0111,R0913
from datetime import datetime
from collections import namedtuple
import numpy as np

def get_metrics_str(all_metrics, to_use=None):
    out_str = ""
    for name, val in all_metrics.items():
        if to_use is None or name in to_use:
            out_str += f" {name} {val:.4g}"
    return out_str

def get_metric_csv_row(metrics):
    now_str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    parts = [now_str, metrics['TP'], metrics['FP'], metrics['TN'],
             metrics['FN'], round(metrics['precision'], 4),
             round(metrics['recall'], 4), round(metrics['f1'], 4),
             metrics['defined'], metrics['duration']]
    return ','.join([str(p) for p in parts]) + '\n'


def get_metrics(tp:int, fp:int, tn:int, fn:int, class_name:str, eps=1e-7) -> dict:
    # for the mtrics function in model utils
    assert not np.isnan(tp)
    assert not np.isnan(fp)
    assert not np.isnan(tn)
    assert not np.isnan(fn)
    total = (tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return {
        "class": class_name,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "dice": f1,
        "true_mean": (tp + fn) / total,
        "pred_mean": (tp + fp) / total
    }
