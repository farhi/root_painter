"""
Copyright (C) 2019 Abraham George Smith

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

import torch
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy

cx_loss = torch.nn.CrossEntropyLoss()

def dice_loss(predictions, labels):
    """ based on loss function from V-Net paper """
    dice_sum = 0
    classes = torch.unique(labels)
    softmaxed = softmax(predictions, 1)
    for c in classes:
        # for each of the labels defined in the annotation.
        # get the specific class prediction
        class_preds = softmaxed[:, c, :]  
        class_label_map = (labels == c)
        class_label_map = class_label_map.float()
        class_preds = class_preds.contiguous().view(-1)
        class_label_map = class_label_map.view(-1)
        intersection = torch.sum(torch.mul(class_preds, class_label_map))
        union = torch.sum(class_preds) + torch.sum(class_label_map)
        dice_sum +=  1 - ((2 * intersection) / (union))
    return dice_sum / classes


def combined_loss(predictions, defined, labels):
    """ mix of dice and cross entropy 
    
    Shapes of input:
        predictions : (bs, classes, out_tile_w, out_tile_w)
        defined: (bs, out_tile_w, out_tile_w))
        labels: (bs, out_tile_w, out_tile_w)

    """
    # regions which are not defined are always set to 0
    # having both the predictions and labels set to 0 for these regions
    # means only the defined regions will be taken into account for model updates.
    for i in range(1, predictions.shape[1]):
        predictions[:, i] *= defined

    # this is not exactly 0 but it is close. See loss flooding for some ideas
    # why always having a bit more loss might be handy.
    # https://arxiv.org/pdf/2002.08709.pdf
    predictions[:, 0] += ((torch.ones(defined.shape) - defined) * 80)

    # undefined region should be 0
    labels *= defined 

    if torch.sum(labels) > 0:
        return (dice_loss(predictions, labels) +
               (0.3 * cx_loss(predictions, labels)))
    # When only background then use only cross entropy as dice is undefined.
    return 0.3 * cx_loss(predictions, labels)
