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


def dice_loss(class_preds, class_labels):
    """ based on loss function from V-Net paper """
    assert class_labels.dtype == torch.float32
    preds = class_preds.contiguous().view(-1)
    labels = class_labels.view(-1)
    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def multiclass_loss(predictions, defined, labels):
    """ mix of dice and cross entropy """
    classes = torch.unique(labels).long()
    # remove any of the predictions for which we don't have ground truth
    # Set outputs to 0 where annotation undefined so that
    # The network can predict whatever it wants without any penalty.
    for c in range(predictions.shape[1]):
        predictions[:, c] *= defined
    softmaxed = softmax(predictions, 1)
    dice_loss_sum = 0
    for c in classes:
        class_preds = softmaxed[:, c]
        class_labels = (labels == c).float()
        dice_loss_sum += dice_loss(class_preds, class_labels)
    dice_loss_sum /= len(classes)
    return dice_loss_sum + (0.3 * cross_entropy(predictions, labels))
