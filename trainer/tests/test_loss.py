"""
Test the loss function

Copyright (C) 2020 Abraham George Smith

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
import numpy as np
import model_utils
from loss import combined_loss
import torch


def test_low_loss_when_matching_labels():
    """
    Test that labels and prediction should give low loss
    when they are the same.
    """
    batch_size = 8
    classes = 3
    out_tile_w = 2
    predictions = torch.zeros(8, 3, out_tile_w, out_tile_w)
    # for this test we predict class 0 (background) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for 
    # the target class in order to achieve close to 0 loss.
    predictions[:, 0] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long) * 8
    # everything is defined. should have no influence
    defined = torch.ones(8, out_tile_w, out_tile_w, dtype=torch.long)
    # we label all tiles to be class 0 (a background class).
    labels = torch.zeros(8, out_tile_w, out_tile_w, dtype=torch.long)
    assert combined_loss(predictions.cuda(), defined.cuda(), labels.cuda()) < 0.01


def test_high_loss_when_not_matching_labels():
    batch_size = 8
    classes = 3
    out_tile_w = 2
    predictions = torch.zeros(8, 3, out_tile_w, out_tile_w)
    # for this test we predict class 0 (background) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for 
    # the target class in order to achieve close to 0 loss.
    predictions[:, 0] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long) * 8
    # everything is defined. should have no influence
    defined = torch.ones(8, out_tile_w, out_tile_w, dtype=torch.long)
    # we label all tiles to be class 1 (a foreground class) that should cause high error.
    labels = torch.ones(8, out_tile_w, out_tile_w, dtype=torch.long)
    assert combined_loss(predictions.cuda(), defined.cuda(), labels.cuda()) > 1.00

def test_low_loss_when_not_matching_labels_but_not_defined():
    # this would give a large error, except that it won't be considered due to the defined map.
    batch_size = 8
    classes = 3
    out_tile_w = 2
    predictions = torch.zeros(8, 3, out_tile_w, out_tile_w)
    # for this test we predict class 0 (background) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for 
    # the target class in order to achieve close to 0 loss.
    predictions[:, 0] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long) * 8
    # everything is defined. should have no influence
    defined = torch.zeros(8, out_tile_w, out_tile_w, dtype=torch.long)
    # we label all tiles to be class 1 (a foreground class) that should cause high error.
    labels = torch.ones(8, out_tile_w, out_tile_w, dtype=torch.long)
    assert combined_loss(predictions.cuda(), defined.cuda(), labels.cuda()) < 0.001

def test_low_loss_when_not_matching_preds_but_undefined():
    # both labels and predictions can be different to bg but
    # and different to each other but still give low loss
    # due to all being undefined. 
    batch_size = 8
    classes = 3
    out_tile_w = 2
    predictions = torch.zeros(8, 3, out_tile_w, out_tile_w)
    # for this test we predict class 2 (a fg class) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for 
    # the target class in order to achieve close to 0 loss.
    predictions[:, 2] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long) * 8
    # everything is defined. should have no influence
    defined = torch.zeros(8, out_tile_w, out_tile_w, dtype=torch.long)
    # we label all tiles to be class 1 (a foreground class) that should cause high error.
    labels = torch.ones(8, out_tile_w, out_tile_w, dtype=torch.long)
    assert combined_loss(predictions.cuda(), defined.cuda(), labels.cuda()) < 0.001
