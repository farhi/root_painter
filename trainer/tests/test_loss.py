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
# pylint: disable=E0401 # import error
import torch
from loss import combined_loss


def test_low_loss_when_matching_labels():
    """
    Test that labels and prediction should give low loss
    when they are the same.
    """
    batch_size = 8
    classes = 1
    out_tile_w = 2
    predictions = torch.zeros(batch_size, classes * 2, out_tile_w, out_tile_w)
    # for this test we predict class 0 (background) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for
    # the target class in order to achieve close to 0 loss.
    predictions[:, 0] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long) * 8
    # we label all tiles to be class 0 (a background class).
    labels = torch.zeros(batch_size, classes * 2, out_tile_w, out_tile_w, dtype=torch.long)
    assert combined_loss(predictions.cuda(), labels.cuda()) < 0.01


def test_high_loss_when_not_matching_labels():
    """ If the labels and predictions don't match then the loss should be high """
    batch_size = 8
    classes = 1
    out_tile_w = 2
    predictions = torch.zeros(batch_size, classes * 2, out_tile_w, out_tile_w)
    # for this test we predict class 0 (background) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for
    # the target class in order to achieve close to 0 loss.
    predictions[:, 0] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long) * 8
    # we label all tiles to be class 1 (a foreground class) that should cause high error.
    labels = torch.ones(batch_size, classes * 2, out_tile_w,
                        out_tile_w, dtype=torch.long)
    assert combined_loss(predictions.cuda(), labels.cuda()) > 1.00
