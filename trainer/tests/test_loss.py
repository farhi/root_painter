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
from loss import multiclass_loss
import torch


def test_low_loss_when_matching_labels():
    """
    Test that labels and prediction should give low loss
    when they are the same.
    """
    batch_size = 8
    classes = 3
    out_tile_w = 2
    predictions = torch.zeros(batch_size, classes, out_tile_w, out_tile_w)
    # for this test we predict class 0 (background) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for
    # the target class in order to achieve close to 0 loss.
    predictions[:, 0] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long) * 8
    # everything is defined. should have no influence
    defined = torch.ones(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    # we label all tiles to be class 0 (a background class).
    labels = torch.zeros(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    assert multiclass_loss(predictions.cuda(), defined.cuda(), labels.cuda()) < 0.01


def test_high_loss_when_not_matching_labels():
    """ If the labels and predictions don't match then the loss should be high """
    batch_size = 8
    classes = 3
    out_tile_w = 2
    predictions = torch.zeros(batch_size, classes, out_tile_w, out_tile_w)
    # for this test we predict class 0 (background) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for
    # the target class in order to achieve close to 0 loss.
    predictions[:, 0] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long) * 8
    # everything is defined. should have no influence
    defined = torch.ones(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    # we label all tiles to be class 1 (a foreground class) that should cause high error.
    labels = torch.ones(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    assert multiclass_loss(predictions.cuda(), defined.cuda(), labels.cuda()) > 1.00


def test_loss_gets_better_with_more_accuracy():
    """
    Test more accurate predictions result in lower loss.

    Undefined regions cause a fairly high loss.
    my understanding is that by setting the predictions to 0
    even though the loss is high, the undefined region of the predictions
    wont be used in gradient updates.
    """
    batch_size = 8
    classes = 3
    out_tile_w = 2
    predictions = torch.zeros(batch_size, classes, out_tile_w, out_tile_w)
    # for this test we predict class 2 (a fg class) for all pixels
    # cx loss actually requires that the predictions be much higher than 1 for
    # the target class in order to achieve close to 0 loss.
    predictions[:, 2] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long)

    # we label all tiles to be class 1 (a foreground class) that should cause high error.
    labels = torch.zeros(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    labels[0, 0] = 1
    defined = (labels > 0)
    labels *= defined
    loss1 = multiclass_loss(predictions.cuda(), defined.cuda(), labels.cuda())

    labels = torch.zeros(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    labels[0, 0] = 2
    defined = (labels > 0)
    labels *= defined
    loss2 = multiclass_loss(predictions.cuda(), defined.cuda(), labels.cuda())

    labels = torch.zeros(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    labels[0, 0] = 2
    labels[0, 1] = 2
    defined = (labels > 0)
    labels *= defined
    loss3 = multiclass_loss(predictions.cuda(), defined.cuda(), labels.cuda())

    labels = torch.zeros(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    labels[:, :] = 2
    defined = (labels > 0)
    labels *= defined
    loss4 = multiclass_loss(predictions.cuda(), defined.cuda(), labels.cuda())

    # loss with only background labeled is a little different as only cx is used.
    predictions = torch.zeros(batch_size, classes, out_tile_w, out_tile_w)
    predictions[:, 2] = torch.ones(out_tile_w, out_tile_w, dtype=torch.long)
    labels = torch.zeros(batch_size, out_tile_w, out_tile_w, dtype=torch.long)
    defined = torch.zeros(out_tile_w, out_tile_w, dtype=torch.long)
    defined[0, 0] = 1
    cx_loss1 = multiclass_loss(predictions.cuda(), defined.cuda(), labels.cuda())
    # should be higher loss as labels are not accurate and now more is defined.
    defined[:, :] = 1
    cx_loss2 = multiclass_loss(predictions.cuda(), defined.cuda(), labels.cuda())
    assert loss1 > loss2 > loss3 > loss4
    assert cx_loss1 < cx_loss2
