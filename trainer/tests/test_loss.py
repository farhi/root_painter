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


def test_three_class_dice_perfect_score():
    """
    Test that labels and predictions where nothing is defined 
    should give 0 loss.
    """
    batch_size = 8
    classes = 3
    predictions = torch.from_numpy(np.zeros((8, 3, 500, 500)))
    # for this test we predict class 0 (probably background) for all pixels
    predictions[:, 0] = torch.from_numpy(np.ones((500, 500)).astype(np.int64))
    defined = torch.from_numpy(np.zeros((8, 500, 500)).astype(np.int64))
    labels = torch.from_numpy(np.ones((8, 500, 500)).astype(np.int64))

    assert combined_loss(predictions, defined, labels) == 0
