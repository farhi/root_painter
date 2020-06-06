"""
Test training a 3D CNN

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

import time

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from unet3d import UNet3D


@pytest.mark.slow
def test_train_to_predict_zeros_from_random():
    """
    Very simple test to check that we can train the CNN model to predict 0
    We will use the same input (random tensor) every time
    """
    num_classes = 3
    cnn = UNet3D(im_channels=1, out_channels=num_classes).cuda()
    test_input = torch.rand((1, 1, 56, 240, 240)).cuda()
    # bs = 1,
    target = torch.zeros((1, 18, 194, 194)).cuda().long()
    start = time.time()
    outputs = cnn(test_input)
    assert outputs.shape[1] == num_classes
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    class_1_pred_sum = torch.sum(outputs[:, 1])
    while class_1_pred_sum > 0:
        optimizer.zero_grad()
        outputs = cnn(test_input)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        optimizer.step()
        class_1_pred_sum = torch.sum(outputs[:, 1])
    assert class_1_pred_sum == 0

@pytest.mark.slow
def test_train_to_predict_ones_from_random():
    """
    Very simple test to check that we can train the CNN model to predict 1
    We will use the same input (random tensor) every time
    """
    num_classes = 31
    cnn = UNet3D(im_channels=1, out_channels=num_classes).cuda()
    bs = 1
    test_input = torch.rand((bs, 1, 56, 240, 240)).cuda()
    target = torch.ones((bs, 18, 194, 194)).cuda().long()
    start = time.time()
    outputs = cnn(test_input)
    assert outputs.shape[1] == num_classes
    assert outputs.shape[0] == bs
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    for i in range(100):
        optimizer.zero_grad()
        outputs = cnn(test_input)
        loss = F.cross_entropy(outputs, target)
        loss.backward()
        optimizer.step()
        class_preds = torch.argmax(outputs, dim=1) # class predictions
        class1_preds_sum = torch.sum(class_preds)
        diff_from_1s = abs(class1_preds_sum - torch.sum(target))
        unique = torch.unique(class_preds)
        if len(unique) == 1 and diff_from_1s == 0:
            print('predicted all 1s')
            return
    assert False, 'test took too long' 
