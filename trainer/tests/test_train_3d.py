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
    batch_size = 1
    target = torch.zeros((batch_size, 18, 194, 194)).cuda().long()
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
    batch_size = 1
    test_input = torch.rand((batch_size, 1, 56, 240, 240)).cuda()
    target = torch.ones((batch_size, 18, 194, 194)).cuda().long()
    outputs = cnn(test_input)
    assert outputs.shape[1] == num_classes
    assert outputs.shape[0] == batch_size
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    for _ in range(100):
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


@pytest.mark.slow
def test_train_identity_from_random():
    """
    Very simple test to check that we can train the CNN model to
    map input to output

    It cannot actually do this but we can test that it gets close.

    """
    cnn = UNet3D(im_channels=1, out_channels=1).cuda()
    batch_size = 1
    # small input for this test to make it go faster.
    test_input = torch.rand((batch_size, 1, 56, 56, 56)).cuda()
    target = test_input[:, :, 19:-19, 19:-19, 19:-19]
    outputs = cnn(test_input)
    assert outputs.shape[0] == batch_size
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.99, nesterov=True)
    criterion = torch.nn.MSELoss() # see [1]

    for _ in range(100):
        optimizer.zero_grad()
        outputs = cnn(test_input)
        # Also consider MarginRankingLoss for similar problems, see [0]
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        diff_mean = torch.mean(torch.abs(outputs - target))
        if diff_mean < 0.1:
            return
    assert False, 'test took too long'

# pylint: disable=W0105 # string has no effect
# pylint: disable=C0301 # line too long
"""
[0] https://discuss.pytorch.org/t/how-to-calculate-pair-wise-differences-between-two-tensors-in-a-vectorized-way/37451/3
[1] https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
"""
