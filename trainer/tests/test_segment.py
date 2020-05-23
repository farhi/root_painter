"""
Test the segment function

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
from unet import UNetGNRes
from model_utils import segment
import torch


def test_CNN_segment_classes():
    """
    test CNN returns data in the correct shape for a single tile.
    Using random weights this time so the output is not checked.

    The number of output channels should correspond to the classes
    """
    num_classes = 2
    cnn = UNetGNRes(im_channels=3, out_channels=num_classes)
    test_input = torch.zeros((1, 3, 572, 572))
    output = cnn(test_input)
    assert output.shape[1] == num_classes

    num_classes = 31
    cnn = UNetGNRes(im_channels=3, out_channels=num_classes)
    test_input = torch.zeros((1, 3, 572, 572))
    output = cnn(test_input)
    assert output.shape[1] == num_classes



