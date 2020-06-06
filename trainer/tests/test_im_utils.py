"""
Test the multiclass dice function.

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

import im_utils


def test_get_class_map():
    """
    For a given class RGB return binary map
    indicating which pixels in the annotation
    are labeled with that specific RGB.
    """
    annot = np.zeros((100, 100, 3), dtype=np.ubyte)
    annot[10, 10] = [255, 20, 20]
    annot[10, 15] = [255, 20, 20]
    annot[20, 20] = [255, 24, 20]

    class_rgb = [255, 20, 20]
    class_map = im_utils.get_class_map(annot, class_rgb)
    assert class_map.shape == (100, 100)
    assert np.sum(class_map) == 2
    assert class_map[10, 10] == 1
    assert class_map[10, 15] == 1


def test_seg_to_rgba():
    """
    The output of the segment method is a channel for each class
    e.g an array of shape [2, 600, 900]
    for 2 classes predicted in an image of size [600, 900]

    This should be displayed to the user
    as an image with each pixel labelled as the most likely class

    the class should be shown using the user defined RGB value.
    """
    seg = np.zeros((5, 6, 9))
    seg[1, 1, 1] = 1
    seg[4, 5, 5] = 1
    classes_rgb = [
        [0, 255, 0, 0], # background
        [244, 0, 0, 255], # one of the foreground classes
        [255, 110, 0, 255], # one of the foreground classes
        [255, 130, 0, 255], # one of the foreground classes
        [255, 150, 0, 255], # one of the foreground classes
    ]
    rgba_image = im_utils.seg_to_rgba(seg, classes_rgb)
    assert rgba_image.shape == (6, 9, 4)
    assert np.array_equal(rgba_image[0, 0], classes_rgb[0])
    assert np.array_equal(rgba_image[1, 1], classes_rgb[1])
    assert np.array_equal(rgba_image[5, 5], classes_rgb[4])
    # all others are 0 (backgroun) but lets just check 1 to make sure
    assert np.array_equal(rgba_image[2, 2], classes_rgb[0])
