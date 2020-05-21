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
    annot = np.zeros((100, 100, 3))
    annot[10, 10] = [255, 20, 20]
    annot[10, 15] = [255, 20, 20]
    annot[20, 20] = [255, 24, 20]
    
    class_rgb = [255, 20, 20]
    class_map = im_utils.get_class_map(annot, class_rgb)
    assert class_map.shape == (100, 100)
    assert np.sum(class_map) == 2
    assert class_map[10, 10] == 1
    assert class_map[10, 15] == 1
