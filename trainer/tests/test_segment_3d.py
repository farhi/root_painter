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
import os
import shutil
import json
import numpy as np
import torch
import time
import pytest
import nibabel as nib
from skimage.io import imread, imsave
from unet import UNetGNRes
from unet3d import UNet3D
from model_utils import segment, segment_3d, create_first_model_with_random_weights
import im_utils
from trainer import Trainer


def test_CNN_segment_classes_3D_3_classes():
    """
    test CNN returns data in the correct shape for a single cube.
    Using random weights this time so the output is not checked.

    The number of output channels should correspond to the classes
    """
    start = time.time()
    num_classes = 3
    cnn = UNet3D(im_channels=1, out_channels=num_classes).cuda()
    test_input = torch.zeros((1, 1, 64, 312, 312)).cuda()
    output = cnn(test_input)
    print('3D unet 3 classes output time = ', time.time() - start, 'output shape', output.shape)
    assert output.shape[1] == num_classes


def test_CNN_segment_classes_3D_9_classes():
    """
    test CNN returns data in the correct shape for a single cube.
    Using random weights this time so the output is not checked.

    The number of output channels should correspond to the classes
    """
    num_classes = 9
    cnn = UNet3D(im_channels=1, out_channels=num_classes).cuda()
    test_input = torch.zeros((1, 1, 64, 312, 312)).cuda()
    start = time.time()
    output = cnn(test_input)
    print('3D unet 9 classes output time = ', time.time() - start, 'output shape', output.shape)
    assert output.shape[1] == num_classes


@pytest.mark.slow
def test_segment_large_3D_image(monkeypatch):
    """ Test that the segmentation method reconstructs the output properly """

    def cnn(tile):
        assert len(tile.shape) == 5
        assert tile.shape[0] == 1
        assert tile.shape[1] == 1
        assert tile.shape[2] == 64, str(tile.shape)
        assert tile.shape[3] == 312
        assert tile.shape[4] == 312

        # identify function as we are only testing the reconstruction methods
        # test returning a smaller part of the image as would happen in reality.
        # take the central 18 slices.
        d_crop = (64 - 18) // 2
        w_crop = (312 - 274) // 2

        central_part = tile[:, :, d_crop:-d_crop, w_crop:-w_crop, w_crop:-w_crop]
        return central_part.unsqueeze(1) # add a (single) classes dimension

    def mock_normalize_tile(tile):
        # skip normalization step.
        return tile
    
    monkeypatch.setattr(im_utils, "normalize_tile", mock_normalize_tile)
    channels = 1 
    depth = 96
    test_input = np.random.rand(channels, depth, 512, 512) # depth, height, width
    test_input = np.expand_dims(test_input, axis=0) # batch dimension
    in_w = 312
    out_w = 274
    in_d = 64
    out_d = 18
    input_tile_shape = (in_d, in_w, in_w)
    output_tile_shape = (out_d, out_w, out_w)
    bs = 1
    seg = segment_3d(cnn, test_input, bs, in_w, out_w, in_d, out_d)
    # not exactly equal because of going via 16bit etc.
    assert np.allclose(seg, test_input, atol=0.01)



def test_3D_segment_instruction():
    """
    A high level integration test for the 
    full 3D image segmentation prcocess.
    We don't check model accuracy here. 
    """
    # create a 'scrap' folder for the test in /tmp 
    # this scrap folder will be used as the sync_directory for this test.
    sync_dir = os.path.join('/tmp', 'test_sync_dir')
    if os.path.isdir(sync_dir):
        shutil.rmtree(sync_dir) 
    os.makedirs(sync_dir)
    
    # create an instructions folder, models folder,  dataset folder
    # and a segmentation folder inside the sync_directory
    instruction_dir = os.path.join(sync_dir, 'instructions')
    dataset_dir = os.path.join(sync_dir, 'dataset')
    seg_dir = os.path.join(sync_dir, 'seg')
    model_dir = os.path.join(sync_dir, 'models')
    for d in [instruction_dir, dataset_dir, model_dir, seg_dir]:
        os.makedirs(d)
    
    # create a model file (random weights is fine)
    # and save the model to the models folder.
    create_first_model_with_random_weights(model_dir, num_classes=3, dimensions=3)

    # create an example input image using numpy and save to the datsets folder
    example_image = np.random.rand(96, 512, 512)
    img = nib.Nifti1Image(example_image, np.eye(4))
    img_path = os.path.join(dataset_dir, 'example_image.nii.gz')
    img.to_filename(img_path)

    # create an instruction as json.
    content = {
        "dataset_dir": dataset_dir,
        "seg_dir": seg_dir,
        "file_names": ['example_image.nii.gz'],
        "model_dir": model_dir,
        "classes": [('bg', [0,180,0,0], 'w'),
                    ('red', [255,0,0,255], '1'),
                    ('blue', [0, 0, 255, 255], '2')]
    }
    # save the instruction (json file) to the instructions folder. 
    hash_str = '_' + str(hash(json.dumps(content)))
    fpath = os.path.join(instruction_dir, 'segment' + hash_str)
    with open(fpath, 'w') as json_file:
        json.dump(content, json_file, indent=4)
 
    # create a trainer object and set the sync directory
    # to be the scrap folder which has been created.
    trainer = Trainer(sync_dir)

    # tell the trainer to check for instructions
    trainer.check_for_instructions()
    print('checking')

    # assert that the segmentation file has been created
    seg_path = os.path.join(seg_dir, 'example_image.nii.gz')
    assert os.path.isfile(seg_path)

    # assert that it it is non-zero (random weights don't do this) 
    image = nib.load(seg_path)
    seg = np.array(image.dataobj)

    assert np.sum(seg) > 0

    # assert that it has the same shape as the input image
    assert seg.shape[0] == 3 # for each class
    assert seg[0].shape == example_image.shape
    # assert that the segment instruction has been deleted.
    assert not os.path.isfile(fpath) 

    # clean up the the scrap folder that was created in /tmp
    shutil.rmtree(sync_dir) 
