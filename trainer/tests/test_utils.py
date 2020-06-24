"""
Shared functions between the tests.

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


def create_tmp_sync_dir():
    """ create a temporary sync dir
        delete if it already exists
    """
    sync_dir = os.path.join('/tmp', 'test_sync_dir')
    if os.path.isdir(sync_dir):
        shutil.rmtree(sync_dir)
    os.makedirs(sync_dir)
    # create an instructions folder, models folder, dataset folder
    # and a segmentation folder inside the sync_directory
    dnames = ['instructions', 'dataset', 'seg',
              'models', 'annots', 'messages']
    for dname in dnames:
        os.makedirs(os.path.join(sync_dir, dname))
    return sync_dir
