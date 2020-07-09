"""
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

# pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914

# too many statements
# pylint: disable=R0915

# catching too general exception
# pylint: disable=W0703

# too many public methods
# pylint: disable=R0904
# pylint: disable=E0401 # import error
# pylint: disable=C0103 # Method name "initUI" doesn't conform to snake_case naming style (invalid-name)

import sys
import os
from pathlib import PurePath
import json
from functools import partial
import copy

import numpy as np
from skimage.io import use_plugin
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

from about import AboutWindow, LicenseWindow
from create_project import CreateProjectWidget
from create_dataset import CreateDatasetWidget
from segment_folder import SegmentFolderWidget
from extract_count import ExtractCountWidget
from extract_regions import ExtractRegionsWidget
from extract_length import ExtractLengthWidget
from extract_comp import ExtractCompWidget
from im_viewer import ImViewer, ImViewerWindow
from nav import NavWidget
from file_utils import last_fname_with_annotations
from file_utils import get_annot_path
from file_utils import maybe_save_annotation_3d
from instructions import send_instruction
from contrast_slider import ContrastSlider
import im_utils
import menus

use_plugin("pil")

class RootPainter(QtWidgets.QMainWindow):

    closed = QtCore.pyqtSignal()

    def __init__(self, sync_dir, contrast_presets):
        super().__init__()
        self.sync_dir = sync_dir
        self.instruction_dir = sync_dir / 'instructions'
        self.send_instruction = partial(send_instruction,
                                        instruction_dir=self.instruction_dir,
                                        sync_dir=sync_dir)
        self.contrast_presets = contrast_presets
        self.tracking = False
        self.seg_mtime = None
        self.pre_segment_count = 0
        self.im_width = None
        self.im_height = None

        self.annot_data = None
        self.seg_data = None
        self.initUI()

    def initUI(self):
        if len(sys.argv) < 2:
            self.init_missing_project_ui()
            return

        fname = sys.argv[1]
        if os.path.splitext(fname)[1] == '.seg_proj':
            proj_file_path = os.path.abspath(sys.argv[1])
            self.open_project(proj_file_path)
        else:
            # only warn if -psn not in the args. -psn is in the args when
            # user opened app in a normal way by clicking on the Application icon.
            if not '-psn' in sys.argv[1]:
                QtWidgets.QMessageBox.about(self, 'Error', sys.argv[1] +
                                            ' is not a valid '
                                            'segmentation project (.seg_proj) file')
            self.init_missing_project_ui()

    def open_project(self, proj_file_path):
        # extract json
        with open(proj_file_path, 'r') as json_file:
            settings = json.load(json_file)
            self.dataset_dir = self.sync_dir / 'datasets' / PurePath(settings['dataset'])

            self.proj_location = self.sync_dir / PurePath(settings['location'])
            self.image_fnames = settings['file_names']
            self.seg_dir = self.proj_location / 'segmentations'
            self.log_dir = self.proj_location / 'logs'
            self.train_annot_dir = self.proj_location / 'annotations' / 'train'
            self.val_annot_dir = self.proj_location / 'annotations' / 'val'

            self.model_dir = self.proj_location / 'models'

            self.message_dir = self.proj_location / 'messages'


            self.classes = settings['classes']

            # If there are any annotations which have already been saved
            # then go through the annotations in the order specified
            # by self.image_fnames
            # and set fname (current image) to be the last image with annotation
            last_with_annot = last_fname_with_annotations(self.image_fnames,
                                                          self.train_annot_dir,
                                                          self.val_annot_dir)
            if last_with_annot:
                fname = last_with_annot
            else:
                fname = self.image_fnames[0]

            # set first image from project to be current image
            self.image_path = os.path.join(self.dataset_dir, fname)
            self.update_window_title()
            self.seg_path = os.path.join(self.seg_dir, fname)
            self.annot_path = get_annot_path(fname, self.train_annot_dir,
                                             self.val_annot_dir)
            self.init_active_project_ui()
            self.track_changes()


    def update_file(self, fpath):
        """ Invoked when the file to view has been changed by the user.
            Show image file and it's associated annotation and segmentation """
        # save annotation for current file before changing to new file.
        self.save_annotation()
        fname = os.path.basename(fpath)
        self.image_path = os.path.join(self.dataset_dir, fname)
        seg_fname = os.path.splitext(fname)[0] + '.nii.gz'
        self.seg_path = os.path.join(self.seg_dir, seg_fname)
        self.annot_path = get_annot_path(fname,
                                         self.train_annot_dir,
                                         self.val_annot_dir)

        self.img_data = im_utils.load_image(self.image_path)
        fname = os.path.basename(self.image_path)

        if self.annot_path and os.path.isfile(self.annot_path):
            self.annot_data = np.array(np.load(self.annot_path))
        else:
            # otherwise create empty annotation array
            # if we are working with 3D data (npy file) and the
            # file hasn't been found then create an empty array to be
            # used for storing the annotation information.
            # channel for bg (0) and fg (1)
            self.annot_data = np.zeros([2] + list(self.img_data.shape))


        if os.path.isfile(self.seg_path):
            self.seg_data = im_utils.load_image(self.seg_path)
        else:
            # it should come later
            self.seg_data = None

        self.axial_viewer.update_image()
        self.sag_viewer.update_image()

        self.update_segmentation()

        self.contrast_slider.update_range(self.img_data)

        self.segment_current_image()
        self.update_window_title()

    def update_segmentation(self):
        if os.path.isfile(self.seg_path):
            self.seg_mtime = os.path.getmtime(self.seg_path)
            self.nav.next_image_button.setText('Save && Next >')
            if hasattr(self, 'vis_widget'):
                self.vis_widget.seg_checkbox.setText('Segmentation (S)')
            self.nav.next_image_button.setEnabled(True)
        else:
            self.nav.next_image_button.setEnabled(False)
            self.nav.next_image_button.setText('Loading Segmentation...')
            if hasattr(self, 'vis_widget'):
                self.vis_widget.seg_checkbox.setText('Segmentation (Loading)')


    def segment_image(self, image_fnames):
        # send instruction to segment the new image.
        seg_classes = copy.deepcopy(self.classes)
        # Tell server to segment the bg with 0 alpha, appplied to 2D only
        assert seg_classes[0][0] == 'Background'
        seg_classes[0][1][3] = 0
        # we assume .npy means 3D
        # We also assume that the files are either all 2D
        # or all 3D
        dimensions = 2
        if image_fnames[0].endswith('.npy'):
            dimensions = 3
            seg_classes = ['Foreground']
        # send instruction to segment the new image.
        content = {
            "dataset_dir": self.dataset_dir,
            "seg_dir": self.seg_dir,
            "file_names": image_fnames,
            "message_dir": self.message_dir,
            "model_dir": self.model_dir,
            "classes": seg_classes,
            "dimensions": dimensions
        }
        self.send_instruction('segment', content)

    def segment_current_image(self):
        dir_path, _ = os.path.split(self.image_path)
        path_list = self.nav.get_path_list(dir_path)
        cur_index = path_list.index(self.image_path)
        to_segment_paths = path_list[cur_index:1+cur_index+self.pre_segment_count]
        to_segment_paths = [f for f in to_segment_paths if
                            os.path.isfile(os.path.join(self.seg_dir, f))]
        to_segment_fnames = [os.path.basename(p) for p in to_segment_paths]
        self.segment_image(to_segment_fnames)

    def show_open_project_widget(self):
        options = QtWidgets.QFileDialog.Options()
        default_loc = self.sync_dir / 'projects'
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load project file",
            str(default_loc),
            "Segmentation project file (*.seg_proj)",
            options=options)

        if file_path:
            self.open_project(file_path)

    def show_create_project_widget(self):
        print("Open the create project widget..")
        self.create_project_widget = CreateProjectWidget(self.sync_dir)
        self.create_project_widget.show()
        self.create_project_widget.created.connect(self.open_project)

    def init_missing_project_ui(self):
        ## Create project menu
        # project has not yet been selected or created
        # need to open minimal interface which allows users
        # to open or create a project.

        menu_bar = self.menuBar()
        self.menu_bar = menu_bar
        self.menu_bar.clear()
        self.project_menu = menu_bar.addMenu("Project")

        # Open project
        self.open_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Open project", self)
        self.open_project_action.setShortcut("Ctrl+O")

        self.project_menu.addAction(self.open_project_action)
        self.open_project_action.triggered.connect(self.show_open_project_widget)

        # Create project
        self.create_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Create project", self)
        self.create_project_action.setShortcut("Ctrl+C")
        self.project_menu.addAction(self.create_project_action)
        self.create_project_action.triggered.connect(self.show_create_project_widget)

        # Network Menu
        self.network_menu = menu_bar.addMenu('Network')
        # # segment folder
        self.segment_folder_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                    'Segment folder', self)

        def show_segment_folder():
            self.segment_folder_widget = SegmentFolderWidget(self.sync_dir,
                                                             self.instruction_dir,
                                                             self.classes)
            self.segment_folder_widget.show()
        self.segment_folder_btn.triggered.connect(show_segment_folder)
        self.network_menu.addAction(self.segment_folder_btn)

        self.add_measurements_menu(menu_bar)
        self.add_extras_menu(menu_bar)

        self.add_about_menu(menu_bar)

        # Add project btns to open window (so it shows something useful)
        project_btn_widget = QtWidgets.QWidget()
        self.setCentralWidget(project_btn_widget)

        layout = QtWidgets.QHBoxLayout()
        project_btn_widget.setLayout(layout)
        open_project_btn = QtWidgets.QPushButton('Open existing project')
        open_project_btn.clicked.connect(self.show_open_project_widget)
        layout.addWidget(open_project_btn)

        create_project_btn = QtWidgets.QPushButton('Create new project')
        create_project_btn.clicked.connect(self.show_create_project_widget)
        layout.addWidget(create_project_btn)

        create_dataset_btn = QtWidgets.QPushButton('Create training dataset')
        def show_create_dataset():
            self.create_dataset_widget = CreateDatasetWidget(self.sync_dir)
            self.create_dataset_widget.show()
        create_dataset_btn.clicked.connect(show_create_dataset)
        layout.addWidget(create_dataset_btn)

        self.setWindowTitle("RootPainter")
        self.resize(layout.sizeHint())

    def add_extras_menu(self, menu_bar):
        extras_menu = menu_bar.addMenu('Extras')
        comp_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Extract composites', self)
        comp_btn.triggered.connect(self.show_extract_comp)
        extras_menu.addAction(comp_btn)

    def add_about_menu(self, menu_bar):
        about_menu = menu_bar.addMenu('About')
        license_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'License', self)
        license_btn.triggered.connect(self.show_license_window)
        about_menu.addAction(license_btn)

        about_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'RootPainter', self)
        about_btn.triggered.connect(self.show_about_window)
        about_menu.addAction(about_btn)

    def show_license_window(self):
        self.license_window = LicenseWindow()
        self.license_window.show()

    def show_about_window(self):
        self.about_window = AboutWindow()
        self.about_window.show()

    def update_window_title(self):
        proj_dirname = os.path.basename(self.proj_location)
        self.setWindowTitle(f"RootPainter {proj_dirname}"
                            f" {os.path.basename(self.image_path)}")

    def closeEvent(self, event):
        if hasattr(self, 'contrast_slider'):
            self.contrast_slider.close()
        if hasattr(self, 'sag_viewer'):
            self.sag_viewer.close()

    def init_active_project_ui(self):
        # container for both nav and im_viewer.
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout = container_layout
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        self.viewers_container = QtWidgets.QWidget()
        self.viewers_layout = QtWidgets.QHBoxLayout()
        self.viewers_container.setLayout(self.viewers_layout)

        self.axial_viewer = ImViewer(self, 'axial')
        self.sag_viewer = ImViewerWindow(self, 'sagittal')
        self.viewers_layout.addWidget(self.axial_viewer)

        container_layout.addWidget(self.viewers_container)
        self.contrast_slider = ContrastSlider(self.contrast_presets)
        self.contrast_slider.changed.connect(self.axial_viewer.update_image_slice)
        self.contrast_slider.changed.connect(self.sag_viewer.update_image_slice)

        self.nav = NavWidget(self.image_fnames)
        self.update_file(self.image_path)

        # bottom bar right
        bottom_bar_r = QtWidgets.QWidget()
        bottom_bar_r_layout = QtWidgets.QVBoxLayout()
        bottom_bar_r.setLayout(bottom_bar_r_layout)
        self.axial_viewer.bottom_bar_layout.addWidget(bottom_bar_r)
        # Nav
        self.nav.file_change.connect(self.update_file)
        self.nav.image_path = self.image_path
        self.nav.update_nav_label()
        # info label
        info_container = QtWidgets.QWidget()
        info_container_layout = QtWidgets.QHBoxLayout()
        info_container_layout.setAlignment(Qt.AlignCenter)
        info_label = QtWidgets.QLabel()
        info_label.setText("")
        info_container_layout.addWidget(info_label)
        # left, top, right, bottom
        info_container_layout.setContentsMargins(0, 0, 0, 0)
        info_container.setLayout(info_container_layout)
        self.info_label = info_label
        # add nav and info label to the axial viewer.
        bottom_bar_r_layout.addWidget(info_container)
        bottom_bar_r_layout.addWidget(self.nav)


        self.add_menu()

        self.resize(container_layout.sizeHint())

        self.axial_viewer.update_cursor()

        def view_fix():
            """ hack for linux bug """
            self.axial_viewer.update_cursor()
            self.axial_viewer.graphics_view.fit_to_view()
        QtCore.QTimer.singleShot(100, view_fix)

    def track_changes(self):
        if self.tracking:
            return
        print('Starting watch for changes')
        self.tracking = True
        def check():
            # check for any messages
            messages = os.listdir(str(self.message_dir))
            for m in messages:
                if hasattr(self, 'info_label'):
                    self.info_label.setText(m)
                try:
                    # Added try catch because this error happened (very rarely)
                    # PermissionError: [WinError 32]
                    # The process cannot access the file because it is
                    # being used by another process
                    os.remove(os.path.join(self.message_dir, m))
                except Exception as e:
                    print('Caught exception when trying to detele msg', e)
            # if a segmentation exists (on disk)
            if hasattr(self, 'seg_path') and os.path.isfile(self.seg_path):
                try:
                    # seg mtime is not actually used any more.
                    new_mtime = os.path.getmtime(self.seg_path)
                    # seg_mtime is None before the seg is loaded.
                    if not self.seg_mtime:
                        print('update seg data wityh', self.seg_path)
                        self.axial_viewer.seg_data = im_utils.load_image(self.seg_path)
                        self.axial_viewer.update_seg_slice()
                        self.seg_mtime = new_mtime
                        self.nav.next_image_button.setText('Save && Next >')
                        self.nav.next_image_button.setEnabled(True)
                        if self.axial_viewer.seg_visible:
                            self.axial_viewer.seg_pixmap_holder.setPixmap(self.seg_pixmap)
                            self.sag_viewer.seg_pixmap_holder.setPixmap(self.seg_pixmap)
                        if hasattr(self, 'vis_widget'):
                            self.vis_widget.seg_checkbox.setText('Segmentation (S)')
                except Exception as e:
                    print('Error: when trying to load segmention ' + str(e))
                    # sometimes problems reading file.
                    # don't worry about this exception
            else:
                print('no seg found', end=",")
            QtCore.QTimer.singleShot(500, check)
        QtCore.QTimer.singleShot(500, check)

    def close_project_window(self):
        self.close()
        self.closed.emit()

    def add_menu(self):
        menu_bar = self.menuBar()
        menu_bar.clear()

        self.project_menu = menu_bar.addMenu("Project")
        # Open project
        self.close_project_action = QtWidgets.QAction(QtGui.QIcon(""), "Close project", self)
        self.project_menu.addAction(self.close_project_action)
        self.close_project_action.triggered.connect(self.close_project_window)
        menus.add_edit_menu(self, self.axial_viewer, menu_bar)

        options_menu = menu_bar.addMenu("Options")

        # pre segment count
        pre_segment_count_action = QtWidgets.QAction(QtGui.QIcon(""), "Pre-Segment", self)
        options_menu.addAction(pre_segment_count_action)
        pre_segment_count_action.triggered.connect(self.open_pre_segment_count_dialog)
        self.menu_bar = menu_bar

        # add brushes menu for axial slice navigation
        menus.add_brush_menu(self.classes, self.axial_viewer, self.menu_bar)

        # add view menu for axial slice navigation.
        view_menu = menus.add_view_menu(self, self.axial_viewer, self.menu_bar)

        self.add_contrast_setting_options(view_menu)

        # Network Menu
        network_menu = menu_bar.addMenu('Network')

        # start training
        start_training_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Start training', self)
        start_training_btn.triggered.connect(self.start_training)
        network_menu.addAction(start_training_btn)

        # stop training
        stop_training_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Stop training', self)
        stop_training_btn.triggered.connect(self.stop_training)
        network_menu.addAction(stop_training_btn)

        # # segment folder
        segment_folder_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Segment folder', self)

        def show_segment_folder():
            self.segment_folder_widget = SegmentFolderWidget(self.sync_dir,
                                                             self.instruction_dir,
                                                             self.classes)
            self.segment_folder_widget.show()
        segment_folder_btn.triggered.connect(show_segment_folder)
        network_menu.addAction(segment_folder_btn)

        # segment current image
        # segment_image_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
        #                                       'Segment current image', self)
        # segment_image_btn.triggered.connect(self.segment_current_image)
        # network_menu.addAction(segment_image_btn)
        self.add_measurements_menu(menu_bar)
        self.add_extras_menu(menu_bar)
        menus.add_windows_menu(self)

    def add_contrast_setting_options(self, view_menu):
        preset_count = 0
        for preset in self.contrast_presets:
            def add_preset_option(new_preset, preset_count):
                preset = new_preset
                preset_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                               f'{preset} contrast settings', self)
                preset_btn.setShortcut(QtGui.QKeySequence(f"Alt+{preset_count}"))
                preset_btn.setStatusTip(f'Use {preset} contrast settings')
                def on_select():
                    self.contrast_slider.preset_selected(preset)
                preset_btn.triggered.connect(on_select)
                view_menu.addAction(preset_btn)
            preset_count += 1
            add_preset_option(preset, preset_count)

    def add_measurements_menu(self, menu_bar):
        # Measurements
        measurements_menu = menu_bar.addMenu('Measurements')
        # object count
        object_count_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                             'Extract count', self)
        def show_extract_count():
            self.extract_count_widget = ExtractCountWidget()
            self.extract_count_widget.show()
        object_count_btn.triggered.connect(show_extract_count)
        measurements_menu.addAction(object_count_btn)

        # length
        length_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                       'Extract length', self)
        def show_extract_length():
            self.extract_length_widget = ExtractLengthWidget()
            self.extract_length_widget.show()
        length_btn.triggered.connect(show_extract_length)
        measurements_menu.addAction(length_btn)

        # region props
        region_props_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                             'Extract region properties', self)
        def show_extract_region_props():
            self.extract_regions_widget = ExtractRegionsWidget()
            self.extract_regions_widget.show()
        region_props_btn.triggered.connect(show_extract_region_props)
        measurements_menu.addAction(region_props_btn)

    def show_extract_comp(self):
        self.extract_comp_widget = ExtractCompWidget()
        self.extract_comp_widget.show()

    def stop_training(self):
        self.info_label.setText("Stopping training...")
        content = {"message_dir": self.message_dir}
        self.send_instruction('stop_training', content)

    def start_training(self):
        self.info_label.setText("Starting training...")
        # 3D just uses the name of the first class
        classes = self.classes
        is_3d = self.image_path.endswith('.npy')
        if is_3d:
            classes = ['Foreground']
            dimensions = 3
        else:
            dimensions = 2
        content = {
            "model_dir": self.model_dir,
            "dataset_dir": self.dataset_dir,
            "train_annot_dir": self.train_annot_dir,
            "val_annot_dir": self.val_annot_dir,
            "seg_dir": self.seg_dir,
            "log_dir": self.log_dir,
            "message_dir": self.message_dir,
            "classes": classes,
            "dimensions": dimensions
        }
        self.send_instruction('start_training', content)

    def open_pre_segment_count_dialog(self):
        new_count, ok = QtWidgets.QInputDialog.getInt(self, "",
                                                      "Select Pre-Segment count",
                                                      self.pre_segment_count,
                                                      0, 100, 1)
        if ok:
            self.pre_segment_count = new_count
        # For some reason the events get confused and
        # scroll+pan gets switched on here.
        # Check if control key is up to disble it.
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if not modifiers & QtCore.Qt.ControlModifier:
            self.axial_viewer.graphics_view.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    def save_annotation(self):

        if self.axial_viewer.scene.annot_pixmap:
            im_utils.store_annot_slice(self.axial_viewer.scene.annot_pixmap,
                                       self.annot_data,
                                       self.axial_viewer.cur_slice_idx,
                                       self.axial_viewer.mode)

        if self.sag_viewer.scene.annot_pixmap:
            im_utils.store_annot_slice(self.sag_viewer.scene.annot_pixmap,
                                       self.annot_data,
                                       self.sag_viewer.cur_slice_idx,
                                       self.sag_viewer.mode)

        # check if it has data yet as when loading the first time
        # it doesn't have anything to save
        if self.annot_data is not None:
            fname = os.path.basename(self.image_path)
            fname = os.path.splitext(fname)[0] + '.npy'
            self.annot_path = maybe_save_annotation_3d(self.annot_data,
                                                       self.annot_path,
                                                       fname,
                                                       self.train_annot_dir,
                                                       self.val_annot_dir)
