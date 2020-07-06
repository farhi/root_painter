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
""" Show image in either sagittal or axial view """
import os
import sys

import numpy as np
import qimage2ndarray
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

from graphics_scene import GraphicsScene
from graphics_view import CustomGraphicsView
from axial_nav import AxialNav

import im_utils

class ImViewer(QtWidgets.QWidget):
    """ Show image, annotation and segmentation data and allow interaction
        Contains graphics scene which is where the image will be drawn
    """
   
    def __init__(self, parent):
        super().__init__()
        self.image_data = None
        self.annot_data = None
        self.seg_data = None
        self.annot_visible = True
        self.seg_visible = False
        self.image_visible = True
        self.image_pixmap_holder = None
        self.seg_pixmap_holder = None
        self.annot_pixmap_holder = None

        self.parent = parent # lazy event handling.
        self.initUI()

    def initUI(self):

        self.graphics_view = CustomGraphicsView()
        self.graphics_view.zoom_change.connect(self.update_cursor)

        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        scene = GraphicsScene()
        scene.parent = self
        self.graphics_view.setScene(scene)
        self.graphics_view.mouse_scroll_event.connect(self.mouse_scroll)

        # Required so graphics scene can track mouse up when mouse is not pressed
        self.graphics_view.setMouseTracking(True)
        self.scene = scene
        
        # slice nav
        self.axial_nav = AxialNav()
        self.layout.addWidget(self.axial_nav)
        self.axial_nav.changed.connect(self.update_slice_index)
        self.layout.addWidget(self.graphics_view)

    def update_slice_index(self):


        # As we have likley moved to a new slice. The history needs erasing. 
        # to avoid undo taking us back to prevous states of another slice.
        if len(self.scene.history) > 0:
            # before updating the slice idx, store current annotation information
            if hasattr(self.scene, 'annot_pixmap'):
                im_utils.store_annot_slice(self.scene.annot_pixmap,
                                           self.annot_data,
                                           self.cur_slice_idx)
            self.scene.history = []

        # update image, seg and annot at current slice.
        self.update_image_slice()
        self.update_seg_slice() 
        self.update_annot_slice()

    def mouse_scroll(self, event):
        scroll_up = event.angleDelta().y() > 0
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        alt_down = (modifiers & QtCore.Qt.AltModifier)
        shift_down = (modifiers & QtCore.Qt.ShiftModifier)
        ctrl_down = (modifiers & QtCore.Qt.ControlModifier)

        if alt_down or shift_down:
            # change by 10% (nearest int) or 1 (min)
            increment = max(1, int(round(self.scene.brush_size / 10)))
            if scroll_up:
                self.scene.brush_size += increment
            else:
                self.scene.brush_size -= increment
            self.scene.brush_size = max(1, self.scene.brush_size)
            self.update_cursor()
        elif ctrl_down:
            if scroll_up:
                self.axial_nav.axial_slider.setValue(self.axial_nav.slice_idx + 1)
            else:
                self.axial_nav.axial_slider.setValue(self.axial_nav.slice_idx - 1)
        else:
            if scroll_up:
                self.graphics_view.zoom *= 1.1
            else:
                self.graphics_view.zoom /= 1.1
            self.graphics_view.update_zoom()

    def show_hide_image(self):
        # show or hide the current image.
        # Could be useful to help inspect the segmentation or annotation
        if self.image_visible:
            self.image_pixmap_holder.setPixmap(self.black_pixmap)
            self.image_visible = False
        else:
            self.image_pixmap_holder.setPixmap(self.graphics_view.image)
            self.image_visible = True


    def show_hide_annot(self):
        # show or hide the current annotations.
        # Could be useful to help inspect the background image
        if self.annot_visible:
            self.annot_pixmap_holder.setPixmap(self.blank_pixmap)
            self.annot_visible = False
        else:
            self.scene.annot_pixmap_holder.setPixmap(self.scene.annot_pixmap)
            self.annot_visible = True

    def show_hide_seg(self):
        # show or hide the current segmentation.
        if self.seg_visible:
            self.seg_pixmap_holder.setPixmap(self.blank_pixmap)
            self.seg_visible = False
        else:
            self.seg_pixmap_holder.setPixmap(self.seg_pixmap)
            self.seg_visible = True

    def update_image(self, image_path, annot_path, seg_path):
        """ update image file data, include image, annot and seg """
        assert os.path.isfile(image_path), f"Cannot find file {self.image_path}"

        # clear history when moving to a new image
        self.scene.history = []
        self.scene.redo_list = []

        self.img_data = im_utils.load_image(image_path)
        fname = os.path.basename(image_path) 

        if self.parent.annot_path and os.path.isfile(annot_path):
            self.annot_data = np.load(annot_path)
        else:
            # otherwise create empty annotation array
            # if we are working with 3D data (npy file) and the
            # file hasn't been found then create an empty array to be
            # used for storing the annotation information.
            # channel for bg (0) and fg (1)
            self.annot_data = np.zeros([2] + list(self.img_data.shape))

        if os.path.isfile(seg_path):
            self.seg_data = im_utils.load_image(seg_path)
        
        self.parent.contrast_slider.update_range(self.img_data)
        self.axial_nav.update_range(self.img_data)

        # used for saving the edited annotation information
        # before changing slice
        self.cur_slice_idx = self.axial_nav.slice_idx
        
        # render image, seg and annot at current slice.
        self.update_image_slice()
        self.update_seg_slice() 
        self.update_annot_slice()

    def update_image_slice(self):
        """ show the already loaded image we are looking at, at the specific
            slice the user has specified and using the specific
            user specified contrast settings.
        """
        img = np.array(self.img_data[self.axial_nav.slice_idx, :, :])
        img = im_utils.norm_slice(img,
                                  self.parent.contrast_slider.min_value,
                                  self.parent.contrast_slider.max_value,
                                  self.parent.contrast_slider.brightness_value)
        q_image = qimage2ndarray.array2qimage(img)
        image_pixmap = QtGui.QPixmap.fromImage(q_image)
        im_size = image_pixmap.size()
        im_width, im_height = im_size.width(), im_size.height()
        assert im_width > 0
        assert im_height > 0
        self.graphics_view.image = image_pixmap # for resize later
        self.im_width = im_width
        self.im_height = im_height
        self.scene.setSceneRect(-15, -15, im_width+30, im_height+30)

        # Used to replace the segmentation or annotation when they are not visible.
        self.blank_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
        self.blank_pixmap.fill(Qt.transparent)

        self.black_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
        self.black_pixmap.fill(Qt.black)

        if self.image_pixmap_holder:
            self.image_pixmap_holder.setPixmap(image_pixmap)
        else:
            self.image_pixmap_holder = self.scene.addPixmap(image_pixmap)
        if not self.image_visible:
            self.image_pixmap_holder.setPixmap(self.black_pixmap)


    def update_seg_slice(self):
        # if seg file is present then load.
        if os.path.isfile(self.parent.seg_path):
            self.seg_mtime = os.path.getmtime(self.parent.seg_path)
            self.update_seg_slice_pixmap()
            self.parent.nav.next_image_button.setText('Save && Next >')
            if hasattr(self, 'vis_widget'):
                self.vis_widget.seg_checkbox.setText('Segmentation (S)')
            self.parent.nav.next_image_button.setEnabled(True)
        else:
            self.seg_mtime = None
            # otherwise use blank
            self.seg_pixmap = QtGui.QPixmap(self.im_width, self.im_height)
            self.seg_pixmap.fill(Qt.transparent)
            painter = QtGui.QPainter()
            painter.begin(self.seg_pixmap)
            font = QtGui.QFont()
            font.setPointSize(48)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255), Qt.SolidPattern))
            if sys.platform == 'win32':
                # For some reason the text has a different size
                # and position on windows
                # so change the background rectangle also.
                painter.drawRect(0, 0, 657, 75)
            else:
                painter.drawRect(10, 10, 465, 55)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 150)))
            painter.drawText(16, 51, 'Loading segmentation')
            painter.end()
            self.nav.next_image_button.setText('Loading Segmentation...')
            if hasattr(self, 'vis_widget'):
                self.vis_widget.seg_checkbox.setText('Segmentation (Loading)')
            self.nav.next_image_button.setEnabled(False)

        if self.seg_pixmap_holder:
            self.seg_pixmap_holder.setPixmap(self.seg_pixmap)
        else:
            self.seg_pixmap_holder = self.scene.addPixmap(self.seg_pixmap)
        if not self.seg_visible:
            self.seg_pixmap_holder.setPixmap(self.blank_pixmap)


    def update_seg_slice_pixmap(self):
        if hasattr(self, 'seg_data'):
            seg_slice = self.seg_data[self.axial_nav.slice_idx, :, :]
            self.seg_pixmap = im_utils.seg_slice_to_pixmap(seg_slice)

        
    def update_cursor(self):
        brush_w = self.scene.brush_size * self.graphics_view.zoom * 0.93
        brush_w = max(brush_w, 3)
        canvas_w = max(brush_w, 30)
        pm = QtGui.QPixmap(canvas_w, canvas_w)
        pm.fill(Qt.transparent)
        painter = QtGui.QPainter(pm)
        painter.drawPixmap(canvas_w, canvas_w, pm)
        brush_rgb = self.scene.brush_color.toRgb()
        r, g, b = brush_rgb.red(), brush_rgb.green(), brush_rgb.blue()
        cursor_color = QtGui.QColor(r, g, b, 120)

        painter.setPen(QtGui.QPen(cursor_color, 3, Qt.SolidLine,
                                  Qt.RoundCap, Qt.RoundJoin))
        ellipse_x = int(round(canvas_w/2 - (brush_w)/2))
        ellipse_y = int(round(canvas_w/2 - (brush_w)/2))
        ellipse_w = brush_w
        ellipse_h = brush_w

        painter.drawEllipse(ellipse_x, ellipse_y, ellipse_w, ellipse_h)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 2,
                                  Qt.SolidLine, Qt.FlatCap))

        # Draw black to show where cursor is even when brush is small
        painter.drawLine(0, (canvas_w/2), canvas_w*2, (canvas_w/2))
        painter.drawLine((canvas_w/2), 0, (canvas_w/2), canvas_w*2)
        painter.end()

        cursor = QtGui.QCursor(pm)
        self.setCursor(cursor)


    def mouse_scroll(self, event):
        scroll_up = event.angleDelta().y() > 0
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        alt_down = (modifiers & QtCore.Qt.AltModifier)
        shift_down = (modifiers & QtCore.Qt.ShiftModifier)
        ctrl_down = (modifiers & QtCore.Qt.ControlModifier)

        if alt_down or shift_down:
            # change by 10% (nearest int) or 1 (min)
            increment = max(1, int(round(self.scene.brush_size / 10)))
            if scroll_up:
                self.scene.brush_size += increment
            else:
                self.scene.brush_size -= increment
            self.scene.brush_size = max(1, self.scene.brush_size)
            self.update_cursor()
        elif ctrl_down:
            if scroll_up:
                self.axial_nav.axial_slider.setValue(self.axial_nav.slice_idx + 1)
            else:
                self.axial_nav.axial_slider.setValue(self.axial_nav.slice_idx - 1)
        else:
            if scroll_up:
                self.graphics_view.zoom *= 1.1
            else:
                self.graphics_view.zoom /= 1.1
            self.graphics_view.update_zoom()


    def update_annot_slice(self):
        """ Update the annotation the user views """
        # if it's 3d then get the slice from the loaded numpy array
        fname = os.path.basename(self.parent.image_path)
        self.cur_slice_idx = self.axial_nav.slice_idx
        annot_slice = self.annot_data[:, self.axial_nav.slice_idx, :, :]
        self.scene.annot_pixmap = im_utils.annot_slice_to_pixmap(annot_slice)
        # if annot_pixmap_holder is ready then setPixmap
        if self.annot_pixmap_holder:
            self.annot_pixmap_holder.setPixmap(self.scene.annot_pixmap)
        else:
            # or create the holder with the loaded pixmap
            self.annot_pixmap_holder = self.scene.addPixmap(self.scene.annot_pixmap)
        self.scene.annot_pixmap_holder = self.annot_pixmap_holder

        # TODO this seems non-intuitive
        self.scene.history.append(self.scene.annot_pixmap.copy())
        if not self.annot_visible:
            self.annot_pixmap_holder.setPixmap(self.blank_pixmap)
