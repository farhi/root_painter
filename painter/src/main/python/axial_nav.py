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

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui

class AxialNav(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, min_slice=0, max_slice=100):
        super().__init__()
        self.min_slice_idx = min_slice
        self.max_slice_idx = max_slice
        self.slice_idx = self.min_slice_idx
        self.initUI()

    def update_range(self, new_image):
        slice_count = new_image.shape[-1]
        self.max_slice_idx = slice_count - 1
        self.axial_slider.setMaximum(self.max_slice_idx)
        if self.slice_idx > self.max_slice_idx:
            self.slice_idx = self.max_slice_idx
            self.axial_slider.setValue(self.slice_idx)

        self.axial_slider.setValue(self.slice_idx)
        self.update_text()

    def update_text(self): 
        self.axial_value_label.setText(f"{self.slice_idx+1}/{self.max_slice_idx+1}")

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel("Axial Slice")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.debounce = QtCore.QTimer()
        self.debounce.setInterval(5)
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self.debounced)

        self.axial_slider_container = QtWidgets.QWidget()
        self.axial_slider_layout = QtWidgets.QVBoxLayout()
        # self.axial_slider_layout.addWidget(QtWidgets.QLabel("Axial"))
        self.axial_slider_container.setLayout(self.axial_slider_layout)
        self.axial_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.axial_slider.setMinimum(self.min_slice_idx)
        self.axial_slider.setMaximum(self.max_slice_idx)
        self.axial_slider.setValue(self.slice_idx)
        self.axial_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.axial_slider.setTickInterval(1)
        self.axial_slider.setFixedHeight(600)
        self.axial_slider.valueChanged.connect(self.value_changed)
        self.axial_slider_layout.addWidget(self.axial_slider)
        self.axial_value_label = QtWidgets.QLabel()
        self.update_text()
        self.axial_value_label.setText(f"{self.slice_idx+1}/{self.max_slice_idx+1}")

        self.axial_slider_layout.addWidget(self.axial_value_label)
        self.layout.addWidget(self.axial_slider_container)
        self.setLayout(self.layout)
        self.setWindowTitle("Axial Position")


    def value_changed(self):
        self.slice_idx = self.axial_slider.value()
        self.update_text()
        self.debounce.start()

    def debounced(self):
        self.changed.emit()
