"""
Copyright (C) 2020 Abraham George Smith

BoundingBox provides a way to specify a subregion of an image.

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
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt


class Handle(QtWidgets.QGraphicsEllipseItem):

    def __init__(self, x, y):
        self.circle_diam = 3
        super().__init__(x, y, self.circle_diam, self.circle_diam)
        self.setPen(QtGui.QPen(QtGui.QColor(250, 250, 250), 0.5, QtCore.Qt.DashLine))
        self.setBrush(QtGui.QBrush(QtGui.QColor(40, 120, 250), style = QtCore.Qt.SolidPattern))
        # self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)

    def mouseMoveEvent(self, event): 
        super().mouseMoveEvent(event)
        print('circle dragged')

    def setPos(self, x, y):
        # use x and y as center of position
        super().setPos(x-(self.circle_diam/2), y-(self.circle_diam/2))


class BoundingBox(QtWidgets.QGraphicsRectItem):

    def __init__(self, x, y):
        start_rect = QtCore.QRectF(x, y, 20, 20)
        super().__init__(start_rect)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)
        self.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 0.2, QtCore.Qt.DashLine))
        self.setBrush(QtGui.QBrush(QtGui.QColor(40, 120, 200, 70), style = QtCore.Qt.SolidPattern))

        self.tl_circle = Handle(x, y)
        self.tl_circle.setParentItem(self)
        
        self.bl_circle = Handle(x, y)
        self.bl_circle.setParentItem(self)
        
        self.tr_circle = Handle(x, y)
        self.tr_circle.setParentItem(self)
       
        self.br_circle = Handle(x, y)
        self.br_circle.setParentItem(self)
        
        
    def resize_drag(self, x, y):
        width = abs(x - self.x_start)
        height = abs(y - self.y_start)
        x = min(x, self.x_start)
        y = min(y, self.y_start)
        self.setRect(x, y, width, height)
        self.tl_circle.setPos(x-self.x_start, y-self.y_start)
        self.bl_circle.setPos(x-self.x_start, (y-self.y_start) + height)

        self.tr_circle.setPos((x-self.x_start) + width, y-self.y_start)
        self.br_circle.setPos((x-self.x_start) + width, (y-self.y_start) + height)

    def set_start(self, x, y):
        self.x_start = x
        self.y_start = y

    def oldMouseMoveEvent(self, event): 
        print('circle selected', self.circle.isSelected())
        if False:
            new_pos = event.scenePos()
            old_pos = event.lastScenePos()
            old_left = self.scenePos().x()
            old_top = self.scenePos().y()
            new_left = new_pos.x() - old_pos.x() + old_left
            new_top = new_pos.y() - old_pos.y() + old_top
            self.setPos(QtCore.QPointF(new_left, new_top))


    def hoverEnterEvent(self, event):
        QtWidgets.QApplication.instance().setOverrideCursor(Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        print('hover leave')

    def mousePressEvent(self, event):
        print('press event')




