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
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        super().hoverEnterEvent(event)
        # TODO: This cursor is wrong. See preview
        QtWidgets.QApplication.instance().setOverrideCursor(Qt.SizeAllCursor)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.on_move(event)

    def hoverLeaveEvent(self, event):
        QtWidgets.QApplication.restoreOverrideCursor()
        
    def mouseMoveEvent(self, event): 
        super().mouseMoveEvent(event)
        if self.on_move is not None:
            self.on_move(event)

    def setPosR(self, x, y):
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
        self.tl_circle.on_move = self.tl_handle_moved
        self.bl_circle = Handle(x, y)
        self.bl_circle.setParentItem(self)
        self.bl_circle.on_move = self.bl_handle_moved
        self.tr_circle = Handle(x, y)
        self.tr_circle.setParentItem(self)
        self.tr_circle.on_move = self.tr_handle_moved
        self.br_circle = Handle(x, y)
        self.br_circle.setParentItem(self)
        self.br_circle.on_move = self.br_handle_moved
        
    def tl_handle_moved(self, event):
        point = QtCore.QPointF(event.scenePos().x(), event.scenePos().y())
        point_item = self.mapFromScene(point)
        new_x = point_item.x()
        old_x = self.rect().x()
        width_increase = old_x - new_x
        new_width = self.rect().width() + width_increase

        new_y = point_item.y()
        old_y = self.rect().y()
        height_increase = old_y - new_y
        new_height = self.rect().height() + height_increase

        self.setRect(point_item.x(),
                     point_item.y(),
                     new_width,
                     new_height)

        self.bl_circle.setPos(self.tl_circle.pos().x(), 
                              self.tl_circle.pos().y() + new_height)
        self.tr_circle.setPos(self.tl_circle.pos().x() + new_width, 
                              self.tl_circle.pos().y())
        self.br_circle.setPos(self.tl_circle.pos().x() + new_width, 
                              self.tl_circle.pos().y() + new_height)

    def bl_handle_moved(self, event):
        point = QtCore.QPointF(event.scenePos().x(), event.scenePos().y())
        point_item = self.mapFromScene(point)
        width_increase = self.rect().x() - point_item.x()
        new_width = self.rect().width() + width_increase
        new_height = point_item.y() - self.rect().y()
        self.setRect(point_item.x(),
                     point_item.y() - new_height,
                     new_width,
                     new_height)
        self.tl_circle.setPos(self.bl_circle.pos().x(), 
                              self.bl_circle.pos().y() - new_height)
        self.tr_circle.setPos(self.bl_circle.pos().x() + new_width, 
                              self.bl_circle.pos().y() - new_height)
        self.br_circle.setPos(self.bl_circle.pos().x() + new_width, 
                              self.bl_circle.pos().y())

    def tr_handle_moved(self, event):
        point = QtCore.QPointF(event.scenePos().x(), event.scenePos().y())
        point_item = self.mapFromScene(point)
        new_width = point_item.x() - self.rect().x()
        new_y = point_item.y()
        old_y = self.rect().y()
        height_increase = old_y - new_y
        new_height = self.rect().height() + height_increase
        self.setRect(point_item.x() - new_width, point_item.y(), new_width, new_height)

        self.br_circle.setPos(self.tr_circle.pos().x(), 
                              self.tr_circle.pos().y() + new_height)
        self.tl_circle.setPos(self.tr_circle.pos().x() - new_width, 
                              self.tr_circle.pos().y())
        self.bl_circle.setPos(self.tr_circle.pos().x() - new_width, 
                              self.tr_circle.pos().y() + new_height)

    def br_handle_moved(self, event):
        point = QtCore.QPointF(event.scenePos().x(), event.scenePos().y())
        point_item = self.mapFromScene(point)
        new_width = point_item.x() - self.rect().x()
        new_y = point_item.y()
        old_y = self.rect().y()
        height_increase = old_y - new_y
        new_height = point_item.y() - self.rect().y()
        self.setRect(point_item.x() - new_width,
                     point_item.y() - new_height,
                     new_width, new_height)
        self.tr_circle.setPos(self.br_circle.pos().x(), 
                              self.br_circle.pos().y() - new_height)
        self.tl_circle.setPos(self.br_circle.pos().x() - new_width, 
                              self.br_circle.pos().y() - new_height)
        self.bl_circle.setPos(self.br_circle.pos().x() - new_width, 
                              self.br_circle.pos().y())


    def resize_drag(self, x, y):
        width = abs(x - self.x_start)
        height = abs(y - self.y_start)
        x = min(x, self.x_start)
        y = min(y, self.y_start)
        self.setRect(x, y, width, height)
        self.tl_circle.setPosR(x-self.x_start, y-self.y_start)
        self.bl_circle.setPosR(x-self.x_start, (y-self.y_start) + height)
        self.tr_circle.setPosR((x-self.x_start) + width, y-self.y_start)
        self.br_circle.setPosR((x-self.x_start) + width, (y-self.y_start) + height)

    def set_start(self, x, y):
        self.x_start = x
        self.y_start = y

    def hoverEnterEvent(self, event):
        QtWidgets.QApplication.instance().setOverrideCursor(Qt.OpenHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        super().hoverLeaveEvent(event)
        QtWidgets.QApplication.restoreOverrideCursor()
 
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        QtWidgets.QApplication.instance().setOverrideCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        QtWidgets.QApplication.restoreOverrideCursor()
        super().mouseReleaseEvent(event)
