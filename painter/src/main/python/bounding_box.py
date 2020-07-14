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

    def __init__(self, x, y, parent, cursor):
        self.circle_diam = 3
        super().__init__(x, y, self.circle_diam, self.circle_diam)
        self.setParentItem(parent)
        self.cursor = cursor
        self.parent = parent
        self.setPen(QtGui.QPen(QtGui.QColor(250, 250, 250), 0.5, QtCore.Qt.DashLine))
        self.setBrush(QtGui.QBrush(QtGui.QColor(40, 120, 250), style = QtCore.Qt.SolidPattern))
        self.setAcceptHoverEvents(True)
        self.drag_start_x = None
        self.drag_start_y = None

    def hoverEnterEvent(self, event):
        super().hoverEnterEvent(event)
        QtWidgets.QApplication.instance().setOverrideCursor(self.cursor)

    def mousePressEvent(self, event):
        self.drag_start_x = event.scenePos().x()
        self.drag_start_y = event.scenePos().y()

    def mouseReleaseEvent(self, event):
        if hasattr(self, 'on_release'):
            self.on_release() 

    def hoverLeaveEvent(self, event):
        QtWidgets.QApplication.restoreOverrideCursor()
        
    def mouseMoveEvent(self, event): 
        new_x = event.scenePos().x()
        new_y = event.scenePos().y()
        diff_x = new_x - self.drag_start_x
        diff_y = new_y - self.drag_start_y
        self.setPos(self.pos().x() + diff_x, self.pos().y() + diff_y)
        self.drag_start_x = new_x
        self.drag_start_y = new_y

        #super().mouseMoveEvent(event)
        if self.on_move is not None:
            self.on_move(event, diff_x, diff_y)

    def setPosR(self, x, y):
        # use x and y as center of position
        super().setPos(x-(self.circle_diam/2), y-(self.circle_diam/2))


class BoundingBox(QtWidgets.QGraphicsRectItem):

    def __init__(self, x, y, parent):
        start_rect = QtCore.QRectF(0, 0, 20, 20)
        super().__init__(start_rect)
        self.parent = parent
        self.first_resize = True
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)
        self.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 0.2, QtCore.Qt.DashLine))
        self.setBrush(QtGui.QBrush(QtGui.QColor(40, 120, 200, 70),
                                   style=QtCore.Qt.SolidPattern))
        self.tl_circle = Handle(x, y, self,  Qt.SizeFDiagCursor)
        self.tl_circle.on_move = self.tl_handle_moved
        self.tl_circle.on_release = self.release_handle
        self.bl_circle = Handle(x, y, self, Qt.SizeBDiagCursor)
        self.bl_circle.on_move = self.bl_handle_moved
        self.bl_circle.on_release = self.release_handle
        self.tr_circle = Handle(x, y, self, Qt.SizeBDiagCursor)
        self.tr_circle.on_release = self.release_handle
        self.tr_circle.on_move = self.tr_handle_moved
        self.br_circle = Handle(x, y, self, Qt.SizeFDiagCursor)
        self.br_circle.on_move = self.br_handle_moved
        self.br_circle.on_release = self.release_handle

    def print_rect(self):
        point = QtCore.QPoint(0, 0)
        scenePos = self.mapToScene(point);
        x = self.rect().x() + scenePos.x()
        y = self.rect().y() + scenePos.y()
        print('x:', x, 'y:', y, 'width:', self.rect().width(), 'height:', self.rect().height())

    def scene_rect(self):
        point = QtCore.QPoint(0, 0)
        scenePos = self.mapToScene(point);
        x = self.rect().x() + scenePos.x()
        y = self.rect().y() + scenePos.y()
        return x, y, self.rect().width(), self.rect().height()

    def tl_handle_moved(self, event, diff_x, diff_y):
        new_x = self.rect().x() + diff_x
        old_x = self.rect().x()
        width_increase = old_x - new_x
        new_width = self.rect().width() + width_increase
        new_y = self.rect().y() + diff_y
        old_y = self.rect().y()
        height_increase = old_y - new_y
        new_height = self.rect().height() + height_increase
        self.setRect(new_x, new_y, new_width, new_height)
        self.bl_circle.setPos(self.tl_circle.pos().x(), 
                              self.tl_circle.pos().y() + new_height)
        self.tr_circle.setPos(self.tl_circle.pos().x() + new_width, 
                              self.tl_circle.pos().y())
        self.br_circle.setPos(self.tl_circle.pos().x() + new_width, 
                              self.tl_circle.pos().y() + new_height)
        self.print_rect()

    def bl_handle_moved(self, event, diff_x, diff_y):
        old_x = self.rect().x()
        new_x = old_x + diff_x
        width_increase = old_x - new_x
        new_width = self.rect().width() + width_increase
        old_y = self.rect().y() 
        new_height = self.rect().height() + diff_y
        self.setRect(new_x, old_y, new_width, new_height)
        self.tl_circle.setPos(self.bl_circle.pos().x(), 
                              self.bl_circle.pos().y() - new_height)
        self.tr_circle.setPos(self.bl_circle.pos().x() + new_width, 
                              self.bl_circle.pos().y() - new_height)
        self.br_circle.setPos(self.bl_circle.pos().x() + new_width, 
                              self.bl_circle.pos().y())
        self.print_rect()

    def tr_handle_moved(self, event, diff_x, diff_y):
        old_x = self.rect().x()
        new_width = self.rect().width() + diff_x
        new_y = self.rect().y() + diff_y
        new_height = self.rect().height() - diff_y
        self.setRect(old_x, new_y, new_width, new_height)
        self.br_circle.setPos(self.tr_circle.pos().x(), 
                              self.tr_circle.pos().y() + new_height)
        self.tl_circle.setPos(self.tr_circle.pos().x() - new_width, 
                              self.tr_circle.pos().y())
        self.bl_circle.setPos(self.tr_circle.pos().x() - new_width, 
                              self.tr_circle.pos().y() + new_height)
        self.print_rect()

    def br_handle_moved(self, event, diff_x, diff_y):
        old_x = self.rect().x()
        old_y = self.rect().y() 
        new_width = self.rect().width() + diff_x
        new_height = self.rect().height() + diff_y
        self.setRect(old_x, old_y, new_width, new_height)
        self.tr_circle.setPos(self.br_circle.pos().x(), 
                              self.br_circle.pos().y() - new_height)
        self.tl_circle.setPos(self.br_circle.pos().x() - new_width, 
                              self.br_circle.pos().y() - new_height)
        self.bl_circle.setPos(self.br_circle.pos().x() - new_width, 
                              self.br_circle.pos().y())
        self.print_rect()


    def release_handle(self):
        """ User could have flipped the rect. sort it out 
            otherwise there are side effects with mouse rollover events
        """
        r = self.rect()
        x = r.x()
        y = r.y()
        width = r.width()
        height = r.height()
        if width < 0:
            x = x + width
            width = -width
            # switch left and right locations
            left = self.br_circle.pos().x()
            right = self.tl_circle.pos().x()
            self.tl_circle.setX(left)
            self.bl_circle.setX(left)
            self.tr_circle.setX(right)
            self.br_circle.setX(right)
        
        if height < 0:
            y = y + height
            height = -height
            # switch top and bottom locations
            top = self.bl_circle.pos().y()
            bottom = self.tl_circle.pos().y()
            self.tl_circle.setY(top)
            self.tr_circle.setY(top)
            self.bl_circle.setY(bottom)
            self.br_circle.setY(bottom)
        self.setRect(x, y, width, height)

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
        self.resize_drag(x, y)

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
