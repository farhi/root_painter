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

# pylint: disable=I1101, C0111, E0611, R0902
""" Canvas where image and annotations can be drawn """
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from bounding_box import BoundingBox


class GraphicsScene(QtWidgets.QGraphicsScene):
    """
    Canvas where image and lines will be drawn
    """
    def __init__(self):
        super().__init__()
        self.drawing = False
        self.box_resizing = True
        self.bounding_box = None
        self.box_enabled = False
        self.move_box = False # when user clicks on the box they are moving it.
        self.brush_size = 25
        # history is a list of pixmaps
        self.history = []
        self.redo_list = []
        self.last_x = None
        self.last_y = None
        self.annot_pixmap = None
        # bounding box start position
        self.box_origin_x = None
        self.box_origin_y = None
        self.mouse_down = False


    def undo(self):
        if len(self.history) > 1:
            self.redo_list.append(self.history.pop().copy())
            # remove top item from history.
            new_state = self.history[-1].copy()
            self.annot_pixmap_holder.setPixmap(new_state)
            self.annot_pixmap = new_state
            self.parent.store_annot_slice()
            self.parent.parent.update_viewer_annot_slice()


    def redo(self):
        if self.redo_list:
            new_state = self.redo_list.pop()
            self.history.append(new_state.copy())
            self.annot_pixmap_holder.setPixmap(new_state)
            self.annot_pixmap = new_state
            self.parent.store_annot_slice()
            # Update all views with new state.
            self.parent.parent.update_viewer_annot_slice()
    
    def start_bounding_box(self):
        #self.box_enabled = True
        # specify parent to keep within a certain area.
        #self.box = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self.parent.parent)
        #self.rect_item = QtWidgets.QGraphicsRectItem(QtCore.QRectF(200, 200, 200, 200))
        #self.rect_item.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        #self.rect_item.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120), 1, QtCore.Qt.DashLine) )
        #self.rect_item.setBrush(QtGui.QBrush(QtGui.QColor(40, 120, 200, 70),
                                #style = QtCore.Qt.SolidPattern))

        self.box_enabled = True
        
        print('start drawing bounding box now')

    def cancel_bounding_box(self):
        print('cancel bounding box not yet implemented') 
        #self.bounding_box = None
        #self.box_enabled = False


    def apply_bounding_box(self):
        x = self.bounding_box.scenePos().x() + self.bounding_box.rect().x()
        y = self.bounding_box.scenePos().y() + self.bounding_box.rect().y()
        w = self.bounding_box.rect().width()
        h = self.bounding_box.rect().height()

        # just for testing, draw the information to a pixmap.

        painter = QtGui.QPainter(self.annot_pixmap)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
        painter.drawPixmap(0, 0, self.annot_pixmap)
        painter.setPen(QtGui.QPen(self.parent.brush_color, 0, Qt.SolidLine,
                                  Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QtGui.QBrush(self.parent.brush_color, Qt.SolidPattern))
        painter.drawRect(x, y, w, h)
        self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
        painter.end()


    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        self.mouse_down = True
        
        if self.box_enabled:
            if self.bounding_box is None:
                print('add box', event.scenePos().x(), event.scenePos().y())
                self.bounding_box = BoundingBox(event.scenePos().x(), event.scenePos().y())
                print('assign x start')
                self.bounding_box.set_start(event.scenePos().x(), event.scenePos().y())
                self.addItem(self.bounding_box) 
        elif not modifiers & QtCore.Qt.ControlModifier and self.parent.annot_visible:
            self.drawing = True
            pos = event.scenePos()
            x, y = pos.x(), pos.y()
            if self.brush_size == 1:
                circle_x = x
                circle_y = y
            else:
                circle_x = x - (self.brush_size / 2) + 0.5
                circle_y = y - (self.brush_size / 2) + 0.5

            painter = QtGui.QPainter(self.annot_pixmap)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            painter.drawPixmap(0, 0, self.annot_pixmap)
            painter.setPen(QtGui.QPen(self.parent.brush_color, 0, Qt.SolidLine,
                                      Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QtGui.QBrush(self.parent.brush_color, Qt.SolidPattern))
            if self.brush_size == 1:
                painter.drawPoint(circle_x, circle_y)
            else:
                painter.drawEllipse(circle_x, circle_y, self.brush_size-1, self.brush_size-1)
            self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
            painter.end()
            self.last_x = x
            self.last_y = y

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.mouse_down = False
        if self.drawing:
            self.drawing = False
            # has to be some limit to history or RAM will run out
            if len(self.history) > 50:
                self.history = self.history[-50:]
            self.history.append(self.annot_pixmap.copy())
            self.redo_list = []
            self.parent.store_annot_slice()
            # update all views with new state.
            self.parent.parent.update_viewer_annot_slice()
        if self.box_resizing:
            self.box_resizing = False

    def map_box_point(self, mouse_event):
        """ The bounding box location must be specified with reference to 
            the top level component, but the mouse events are relative 
            to the nested component. Map the mouse events to
            values appropriate for setting the box coordinates
            
            There is likley a built in PyQt function that I could use instead
            of this hack but I haven't been able to find it yet.
        """
        p = self.parent.graphics_view.mapFromScene(mouse_event.scenePos())
        x = p.x()
        y = p.y()
        for widget in [self.parent,
                       self.parent.graphics_view,
                       self.parent.inner_container,
                       self.parent.parent.container]:
            x += widget.geometry().x()
            y += widget.geometry().y()
        return x, y

    def box_point_to_pixmap_loc(self, x, y):
        for widget in [self.parent,
                       self.parent.graphics_view,
                       self.parent.inner_container,
                       self.parent.parent.container]:
            x -= widget.geometry().x()
            y -= widget.geometry().y()
        return x, y

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.box_enabled:
            if self.mouse_down and self.box_resizing:
                x_start, y_start = self.bounding_box.get_start()
                print('x start y start', x_start, y_start)
                 
                pos = event.scenePos()
                x, y = pos.x(), pos.y()

                #box_x = self.bounding_box.rect().x()
                #box_y = self.bounding_box.rect().y()

                #print('moving', box_x, box_y)
                print('x', x, 'x_start', x_start)
                width = abs(x - x_start)
                height = abs(y - y_start)
                x = min(x, x_start)
                y = min(y, y_start)

                print('min', 'x', x)
                print(x, width)

                self.bounding_box.setRect(x, y, width, height)

        elif self.drawing:
            painter = QtGui.QPainter(self.annot_pixmap)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
            painter.drawPixmap(0, 0, self.annot_pixmap)
            pen = QtGui.QPen(self.parent.brush_color, self.brush_size, Qt.SolidLine,
                             Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            pos = event.scenePos()
            x, y = pos.x(), pos.y()

            # Based on empirical observation
            if self.brush_size % 2 == 0:
                painter.drawLine(self.last_x+0.5, self.last_y+0.5, x+0.5, y+0.5)
            else:
                painter.drawLine(self.last_x, self.last_y, x, y)

            self.annot_pixmap_holder.setPixmap(self.annot_pixmap)
            painter.end()
            self.last_x = x
            self.last_y = y
