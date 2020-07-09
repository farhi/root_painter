from functools import partial

from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import Qt


def add_edit_menu(window, im_viewer, menu_bar):
    edit_menu = menu_bar.addMenu("Edit")
    # Undo
    undo_action = QtWidgets.QAction(QtGui.QIcon(""), "Undo", window)
    undo_action.setShortcut("Z")
    edit_menu.addAction(undo_action)
    undo_action.triggered.connect(im_viewer.scene.undo)
    # Redo
    redo_action = QtWidgets.QAction(QtGui.QIcon(""), "Redo", window)
    redo_action.setShortcut("Ctrl+Shift+Z")
    edit_menu.addAction(redo_action)
    redo_action.triggered.connect(im_viewer.scene.redo)
    return edit_menu


def add_windows_menu(main_window):
    # contrast slider
    menu = main_window.menu_bar.addMenu("Windows")
    contrast_settings_action = QtWidgets.QAction(QtGui.QIcon(""), "Contrast settings", main_window)
    menu.addAction(contrast_settings_action)
    contrast_settings_action.triggered.connect(main_window.contrast_slider.show)


def add_brush_menu(classes, im_viewer, menu_bar):
    brush_menu = menu_bar.addMenu("Brushes")

    def add_brush(name, color_val, shortcut=None):
        color_action = QtWidgets.QAction(QtGui.QIcon(""), name, im_viewer)
        if shortcut:
            color_action.setShortcut(shortcut)
        brush_menu.addAction(color_action)
        color_action.triggered.connect(partial(im_viewer.set_color,
                                               color=QtGui.QColor(*color_val)))
        if im_viewer.brush_color is None:
            im_viewer.brush_color = QtGui.QColor(*color_val)

    for name, rgba, shortcut in classes:
        add_brush(name, rgba, shortcut)
    add_brush('Eraser', (255, 205, 180, 0), 'E')


def add_view_menu(window, im_viewer, menu_bar):
    """ Create view menu with options for
        * fit to view
        * actual size
        * toggle segmentation visibility
        * toggle annotation visibility
        * toggle image visibility
    """
    view_menu = menu_bar.addMenu('View')

    # Fit to view
    fit_to_view_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Fit to View', window)
    fit_to_view_btn.setShortcut('Ctrl+F')
    fit_to_view_btn.setStatusTip('Fit image to view')
    fit_to_view_btn.triggered.connect(im_viewer.graphics_view.fit_to_view)
    view_menu.addAction(fit_to_view_btn)

    # Actual size
    actual_size_view_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'), 'Actual size', window)
    actual_size_view_btn.setShortcut('Ctrl+A')
    actual_size_view_btn.setStatusTip('Show image at actual size')
    actual_size_view_btn.triggered.connect(im_viewer.graphics_view.show_actual_size)
    actual_size_view_btn.triggered.connect(im_viewer.graphics_view.show_actual_size)
    view_menu.addAction(actual_size_view_btn)


    # toggle segmentation visibility
    toggle_seg_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                  'Toggle segmentation visibility', window)
    toggle_seg_visibility_btn.setShortcut('S')
    toggle_seg_visibility_btn.setStatusTip('Show or hide segmentation')
    toggle_seg_visibility_btn.triggered.connect(im_viewer.show_hide_seg)
    view_menu.addAction(toggle_seg_visibility_btn)


    # toggle annotation visibility
    toggle_annot_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                    'Toggle annotation visibility', window)
    toggle_annot_visibility_btn.setShortcut('A')
    toggle_annot_visibility_btn.setStatusTip('Show or hide annotation')
    toggle_annot_visibility_btn.triggered.connect(im_viewer.show_hide_annot)
    view_menu.addAction(toggle_annot_visibility_btn)

    # toggle image visibility
    toggle_image_visibility_btn = QtWidgets.QAction(QtGui.QIcon('missing.png'),
                                                    'Toggle image visibility', window)
    toggle_image_visibility_btn.setShortcut('I')
    toggle_image_visibility_btn.setStatusTip('Show or hide image')
    toggle_image_visibility_btn.triggered.connect(im_viewer.show_hide_image)
    view_menu.addAction(toggle_image_visibility_btn)
    return view_menu
