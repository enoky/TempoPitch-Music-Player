from __future__ import annotations

from PySide6 import QtGui

from models import Theme, THEMES
from utils import adjust_color


def build_palette(theme: Theme) -> QtGui.QPalette:
    window_color = QtGui.QColor(theme.window)
    base_color = QtGui.QColor(theme.base)
    text_color = QtGui.QColor(theme.text)
    highlight_color = QtGui.QColor(theme.highlight)
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, window_color)
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.Base, base_color)
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, window_color.darker(110))
    palette.setColor(QtGui.QPalette.ColorRole.Text, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.Button, window_color)
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, text_color)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, highlight_color)
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
    return palette


def build_stylesheet(theme: Theme) -> str:
    border = adjust_color(theme.card, lighter=120)
    button = adjust_color(theme.card, lighter=112)
    button_hover = adjust_color(button, lighter=108)
    accent = theme.accent
    slider_track = adjust_color(theme.text, darker=220)
    slider_border = adjust_color(slider_track, darker=130)
    slider_handle_border = adjust_color(accent, darker=150)
    return f"""
        QMainWindow {{
            background: {theme.window};
        }}
        QToolButton, QPushButton {{
            padding: 6px 10px;
            border-radius: 8px;
            background: {button};
            border: 1px solid {border};
        }}
        QToolButton:hover, QPushButton:hover {{
            background: {button_hover};
        }}
        QToolButton:checked {{
            background: {accent};
            color: #0b0b0b;
        }}
        QSlider::handle:horizontal {{
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
            background: {accent};
            border: 1px solid {slider_handle_border};
        }}
        QSlider::handle:vertical {{
            width: 16px;
            height: 16px;
            margin: 0 -5px;
            border-radius: 8px;
            background: {accent};
            border: 1px solid {slider_handle_border};
        }}
        QSlider::groove:horizontal {{
            height: 6px;
            background: {slider_track};
            border: 1px solid {slider_border};
            border-radius: 3px;
        }}
        QSlider::groove:vertical {{
            width: 6px;
            background: {slider_track};
            border: 1px solid {slider_border};
            border-radius: 3px;
        }}
        QSlider::sub-page:horizontal {{
            background: {accent};
            border-radius: 3px;
        }}
        QSlider::add-page:horizontal {{
            background: {slider_track};
            border-radius: 3px;
        }}
        QSlider::sub-page:vertical {{
            background: {slider_track};
            border-radius: 3px;
        }}
        QSlider::add-page:vertical {{
            background: {slider_track};
            border-radius: 3px;
        }}
        QGroupBox {{
            margin-top: 16px;
            padding: 12px;
            background: {theme.card};
            border: 1px solid {border};
            border-radius: 12px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            margin-left: 8px;
            font-weight: 600;
        }}
        QComboBox {{
            padding: 4px 10px;
            border-radius: 8px;
            background: {theme.base};
            border: 1px solid {border};
        }}
        QComboBox::drop-down {{
            width: 22px;
            border-left: 1px solid {border};
        }}
        QComboBox QAbstractItemView {{
            background: {theme.base};
            color: {theme.text};
            border: 1px solid {border};
            selection-background-color: {theme.highlight};
            selection-color: #0b0b0b;
        }}
        QMenu {{
            background: {theme.base};
            color: {theme.text};
            border: 1px solid {border};
        }}
        QMenu::item:selected {{
            background: {theme.highlight};
            color: #0b0b0b;
        }}
        QListWidget {{
            padding: 8px;
            border-radius: 10px;
            border: 1px solid {border};
            background: {theme.base};
        }}
        QListWidget::item {{
            margin: 0px;
            padding: 0px;
            border: none;
        }}
        QListWidget::item:selected {{
            border: none;
        }}
        QListWidget::item:hover {{
            border: none;
        }}
        QLabel#track_title {{
            font-size: 18px;
            font-weight: 700;
            color: {theme.text};
        }}
        QLabel#track_artist {{
            font-size: 14px;
            font-weight: 600;
            color: {adjust_color(theme.text, lighter=112)};
        }}
        QLabel#track_album {{
            font-size: 13px;
            color: {adjust_color(theme.text, lighter=108)};
        }}
        QLabel#track_meta {{
            font-size: 12px;
            color: {adjust_color(theme.text, lighter=120)};
        }}
        QLabel#status_label {{
            color: {adjust_color(theme.text, lighter=120)};
        }}
        QFrame#header_frame {{
            border: 1px solid {border};
            border-radius: 14px;
            background: {theme.card};
            padding: 12px;
        }}
        QFrame#media_frame {{
            border: 1px solid {border};
            border-radius: 16px;
            background: {adjust_color(theme.base, lighter=106)};
        }}
        QLabel#playlist_header {{
            font-size: 14px;
            font-weight: 600;
            color: {theme.text};
        }}
        QSplitter::handle {{
            background: {adjust_color(theme.window, lighter=110)};
        }}
    """
