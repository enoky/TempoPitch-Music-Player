from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from models import THEMES, Theme
from utils import adjust_color


class ThemePreviewWidget(QtWidgets.QWidget):
    clicked = QtCore.Signal(str)  # Emits theme name

    def __init__(self, theme: Theme, selected: bool = False, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.theme = theme
        self.selected = selected
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(140, 100)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        
        # Outer border / Selection
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect).adjusted(1, 1, -1, -1), 8, 8)
        
        painter.fillPath(path, QtGui.QColor(self.theme.window))
        
        if self.selected:
            pen = QtGui.QPen(QtGui.QColor(self.theme.accent), 3)
            painter.setPen(pen)
            painter.drawPath(path)
        else:
            pen = QtGui.QPen(QtGui.QColor(adjust_color(self.theme.text, alpha=30)), 1)
            painter.setPen(pen)
            painter.drawPath(path)

        # Draw Mini UI Representation
        
        # 1. Sidebar (Leftern part)
        sidebar_rect = QtCore.QRectF(rect.x() + 4, rect.y() + 4, 30, rect.height() - 8)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(self.theme.base))
        painter.drawRoundedRect(sidebar_rect, 4, 4)
        
        # 2. Card / Content (Right part)
        card_rect = QtCore.QRectF(rect.x() + 38, rect.y() + 24, rect.width() - 42, rect.height() - 28)
        painter.setBrush(QtGui.QColor(self.theme.card))
        painter.drawRoundedRect(card_rect, 4, 4)
        
        # 3. Header Text Lines (Abstract)
        painter.setBrush(QtGui.QColor(self.theme.text))
        painter.drawRoundedRect(rect.x() + 38, rect.y() + 8, 60, 4, 2, 2)
        
        # 4. Accent Item (e.g. valid button or slider)
        painter.setBrush(QtGui.QColor(self.theme.accent))
        painter.drawRoundedRect(rect.x() + 38, rect.y() + 16, 20, 4, 2, 2)

        # Theme Name Overlay (Bottom)
        painter.setPen(QtGui.QColor(self.theme.text))
        font = painter.font()
        font.setPointSize(9)
        if self.selected:
            font.setBold(True)
        painter.setFont(font)
        
        text_rect = QtCore.QRect(0, rect.height() - 24, rect.width(), 20)
        # Background for text readability if needed? distinct style usually enough.
        # Let's actually draw name centered at bottom
        
        # To make it readable on all backgrounds, maybe draw a small overlay or just rely on the window bg being fairly solid.
        pass

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self.theme.name)


class ThemeSelectorWidget(QtWidgets.QWidget):
    themeChanged = QtCore.Signal(str)

    def __init__(self, current_theme: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._current_theme = current_theme
        self._widgets: dict[str, ThemePreviewWidget] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Scroll Area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(content)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(12)
        
        # Sort and group themes
        light_themes = []
        dark_themes = []
        
        for name, theme in THEMES.items():
            # Quick luminance check of window color
            c = QtGui.QColor(theme.window)
            lum = (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue())
            if lum > 128:
                light_themes.append(name)
            else:
                dark_themes.append(name)

        row = 0
        
        # Dark Themes Section
        if dark_themes:
            label = QtWidgets.QLabel("Dark Themes")
            font = label.font()
            font.setBold(True)
            font.setPointSize(10)
            label.setFont(font)
            self.grid_layout.addWidget(label, row, 0, 1, -1)
            row += 1
            
            col = 0
            for name in dark_themes:
                self._add_preview(name, row, col)
                col += 1
                if col > 3: # 4 columns
                    col = 0
                    row += 1
            row += 1

        # Light Themes Section
        if light_themes:
            # Spacer
            if dark_themes:
                self.grid_layout.addItem(QtWidgets.QSpacerItem(20, 20), row, 0)
                row += 1
                
            label = QtWidgets.QLabel("Light Themes")
            font = label.font()
            font.setBold(True)
            font.setPointSize(10)
            label.setFont(font)
            self.grid_layout.addWidget(label, row, 0, 1, -1)
            row += 1
            
            col = 0
            for name in light_themes:
                self._add_preview(name, row, col)
                col += 1
                if col > 3:
                    col = 0
                    row += 1
            row += 1

        self.grid_layout.setRowStretch(row, 1) # Push everything up
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # Name label below
        self.name_label = QtWidgets.QLabel(current_theme)
        self.name_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.name_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.name_label.setFont(font)
        layout.addWidget(self.name_label)


    def _add_preview(self, name: str, row: int, col: int):
        theme = THEMES[name]
        is_selected = (name == self._current_theme)
        widget = ThemePreviewWidget(theme, selected=is_selected)
        widget.setToolTip(name)
        widget.clicked.connect(self._on_item_clicked)
        self.grid_layout.addWidget(widget, row, col)
        self._widgets[name] = widget

    def _on_item_clicked(self, name: str):
        if name == self._current_theme:
            return
            
        # Deselect old
        if self._current_theme in self._widgets:
            self._widgets[self._current_theme].selected = False
            self._widgets[self._current_theme].update()
            
        self._current_theme = name
        
        # Select new
        if self._current_theme in self._widgets:
            self._widgets[self._current_theme].selected = True
            self._widgets[self._current_theme].update()
            
        self.name_label.setText(name)
        self.themeChanged.emit(name)

    def set_theme(self, name: str):
        if name != self._current_theme:
            self._on_item_clicked(name)
