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
        self._hovered = False
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(120, 90)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        
        # Shadow for depth (only when hovered or selected)
        if self._hovered or self.selected:
            shadow_offset = 3 if self.selected else 2
            shadow_path = QtGui.QPainterPath()
            shadow_path.addRoundedRect(
                QtCore.QRectF(rect).adjusted(shadow_offset, shadow_offset, 0, 0), 
                10, 10
            )
            painter.fillPath(shadow_path, QtGui.QColor(0, 0, 0, 40))
        
        # Main card background
        card_path = QtGui.QPainterPath()
        card_rect = QtCore.QRectF(rect).adjusted(2, 2, -4, -4)
        card_path.addRoundedRect(card_rect, 10, 10)
        
        # Fill with window color
        painter.fillPath(card_path, QtGui.QColor(self.theme.window))
        
        # Selection / Hover border
        if self.selected:
            pen = QtGui.QPen(QtGui.QColor(self.theme.accent), 3)
            painter.setPen(pen)
            painter.drawPath(card_path)
        elif self._hovered:
            pen = QtGui.QPen(QtGui.QColor(self.theme.highlight), 2)
            painter.setPen(pen)
            painter.drawPath(card_path)
        else:
            pen = QtGui.QPen(QtGui.QColor(adjust_color(self.theme.text, alpha=25)), 1)
            painter.setPen(pen)
            painter.drawPath(card_path)

        # Inner content area coordinates
        inner_x = card_rect.x() + 8
        inner_y = card_rect.y() + 8
        inner_w = card_rect.width() - 16
        inner_h = card_rect.height() - 32  # Leave room for name

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        
        # Mini app visualization
        # Sidebar
        sidebar_w = 22
        sidebar_rect = QtCore.QRectF(inner_x, inner_y, sidebar_w, inner_h)
        painter.setBrush(QtGui.QColor(self.theme.base))
        painter.drawRoundedRect(sidebar_rect, 4, 4)
        
        # Sidebar accent indicator
        painter.setBrush(QtGui.QColor(self.theme.accent))
        painter.drawRoundedRect(inner_x + 4, inner_y + 8, sidebar_w - 8, 4, 2, 2)
        painter.drawRoundedRect(inner_x + 4, inner_y + 16, sidebar_w - 8, 3, 1, 1)
        
        # Main content area
        content_x = inner_x + sidebar_w + 6
        content_w = inner_w - sidebar_w - 6
        
        # Header bar simulation
        painter.setBrush(QtGui.QColor(self.theme.text))
        painter.drawRoundedRect(content_x, inner_y, content_w * 0.6, 4, 2, 2)
        
        # Highlight element (like a button or active element)
        painter.setBrush(QtGui.QColor(self.theme.highlight))
        painter.drawRoundedRect(content_x, inner_y + 8, content_w * 0.35, 4, 2, 2)
        
        # Card area
        card_inner = QtCore.QRectF(content_x, inner_y + 16, content_w, inner_h - 16)
        painter.setBrush(QtGui.QColor(self.theme.card))
        painter.drawRoundedRect(card_inner, 4, 4)
        
        # Small details inside card
        painter.setBrush(QtGui.QColor(adjust_color(self.theme.text, alpha=60)))
        painter.drawRoundedRect(content_x + 4, inner_y + 22, content_w - 8, 3, 1, 1)
        painter.drawRoundedRect(content_x + 4, inner_y + 28, content_w * 0.5, 3, 1, 1)

        # Theme name at bottom
        name_rect = QtCore.QRectF(card_rect.x(), card_rect.bottom() - 20, card_rect.width(), 18)
        
        # Name background pill
        name_bg_path = QtGui.QPainterPath()
        name_bg_rect = QtCore.QRectF(
            card_rect.x() + (card_rect.width() - 80) / 2,
            card_rect.bottom() - 22,
            80,
            18
        )
        name_bg_path.addRoundedRect(name_bg_rect, 9, 9)
        
        bg_color = QtGui.QColor(self.theme.card)
        bg_color.setAlpha(200)
        painter.fillPath(name_bg_path, bg_color)
        
        # Theme name text
        painter.setPen(QtGui.QColor(self.theme.text))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(self.selected)
        painter.setFont(font)
        painter.drawText(name_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.theme.name)

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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(20)

        # Sort and group themes
        # Since THEMES is an ordered dict (Python 3.7+) and we defined them in order
        # of Dark (8), Medium (8), Light (8), we can just slice them.
        all_themes = list(THEMES.keys())
        
        dark_themes = all_themes[:8]
        medium_themes = all_themes[8:16]
        light_themes = all_themes[16:]

        # Dark Themes Section
        if dark_themes:
            dark_section = self._create_section("Dark Themes", dark_themes)
            layout.addWidget(dark_section)

        # Medium Themes Section
        if medium_themes:
            medium_section = self._create_section("Medium Themes", medium_themes)
            layout.addWidget(medium_section)

        # Light Themes Section
        if light_themes:
            light_section = self._create_section("Light Themes", light_themes)
            layout.addWidget(light_section)

        layout.addStretch(1)

    def _create_section(self, title: str, theme_names: list[str]) -> QtWidgets.QWidget:
        section = QtWidgets.QWidget()
        section_layout = QtWidgets.QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(12)
        
        # Section header with styled label
        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 4, 0)
        header_layout.setSpacing(8)
        
        label = QtWidgets.QLabel(title)
        font = label.font()
        font.setBold(True)
        font.setPointSize(11)
        label.setFont(font)
        label.setStyleSheet("color: palette(text); opacity: 0.9;")
        
        # Decorative line after label
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("background-color: palette(mid); max-height: 1px;")
        
        header_layout.addWidget(label)
        header_layout.addWidget(line, 1)
        
        section_layout.addWidget(header)
        
        # Theme grid using flow layout
        flow = FlowLayout(spacing=12)
        for name in theme_names:
            widget = self._create_preview(name)
            flow.addWidget(widget)
        section_layout.addLayout(flow)
        
        return section

    def _create_preview(self, name: str) -> ThemePreviewWidget:
        theme = THEMES[name]
        is_selected = (name == self._current_theme)
        widget = ThemePreviewWidget(theme, selected=is_selected)
        widget.setToolTip(f"Switch to {name} theme")
        widget.clicked.connect(self._on_item_clicked)
        self._widgets[name] = widget
        return widget

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
            
        self.themeChanged.emit(name)

    def set_theme(self, name: str):
        if name != self._current_theme:
            self._on_item_clicked(name)


class FlowLayout(QtWidgets.QLayout):
    """A flow layout that arranges widgets horizontally, wrapping to next line when needed."""
    
    def __init__(self, parent=None, spacing: int = 12):
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._spacing = spacing

    def addItem(self, item: QtWidgets.QLayoutItem):
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def spacing(self) -> int:
        return self._spacing

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        
        for item in self._items:
            widget = item.widget()
            if widget is None:
                continue
                
            space_x = self._spacing
            space_y = self._spacing
            
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))
            
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        
        return y + line_height - rect.y() + bottom
