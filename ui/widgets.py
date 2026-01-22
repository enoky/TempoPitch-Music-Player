from __future__ import annotations

import math
import os
import json
import random
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
try:
    from PySide6 import QtSvg
except ImportError:  # pragma: no cover - optional dependency for SVG icons
    QtSvg = None

from models import PlayerState, RepeatMode, Track, TrackMetadata, format_track_title
from utils import clamp, format_time, safe_float

# UI Widgets
# -----------------------------

PLAYING_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
SVG_ICON_TEMPLATES = {
    "file": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
        </svg>
    """,
    "folder": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
        </svg>
    """,
    "trash": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
            <path d="M10 11v6"/>
            <path d="M14 11v6"/>
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
        </svg>
    """,
    "play": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polygon points="6 4 20 12 6 20 6 4"/>
        </svg>
    """,
    "pause": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <line x1="9" y1="5" x2="9" y2="19"/>
            <line x1="15" y1="5" x2="15" y2="19"/>
        </svg>
    """,
    "stop": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <rect x="6" y="6" width="12" height="12"/>
        </svg>
    """,
    "prev": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polygon points="19 5 9 12 19 19 19 5"/>
            <line x1="5" y1="5" x2="5" y2="19"/>
        </svg>
    """,
    "next": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 5 15 12 5 19 5 5"/>
            <line x1="19" y1="5" x2="19" y2="19"/>
        </svg>
    """,
    "volume_on": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
            <path d="M15 9a4 4 0 0 1 0 6"/>
            <path d="M17.5 6.5a7 7 0 0 1 0 11"/>
        </svg>
    """,
    "volume_off": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
            <line x1="16" y1="9" x2="21" y2="14"/>
            <line x1="21" y1="9" x2="16" y2="14"/>
        </svg>
    """,
    "search": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
    """,
    "close": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
    """,
    "refresh": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10"/>
            <polyline points="1 20 1 14 7 14"/>
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
        </svg>
    """,
    "list": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <line x1="8" y1="6" x2="21" y2="6"/>
            <line x1="8" y1="12" x2="21" y2="12"/>
            <line x1="8" y1="18" x2="21" y2="18"/>
            <line x1="3" y1="6" x2="3.01" y2="6"/>
            <line x1="3" y1="12" x2="3.01" y2="12"/>
            <line x1="3" y1="18" x2="3.01" y2="18"/>
        </svg>
    """,
    "user": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
        </svg>
    """,
    "disc": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <circle cx="12" cy="12" r="3"/>
        </svg>
    """,
    "tag": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>
            <line x1="7" y1="7" x2="7.01" y2="7"/>
        </svg>
    """,
    "video": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polygon points="23 7 16 12 23 17 23 7"/>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
        </svg>
    """,
    "fullscreen": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polyline points="15 3 21 3 21 9"/>
            <polyline points="9 21 3 21 3 15"/>
            <line x1="21" y1="3" x2="14" y2="10"/>
            <line x1="3" y1="21" x2="10" y2="14"/>
        </svg>
    """,
    "exit_fullscreen": """
        <svg viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
            <polyline points="4 14 10 14 10 20"/>
            <polyline points="20 10 14 10 14 4"/>
            <line x1="14" y1="10" x2="21" y2="3"/>
            <line x1="3" y1="21" x2="10" y2="14"/>
        </svg>
    """,
}


def render_svg_icon(svg_template: str, color: QtGui.QColor, size_px: int) -> QtGui.QIcon:
    if QtSvg is None:
        return QtGui.QIcon()
    svg = svg_template.format(color=color.name())
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    pixmap = QtGui.QPixmap(size_px, size_px)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    renderer.render(painter, QtCore.QRectF(0, 0, size_px, size_px))
    painter.end()
    return QtGui.QIcon(pixmap)


class PlaylistItemDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, list_widget: QtWidgets.QListWidget):
        super().__init__(list_widget)
        self._list = list_widget

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        if index.data(PLAYING_ROLE):
            color = QtGui.QColor(self._list.palette().color(QtGui.QPalette.ColorRole.Highlight))
            color.setAlpha(140)
            painter.save()
            painter.fillRect(option.rect, color)
            painter.restore()
        super().paint(painter, option, index)

class VisualizerWidget(QtWidgets.QWidget):
    def __init__(self, engine: PlayerEngine, parent=None):
        super().__init__(parent)
        self.engine = engine

        # Spectrum state
        self._fft_size = 1024
        self._bar_count = 48
        self._bar_levels = np.zeros(self._bar_count, dtype=np.float32)
        self._fft_window = np.hanning(self._fft_size).astype(np.float32)
        self._fft_input = np.zeros(self._fft_size, dtype=np.float32)
        self._fft_windowed = np.zeros(self._fft_size, dtype=np.float32)
        self._fft_magnitudes = np.zeros(self._fft_size // 2 + 1, dtype=np.float32)
        self._bin_edges = np.linspace(0, self._fft_size // 2, self._bar_count + 1, dtype=int)
        self._bin_counts = np.diff(self._bin_edges).astype(np.float32)
        self._bin_counts[self._bin_counts == 0] = 1.0
        self._levels = np.zeros(self._bar_count, dtype=np.float32)

        # Loudness state
        self._viz_mode = 0  # 0=Spectrum, 1=Loudness
        self._l_level = 0.0
        self._r_level = 0.0

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._pull_frames)
        self._timer.start()
        self.setMinimumHeight(140)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._viz_mode = 1 - self._viz_mode
            self._l_level = 0.0
            self._r_level = 0.0
            self._bar_levels.fill(0.0)
            self.update()
        super().mousePressEvent(event)

    def _pull_frames(self) -> None:
        if not self.engine:
            return

        delay_sec = self.engine.get_output_latency_seconds()

        if self._viz_mode == 0:
            # Spectrum
            frames = self.engine.get_visualizer_frames(
                frames=self._fft_size,
                mono=True,
                delay_sec=delay_sec,
            )
            if frames.size == 0:
                self._bar_levels *= 0.75
                self.update()
                return

            mono = frames[:, 0] if frames.ndim == 2 else frames.reshape(-1)
            if mono.size < self._fft_size:
                self._fft_input.fill(0.0)
                self._fft_input[-mono.size:] = mono
            else:
                self._fft_input[:] = mono[-self._fft_size:]

            np.multiply(self._fft_input, self._fft_window, out=self._fft_windowed)
            spectrum = np.fft.rfft(self._fft_windowed)
            np.abs(spectrum, out=self._fft_magnitudes)
            magnitudes = self._fft_magnitudes[1:]
            np.log1p(magnitudes, out=magnitudes)
            if magnitudes.size > 0:
                peak = float(magnitudes.max())
                if peak > 0.0:
                    magnitudes /= peak
                np.add.reduceat(magnitudes, self._bin_edges[:-1], out=self._levels)
                self._levels /= self._bin_counts
                self._bar_levels *= 0.75
                np.maximum(self._bar_levels, self._levels, out=self._bar_levels)

        else:
            # Loudness
            frames = self.engine.get_visualizer_frames(
                frames=1024,
                mono=False,
                delay_sec=delay_sec,
            )
            if frames.size == 0:
                self._l_level *= 0.75
                self._r_level *= 0.75
            else:
                # Use peak detection for snappier meter
                peaks = np.max(np.abs(frames), axis=0) if frames.ndim == 2 else np.array([0.0, 0.0])
                if peaks.size >= 2:
                    current_l = float(peaks[0])
                    current_r = float(peaks[1])
                    # Smooth decay
                    self._l_level = max(current_l, self._l_level * 0.85)
                    self._r_level = max(current_r, self._r_level * 0.85)

        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        palette = self.palette()
        background = palette.color(QtGui.QPalette.ColorRole.Base)
        
        painter.fillRect(self.rect(), background)
        
        if self._viz_mode == 0:
            self._draw_spectrum(painter)
        else:
            self._draw_loudness(painter)

    def _draw_spectrum(self, painter: QtGui.QPainter) -> None:
        rect = self.rect().adjusted(12, 12, -12, -12)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        text_color = self.palette().color(QtGui.QPalette.ColorRole.Text)
        highlight = self.palette().color(QtGui.QPalette.ColorRole.Highlight)

        baseline_y = rect.bottom()
        painter.setPen(QtGui.QPen(text_color, 1))
        painter.drawLine(rect.left(), baseline_y, rect.right(), baseline_y)

        bar_count = len(self._bar_levels)
        if bar_count == 0:
            return
        bar_width = rect.width() / bar_count
        for i, level in enumerate(self._bar_levels):
            bar_height = rect.height() * float(level)
            if bar_height <= 1:
                continue
            x = rect.left() + i * bar_width
            y = rect.bottom() - bar_height
            color = QtGui.QColor(highlight)
            color = color.lighter(110 + int(60 * (i / max(1, bar_count - 1))))
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QtCore.QRectF(x + 1, y, bar_width - 2, bar_height), 2.0, 2.0)

    def _draw_loudness(self, painter: QtGui.QPainter) -> None:
        rect = self.rect().adjusted(20, 20, -20, -20)
        if rect.width() <= 0 or rect.height() <= 0:
            return
            
        highlight = self.palette().color(QtGui.QPalette.ColorRole.Highlight)
        text_color = self.palette().color(QtGui.QPalette.ColorRole.Text)
        
        # Center the meter vertically
        bar_height = max(10, rect.height() // 5)
        mid_y = rect.center().y()
        gap = 10
        
        y_l = mid_y - bar_height - (gap / 2)
        y_r = mid_y + (gap / 2)
        
        # L/R Labels
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(text_color)
        painter.drawText(QtCore.QRectF(rect.left(), y_l, 30, bar_height), QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, "L")
        painter.drawText(QtCore.QRectF(rect.left(), y_r, 30, bar_height), QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, "R")
        
        # Bar areas
        bar_start_x = rect.left() + 25
        bar_width_total = rect.width() - 25
        bar_rect_l = QtCore.QRectF(bar_start_x, y_l, bar_width_total, bar_height)
        bar_rect_r = QtCore.QRectF(bar_start_x, y_r, bar_width_total, bar_height)
        
        # Draw track backgrounds (darker track)
        painter.setBrush(QtGui.QBrush(text_color.darker(350)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bar_rect_l, 4, 4)
        painter.drawRoundedRect(bar_rect_r, 4, 4)
        
        # Calculate active widths (logarithmic scale feels more natural for volume)
        # But for simplicity and visual punch, simple sqrt or linear can work.
        # Let's map 0..1 to the bar.
        
        val_l = min(1.0, math.sqrt(self._l_level))
        val_r = min(1.0, math.sqrt(self._r_level))
        
        width_l = bar_rect_l.width() * val_l
        width_r = bar_rect_r.width() * val_r
        
        # Gradients
        grad_l = QtGui.QLinearGradient(bar_rect_l.topLeft(), bar_rect_l.topRight())
        grad_l.setColorAt(0.0, highlight.darker(150))
        grad_l.setColorAt(1.0, highlight.lighter(130))
        
        grad_r = QtGui.QLinearGradient(bar_rect_r.topLeft(), bar_rect_r.topRight())
        grad_r.setColorAt(0.0, highlight.darker(150))
        grad_r.setColorAt(1.0, highlight.lighter(130))

        # Draw active bars
        if width_l > 1:
             painter.setBrush(QtGui.QBrush(grad_l))
             painter.drawRoundedRect(QtCore.QRectF(bar_rect_l.x(), bar_rect_l.y(), width_l, bar_rect_l.height()), 4, 4)

        if width_r > 1:
             painter.setBrush(QtGui.QBrush(grad_r))
             painter.drawRoundedRect(QtCore.QRectF(bar_rect_r.x(), bar_rect_r.y(), width_r, bar_rect_r.height()), 4, 4)



class VideoWidget(QtWidgets.QWidget):
    def __init__(self, engine: PlayerEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._image: Optional[QtGui.QImage] = None
        self._timestamp: Optional[float] = None
        self._timer: Optional[QtCore.QTimer] = None
        if self.engine:
            self.engine.videoFrameReady.connect(self._on_frame_ready)
            self.engine.stateChanged.connect(self._on_state_changed)
            self.engine.trackChanged.connect(self._on_track_changed)
        self.setMinimumSize(200, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._pull_frame)
        self._timer.start()

    def stop_timer(self) -> None:
        if self._timer and self._timer.isActive():
            self._timer.stop()

    def clear(self) -> None:
        self._image = None
        self._timestamp = None
        self.update()

    def _pull_frame(self) -> None:
        if not self.engine:
            return
        image, timestamp = self.engine.get_video_frame()
        if image is None:
            if self._image is not None:
                self._image = None
                self._timestamp = None
                self.update()
            return
        if timestamp != self._timestamp:
            self._image = image
            self._timestamp = timestamp
            self.update()

    def _on_frame_ready(self, image: QtGui.QImage, timestamp: float) -> None:
        if timestamp != self._timestamp:
            self._image = image
            self._timestamp = timestamp
            self.update()

    def _on_state_changed(self, state: PlayerState) -> None:
        if state in (PlayerState.STOPPED, PlayerState.ERROR):
            self.clear()

    def _on_track_changed(self, track: Optional[Track]) -> None:
        self.clear()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        # Use black background for video
        painter.fillRect(self.rect(), QtCore.Qt.GlobalColor.black)

        if self._image is None or self._image.isNull():
            painter.setPen(QtGui.QColor(255, 255, 255))  # White text on black
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "No Video")
            return

        target = self.rect()
        scaled = self._image.scaled(
            target.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        x = target.x() + (target.width() - scaled.width()) // 2
        y = target.y() + (target.height() - scaled.height()) // 2
        painter.drawImage(QtCore.QPoint(x, y), scaled)


class VideoPopoutDialog(QtWidgets.QDialog):
    closed = QtCore.Signal()

    def __init__(self, engine: PlayerEngine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._is_fullscreen = False
        self._normal_geometry: Optional[QtCore.QRect] = None
        self._controls_visible = True
        
        self.setWindowTitle("Pop-out Video")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        
        # Set black background for entire dialog
        self.setStyleSheet("VideoPopoutDialog { background-color: black; }")
        
        # Video widget - takes up whole dialog
        self.video_widget = VideoWidget(engine, self)
        self.video_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        
        # Control buttons - use white icons on dark background
        icon_size = 24
        icon_color = QtGui.QColor(255, 255, 255)  # White for visibility on black
        
        self._play_icon = render_svg_icon(SVG_ICON_TEMPLATES["play"], icon_color, icon_size)
        self._pause_icon = render_svg_icon(SVG_ICON_TEMPLATES["pause"], icon_color, icon_size)
        self._stop_icon = render_svg_icon(SVG_ICON_TEMPLATES["stop"], icon_color, icon_size)
        self._fullscreen_icon = render_svg_icon(SVG_ICON_TEMPLATES["fullscreen"], icon_color, icon_size)
        self._exit_fullscreen_icon = render_svg_icon(SVG_ICON_TEMPLATES["exit_fullscreen"], icon_color, icon_size)
        
        self.play_pause_btn = QtWidgets.QPushButton()
        self.play_pause_btn.setIcon(self._play_icon)
        self.play_pause_btn.setIconSize(QtCore.QSize(icon_size, icon_size))
        self.play_pause_btn.setToolTip("Play/Pause")
        self.play_pause_btn.setFixedSize(40, 32)
        self.play_pause_btn.clicked.connect(self._on_play_pause_clicked)
        
        self.stop_btn = QtWidgets.QPushButton()
        self.stop_btn.setIcon(self._stop_icon)
        self.stop_btn.setIconSize(QtCore.QSize(icon_size, icon_size))
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setFixedSize(40, 32)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        
        self.fullscreen_btn = QtWidgets.QPushButton()
        self.fullscreen_btn.setIcon(self._fullscreen_icon)
        self.fullscreen_btn.setIconSize(QtCore.QSize(icon_size, icon_size))
        self.fullscreen_btn.setToolTip("Toggle Fullscreen (ESC to exit)")
        self.fullscreen_btn.setFixedSize(40, 32)
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
        
        # Style buttons for dark background
        btn_style = """
            QPushButton {
                background-color: rgba(60, 60, 60, 180);
                border: 1px solid rgba(100, 100, 100, 150);
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 200);
                border: 1px solid rgba(120, 120, 120, 180);
            }
            QPushButton:pressed {
                background-color: rgba(50, 50, 50, 220);
            }
        """
        self.play_pause_btn.setStyleSheet(btn_style)
        self.stop_btn.setStyleSheet(btn_style)
        self.fullscreen_btn.setStyleSheet(btn_style)
        
        # Controls layout - will be overlaid at the bottom of the video
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.fullscreen_btn)
        
        # Controls widget - overlays on top of video at bottom
        self._controls_widget = QtWidgets.QWidget(self)
        self._controls_widget.setLayout(controls_layout)
        self._controls_widget.setStyleSheet("background-color: rgba(0, 0, 0, 160); border-radius: 6px;")
        self._controls_widget.setFixedHeight(48)
        
        # Setup opacity effect for fade animation
        self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self._controls_widget)
        self._opacity_effect.setOpacity(1.0)
        self._controls_widget.setGraphicsEffect(self._opacity_effect)
        
        # Setup fade animation
        self._fade_animation = QtCore.QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(300)  # 300ms fade
        self._fade_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)
        self._fade_animation.finished.connect(self._on_fade_finished)
        
        # Hide timer - fires after 2 seconds of inactivity
        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(2000)  # 2 seconds
        self._hide_timer.timeout.connect(self._fade_out_controls)
        
        # Main layout - just the video, controls overlay separately
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.video_widget)
        
        # Enable mouse tracking for the entire dialog
        self.setMouseTracking(True)
        self.video_widget.setMouseTracking(True)
        self._controls_widget.setMouseTracking(True)
        
        # Connect to engine state changes to update play/pause icon
        if self._engine:
            self._engine.stateChanged.connect(self._on_engine_state_changed)
            self._update_play_pause_icon()
        
        # Start the hide timer
        self._hide_timer.start()
    
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Reposition controls overlay at bottom of dialog
        self._update_controls_position()
    
    def _update_controls_position(self) -> None:
        # Position controls at bottom center of dialog
        margin = 12
        controls_width = self.width() - (margin * 2)
        controls_height = self._controls_widget.height()
        x = margin
        y = self.height() - controls_height - margin
        self._controls_widget.setGeometry(x, y, controls_width, controls_height)
    
    def _on_fade_finished(self) -> None:
        # Hide controls completely when faded out
        if not self._controls_visible:
            self._controls_widget.hide()
    
    def _fade_out_controls(self) -> None:
        if not self._controls_visible:
            return
        self._controls_visible = False
        self._fade_animation.stop()
        self._fade_animation.setStartValue(self._opacity_effect.opacity())
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.start()
    
    def _fade_in_controls(self) -> None:
        if self._controls_visible:
            # Already visible, just restart the hide timer
            self._hide_timer.start()
            return
        self._controls_visible = True
        self._controls_widget.show()
        self._fade_animation.stop()
        self._fade_animation.setStartValue(self._opacity_effect.opacity())
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()
        self._hide_timer.start()
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._fade_in_controls()
        super().mouseMoveEvent(event)
    
    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._fade_in_controls()
        super().enterEvent(event)
    
    def _on_play_pause_clicked(self) -> None:
        if not self._engine:
            return
        state = self._engine.state
        if state == PlayerState.PLAYING:
            self._engine.pause()
        else:
            self._engine.play()
    
    def _on_stop_clicked(self) -> None:
        if self._engine:
            self._engine.stop()
    
    def _on_engine_state_changed(self, state: PlayerState) -> None:
        self._update_play_pause_icon()
    
    def _update_play_pause_icon(self) -> None:
        if not self._engine:
            return
        if self._engine.state == PlayerState.PLAYING:
            self.play_pause_btn.setIcon(self._pause_icon)
            self.play_pause_btn.setToolTip("Pause")
        else:
            self.play_pause_btn.setIcon(self._play_icon)
            self.play_pause_btn.setToolTip("Play")
    
    def _toggle_fullscreen(self) -> None:
        if self._is_fullscreen:
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()
    
    def _enter_fullscreen(self) -> None:
        if self._is_fullscreen:
            return
        self._normal_geometry = self.geometry()
        self._is_fullscreen = True
        self.showFullScreen()
        self.fullscreen_btn.setIcon(self._exit_fullscreen_icon)
        self.fullscreen_btn.setToolTip("Exit Fullscreen (ESC)")
    
    def _exit_fullscreen(self) -> None:
        if not self._is_fullscreen:
            return
        self._is_fullscreen = False
        self.showNormal()
        if self._normal_geometry:
            self.setGeometry(self._normal_geometry)
        self.fullscreen_btn.setIcon(self._fullscreen_icon)
        self.fullscreen_btn.setToolTip("Toggle Fullscreen (ESC to exit)")
    
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape and self._is_fullscreen:
            self._exit_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._hide_timer.stop()
        self._fade_animation.stop()
        self.video_widget.stop_timer()
        if self._engine:
            try:
                self._engine.stateChanged.disconnect(self._on_engine_state_changed)
            except RuntimeError:
                pass  # Already disconnected
        self.closed.emit()
        super().closeEvent(event)


class TempoPitchWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float, bool, bool, bool)  # tempo, pitch_st, key_lock, tape_mode, lock_432

    def __init__(self, parent=None):
        super().__init__("Tempo & Pitch", parent)

        self.tempo_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.tempo_slider.setRange(50, 200)
        self.tempo_slider.setValue(100)
        self.tempo_slider.setToolTip("Adjust tempo (0.50× to 2.00×).")
        self.tempo_slider.setAccessibleName("Tempo slider")

        self.pitch_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-120, 120)  # -12..+12 semitones in 0.1 steps
        self.pitch_slider.setValue(0)
        self.pitch_slider.setToolTip("Adjust pitch in semitones.")
        self.pitch_slider.setAccessibleName("Pitch slider")

        self.tempo_label = QtWidgets.QLabel("Tempo: 1.00×")
        self.pitch_label = QtWidgets.QLabel("Pitch: +0.0 st")

        self.key_lock = QtWidgets.QCheckBox("Key Lock (tempo ≠ pitch)")
        self.key_lock.setChecked(True)
        self.key_lock.setToolTip("Keep pitch steady while changing tempo.")
        self.key_lock.setAccessibleName("Key lock")

        self.tape_mode = QtWidgets.QCheckBox("Tape Mode (rate)")
        self.tape_mode.setChecked(False)
        self.tape_mode.setToolTip("Link pitch to tempo changes.")
        self.tape_mode.setAccessibleName("Tape mode")

        self.lock_432 = QtWidgets.QCheckBox("Lock pitch to A4=432 Hz")
        self.lock_432.setChecked(False)
        self.lock_432.setToolTip("Lock pitch to A4=432 Hz and disable manual pitch edits.")
        self.lock_432.setAccessibleName("Lock pitch to A4 432")

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setToolTip("Reset tempo and pitch to defaults.")
        self.reset_btn.setAccessibleName("Reset tempo and pitch")

        form = QtWidgets.QFormLayout()
        form.addRow(self.tempo_label, self.tempo_slider)
        form.addRow(self.pitch_label, self.pitch_slider)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.key_lock)
        row.addWidget(self.tape_mode)
        row.addWidget(self.lock_432)
        row.addStretch(1)
        row.addWidget(self.reset_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(row)

        self.tempo_slider.valueChanged.connect(self._emit)
        self.pitch_slider.valueChanged.connect(self._emit)
        self.key_lock.toggled.connect(self._emit)
        self.tape_mode.toggled.connect(self._on_tape)
        self.lock_432.toggled.connect(self._on_lock_432)
        self.reset_btn.clicked.connect(self._on_reset)

        self._update_pitch_controls_enabled()
        self._emit()

    @property
    def _lock_432_semitones(self) -> float:
        return 12.0 * math.log2(432.0 / 440.0)

    def _on_tape(self, on: bool):
        if on:
            self.lock_432.setChecked(False)
        self._update_pitch_controls_enabled()
        self.key_lock.setEnabled(not on)
        self._emit()

    def _on_lock_432(self, on: bool):
        if on:
            self.pitch_slider.setValue(int(round(self._lock_432_semitones * 10)))
        self._update_pitch_controls_enabled()
        self._emit()

    def _update_pitch_controls_enabled(self):
        tape_on = self.tape_mode.isChecked()
        lock_on = self.lock_432.isChecked()
        self.pitch_slider.setEnabled(not tape_on and not lock_on)
        self.lock_432.setEnabled(not tape_on)

    def _on_reset(self):
        self.tempo_slider.setValue(100)
        self.pitch_slider.setValue(0)
        self.key_lock.setChecked(True)
        self.tape_mode.setChecked(False)
        self.lock_432.setChecked(False)

    def _emit(self):
        tempo = self.tempo_slider.value() / 100.0
        lock_432 = self.lock_432.isChecked()
        pitch = self._lock_432_semitones if lock_432 else self.pitch_slider.value() / 10.0
        key_lock = self.key_lock.isChecked()
        tape = self.tape_mode.isChecked()

        self.tempo_label.setText(f"Tempo: {tempo:.2f}×")
        if tape:
            st = 12.0 * math.log2(max(1e-6, tempo))
            self.pitch_label.setText(f"Pitch: {st:+.2f} st (linked)")
        elif lock_432:
            self.pitch_label.setText(f"Pitch: {pitch:+.2f} st (A4=432 Hz)")
        else:
            self.pitch_label.setText(f"Pitch: {pitch:+.1f} st")

        self.controlsChanged.emit(tempo, pitch, key_lock, tape, lock_432)



class EqualizerSlider(QtWidgets.QSlider):
    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Orientation.Vertical, parent)
        self.setRange(-120, 120)  # -12.0 to +12.0 dB in 0.1 steps
        self.setValue(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMinimumWidth(30)

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Retrieve palette colors
        palette = self.palette()
        bg_color = palette.color(QtGui.QPalette.ColorRole.Base).darker(110)
        active_color = palette.color(QtGui.QPalette.ColorRole.Highlight)
        handle_color = palette.color(QtGui.QPalette.ColorRole.Button)
        handle_border = palette.color(QtGui.QPalette.ColorRole.Mid)

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # Track geometry
        track_w = 4
        track_x = (w - track_w) / 2
        # Add some padding top/bottom so handle doesn't clip
        padding = 7 
        track_h = h - (padding * 2)
        track_y = padding

        # Draw full track background
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(bg_color)
        track_rect = QtCore.QRectF(track_x, track_y, track_w, track_h)
        painter.drawRoundedRect(track_rect, 2, 2)

        # Value mapping
        # Min (-120) at bottom, Max (+120) at top
        val = self.value()
        mn = self.minimum()
        mx = self.maximum()
        rng = mx - mn
        if rng == 0:
            return

        # Y position calculation (0 at bottom effectively for calc, then flip)
        # But Qt coords: 0 is top.
        # normalized position from bottom (0.0 to 1.0)
        norm_val = (val - mn) / rng
        center_norm = (0 - mn) / rng

        def get_y(norm):
            return track_y + track_h - (norm * track_h)

        y_now = get_y(norm_val)
        y_center = get_y(center_norm)

        # Draw active fill from center to current
        painter.setBrush(active_color)
        
        # Avoid drawing 0-height rect
        if val != 0:
            top = min(y_now, y_center)
            dist = abs(y_now - y_center)
            fill_rect = QtCore.QRectF(track_x, top, track_w, dist)
            painter.drawRoundedRect(fill_rect, 2, 2)
        
        # Center marker (small horizontal line behind track or on top?)
        # Let's draw it on the track background before fill? 
        # Actually a small tick mark at center is helpful.
        painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.ColorRole.PlaceholderText), 1))
        painter.drawLine(
            QtCore.QPointF(track_x - 4, y_center), 
            QtCore.QPointF(track_x + track_w + 4, y_center)
        )

        # Handle
        handle_size = 14
        handle_x = (w - handle_size) / 2
        handle_y_pos = y_now - (handle_size / 2)
        handle_rect = QtCore.QRectF(handle_x, handle_y_pos, handle_size, handle_size)

        gradient = QtGui.QLinearGradient(handle_rect.topLeft(), handle_rect.bottomLeft())
        gradient.setColorAt(0, handle_color.lighter(105))
        gradient.setColorAt(1, handle_color.darker(105))
        
        painter.setPen(QtGui.QPen(handle_border, 1.5))
        painter.setBrush(gradient)
        painter.drawEllipse(handle_rect)


class EqualizerWidget(QtWidgets.QGroupBox):
    gainsChanged = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__("Equalizer", parent)

        self.presets_map = {
            "Flat": [0.0] * 10,
            "Bass Boost": [6.0, 5.0, 4.0, 2.0, 0.0, -1.0, -2.0, -2.0, -2.0, -2.0],
            "Treble Boost": [-2.0, -2.0, -1.0, 0.0, 1.0, 3.0, 5.0, 6.0, 6.0, 6.0],
            "Vocal": [-2.0, -1.0, 0.0, 2.0, 4.0, 4.0, 2.0, 0.0, -1.0, -2.0],
            "Rock": [4.0, 3.0, 2.0, 0.0, -1.0, 1.0, 3.0, 4.0, 4.0, 3.0],
            "Pop": [-1.0, 0.0, 2.0, 3.0, 4.0, 2.0, 0.0, -1.0, -2.0, -2.0],
        }

        self.presets = QtWidgets.QComboBox()
        self.presets.addItems(list(self.presets_map.keys()) + ["Custom"])

        self.reset_btn = QtWidgets.QPushButton("Reset")

        self._gains_timer = QtCore.QTimer(self)
        self._gains_timer.setSingleShot(True)
        self._gains_timer.setInterval(75)
        self._gains_timer.timeout.connect(self._emit_gains)

        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.setToolTip("Save current settings as a new preset")
        
        self.clear_btn = QtWidgets.QPushButton("Clear Custom")
        self.clear_btn.setToolTip("Remove all custom presets")

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Presets"))
        header.addWidget(self.presets)
        header.addStretch(1)
        header.addWidget(self.save_btn)
        header.addWidget(self.clear_btn)
        header.addWidget(self.reset_btn)

        bands = ["31", "62", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
        self.band_sliders: List[EqualizerSlider] = []
        self.db_labels: List[QtWidgets.QLabel] = []

        sliders_layout = QtWidgets.QHBoxLayout()
        sliders_layout.setSpacing(8)
        
        for band in bands:
            slider = EqualizerSlider()
            slider.setToolTip(f"{band} Hz band")
            slider.setAccessibleName(f"{band} Hz band")

            db_label = QtWidgets.QLabel("0.0")
            db_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            font = db_label.font()
            font.setPointSize(9)
            db_label.setFont(font)
            # Use placeholder color or slightly dimmer
            
            band_label = QtWidgets.QLabel(band)
            band_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            band_label.setStyleSheet("color: #888888; font-weight: bold;")

            column = QtWidgets.QVBoxLayout()
            column.setSpacing(4)
            column.addWidget(db_label)
            column.addWidget(slider, 1, QtCore.Qt.AlignmentFlag.AlignHCenter)
            column.addWidget(band_label)
            sliders_layout.addLayout(column, 1) # Equal stretch

            self.band_sliders.append(slider)
            self.db_labels.append(db_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.addLayout(header)
        layout.addSpacing(10)
        layout.addLayout(sliders_layout)

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)

        self.reset_btn.clicked.connect(self._on_reset)
        self.save_btn.clicked.connect(self._on_save_preset)
        self.clear_btn.clicked.connect(self._on_clear_custom)
        self.presets.currentTextChanged.connect(self._on_preset_changed)
        
        for i, slider in enumerate(self.band_sliders):
            # Connect using closure to capture index
            slider.valueChanged.connect(lambda val, idx=i: self._on_slider_val_changed(idx, val))
            slider.sliderReleased.connect(self._on_slider_released)

        self._load_custom_presets()
        self._apply_gains(self.presets_map["Flat"], emit=False)

    def _load_custom_presets(self):
        try:
            path = os.path.join(os.getcwd(), "equalizer_presets.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    custom = json.load(f)
                    if isinstance(custom, dict):
                        for name, gains in custom.items():
                            if isinstance(gains, list) and len(gains) == 10:
                                self.presets_map[name] = gains
                                self.presets.addItem(name)
        except Exception:
            pass  # Ignore errors loading presets

    def _save_custom_presets(self):
        try:
            path = os.path.join(os.getcwd(), "equalizer_presets.json")
            # Filter out built-in presets
            base_presets = ["Flat", "Bass Boost", "Treble Boost", "Vocal", "Rock", "Pop"]
            to_save = {
                k: v for k, v in self.presets_map.items() 
                if k not in base_presets
            }
            with open(path, "w") as f:
                json.dump(to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")

    def _on_save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Preset", "Enter preset name:",
            QtWidgets.QLineEdit.EchoMode.Normal
        )
        if ok and name:
            name = name.strip()
            if not name:
                return
            gains = self.gains()
            self.presets_map[name] = gains
            
            # Update combo box if new
            if self.presets.findText(name) == -1:
                self.presets.addItem(name)
            
            self.presets.setCurrentText(name)
            self._save_custom_presets()

    def _on_clear_custom(self):
        # Confirm action
        reply = QtWidgets.QMessageBox.question(
            self, "Clear Custom Presets",
            "Are you sure you want to remove all custom presets?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # Remove from map and combo
        base_presets = ["Flat", "Bass Boost", "Treble Boost", "Vocal", "Rock", "Pop"]
        
        # Identify custom keys
        custom_keys = [k for k in self.presets_map.keys() if k not in base_presets]
        
        # Remove from map
        for k in custom_keys:
            del self.presets_map[k]
            
        # Remove from combo
        # We can iterate backwards to remove safely or just rebuild
        # Rebuilding is safer/easier
        self.presets.blockSignals(True)
        self.presets.clear()
        self.presets.addItems(list(self.presets_map.keys()) + ["Custom"])
        self.presets.setCurrentText("Flat") # Reset to flat
        self.presets.blockSignals(False)
        
        # Update file
        self._save_custom_presets()
        # Also apply flat since we reset selection
        self._on_reset()

    def _on_reset(self):
        self.presets.setCurrentText("Flat")

    def _on_preset_changed(self, name: str):
        if name in self.presets_map:
            self._apply_gains(self.presets_map[name], emit=True)
        else:
            self._emit_gains()

    def _on_slider_val_changed(self, index: int, value: int):
        # Update label
        db_val = value / 10.0
        self.db_labels[index].setText(f"{db_val:+.1f}")
        
        # Check Custom preset status
        current = self.presets.currentText()
        if current in self.presets_map and not self._gains_match(self.presets_map[current]):
            self.presets.blockSignals(True)
            self.presets.setCurrentText("Custom")
            self.presets.blockSignals(False)
        
        self._gains_timer.start()

    def _on_slider_released(self):
        self._gains_timer.stop()
        self._emit_gains()

    def _apply_gains(self, gains: list[float], emit: bool = True):
        if len(gains) != len(self.band_sliders):
            return
        for i, (slider, gain) in enumerate(zip(self.band_sliders, gains)):
            slider.blockSignals(True)
            val = int(round(clamp(float(gain), -12.0, 12.0) * 10))
            slider.setValue(val)
            self.db_labels[i].setText(f"{val/10.0:+.1f}")
            slider.blockSignals(False)
        if emit:
            self._gains_timer.stop()
            self._emit_gains()

    def _emit_gains(self):
        self.gainsChanged.emit(self.gains())

    def gains(self) -> list[float]:
        return [slider.value() / 10.0 for slider in self.band_sliders]

    def set_gains(self, gains: list[float], preset: Optional[str] = None, emit: bool = False):
        if preset:
            if preset in self.presets_map:
                self.presets.blockSignals(True)
                self.presets.setCurrentText(preset)
                self.presets.blockSignals(False)
                self._apply_gains(self.presets_map[preset], emit=emit)
                return
            self.presets.blockSignals(True)
            self.presets.setCurrentText("Custom")
            self.presets.blockSignals(False)
        self._apply_gains(gains, emit=emit)

    def _gains_match(self, gains: list[float]) -> bool:
        if len(gains) != len(self.band_sliders):
            return False
        # Compare with tolerance due to float conversion or slider step
        current_gains = self.gains()
        return all(abs(g - cg) < 0.15 for g, cg in zip(gains, current_gains))


class ReverbWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float, float)  # decay_sec, pre_delay_ms, wet

    def __init__(self, parent=None):
        super().__init__("Reverb", parent)

        self.decay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.decay_slider.setRange(20, 600)
        self.decay_slider.setValue(140)
        self.decay_slider.setToolTip("Adjust decay time (0.2s to 6.0s).")
        self.decay_slider.setAccessibleName("Reverb decay time")

        self.predelay_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.predelay_slider.setRange(0, 120)
        self.predelay_slider.setValue(20)
        self.predelay_slider.setToolTip("Adjust pre-delay (0ms to 120ms).")
        self.predelay_slider.setAccessibleName("Reverb pre-delay")

        self.mix_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(25)
        self.mix_slider.setToolTip("Adjust wet/dry mix (0% dry to 100% wet).")
        self.mix_slider.setAccessibleName("Reverb wet/dry mix")

        self.decay_label = QtWidgets.QLabel("Decay: 1.40s")
        self.predelay_label = QtWidgets.QLabel("Pre-delay: 20 ms")
        self.mix_label = QtWidgets.QLabel("Mix: 25%")

        form = QtWidgets.QFormLayout()
        form.addRow(self.decay_label, self.decay_slider)
        form.addRow(self.predelay_label, self.predelay_slider)
        form.addRow(self.mix_label, self.mix_slider)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)

        self.decay_slider.valueChanged.connect(self._emit)
        self.predelay_slider.valueChanged.connect(self._emit)
        self.mix_slider.valueChanged.connect(self._emit)

        self._emit()

    def _emit(self):
        decay = self.decay_slider.value() / 100.0
        predelay = float(self.predelay_slider.value())
        mix = self.mix_slider.value() / 100.0

        self.decay_label.setText(f"Decay: {decay:.2f}s")
        self.predelay_label.setText(f"Pre-delay: {predelay:.0f} ms")
        self.mix_label.setText(f"Mix: {mix * 100:.0f}%")

        self.controlsChanged.emit(decay, predelay, mix)


class ChorusWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float, float)  # rate_hz, depth_ms, mix

    def __init__(self, parent=None):
        super().__init__("Chorus", parent)

        self.rate_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.rate_slider.setRange(5, 500)
        self.rate_slider.setValue(80)
        self.rate_slider.setToolTip("Adjust LFO rate (0.05 Hz to 5.00 Hz).")
        self.rate_slider.setAccessibleName("Chorus rate")

        self.depth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.depth_slider.setRange(0, 200)
        self.depth_slider.setValue(80)
        self.depth_slider.setToolTip("Adjust modulation depth (0ms to 20ms).")
        self.depth_slider.setAccessibleName("Chorus depth")

        self.mix_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(25)
        self.mix_slider.setToolTip("Adjust wet/dry mix (0% dry to 100% wet).")
        self.mix_slider.setAccessibleName("Chorus wet/dry mix")

        self.rate_label = QtWidgets.QLabel("Rate: 0.80 Hz")
        self.depth_label = QtWidgets.QLabel("Depth: 8.0 ms")
        self.mix_label = QtWidgets.QLabel("Mix: 25%")

        form = QtWidgets.QFormLayout()
        form.addRow(self.rate_label, self.rate_slider)
        form.addRow(self.depth_label, self.depth_slider)
        form.addRow(self.mix_label, self.mix_slider)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)

        self.rate_slider.valueChanged.connect(self._emit)
        self.depth_slider.valueChanged.connect(self._emit)
        self.mix_slider.valueChanged.connect(self._emit)

        self._emit()

    def _emit(self):
        rate = self.rate_slider.value() / 100.0
        depth = self.depth_slider.value() / 10.0
        mix = self.mix_slider.value() / 100.0

        self.rate_label.setText(f"Rate: {rate:.2f} Hz")
        self.depth_label.setText(f"Depth: {depth:.1f} ms")
        self.mix_label.setText(f"Mix: {mix * 100:.0f}%")

        self.controlsChanged.emit(rate, depth, mix)


class StereoWidthWidget(QtWidgets.QGroupBox):
    widthChanged = QtCore.Signal(float)  # width 0..2

    def __init__(self, parent=None):
        super().__init__("Stereo Width", parent)

        self.width_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.width_slider.setRange(0, 200)
        self.width_slider.setValue(100)
        self.width_slider.setToolTip("Adjust stereo width (0% mono to 200% wide).")
        self.width_slider.setAccessibleName("Stereo width")

        self.width_label = QtWidgets.QLabel("Width: 100%")

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(self.width_label, self.width_slider)

        self.width_slider.valueChanged.connect(self._emit)
        self._emit()

    def _emit(self):
        width = self.width_slider.value() / 100.0
        self.width_label.setText(f"Width: {width * 100:.0f}%")
        self.widthChanged.emit(width)


class StereoPannerWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float)  # azimuth, spread

    def __init__(self, parent=None):
        super().__init__("Stereo Panner", parent)

        self.azimuth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.azimuth_slider.setRange(-90, 90)
        self.azimuth_slider.setValue(0)
        self.azimuth_slider.setToolTip("Pan left/right (-90° left to 90° right).")
        self.azimuth_slider.setAccessibleName("Stereo panner azimuth")

        self.spread_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.spread_slider.setRange(0, 100)
        self.spread_slider.setValue(100)
        self.spread_slider.setToolTip("Adjust source spread (0% mono to 100% wide).")
        self.spread_slider.setAccessibleName("Stereo panner spread")

        self.azimuth_label = QtWidgets.QLabel("Azimuth: 0°")
        self.spread_label = QtWidgets.QLabel("Spread: 100%")

        form = QtWidgets.QFormLayout(self)
        form.addRow(self.azimuth_label, self.azimuth_slider)
        form.addRow(self.spread_label, self.spread_slider)

        self.azimuth_slider.valueChanged.connect(self._emit)
        self.spread_slider.valueChanged.connect(self._emit)
        self._emit()

    def _emit(self):
        azimuth = float(self.azimuth_slider.value())
        spread = self.spread_slider.value() / 100.0
        self.azimuth_label.setText(f"Azimuth: {azimuth:.0f}°")
        self.spread_label.setText(f"Spread: {spread * 100:.0f}%")
        self.controlsChanged.emit(azimuth, spread)


class DynamicEqWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float, float, float, float)

    def __init__(self, parent=None):
        super().__init__("Dynamic EQ", parent)

        self._freq_min = 20.0
        self._freq_max = 20000.0

        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, 1000)
        self.freq_slider.setValue(self._freq_to_slider(1000.0))
        self.freq_slider.setToolTip("Set center frequency (20 Hz to 20 kHz).")
        self.freq_slider.setAccessibleName("Dynamic EQ frequency")

        self.q_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.q_slider.setRange(10, 200)
        self.q_slider.setValue(10)
        self.q_slider.setToolTip("Set Q (0.1 to 20.0).")
        self.q_slider.setAccessibleName("Dynamic EQ Q")

        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-120, 120)
        self.gain_slider.setValue(0)
        self.gain_slider.setToolTip("Set base gain (-12 dB to +12 dB).")
        self.gain_slider.setAccessibleName("Dynamic EQ gain")

        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(-600, 0)
        self.threshold_slider.setValue(-240)
        self.threshold_slider.setToolTip("Set threshold (-60 dB to 0 dB).")
        self.threshold_slider.setAccessibleName("Dynamic EQ threshold")

        self.ratio_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(10, 200)
        self.ratio_slider.setValue(40)
        self.ratio_slider.setToolTip("Set ratio (1:1 to 20:1).")
        self.ratio_slider.setAccessibleName("Dynamic EQ ratio")

        self.freq_label = QtWidgets.QLabel("Freq: 1.00 kHz")
        self.q_label = QtWidgets.QLabel("Q: 1.0")
        self.gain_label = QtWidgets.QLabel("Gain: +0.0 dB")
        self.threshold_label = QtWidgets.QLabel("Threshold: -24.0 dB")
        self.ratio_label = QtWidgets.QLabel("Ratio: 4.0:1")

        form = QtWidgets.QFormLayout()
        form.addRow(self.freq_label, self.freq_slider)
        form.addRow(self.q_label, self.q_slider)
        form.addRow(self.gain_label, self.gain_slider)
        form.addRow(self.threshold_label, self.threshold_slider)
        form.addRow(self.ratio_label, self.ratio_slider)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)

        self.freq_slider.valueChanged.connect(self._emit)
        self.q_slider.valueChanged.connect(self._emit)
        self.gain_slider.valueChanged.connect(self._emit)
        self.threshold_slider.valueChanged.connect(self._emit)
        self.ratio_slider.valueChanged.connect(self._emit)

        self._emit()

    def _freq_to_slider(self, freq: float) -> int:
        freq = clamp(float(freq), self._freq_min, self._freq_max)
        log_min = math.log10(self._freq_min)
        log_max = math.log10(self._freq_max)
        pos = (math.log10(freq) - log_min) / (log_max - log_min)
        return int(round(pos * 1000.0))

    def _slider_to_freq(self, value: int) -> float:
        pos = clamp(float(value) / 1000.0, 0.0, 1.0)
        log_min = math.log10(self._freq_min)
        log_max = math.log10(self._freq_max)
        return float(10.0 ** (log_min + pos * (log_max - log_min)))

    def _format_freq(self, freq: float) -> str:
        if freq >= 1000.0:
            return f"{freq / 1000.0:.2f} kHz"
        return f"{freq:.0f} Hz"

    def _emit(self):
        freq = self._slider_to_freq(self.freq_slider.value())
        q = self.q_slider.value() / 10.0
        gain = self.gain_slider.value() / 10.0
        threshold = self.threshold_slider.value() / 10.0
        ratio = self.ratio_slider.value() / 10.0

        self.freq_label.setText(f"Freq: {self._format_freq(freq)}")
        self.q_label.setText(f"Q: {q:.1f}")
        self.gain_label.setText(f"Gain: {gain:+.1f} dB")
        self.threshold_label.setText(f"Threshold: {threshold:.1f} dB")
        self.ratio_label.setText(f"Ratio: {ratio:.1f}:1")

        self.controlsChanged.emit(freq, q, gain, threshold, ratio)


class CompressorWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float, float, float, float)

    def __init__(self, parent=None):
        super().__init__("Compressor", parent)

        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(-600, 0)
        self.threshold_slider.setValue(-180)
        self.threshold_slider.setToolTip("Set threshold (-60 dB to 0 dB).")
        self.threshold_slider.setAccessibleName("Compressor threshold")

        self.ratio_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(10, 200)
        self.ratio_slider.setValue(40)
        self.ratio_slider.setToolTip("Set ratio (1:1 to 20:1).")
        self.ratio_slider.setAccessibleName("Compressor ratio")

        self.attack_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.attack_slider.setRange(1, 2000)
        self.attack_slider.setValue(100)
        self.attack_slider.setToolTip("Set attack time (0.1 ms to 200 ms).")
        self.attack_slider.setAccessibleName("Compressor attack")

        self.release_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.release_slider.setRange(1, 1000)
        self.release_slider.setValue(120)
        self.release_slider.setToolTip("Set release time (1 ms to 1000 ms).")
        self.release_slider.setAccessibleName("Compressor release")

        self.makeup_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.makeup_slider.setRange(0, 240)
        self.makeup_slider.setValue(0)
        self.makeup_slider.setToolTip("Set makeup gain (0 dB to 24 dB).")
        self.makeup_slider.setAccessibleName("Compressor makeup gain")

        self.threshold_label = QtWidgets.QLabel("Threshold: -18.0 dB")
        self.ratio_label = QtWidgets.QLabel("Ratio: 4.0:1")
        self.attack_label = QtWidgets.QLabel("Attack: 10.0 ms")
        self.release_label = QtWidgets.QLabel("Release: 120 ms")
        self.makeup_label = QtWidgets.QLabel("Makeup: 0.0 dB")

        self.meter_label = QtWidgets.QLabel("Gain Reduction: 0.0 dB")
        self.meter = QtWidgets.QProgressBar()
        self.meter.setRange(0, 240)
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self._meter_provider = None
        self._meter_timer = QtCore.QTimer(self)
        self._meter_timer.setInterval(80)
        self._meter_timer.timeout.connect(self._update_meter)

        form = QtWidgets.QFormLayout()
        form.addRow(self.threshold_label, self.threshold_slider)
        form.addRow(self.ratio_label, self.ratio_slider)
        form.addRow(self.attack_label, self.attack_slider)
        form.addRow(self.release_label, self.release_slider)
        form.addRow(self.makeup_label, self.makeup_slider)

        meter_layout = QtWidgets.QVBoxLayout()
        meter_layout.addWidget(self.meter_label)
        meter_layout.addWidget(self.meter)
        meter_box = QtWidgets.QWidget()
        meter_box.setLayout(meter_layout)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(meter_box)

        self.threshold_slider.valueChanged.connect(self._emit)
        self.ratio_slider.valueChanged.connect(self._emit)
        self.attack_slider.valueChanged.connect(self._emit)
        self.release_slider.valueChanged.connect(self._emit)
        self.makeup_slider.valueChanged.connect(self._emit)

        self._emit()
        self._set_meter_visible(False)

    def set_meter_provider(self, provider) -> None:
        self._meter_provider = provider
        self._set_meter_visible(provider is not None)
        if provider is not None:
            self._meter_timer.start()
        else:
            self._meter_timer.stop()

    def _set_meter_visible(self, visible: bool) -> None:
        self.meter_label.setVisible(visible)
        self.meter.setVisible(visible)

    def _emit(self):
        threshold = self.threshold_slider.value() / 10.0
        ratio = self.ratio_slider.value() / 10.0
        attack = self.attack_slider.value() / 10.0
        release = float(self.release_slider.value())
        makeup = self.makeup_slider.value() / 10.0

        self.threshold_label.setText(f"Threshold: {threshold:.1f} dB")
        self.ratio_label.setText(f"Ratio: {ratio:.1f}:1")
        self.attack_label.setText(f"Attack: {attack:.1f} ms")
        self.release_label.setText(f"Release: {release:.0f} ms")
        self.makeup_label.setText(f"Makeup: {makeup:.1f} dB")

        self.controlsChanged.emit(threshold, ratio, attack, release, makeup)

    def _update_meter(self) -> None:
        if not self._meter_provider:
            return
        value = self._meter_provider()
        if value is None:
            self._set_meter_visible(False)
            return
        reduction_db = clamp(float(value), 0.0, 24.0)
        self.meter.setValue(int(round(reduction_db * 10)))
        self.meter_label.setText(f"Gain Reduction: {reduction_db:.1f} dB")


class SaturationWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float, float, bool)

    def __init__(self, parent=None):
        super().__init__("Saturation", parent)

        self.drive_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.drive_slider.setRange(0, 240)
        self.drive_slider.setValue(60)
        self.drive_slider.setToolTip("Set saturation drive (0 dB to 24 dB).")
        self.drive_slider.setAccessibleName("Saturation drive")

        self.trim_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.trim_slider.setRange(-240, 240)
        self.trim_slider.setValue(0)
        self.trim_slider.setToolTip("Set output trim (-24 dB to +24 dB).")
        self.trim_slider.setAccessibleName("Saturation output trim")

        self.tone_toggle = QtWidgets.QCheckBox("Tone shaping")
        self.tone_toggle.setChecked(False)
        self.tone_toggle.setToolTip("Enable tone shaping after saturation.")
        self.tone_toggle.setAccessibleName("Saturation tone toggle")

        self.tone_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.tone_slider.setRange(-100, 100)
        self.tone_slider.setValue(0)
        self.tone_slider.setToolTip("Adjust tone (darker to brighter).")
        self.tone_slider.setAccessibleName("Saturation tone")

        self.drive_label = QtWidgets.QLabel("Drive: +6.0 dB")
        self.trim_label = QtWidgets.QLabel("Trim: +0.0 dB")
        self.tone_label = QtWidgets.QLabel("Tone: 0%")

        form = QtWidgets.QFormLayout()
        form.addRow(self.drive_label, self.drive_slider)
        form.addRow(self.trim_label, self.trim_slider)
        form.addRow(self.tone_label, self.tone_slider)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tone_toggle)
        layout.addLayout(form)

        self.drive_slider.valueChanged.connect(self._emit)
        self.trim_slider.valueChanged.connect(self._emit)
        self.tone_slider.valueChanged.connect(self._emit)
        self.tone_toggle.toggled.connect(self._emit)

        self._emit()

    def _emit(self):
        drive_db = self.drive_slider.value() / 10.0
        trim_db = self.trim_slider.value() / 10.0
        tone = self.tone_slider.value() / 100.0
        tone_enabled = self.tone_toggle.isChecked()

        self.drive_label.setText(f"Drive: {drive_db:+.1f} dB")
        self.trim_label.setText(f"Trim: {trim_db:+.1f} dB")
        self.tone_slider.setEnabled(tone_enabled)
        if tone_enabled:
            self.tone_label.setText(f"Tone: {tone * 100:+.0f}%")
        else:
            self.tone_label.setText("Tone: Off")

        self.controlsChanged.emit(drive_db, trim_db, tone, tone_enabled)


class SubharmonicWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float, float)  # mix, intensity, cutoff_hz

    def __init__(self, parent=None):
        super().__init__("Subharmonic", parent)

        self.mix_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(25)
        self.mix_slider.setToolTip("Blend the octave-down layer with the original.")
        self.mix_slider.setAccessibleName("Subharmonic mix")

        self.intensity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 150)
        self.intensity_slider.setValue(60)
        self.intensity_slider.setToolTip("Set the subharmonic intensity.")
        self.intensity_slider.setAccessibleName("Subharmonic intensity")

        self.cutoff_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.cutoff_slider.setRange(60, 240)
        self.cutoff_slider.setValue(140)
        self.cutoff_slider.setToolTip("Low-pass cutoff for the generated sub (Hz).")
        self.cutoff_slider.setAccessibleName("Subharmonic low-pass cutoff")

        self.mix_label = QtWidgets.QLabel("Mix: 25%")
        self.intensity_label = QtWidgets.QLabel("Intensity: 60%")
        self.cutoff_label = QtWidgets.QLabel("Low-pass: 140 Hz")

        form = QtWidgets.QFormLayout()
        form.addRow(self.mix_label, self.mix_slider)
        form.addRow(self.intensity_label, self.intensity_slider)
        form.addRow(self.cutoff_label, self.cutoff_slider)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)

        self.mix_slider.valueChanged.connect(self._emit)
        self.intensity_slider.valueChanged.connect(self._emit)
        self.cutoff_slider.valueChanged.connect(self._emit)

        self._emit()

    def _emit(self):
        mix = self.mix_slider.value() / 100.0
        intensity = self.intensity_slider.value() / 100.0
        cutoff = float(self.cutoff_slider.value())

        self.mix_label.setText(f"Mix: {mix * 100:.0f}%")
        self.intensity_label.setText(f"Intensity: {intensity * 100:.0f}%")
        self.cutoff_label.setText(f"Low-pass: {cutoff:.0f} Hz")

        self.controlsChanged.emit(mix, intensity, cutoff)


class LimiterWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, object)

    def __init__(self, parent=None):
        super().__init__("Limiter", parent)

        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(-600, 0)
        self.threshold_slider.setValue(-10)
        self.threshold_slider.setToolTip("Set limiter threshold (-60 dB to 0 dB).")
        self.threshold_slider.setAccessibleName("Limiter threshold")

        self.release_toggle = QtWidgets.QCheckBox("Release smoothing")
        self.release_toggle.setChecked(True)
        self.release_toggle.setToolTip("Enable release smoothing (optional).")
        self.release_toggle.setAccessibleName("Limiter release toggle")

        self.release_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.release_slider.setRange(1, 1000)
        self.release_slider.setValue(80)
        self.release_slider.setToolTip("Set release time (1 ms to 1000 ms).")
        self.release_slider.setAccessibleName("Limiter release time")

        self.threshold_label = QtWidgets.QLabel("Threshold: -1.0 dB")
        self.release_label = QtWidgets.QLabel("Release: 80 ms")

        form = QtWidgets.QFormLayout()
        form.addRow(self.threshold_label, self.threshold_slider)
        form.addRow(self.release_label, self.release_slider)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.release_toggle)
        layout.addLayout(form)

        self.threshold_slider.valueChanged.connect(self._emit)
        self.release_slider.valueChanged.connect(self._emit)
        self.release_toggle.toggled.connect(self._emit)

        self._emit()

    def _emit(self):
        threshold = self.threshold_slider.value() / 10.0
        use_release = self.release_toggle.isChecked()
        release_ms = float(self.release_slider.value())

        self.threshold_label.setText(f"Threshold: {threshold:.1f} dB")
        self.release_slider.setEnabled(use_release)
        if use_release:
            self.release_label.setText(f"Release: {release_ms:.0f} ms")
            release_value = release_ms
        else:
            self.release_label.setText("Release: Off")
            release_value = None

        self.controlsChanged.emit(threshold, release_value)


class TransportWidget(QtWidgets.QWidget):
    playPauseToggled = QtCore.Signal(bool)
    stopClicked = QtCore.Signal()
    prevClicked = QtCore.Signal()
    nextClicked = QtCore.Signal()
    seekRequested = QtCore.Signal(float)  # fraction 0..1
    volumeChanged = QtCore.Signal(float)
    muteToggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._transport_icon_size = 18

        self.prev_btn = QtWidgets.QToolButton()
        self.play_pause_btn = QtWidgets.QToolButton()
        self.play_pause_btn.setCheckable(True)
        self.stop_btn = QtWidgets.QToolButton()
        self.next_btn = QtWidgets.QToolButton()
        transport_buttons = [
            self.prev_btn,
            self.play_pause_btn,
            self.stop_btn,
            self.next_btn,
        ]
        for button in transport_buttons:
            button.setMinimumSize(36, 36)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setIconSize(QtCore.QSize(self._transport_icon_size, self._transport_icon_size))
        self.prev_btn.setToolTip("Previous track (Ctrl+P).")
        self.prev_btn.setAccessibleName("Previous track")
        self.play_pause_btn.setToolTip("Play/Pause (Space).")
        self.play_pause_btn.setAccessibleName("Play/Pause")
        self.stop_btn.setToolTip("Stop playback.")
        self.stop_btn.setAccessibleName("Stop")
        self.next_btn.setToolTip("Next track (Ctrl+N).")
        self.next_btn.setAccessibleName("Next track")

        self.pos_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pos_slider.setRange(0, 1000)
        self.pos_slider.setValue(0)
        self.pos_slider.setToolTip("Seek position.")
        self.pos_slider.setAccessibleName("Seek position")

        self.time_label = QtWidgets.QLabel("0:00 / 0:00")
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(120)
        self.volume_slider.setToolTip("Adjust volume.")
        self.volume_slider.setAccessibleName("Volume")

        self.mute_btn = QtWidgets.QToolButton()
        self.mute_btn.setCheckable(True)
        self.mute_btn.setToolTip("Mute audio.")
        self.mute_btn.setAccessibleName("Mute")
        self.mute_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.mute_btn.setIconSize(QtCore.QSize(self._transport_icon_size, self._transport_icon_size))

        self._update_transport_icons()

        btns = QtWidgets.QHBoxLayout()
        for b in [self.prev_btn, self.play_pause_btn, self.stop_btn, self.next_btn]:
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(QtWidgets.QLabel("Vol"))
        btns.addWidget(self.volume_slider)
        btns.addWidget(self.mute_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(btns)

        seek_row = QtWidgets.QHBoxLayout()
        seek_row.addWidget(self.pos_slider, 1)
        seek_row.addWidget(self.time_label)
        layout.addLayout(seek_row)

        self.prev_btn.clicked.connect(self.prevClicked)
        self.play_pause_btn.toggled.connect(self.playPauseToggled)
        self.play_pause_btn.toggled.connect(self._update_transport_icons)
        self.stop_btn.clicked.connect(self.stopClicked)
        self.next_btn.clicked.connect(self.nextClicked)

        self.volume_slider.valueChanged.connect(lambda v: self.volumeChanged.emit(v / 100.0))
        self.mute_btn.toggled.connect(self._on_mute)

        self._dragging = False
        self.pos_slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self.pos_slider.sliderReleased.connect(self._on_seek_end)

    def changeEvent(self, event: QtCore.QEvent):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.PaletteChange, QtCore.QEvent.Type.StyleChange):
            self._update_transport_icons()

    def _icon_color_for_button(self, button: QtWidgets.QToolButton) -> QtGui.QColor:
        if button.isChecked():
            return QtGui.QColor("#0b0b0b")
        return self.palette().color(QtGui.QPalette.ColorRole.ButtonText)

    def _standard_icon(
        self,
        name: str,
        fallback: QtWidgets.QStyle.StandardPixmap,
    ) -> QtGui.QIcon:
        enum = getattr(QtWidgets.QStyle.StandardPixmap, name, fallback)
        return self.style().standardIcon(enum)

    def set_play_pause_state(self, playing: bool) -> None:
        if self.play_pause_btn.isChecked() != playing:
            self.play_pause_btn.blockSignals(True)
            self.play_pause_btn.setChecked(playing)
            self.play_pause_btn.blockSignals(False)
            self._update_transport_icons()

    def _update_transport_icons(self, *_: object) -> None:
        if QtSvg is None:
            self.prev_btn.setIcon(
                self._standard_icon(
                    "SP_MediaSkipBackward",
                    QtWidgets.QStyle.StandardPixmap.SP_ArrowLeft,
                )
            )
            self.next_btn.setIcon(
                self._standard_icon(
                    "SP_MediaSkipForward",
                    QtWidgets.QStyle.StandardPixmap.SP_ArrowRight,
                )
            )
            play_enum = "SP_MediaPause" if self.play_pause_btn.isChecked() else "SP_MediaPlay"
            self.play_pause_btn.setIcon(
                self._standard_icon(play_enum, QtWidgets.QStyle.StandardPixmap.SP_ArrowRight)
            )
            self.stop_btn.setIcon(
                self._standard_icon("SP_MediaStop", QtWidgets.QStyle.StandardPixmap.SP_DialogCloseButton)
            )
            mute_enum = "SP_MediaVolumeMuted" if self.mute_btn.isChecked() else "SP_MediaVolume"
            self.mute_btn.setIcon(
                self._standard_icon(mute_enum, QtWidgets.QStyle.StandardPixmap.SP_ArrowRight)
            )
            return

        size_px = self._transport_icon_size
        self.prev_btn.setIcon(
            render_svg_icon(
                SVG_ICON_TEMPLATES["prev"],
                self._icon_color_for_button(self.prev_btn),
                size_px,
            )
        )
        self.next_btn.setIcon(
            render_svg_icon(
                SVG_ICON_TEMPLATES["next"],
                self._icon_color_for_button(self.next_btn),
                size_px,
            )
        )
        play_key = "pause" if self.play_pause_btn.isChecked() else "play"
        self.play_pause_btn.setIcon(
            render_svg_icon(
                SVG_ICON_TEMPLATES[play_key],
                self._icon_color_for_button(self.play_pause_btn),
                size_px,
            )
        )
        self.stop_btn.setIcon(
            render_svg_icon(
                SVG_ICON_TEMPLATES["stop"],
                self._icon_color_for_button(self.stop_btn),
                size_px,
            )
        )
        mute_key = "volume_off" if self.mute_btn.isChecked() else "volume_on"
        self.mute_btn.setIcon(
            render_svg_icon(
                SVG_ICON_TEMPLATES[mute_key],
                self._icon_color_for_button(self.mute_btn),
                size_px,
            )
        )

    def _on_mute(self, on: bool):
        self._update_transport_icons()
        self.muteToggled.emit(on)

    def _on_seek_end(self):
        self._dragging = False
        frac = self.pos_slider.value() / 1000.0
        self.seekRequested.emit(frac)

    def set_play_pause_state(self, playing: bool):
        self.play_pause_btn.blockSignals(True)
        self.play_pause_btn.setChecked(playing)
        self.play_pause_btn.blockSignals(False)
        self._update_transport_icons()

    def set_time(self, pos_sec: float, dur_sec: float):
        self.time_label.setText(f"{format_time(pos_sec)} / {format_time(dur_sec)}")
        if dur_sec > 0 and not self._dragging:
            frac = clamp(pos_sec / dur_sec, 0.0, 1.0)
            self.pos_slider.setValue(int(round(frac * 1000)))


class PlaylistWidget(QtWidgets.QWidget):
    addFilesRequested = QtCore.Signal()
    addFolderRequested = QtCore.Signal()
    clearRequested = QtCore.Signal()
    trackActivated = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        header = QtWidgets.QLabel("Playlist")
        header.setObjectName("playlist_header")

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        row_height = self.list.fontMetrics().height() + 4
        self._row_height = row_height
        self.list.setIconSize(QtCore.QSize(row_height, row_height))
        self.list.setUniformItemSizes(True)
        self.list.setSpacing(0)
        self.list.setItemDelegate(PlaylistItemDelegate(self.list))

        self._action_icon_size = 14
        self._add_files_btn = QtWidgets.QToolButton()
        self._add_files_btn.setText("Files")
        self._add_files_btn.setToolTip("Add files")
        self._add_files_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._add_files_btn.setAutoRaise(True)

        self._add_folder_btn = QtWidgets.QToolButton()
        self._add_folder_btn.setText("Folder")
        self._add_folder_btn.setToolTip("Add folder")
        self._add_folder_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._add_folder_btn.setAutoRaise(True)

        self._clear_btn = QtWidgets.QToolButton()
        self._clear_btn.setText("Clear")
        self._clear_btn.setToolTip("Clear playlist")
        self._clear_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._clear_btn.setAutoRaise(True)

        self._update_action_icons()

        for btn in (self._add_files_btn, self._add_folder_btn, self._clear_btn):
            btn.setIconSize(QtCore.QSize(self._action_icon_size, self._action_icon_size))

        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(header)
        header_row.addStretch(1)
        header_row.addWidget(self._add_files_btn)
        header_row.addWidget(self._add_folder_btn)
        header_row.addWidget(self._clear_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header_row)
        layout.addWidget(self.list, 1)

        self._add_files_btn.clicked.connect(self.addFilesRequested)
        self._add_folder_btn.clicked.connect(self.addFolderRequested)
        self._clear_btn.clicked.connect(self.clearRequested)

        self.list.itemDoubleClicked.connect(self._on_double)

        self.setAcceptDrops(True)
        self._dropped_paths: List[str] = []
        self._items_by_path: dict[str, list[QtWidgets.QListWidgetItem]] = {}
        self._playing_item: Optional[QtWidgets.QListWidgetItem] = None

    def changeEvent(self, event: QtCore.QEvent):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.PaletteChange, QtCore.QEvent.Type.StyleChange):
            self._update_action_icons()

    def _update_action_icons(self) -> None:
        if QtSvg is None:
            style = self.style()
            self._add_files_btn.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon))
            self._add_folder_btn.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon))
            reset_enum = getattr(
                QtWidgets.QStyle.StandardPixmap,
                "SP_DialogResetButton",
                QtWidgets.QStyle.StandardPixmap.SP_DialogCloseButton,
            )
            clear_icon_enum = getattr(QtWidgets.QStyle.StandardPixmap, "SP_TrashIcon", reset_enum)
            self._clear_btn.setIcon(style.standardIcon(clear_icon_enum))
            return

        color = self.palette().color(QtGui.QPalette.ColorRole.ButtonText)
        size_px = self._action_icon_size
        self._add_files_btn.setIcon(render_svg_icon(SVG_ICON_TEMPLATES["file"], color, size_px))
        self._add_folder_btn.setIcon(render_svg_icon(SVG_ICON_TEMPLATES["folder"], color, size_px))
        self._clear_btn.setIcon(render_svg_icon(SVG_ICON_TEMPLATES["trash"], color, size_px))

    def _on_double(self, item: QtWidgets.QListWidgetItem):
        self.trackActivated.emit(self.list.row(item))

    def add_tracks(self, tracks: List[Track]):
        icon_size = self.list.iconSize()
        for t in tracks:
            item_text = self._format_item_text(t)
            it = QtWidgets.QListWidgetItem(item_text)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, t)
            it.setSizeHint(QtCore.QSize(0, self._row_height))

            # Small album-art thumbnail in the playlist (if embedded artwork exists)
            self._apply_cover_art(it, t, icon_size)

            self.list.addItem(it)
            self._items_by_path.setdefault(t.path, []).append(it)

    def clear(self):
        self.list.clear()
        self._items_by_path.clear()
        self._playing_item = None

    def count(self) -> int:
        return self.list.count()

    def get_track(self, idx: int) -> Optional[Track]:
        it = self.list.item(idx)
        return it.data(QtCore.Qt.ItemDataRole.UserRole) if it else None

    def index_for_path(self, path: str) -> int:
        items = self._items_by_path.get(path)
        if not items:
            return -1
        return self.list.row(items[0])

    def track_paths(self) -> List[str]:
        paths: List[str] = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if not it:
                continue
            track = it.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(track, Track):
                paths.append(track.path)
        return paths

    def current_index(self) -> int:
        return self.list.currentRow()

    def select_index(self, idx: int):
        if 0 <= idx < self.list.count():
            self.list.setCurrentRow(idx)

    def set_playing_index(self, idx: int) -> None:
        self._clear_playing_style(self._playing_item)
        item = self.list.item(idx) if 0 <= idx < self.list.count() else None
        self._playing_item = item
        self._apply_playing_style(item)
        self.list.viewport().update()

    def refresh_playing_highlight(self) -> None:
        self._apply_playing_style(self._playing_item)
        self.list.viewport().update()

    def _apply_playing_style(self, item: Optional[QtWidgets.QListWidgetItem]) -> None:
        if not item:
            return
        item.setData(PLAYING_ROLE, True)

    def _clear_playing_style(self, item: Optional[QtWidgets.QListWidgetItem]) -> None:
        if not item:
            return
        item.setData(PLAYING_ROLE, False)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dropEvent(self, e: QtGui.QDropEvent):
        if e.mimeData().hasUrls():
            self._dropped_paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
            self.addFilesRequested.emit()
            e.acceptProposedAction()
        else:
            super().dropEvent(e)

    def consume_dropped_paths(self) -> List[str]:
        p = self._dropped_paths
        self._dropped_paths = []
        return p

    @staticmethod
    def _format_item_text(track: Track) -> str:
        return f"{format_track_title(track)} - {format_time(track.duration_sec)}"

    @staticmethod
    def _apply_metadata(track: Track, metadata: TrackMetadata) -> None:
        title = metadata.title or track.title or os.path.basename(track.path)
        track.title = title
        track.title_display = title
        track.duration_sec = metadata.duration_sec
        track.artist = metadata.artist
        track.album = metadata.album
        track.cover_art = metadata.cover_art
        track.has_video = metadata.has_video
        track.video_fps = metadata.video_fps
        track.video_size = metadata.video_size

    @staticmethod
    def _apply_cover_art(
        item: QtWidgets.QListWidgetItem,
        track: Track,
        icon_size: QtCore.QSize,
    ) -> None:
        if not track.cover_art:
            return
        pm = QtGui.QPixmap()
        if pm.loadFromData(track.cover_art):
            pm = pm.scaled(
                icon_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            item.setIcon(QtGui.QIcon(pm))

    def update_track_metadata(self, path: str, metadata: TrackMetadata) -> None:
        items = self._items_by_path.get(path)
        if not items:
            return
        icon_size = self.list.iconSize()
        for item in items:
            track = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not isinstance(track, Track):
                continue
            self._apply_metadata(track, metadata)
            item.setText(self._format_item_text(track))
            item.setSizeHint(QtCore.QSize(0, self._row_height))
            self._apply_cover_art(item, track, icon_size)


# -----------------------------
