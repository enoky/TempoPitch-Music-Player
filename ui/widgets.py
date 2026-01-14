from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from models import PlayerState, RepeatMode, Track, format_track_title
from utils import clamp, format_time, safe_float

# UI Widgets
# -----------------------------

class VisualizerWidget(QtWidgets.QWidget):
    def __init__(self, engine: PlayerEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._fft_size = 1024
        self._bar_count = 48
        self._bar_levels = np.zeros(self._bar_count, dtype=np.float32)
        self._fft_window = np.hanning(self._fft_size).astype(np.float32)
        self._bin_edges = np.linspace(0, self._fft_size // 2, self._bar_count + 1, dtype=int)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._pull_frames)
        self._timer.start()
        self.setMinimumHeight(140)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def _pull_frames(self) -> None:
        if not self.engine:
            return
        delay_sec = self.engine.get_output_latency_seconds()
        frames = self.engine.get_visualizer_frames(
            frames=self._fft_size,
            mono=True,
            delay_sec=delay_sec,
        )
        if frames.size == 0:
            self._bar_levels *= 0.85
            self.update()
            return
        mono = frames.reshape(-1)
        if mono.size < self._fft_size:
            padded = np.zeros(self._fft_size, dtype=np.float32)
            padded[-mono.size:] = mono
            mono = padded
        else:
            mono = mono[-self._fft_size:]
        spectrum = np.fft.rfft(mono * self._fft_window)
        magnitudes = np.abs(spectrum)[1:]
        magnitudes = np.log1p(magnitudes)
        if magnitudes.size == 0:
            return
        peak = np.max(magnitudes)
        if peak > 0:
            magnitudes /= peak
        bin_edges = self._bin_edges
        levels = np.zeros(self._bar_count, dtype=np.float32)
        for i in range(self._bar_count):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            if end > start:
                levels[i] = float(np.mean(magnitudes[start:end]))
        self._bar_levels = np.maximum(levels, self._bar_levels * 0.85)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        palette = self.palette()
        background = palette.color(QtGui.QPalette.ColorRole.Base)
        text_color = palette.color(QtGui.QPalette.ColorRole.Text)
        highlight = palette.color(QtGui.QPalette.ColorRole.Highlight)
        painter.fillRect(self.rect(), background)

        rect = self.rect().adjusted(12, 12, -12, -12)
        if rect.width() <= 0 or rect.height() <= 0:
            return

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
        palette = self.palette()
        painter.fillRect(self.rect(), palette.color(QtGui.QPalette.ColorRole.Base))

        if self._image is None or self._image.isNull():
            painter.setPen(palette.color(QtGui.QPalette.ColorRole.Text))
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
        self.setWindowTitle("Pop-out Video")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.video_widget = VideoWidget(engine, self)
        self.video_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_widget)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.video_widget.stop_timer()
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


class EqualizerWidget(QtWidgets.QGroupBox):
    gainsChanged = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__("Equalizer", parent)

        self.presets_map = {
            "Flat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Presets"))
        header.addWidget(self.presets)
        header.addStretch(1)
        header.addWidget(self.reset_btn)

        bands = ["31", "62", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
        self.band_sliders: List[QtWidgets.QSlider] = []

        sliders_layout = QtWidgets.QHBoxLayout()
        sliders_layout.setSpacing(6)
        for band in bands:
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
            slider.setRange(-12, 12)
            slider.setValue(0)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBothSides)
            slider.setTickInterval(3)
            slider.setToolTip(f"{band} Hz band")
            slider.setAccessibleName(f"{band} Hz band")

            band_label = QtWidgets.QLabel(band)
            band_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

            column = QtWidgets.QVBoxLayout()
            column.addWidget(slider, 1, QtCore.Qt.AlignmentFlag.AlignHCenter)
            column.addWidget(band_label)
            sliders_layout.addLayout(column)
            self.band_sliders.append(slider)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header)
        layout.addLayout(sliders_layout)

        self.setMaximumWidth(360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)

        self.reset_btn.clicked.connect(self._on_reset)
        self.presets.currentTextChanged.connect(self._on_preset_changed)
        for slider in self.band_sliders:
            slider.valueChanged.connect(self._on_slider_changed)
            slider.sliderReleased.connect(self._on_slider_released)

        self._apply_gains(self.presets_map["Flat"], emit=False)

    def _on_reset(self):
        self.presets.setCurrentText("Flat")

    def _on_preset_changed(self, name: str):
        if name in self.presets_map:
            self._apply_gains(self.presets_map[name], emit=True)
        else:
            self._emit_gains()

    def _on_slider_changed(self, _value: int):
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
        for slider, gain in zip(self.band_sliders, gains):
            slider.blockSignals(True)
            slider.setValue(int(round(clamp(float(gain), -12.0, 12.0))))
            slider.blockSignals(False)
        if emit:
            self._gains_timer.stop()
            self._emit_gains()

    def _emit_gains(self):
        self.gainsChanged.emit(self.gains())

    def gains(self) -> list[float]:
        return [float(slider.value()) for slider in self.band_sliders]

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
        return all(int(round(g)) == slider.value() for g, slider in zip(gains, self.band_sliders))


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

        self.prev_btn = QtWidgets.QToolButton(text="⏮")
        self.play_pause_btn = QtWidgets.QToolButton(text="▶")
        self.play_pause_btn.setCheckable(True)
        self.stop_btn = QtWidgets.QToolButton(text="⏹")
        self.next_btn = QtWidgets.QToolButton(text="⏭")
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

        self.mute_btn = QtWidgets.QToolButton(text="🔈")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setToolTip("Mute audio.")
        self.mute_btn.setAccessibleName("Mute")

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
        self.stop_btn.clicked.connect(self.stopClicked)
        self.next_btn.clicked.connect(self.nextClicked)

        self.volume_slider.valueChanged.connect(lambda v: self.volumeChanged.emit(v / 100.0))
        self.mute_btn.toggled.connect(self._on_mute)

        self._dragging = False
        self.pos_slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self.pos_slider.sliderReleased.connect(self._on_seek_end)

    def _on_mute(self, on: bool):
        self.mute_btn.setText("🔇" if on else "🔈")
        self.muteToggled.emit(on)

    def _on_seek_end(self):
        self._dragging = False
        frac = self.pos_slider.value() / 1000.0
        self.seekRequested.emit(frac)

    def set_play_pause_state(self, playing: bool):
        self.play_pause_btn.blockSignals(True)
        self.play_pause_btn.setChecked(playing)
        self.play_pause_btn.blockSignals(False)
        self.play_pause_btn.setText("⏸" if playing else "▶")

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
        self.list.setIconSize(QtCore.QSize(32, 32))
        self.list.setUniformItemSizes(True)
        self.list.setSpacing(2)

        style = self.style()
        add_files = QtWidgets.QToolButton()
        add_files.setText("Files")
        add_files.setToolTip("Add files")
        add_files.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon))
        add_files.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        add_files.setAutoRaise(True)

        add_folder = QtWidgets.QToolButton()
        add_folder.setText("Folder")
        add_folder.setToolTip("Add folder")
        add_folder.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon))
        add_folder.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        add_folder.setAutoRaise(True)

        reset_enum = getattr(
            QtWidgets.QStyle.StandardPixmap,
            "SP_DialogResetButton",
            QtWidgets.QStyle.StandardPixmap.SP_DialogCloseButton,
        )
        clear_icon_enum = getattr(QtWidgets.QStyle.StandardPixmap, "SP_TrashIcon", reset_enum)
        clear_btn = QtWidgets.QToolButton()
        clear_btn.setText("Clear")
        clear_btn.setToolTip("Clear playlist")
        clear_btn.setIcon(style.standardIcon(clear_icon_enum))
        clear_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        clear_btn.setAutoRaise(True)

        for btn in (add_files, add_folder, clear_btn):
            btn.setIconSize(QtCore.QSize(14, 14))

        header_row = QtWidgets.QHBoxLayout()
        header_row.addWidget(header)
        header_row.addStretch(1)
        header_row.addWidget(add_files)
        header_row.addWidget(add_folder)
        header_row.addWidget(clear_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header_row)
        layout.addWidget(self.list, 1)

        add_files.clicked.connect(self.addFilesRequested)
        add_folder.clicked.connect(self.addFolderRequested)
        clear_btn.clicked.connect(self.clearRequested)

        self.list.itemDoubleClicked.connect(self._on_double)

        self.setAcceptDrops(True)
        self._dropped_paths: List[str] = []

    def _on_double(self, item: QtWidgets.QListWidgetItem):
        self.trackActivated.emit(self.list.row(item))

    def add_tracks(self, tracks: List[Track]):
        icon_size = self.list.iconSize()
        for t in tracks:
            item_text = f"{format_track_title(t)} — {format_time(t.duration_sec)}"
            it = QtWidgets.QListWidgetItem(item_text)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, t)

            # Small album-art thumbnail in the playlist (if embedded artwork exists)
            if t.cover_art:
                pm = QtGui.QPixmap()
                if pm.loadFromData(t.cover_art):
                    pm = pm.scaled(
                        icon_size,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                    it.setIcon(QtGui.QIcon(pm))

            self.list.addItem(it)

    def clear(self):
        self.list.clear()

    def count(self) -> int:
        return self.list.count()

    def get_track(self, idx: int) -> Optional[Track]:
        it = self.list.item(idx)
        return it.data(QtCore.Qt.ItemDataRole.UserRole) if it else None

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


# -----------------------------
