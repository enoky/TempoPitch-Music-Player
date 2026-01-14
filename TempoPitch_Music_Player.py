
"""
PySide6 Music Player with real-time tempo + pitch controls (SoundTouch preferred).

Backend pipeline:
- Decode: ffmpeg -> float32 PCM (stereo) at fixed sample rate
- DSP: SoundTouch (streaming) if available, else PhaseVocoder+Resampler fallback
- Output: sounddevice (PortAudio) callback pulling from a thread-safe ring buffer

Requirements:
  pip install PySide6 numpy sounddevice
  ffmpeg + ffprobe installed and on PATH

SoundTouch:
- Windows: place SoundTouch.dll next to this script, or set SOUNDTOUCH_DLL to its full path.
- macOS/Linux: install SoundTouch shared library (e.g. via brew/apt) or set SOUNDTOUCH_DLL.

Env vars:
- TEMPOPITCH_DSP = "soundtouch" | "phasevocoder" | "auto" (default auto)
- SOUNDTOUCH_DLL = explicit path to SoundTouch shared library (.dll/.so/.dylib)
"""

from __future__ import annotations

import os
import sys
import math
import time
import threading
import subprocess
import shutil
import ctypes
import ctypes.util
import random
import json
import logging
import re
from dataclasses import dataclass, replace
from collections import deque
from typing import Optional, List, Callable

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    _sounddevice_import_error = e

from PySide6 import QtCore, QtGui, QtWidgets

from config import (
    AUTO_BUFFER_PRESET,
    AUTO_BUFFER_THRESHOLD,
    AUTO_BUFFER_WINDOW_SEC,
    BLOCKSIZE_FRAMES,
    BUFFER_PRESETS,
    DEFAULT_BUFFER_PRESET,
    EQ_PROFILE,
    EQ_PROFILE_LOG_EVERY,
    EQ_PROFILE_LOW_WATERMARK_SEC,
    LATENCY,
)
from models import (
    AudioParams,
    BufferPreset,
    PlayerState,
    RepeatMode,
    THEMES,
    Track,
    TrackMetadata,
    format_track_title,
)
from dsp import (
    DSPBase,
    EqualizerDSP,
    GainEffect,
    CompressorEffect,
    DynamicEqEffect,
    LimiterEffect,
    SaturationEffect,
    SubharmonicEffect,
    ReverbEffect,
    StereoWidenerEffect,
    StereoPannerEffect,
    ChorusEffect,
    EffectsChain,
    make_dsp,
    build_track,
    make_ffmpeg_cmd,
    make_ffmpeg_video_cmd,
)
from theme import build_palette, build_stylesheet
from buffers import AudioRingBuffer, VisualizerBuffer, VideoFrameBuffer
from audio.engine import PlayerEngine
from utils import (
    adjust_color,
    clamp,
    env_flag,
    format_time,
    have_exe,
    safe_float,
    semitones_to_factor,
)

logger = logging.getLogger(__name__)





 


# -----------------------------
# DSP moved to dsp.py

# Audio engine moved to audio/engine.py

# UI Widgets
# -----------------------------

class VisualizerWidget(QtWidgets.QWidget):
    def __init__(self, engine: PlayerEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._fft_size = 2048
        self._bar_count = 48
        self._bar_levels = np.zeros(self._bar_count, dtype=np.float32)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._pull_frames)
        self._timer.start()
        self.setMinimumHeight(140)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def _pull_frames(self) -> None:
        if not self.engine:
            return
        frames = self.engine.get_visualizer_frames(frames=self._fft_size, mono=True)
        if frames.size == 0:
            self._bar_levels *= 0.9
            self.update()
            return
        mono = frames.reshape(-1)
        if mono.size < self._fft_size:
            padded = np.zeros(self._fft_size, dtype=np.float32)
            padded[-mono.size:] = mono
            mono = padded
        else:
            mono = mono[-self._fft_size:]
        window = np.hanning(self._fft_size).astype(np.float32)
        spectrum = np.fft.rfft(mono * window)
        magnitudes = np.abs(spectrum)[1:]
        magnitudes = np.log1p(magnitudes)
        if magnitudes.size == 0:
            return
        peak = np.max(magnitudes)
        if peak > 0:
            magnitudes /= peak
        bin_edges = np.linspace(0, magnitudes.size, self._bar_count + 1, dtype=int)
        levels = np.zeros(self._bar_count, dtype=np.float32)
        for i in range(self._bar_count):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            if end > start:
                levels[i] = float(np.mean(magnitudes[start:end]))
        self._bar_levels = np.maximum(levels, self._bar_levels * 0.92)
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
        self.list.setIconSize(QtCore.QSize(42, 42))
        self.list.setUniformItemSizes(True)

        add_files = QtWidgets.QPushButton("Add Files…")
        add_folder = QtWidgets.QPushButton("Add Folder…")
        clear_btn = QtWidgets.QPushButton("Clear")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(add_files)
        top.addWidget(add_folder)
        top.addStretch(1)
        top.addWidget(clear_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(header)
        layout.addLayout(top)
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
# Main Window
# -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 Tempo/Pitch Music Player (SoundTouch)")
        self.resize(1280, 640)

        self.settings = QtCore.QSettings("ChatGPT", "TempoPitchPlayer")
        self._theme_name = str(self.settings.value("ui/theme", "Ocean"))
        metrics_enabled = self.settings.value("audio/metrics_enabled", True, type=bool)

        self.engine = PlayerEngine(
            sample_rate=44100,
            channels=2,
            metrics_enabled=metrics_enabled,
            parent=self,
        )

        self.transport = TransportWidget()
        self.visualizer = VisualizerWidget(self.engine)
        self.dsp_widget = TempoPitchWidget()
        self.dynamic_eq_widget = DynamicEqWidget()
        self.compressor_widget = CompressorWidget()
        self.saturation_widget = SaturationWidget()
        self.subharmonic_widget = SubharmonicWidget()
        self.limiter_widget = LimiterWidget()
        self.reverb_widget = ReverbWidget()
        self.chorus_widget = ChorusWidget()
        self.stereo_panner_widget = StereoPannerWidget()
        self.stereo_width_widget = StereoWidthWidget()
        self.equalizer = EqualizerWidget()
        self.playlist = PlaylistWidget()

        self.now_playing = QtWidgets.QLabel("No track loaded")
        self.now_playing.setObjectName("now_playing")
        self.now_playing.setWordWrap(True)
        font = self.now_playing.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.now_playing.setFont(font)

        self._media_size = QtCore.QSize(200, 120)
        self.artwork_label = QtWidgets.QLabel("No Artwork")
        self.artwork_label.setObjectName("artwork_label")
        self.artwork_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.artwork_label.setFixedSize(self._media_size)
        self.artwork_label.setWordWrap(True)
        self.video_widget = VideoWidget(self.engine)
        self.video_widget.setFixedSize(self._media_size)
        self.popout_video_btn = QtWidgets.QPushButton("Pop-out Video")
        self.popout_video_btn.setToolTip("Open video in a separate window.")
        self.popout_video_btn.setEnabled(False)
        self.popout_video_btn.setVisible(False)
        self._video_popout: Optional[VideoPopoutDialog] = None

        self.status = QtWidgets.QLabel("Ready.")
        self.status.setObjectName("status_label")
        self.fx_status = QtWidgets.QLabel("Enabled FX: None")
        self.fx_status.setObjectName("fx_status_label")
        self.fx_status.setWordWrap(True)

        self.header_frame = QtWidgets.QFrame()
        self.header_frame.setObjectName("header_frame")
        self.header_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum,
        )
        header_layout = QtWidgets.QVBoxLayout(self.header_frame)
        header_top_row = QtWidgets.QHBoxLayout()
        self.media_stack_widget = QtWidgets.QWidget()
        self.media_stack = QtWidgets.QStackedLayout(self.media_stack_widget)
        self.media_stack.addWidget(self.artwork_label)
        self.media_stack.addWidget(self.video_widget)
        self.media_stack.setCurrentWidget(self.artwork_label)
        header_top_row.addWidget(self.media_stack_widget)
        header_text_column = QtWidgets.QVBoxLayout()
        header_text_column.addWidget(self.now_playing)
        header_text_column.addWidget(self.status)
        header_text_column.addWidget(self.fx_status)
        header_text_column.addWidget(self.popout_video_btn)
        header_text_column.addStretch(1)
        header_top_row.addLayout(header_text_column)
        header_layout.addLayout(header_top_row)

        self.appearance_group = QtWidgets.QGroupBox("Appearance")
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(THEMES.keys())
        if self._theme_name not in THEMES:
            self._theme_name = next(iter(THEMES.keys()))
        self.theme_combo.setCurrentText(self._theme_name)
        self.theme_combo.setToolTip("Choose a color theme.")
        self.theme_combo.setAccessibleName("Theme selector")
        appearance_layout = QtWidgets.QFormLayout(self.appearance_group)
        appearance_layout.addRow("Theme", self.theme_combo)

        self.audio_group = QtWidgets.QGroupBox("Audio")
        self.buffer_preset_combo = QtWidgets.QComboBox()
        self.buffer_preset_combo.addItems(list(BUFFER_PRESETS.keys()))
        self.buffer_preset_combo.setToolTip("Balance output latency vs stability.")
        self.metrics_checkbox = QtWidgets.QCheckBox("Enable metrics logging")
        self.metrics_checkbox.setToolTip("Log audio engine metrics periodically.")
        self.metrics_checkbox.setChecked(bool(metrics_enabled))
        audio_layout = QtWidgets.QFormLayout(self.audio_group)
        audio_layout.addRow("Buffer preset", self.buffer_preset_combo)
        audio_layout.addRow(self.metrics_checkbox)

        self._shuffle = bool(self.settings.value("playback/shuffle", False, type=bool))
        repeat_setting = self.settings.value("playback/repeat", RepeatMode.OFF.value)
        self._repeat_mode = RepeatMode.from_setting(repeat_setting)
        self._shuffle_history: List[int] = []
        self._shuffle_bag: List[int] = []

        app = QtWidgets.QApplication.instance()
        if app:
            self._apply_theme(self._theme_name)

        self.effects_tabs = QtWidgets.QTabWidget()
        self.effects_tabs.setObjectName("effects_tabs")
        self.effects_tabs.addTab(self.dynamic_eq_widget, "Dynamic EQ")
        self.effects_tabs.addTab(self.compressor_widget, "Compressor")
        self.effects_tabs.addTab(self.saturation_widget, "Saturation")
        self.effects_tabs.addTab(self.subharmonic_widget, "Subharmonic")
        self.effects_tabs.addTab(self.limiter_widget, "Limiter")
        self.effects_tabs.addTab(self.reverb_widget, "Reverb")
        self.effects_tabs.addTab(self.chorus_widget, "Chorus")
        self.effects_tabs.addTab(self.stereo_panner_widget, "Stereo Panner")
        self.effects_tabs.addTab(self.stereo_width_widget, "Stereo Width")
        self.effects_tabs.addTab(self.equalizer, "Equalizer")

        self.effects_toggle_group = QtWidgets.QGroupBox("FX Enable")
        self.effects_toggle_group.setObjectName("effects_toggle_group")
        effects_toggle_layout = QtWidgets.QGridLayout(self.effects_toggle_group)
        effects_toggle_layout.setContentsMargins(8, 8, 8, 8)
        effects_toggle_layout.setHorizontalSpacing(16)
        effects_toggle_layout.setVerticalSpacing(6)
        self.effect_toggles: dict[str, QtWidgets.QCheckBox] = {}
        effect_names = [
            "Compressor",
            "Dynamic EQ",
            "Subharmonic",
            "Reverb",
            "Chorus",
            "Saturation",
            "Limiter",
        ]
        for index, name in enumerate(effect_names):
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setAccessibleName(f"{name} enable")
            checkbox.toggled.connect(lambda checked, effect_name=name: self._on_effect_toggled(effect_name, checked))
            row = index // 2
            col = index % 2
            effects_toggle_layout.addWidget(checkbox, row, col)
            self.effect_toggles[name] = checkbox

        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(16, 16, 16, 16)
        left.setSpacing(12)
        left.addWidget(self.transport)
        left.addWidget(self.visualizer)
        left.addWidget(self.header_frame)
        left.addWidget(self.dsp_widget)
        left.addWidget(self.effects_toggle_group)
        left.addWidget(self.effects_tabs)
        left.addWidget(self.audio_group)
        left.addWidget(self.appearance_group)
        left.addStretch(1)

        leftw = QtWidgets.QWidget()
        leftw.setLayout(left)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(leftw)
        splitter.addWidget(self.playlist)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)
        self.setCentralWidget(splitter)

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        open_files = QtGui.QAction("Open Files…", self)
        open_folder = QtGui.QAction("Open Folder…", self)
        quit_act = QtGui.QAction("Quit", self)
        file_menu.addAction(open_files)
        file_menu.addAction(open_folder)
        file_menu.addSeparator()
        file_menu.addAction(quit_act)

        playback_menu = self.menuBar().addMenu("&Playback")
        shuffle_act = QtGui.QAction("Shuffle", self, checkable=True)
        shuffle_act.setChecked(self._shuffle)

        repeat_group = QtGui.QActionGroup(self)
        repeat_off_act = QtGui.QAction("Repeat Off", self, checkable=True)
        repeat_all_act = QtGui.QAction("Repeat All", self, checkable=True)
        repeat_one_act = QtGui.QAction("Repeat One", self, checkable=True)
        for act in (repeat_off_act, repeat_all_act, repeat_one_act):
            repeat_group.addAction(act)
        repeat_map = {
            RepeatMode.OFF: repeat_off_act,
            RepeatMode.ALL: repeat_all_act,
            RepeatMode.ONE: repeat_one_act,
        }
        repeat_map[self._repeat_mode].setChecked(True)

        playback_menu.addAction(shuffle_act)
        playback_menu.addSeparator()
        playback_menu.addAction(repeat_off_act)
        playback_menu.addAction(repeat_all_act)
        playback_menu.addAction(repeat_one_act)

        help_menu = self.menuBar().addMenu("&Help")
        about_act = QtGui.QAction("About", self)
        help_menu.addAction(about_act)

        self._restore_ui_settings()

        open_files.triggered.connect(self._add_files_dialog)
        open_folder.triggered.connect(self._add_folder_dialog)
        quit_act.triggered.connect(self.close)
        about_act.triggered.connect(self._about)
        shuffle_act.toggled.connect(self._set_shuffle)
        repeat_off_act.triggered.connect(lambda: self._set_repeat_mode(RepeatMode.OFF))
        repeat_all_act.triggered.connect(lambda: self._set_repeat_mode(RepeatMode.ALL))
        repeat_one_act.triggered.connect(lambda: self._set_repeat_mode(RepeatMode.ONE))

        # Wiring
        self.transport.playPauseToggled.connect(self._toggle_play_pause)
        self.transport.stopClicked.connect(self.engine.stop)
        self.transport.prevClicked.connect(self._on_prev)
        self.transport.nextClicked.connect(self._on_next)
        self.transport.volumeChanged.connect(self.engine.set_volume)
        self.transport.volumeChanged.connect(self._on_volume_changed)
        self.transport.muteToggled.connect(self.engine.set_muted)
        self.transport.muteToggled.connect(self._on_mute_toggled)
        self.transport.seekRequested.connect(self._on_seek_fraction)

        self.dsp_widget.controlsChanged.connect(
            lambda tempo, pitch, key_lock, tape_mode, _lock_432: self.engine.set_dsp_controls(
                tempo,
                pitch,
                key_lock,
                tape_mode,
            )
        )
        self.dsp_widget.controlsChanged.connect(self._on_dsp_controls_changed)

        self.equalizer.gainsChanged.connect(self._on_eq_gains_changed)
        self.dynamic_eq_widget.controlsChanged.connect(self._on_dynamic_eq_controls_changed)
        self.dynamic_eq_widget.controlsChanged.connect(
            lambda freq, q, gain, threshold, ratio: self.engine.set_dynamic_eq_controls(
                freq,
                q,
                gain,
                threshold,
                ratio,
            )
        )
        self.compressor_widget.controlsChanged.connect(self._on_compressor_controls_changed)
        self.compressor_widget.controlsChanged.connect(
            lambda threshold, ratio, attack, release, makeup: self.engine.set_compressor_controls(
                threshold,
                ratio,
                attack,
                release,
                makeup,
            )
        )
        self.saturation_widget.controlsChanged.connect(self._on_saturation_controls_changed)
        self.saturation_widget.controlsChanged.connect(
            lambda drive, trim, tone, tone_enabled: self.engine.set_saturation_controls(
                drive,
                trim,
                tone,
                tone_enabled,
            )
        )
        self.subharmonic_widget.controlsChanged.connect(self._on_subharmonic_controls_changed)
        self.subharmonic_widget.controlsChanged.connect(
            lambda mix, intensity, cutoff: self.engine.set_subharmonic_controls(
                mix,
                intensity,
                cutoff,
            )
        )
        self.limiter_widget.controlsChanged.connect(self._on_limiter_controls_changed)
        self.limiter_widget.controlsChanged.connect(
            lambda threshold, release: self.engine.set_limiter_controls(threshold, release)
        )
        self.reverb_widget.controlsChanged.connect(self._on_reverb_controls_changed)
        self.reverb_widget.controlsChanged.connect(
            lambda decay, predelay, mix: self.engine.set_reverb_controls(decay, predelay, mix)
        )
        self.chorus_widget.controlsChanged.connect(self._on_chorus_controls_changed)
        self.chorus_widget.controlsChanged.connect(
            lambda rate, depth, mix: self.engine.set_chorus_controls(rate, depth, mix)
        )
        self.stereo_panner_widget.controlsChanged.connect(self._on_stereo_panner_changed)
        self.stereo_panner_widget.controlsChanged.connect(
            lambda azimuth, spread: self.engine.set_stereo_panner_controls(azimuth, spread)
        )
        self.stereo_width_widget.widthChanged.connect(self._on_stereo_width_changed)
        self.stereo_width_widget.widthChanged.connect(self.engine.set_stereo_width)

        self.playlist.addFilesRequested.connect(self._on_add_files_requested)
        self.playlist.addFolderRequested.connect(self._add_folder_dialog)
        self.playlist.clearRequested.connect(self._on_clear)
        self.playlist.trackActivated.connect(self._on_track_activated)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.buffer_preset_combo.currentTextChanged.connect(self._on_buffer_preset_changed)
        self.metrics_checkbox.toggled.connect(self._on_metrics_toggled)
        self.popout_video_btn.clicked.connect(self._toggle_video_window)

        self.engine.trackChanged.connect(self._on_track_changed)
        self.engine.stateChanged.connect(self._on_state_changed)
        self.engine.errorOccurred.connect(self._on_error)
        self.engine.bufferPresetChanged.connect(self._on_engine_buffer_preset_changed)
        self.engine.effectAutoEnabled.connect(self._on_effect_auto_enabled)

        # Timer
        self._dur = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._metrics_timer = QtCore.QTimer(self)
        self._metrics_timer.setInterval(300)
        self._metrics_timer.timeout.connect(self.engine.log_metrics_if_needed)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_play_pause)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, activated=self._add_files_dialog)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self, activated=self._add_folder_dialog)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self, activated=self._on_next)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+P"), self, activated=self._on_prev)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Right"), self, activated=lambda: self._seek_relative(10))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Left"), self, activated=lambda: self._seek_relative(-10))

        self._current_index = -1
        if self._shuffle:
            self._reset_shuffle_bag()

        self._apply_ui_settings()
        self.compressor_widget.set_meter_provider(self.engine.get_compressor_gain_reduction_db)
        self._initial_warnings()
        self._restore_playlist_session()
        self._on_state_changed(self.engine.state)
        self._update_enabled_fx_label()
        self._schedule_debug_autoplay()

    def _schedule_debug_autoplay(self) -> None:
        autoplay_enabled = env_flag("TEMPOPITCH_DEBUG_AUTOPLAY")
        autoplay_seconds = safe_float(os.environ.get("TEMPOPITCH_DEBUG_AUTOPLAY_SECONDS", "0"), 0.0)
        if autoplay_enabled and autoplay_seconds <= 0:
            autoplay_seconds = 600.0
        if not autoplay_enabled and autoplay_seconds <= 0:
            return

        def start_playback():
            if self.engine.track is None:
                idx = self.playlist.current_index()
                if idx >= 0:
                    track = self.playlist.get_track(idx)
                    if track:
                        self.engine.load_track(track.path)
            self.engine.play()
            if autoplay_seconds > 0:
                QtCore.QTimer.singleShot(int(autoplay_seconds * 1000), self.engine.stop)

        QtCore.QTimer.singleShot(250, start_playback)

    def _initial_warnings(self):
        warnings = []
        if sd is None:
            warnings.append(f"sounddevice missing ({_sounddevice_import_error})")
        if not have_exe("ffmpeg"):
            warnings.append("ffmpeg not found in PATH")
        if not have_exe("ffprobe"):
            warnings.append("ffprobe not found in PATH (duration may be unknown)")
        dsp_name = self.engine.dsp_name()
        if "SoundTouch" in dsp_name:
            warnings.append(f"DSP: {dsp_name}")
        else:
            warnings.append(f"DSP: {dsp_name}")

        self.status.setText(("⚠ " + " | ".join(warnings)) if warnings else "Ready.")

    def _apply_theme(self, theme_name: str):
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        theme = THEMES.get(theme_name, next(iter(THEMES.values())))
        app.setPalette(build_palette(theme))
        app.setStyleSheet(build_stylesheet(theme))
        self._theme_name = theme.name

    def _on_theme_changed(self, theme_name: str):
        if theme_name not in THEMES:
            return
        self._apply_theme(theme_name)
        self.settings.setValue("ui/theme", theme_name)

    def _about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "PySide6 Tempo/Pitch Music Player\n\n"
            "Decode: ffmpeg\nDSP: SoundTouch (preferred)\nOutput: sounddevice\n\n"
            "If SoundTouch isn't found, the app falls back to a phase vocoder DSP.\n\n"
            "Shortcuts:\n"
            "Space: Play/Pause\n"
            "Ctrl+O: Open files\n"
            "Ctrl+L: Open folder\n"
            "Ctrl+N: Next track\n"
            "Ctrl+P: Previous track\n"
            "Ctrl+Left/Right: Seek ±10s"
        )

    # Playlist actions
    def _on_add_files_requested(self):
        dropped = self.playlist.consume_dropped_paths()
        if dropped:
            self._add_paths(dropped)
        else:
            self._add_files_dialog()

    def _add_files_dialog(self):
        last_dir = self.settings.value("last_dir", os.path.expanduser("~"))
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select audio files", last_dir,
            "Audio/Video Files (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.mp4 *.mkv *.mov *.webm *.avi);;All Files (*.*)"
        )
        if not paths:
            return
        self.settings.setValue("last_dir", os.path.dirname(paths[0]))
        self._add_paths(paths)

    def _add_folder_dialog(self):
        last_dir = self.settings.value("last_dir", os.path.expanduser("~"))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", last_dir)
        if not folder:
            return
        self.settings.setValue("last_dir", folder)
        exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mkv", ".mov", ".webm", ".avi"}
        paths = []
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    paths.append(os.path.join(root, f))
        paths.sort()
        self._add_paths(paths)

    def _add_paths(self, paths: List[str]):
        tracks: List[Track] = []
        for p in paths:
            if os.path.isdir(p) or not os.path.exists(p):
                continue
            tracks.append(build_track(p))

        if not tracks:
            return

        self.playlist.add_tracks(tracks)
        if self._shuffle:
            self._reset_shuffle_bag()
        if self._current_index < 0 and self.playlist.count() > 0:
            self._current_index = 0
            self.playlist.select_index(0)
            t = self.playlist.get_track(0)
            if t:
                self.engine.load_track(t.path)

    def _on_clear(self):
        self.engine.stop()
        self.engine.track = None
        self.playlist.clear()
        self._current_index = -1
        self._shuffle_history.clear()
        self._shuffle_bag = []
        self.now_playing.setText("No track loaded")
        self._set_media_mode(False)
        self.video_widget.clear()
        self._set_artwork(None)
        self._dur = 0.0

    # Playback
    def _toggle_play_pause(self, _checked: Optional[bool] = None):
        if self.engine.state == PlayerState.PLAYING:
            self.engine.pause()
        else:
            self._on_play()

    def _on_play(self):
        if self.engine.track is None:
            idx = self.playlist.current_index()
            if idx >= 0:
                t = self.playlist.get_track(idx)
                if t:
                    self._current_index = idx
                    self.engine.load_track(t.path)
        self.engine.play()

    def _on_volume_changed(self, _value: float):
        self.settings.setValue("audio/volume_slider", self.transport.volume_slider.value())

    def _on_mute_toggled(self, muted: bool):
        self.settings.setValue("audio/muted", bool(muted))

    def _on_buffer_preset_changed(self, preset: str):
        if preset not in BUFFER_PRESETS:
            preset = DEFAULT_BUFFER_PRESET
            self.buffer_preset_combo.blockSignals(True)
            self.buffer_preset_combo.setCurrentText(preset)
            self.buffer_preset_combo.blockSignals(False)
        self.settings.setValue("audio/buffer_preset", preset)
        self.engine.set_buffer_preset(preset)

    def _on_engine_buffer_preset_changed(self, preset: str) -> None:
        if preset not in BUFFER_PRESETS:
            return
        current = self.buffer_preset_combo.currentText()
        if current != preset:
            self.buffer_preset_combo.blockSignals(True)
            self.buffer_preset_combo.setCurrentText(preset)
            self.buffer_preset_combo.blockSignals(False)
        self.settings.setValue("audio/buffer_preset", preset)

    def _on_metrics_toggled(self, enabled: bool):
        self.settings.setValue("audio/metrics_enabled", bool(enabled))
        self.engine.set_metrics_enabled(bool(enabled))

    def _on_dsp_controls_changed(self, tempo: float, pitch: float, key_lock: bool, tape_mode: bool, lock_432: bool):
        self.settings.setValue("dsp/tempo", float(tempo))
        self.settings.setValue("dsp/pitch", float(pitch))
        self.settings.setValue("dsp/key_lock", bool(key_lock))
        self.settings.setValue("dsp/tape_mode", bool(tape_mode))
        self.settings.setValue("dsp/lock_432", bool(lock_432))

    def _on_eq_gains_changed(self, gains: list[float]):
        gains_db = [float(g) for g in gains]
        self.engine.set_eq_gains(gains_db)
        self.settings.setValue("eq/gains", gains_db)
        self.settings.setValue("eq/preset", self.equalizer.presets.currentText())

    def _on_dynamic_eq_controls_changed(
        self,
        freq_hz: float,
        q: float,
        gain_db: float,
        threshold_db: float,
        ratio: float,
    ):
        self.settings.setValue("dynamic_eq/freq", float(freq_hz))
        self.settings.setValue("dynamic_eq/q", float(q))
        self.settings.setValue("dynamic_eq/gain", float(gain_db))
        self.settings.setValue("dynamic_eq/threshold", float(threshold_db))
        self.settings.setValue("dynamic_eq/ratio", float(ratio))

    def _on_compressor_controls_changed(
        self,
        threshold: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        makeup_db: float,
    ):
        self.settings.setValue("compressor/threshold", float(threshold))
        self.settings.setValue("compressor/ratio", float(ratio))
        self.settings.setValue("compressor/attack", float(attack_ms))
        self.settings.setValue("compressor/release", float(release_ms))
        self.settings.setValue("compressor/makeup", float(makeup_db))

    def _on_saturation_controls_changed(
        self,
        drive_db: float,
        trim_db: float,
        tone: float,
        tone_enabled: bool,
    ):
        self.settings.setValue("saturation/drive", float(drive_db))
        self.settings.setValue("saturation/trim", float(trim_db))
        self.settings.setValue("saturation/tone", float(tone))
        self.settings.setValue("saturation/tone_enabled", bool(tone_enabled))

    def _on_subharmonic_controls_changed(self, mix: float, intensity: float, cutoff_hz: float):
        self.settings.setValue("subharmonic/mix", float(mix))
        self.settings.setValue("subharmonic/intensity", float(intensity))
        self.settings.setValue("subharmonic/cutoff", float(cutoff_hz))

    def _on_limiter_controls_changed(self, threshold: float, release_ms: Optional[float]):
        self.settings.setValue("limiter/threshold", float(threshold))
        self.settings.setValue("limiter/release_enabled", release_ms is not None)
        if release_ms is None:
            release_ms = float(self.limiter_widget.release_slider.value())
        self.settings.setValue("limiter/release", float(release_ms))

    def _on_reverb_controls_changed(self, decay: float, pre_delay_ms: float, mix: float):
        self.settings.setValue("reverb/decay", float(decay))
        self.settings.setValue("reverb/predelay", float(pre_delay_ms))
        self.settings.setValue("reverb/mix", float(mix))

    def _on_chorus_controls_changed(self, rate: float, depth_ms: float, mix: float):
        self.settings.setValue("chorus/rate", float(rate))
        self.settings.setValue("chorus/depth", float(depth_ms))
        self.settings.setValue("chorus/mix", float(mix))

    def _on_stereo_panner_changed(self, azimuth: float, spread: float):
        self.settings.setValue("panner/azimuth", float(azimuth))
        self.settings.setValue("panner/spread", float(spread))

    def _on_stereo_width_changed(self, width: float):
        self.settings.setValue("stereo/width", float(width))

    def _on_effect_toggled(self, effect_name: str, enabled: bool) -> None:
        self.engine.enable_effect(effect_name, enabled)
        self.settings.setValue(self._effect_setting_key(effect_name), bool(enabled))
        self._update_enabled_fx_label()

    def _on_effect_auto_enabled(self, effect_name: str) -> None:
        checkbox = self.effect_toggles.get(effect_name)
        if checkbox is None:
            return
        checkbox.blockSignals(True)
        checkbox.setChecked(True)
        checkbox.blockSignals(False)
        self.settings.setValue(self._effect_setting_key(effect_name), True)
        self._update_enabled_fx_label()

    def _set_shuffle(self, on: bool):
        self._shuffle = bool(on)
        self.settings.setValue("playback/shuffle", self._shuffle)
        self._shuffle_history.clear()
        self._reset_shuffle_bag()

    def _set_repeat_mode(self, mode: RepeatMode):
        self._repeat_mode = mode
        self.settings.setValue("playback/repeat", mode.value)

    def _reset_shuffle_bag(self):
        count = self.playlist.count()
        if count <= 0:
            self._shuffle_bag = []
            return
        current = self._current_index if self._current_index >= 0 else self.playlist.current_index()
        indices = [i for i in range(count) if i != current]
        random.shuffle(indices)
        self._shuffle_bag = indices

    def _next_shuffle_index(self, current: int) -> Optional[int]:
        if current < 0:
            current = 0
        if not self._shuffle_bag:
            return None
        return self._shuffle_bag.pop()

    def _advance_track(self, direction: int, auto: bool = False):
        count = self.playlist.count()
        if count == 0:
            return
        current = self._current_index if self._current_index >= 0 else self.playlist.current_index()
        current = 0 if current < 0 else current

        if self._repeat_mode == RepeatMode.ONE:
            self._play_index(current, push_history=False)
            return

        if self._shuffle:
            if count == 1:
                if self._repeat_mode == RepeatMode.ALL:
                    self._play_index(current, push_history=False)
                return
            if direction < 0:
                if self._shuffle_history:
                    idx = self._shuffle_history.pop()
                    self._play_index(idx, push_history=False)
                return

            idx = self._next_shuffle_index(current)
            if idx is None:
                if self._repeat_mode == RepeatMode.ALL:
                    self._reset_shuffle_bag()
                    idx = self._next_shuffle_index(current)
                if idx is None:
                    if auto:
                        self.engine.stop()
                    return
            self._shuffle_history.append(current)
            self._play_index(idx, push_history=False)
            return

        idx = current + direction
        if idx < 0:
            if self._repeat_mode == RepeatMode.ALL:
                idx = count - 1
            else:
                return
        if idx >= count:
            if self._repeat_mode == RepeatMode.ALL:
                idx = 0
            else:
                if auto:
                    self.engine.stop()
                return
        self._play_index(idx, push_history=False)

    def _on_prev(self):
        self._advance_track(direction=-1)

    def _on_next(self):
        self._advance_track(direction=1)

    def _on_track_activated(self, idx: int):
        self._shuffle_history.clear()
        if self._shuffle:
            self._reset_shuffle_bag()
        self._play_index(idx, push_history=False)

    def _play_index(self, idx: int, push_history: bool = True):
        t = self.playlist.get_track(idx)
        if not t:
            return
        if push_history and self._shuffle and self._current_index >= 0 and idx != self._current_index:
            self._shuffle_history.append(self._current_index)
        self._current_index = idx
        self.playlist.select_index(idx)
        self.engine.load_track(t.path)
        self.engine.play()

    def _on_seek_fraction(self, frac: float):
        if self.engine.track is None:
            return
        dur = self.engine.track.duration_sec
        if dur > 0:
            self.engine.seek(frac * dur)

    def _seek_relative(self, delta_sec: float):
        if self.engine.track is None:
            return
        self.engine.seek(self.engine.get_position() + delta_sec)

    # Engine signals
    def _on_track_changed(self, track: Track):
        artist = track.artist.strip() or "Unknown Artist"
        title = track.title_display.strip() or track.title or os.path.basename(track.path)
        album = track.album.strip() or "Unknown Album"
        self.now_playing.setText(f"{artist} — {title}\n{album}")
        self._set_media_mode(track.has_video)
        self._set_artwork(track.cover_art)
        self._dur = track.duration_sec

    def _set_media_mode(self, has_video: bool) -> None:
        if has_video:
            self.video_widget.clear()
            self.video_widget.setVisible(True)
            self.artwork_label.setVisible(False)
            self.media_stack.setCurrentWidget(self.video_widget)
        else:
            self.video_widget.setVisible(False)
            self.artwork_label.setVisible(True)
            self.media_stack.setCurrentWidget(self.artwork_label)
        self._update_video_popout_state(has_video)

    def _update_video_popout_state(self, has_video: bool) -> None:
        has_track = self.engine.track is not None
        show_video = has_track and has_video
        self.popout_video_btn.setEnabled(show_video)
        self.popout_video_btn.setVisible(show_video)
        if not show_video and self._video_popout is not None:
            self._video_popout.close()

    def _toggle_video_window(self) -> None:
        if self._video_popout is not None and self._video_popout.isVisible():
            self._video_popout.raise_()
            self._video_popout.activateWindow()
            return
        if self.engine.track is None or not self.engine.track.has_video:
            return
        self._video_popout = VideoPopoutDialog(self.engine, self)
        self._video_popout.closed.connect(self._on_video_popout_closed)
        self._video_popout.resize(640, 360)
        self._video_popout.show()

    def _on_video_popout_closed(self) -> None:
        self._video_popout = None

    def _set_artwork(self, data: Optional[bytes]):
        if data:
            pixmap = QtGui.QPixmap()
            if pixmap.loadFromData(data):
                scaled = pixmap.scaled(
                    self.artwork_label.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                self.artwork_label.setPixmap(scaled)
                self.artwork_label.setText("")
                return
        self.artwork_label.setPixmap(QtGui.QPixmap())
        self.artwork_label.setText("No Artwork")

    def _on_state_changed(self, st: PlayerState):
        # basic status text is handled in tick (includes buffer)
        has_track = self.engine.track is not None
        for control in (
            self.transport.prev_btn,
            self.transport.next_btn,
            self.transport.stop_btn,
            self.transport.pos_slider,
        ):
            control.setEnabled(has_track)

        is_playing = has_track and st in (PlayerState.PLAYING, PlayerState.LOADING)
        self.transport.set_play_pause_state(is_playing)
        if is_playing:
            if not self._metrics_timer.isActive():
                self._metrics_timer.start()
        elif self._metrics_timer.isActive():
            self._metrics_timer.stop()

    def _on_error(self, msg: str):
        self.status.setText(f"❌ {msg}")
        QtWidgets.QMessageBox.warning(self, "Playback error", msg)

    def _tick(self):
        self.engine.update_position_from_clock()
        pos = self.engine.get_position()
        self.transport.set_time(pos, self._dur)

        if self.engine.state in (PlayerState.PLAYING, PlayerState.LOADING):
            buf = self.engine.get_buffer_seconds()
            self.status.setText(f"{self.engine.dsp_name()} | {'Loading…' if self.engine.state==PlayerState.LOADING else 'Playing'} | Buffer: {buf:.2f}s")
        elif self.engine.state == PlayerState.PAUSED:
            self.status.setText(f"{self.engine.dsp_name()} | Paused")
        elif self.engine.state == PlayerState.STOPPED:
            self.status.setText(f"{self.engine.dsp_name()} | Stopped")

        self._update_enabled_fx_label()

        # Auto-advance (best-effort)
        if self._dur > 0 and pos >= self._dur - 0.25 and self.engine.state == PlayerState.PLAYING:
            self._advance_track(direction=1, auto=True)

    def _update_enabled_fx_label(self) -> None:
        enabled = self.engine.get_enabled_effects()
        label_text = f"Enabled FX: {', '.join(enabled)}" if enabled else "Enabled FX: None"
        if self.fx_status.text() != label_text:
            self.fx_status.setText(label_text)

    def closeEvent(self, e: QtGui.QCloseEvent):
        self._save_ui_settings()
        self.settings.setValue("playlist/paths", self.playlist.track_paths())
        current_index = self.playlist.current_index()
        if current_index < 0:
            current_index = self._current_index
        self.settings.setValue("playlist/current_index", current_index)
        self.settings.setValue("playlist/position_sec", self.engine.get_position())
        self.engine.stop()
        super().closeEvent(e)

    def _restore_ui_settings(self):
        volume_value = self.settings.value("audio/volume_slider", self.transport.volume_slider.value(), type=int)
        self.transport.volume_slider.setValue(int(volume_value))
        self.transport.mute_btn.setChecked(self.settings.value("audio/muted", False, type=bool))

        buffer_preset = str(self.settings.value("audio/buffer_preset", DEFAULT_BUFFER_PRESET))
        if buffer_preset not in BUFFER_PRESETS:
            buffer_preset = DEFAULT_BUFFER_PRESET
        self.buffer_preset_combo.blockSignals(True)
        self.buffer_preset_combo.setCurrentText(buffer_preset)
        self.buffer_preset_combo.blockSignals(False)
        self.engine.set_buffer_preset(buffer_preset)
        metrics_enabled = self.settings.value("audio/metrics_enabled", True, type=bool)
        self.metrics_checkbox.blockSignals(True)
        self.metrics_checkbox.setChecked(bool(metrics_enabled))
        self.metrics_checkbox.blockSignals(False)
        self.engine.set_metrics_enabled(bool(metrics_enabled))

        tempo = self.settings.value("dsp/tempo", 1.0, type=float)
        pitch = self.settings.value("dsp/pitch", 0.0, type=float)
        key_lock = self.settings.value("dsp/key_lock", True, type=bool)
        tape_mode = self.settings.value("dsp/tape_mode", False, type=bool)
        lock_432 = self.settings.value("dsp/lock_432", False, type=bool)

        tempo_value = int(round(clamp(float(tempo), 0.5, 2.0) * 100))
        pitch_value = int(round(clamp(float(pitch), -12.0, 12.0) * 10))

        self.dsp_widget.tempo_slider.setValue(tempo_value)
        self.dsp_widget.pitch_slider.setValue(pitch_value)
        self.dsp_widget.key_lock.setChecked(bool(key_lock))
        self.dsp_widget.lock_432.setChecked(bool(lock_432))
        self.dsp_widget.tape_mode.setChecked(bool(tape_mode))

        eq_preset = str(self.settings.value("eq/preset", "Flat"))
        eq_gains_raw = self.settings.value("eq/gains", [0.0] * len(self.equalizer.band_sliders))
        eq_gains = self._normalize_eq_gains(eq_gains_raw, len(self.equalizer.band_sliders))
        if eq_preset in self.equalizer.presets_map:
            self.equalizer.set_gains(eq_gains, preset=eq_preset, emit=False)
        else:
            self.equalizer.set_gains(eq_gains, preset="Custom", emit=False)

        dynamic_eq_freq = self.settings.value("dynamic_eq/freq", 1000.0, type=float)
        dynamic_eq_q = self.settings.value("dynamic_eq/q", 1.0, type=float)
        dynamic_eq_gain = self.settings.value("dynamic_eq/gain", 0.0, type=float)
        dynamic_eq_threshold = self.settings.value("dynamic_eq/threshold", -24.0, type=float)
        dynamic_eq_ratio = self.settings.value("dynamic_eq/ratio", 4.0, type=float)

        self.dynamic_eq_widget.freq_slider.setValue(
            self.dynamic_eq_widget._freq_to_slider(
                clamp(dynamic_eq_freq, 20.0, 20000.0)
            )
        )
        self.dynamic_eq_widget.q_slider.setValue(
            int(round(clamp(dynamic_eq_q, 0.1, 20.0) * 10))
        )
        self.dynamic_eq_widget.gain_slider.setValue(
            int(round(clamp(dynamic_eq_gain, -12.0, 12.0) * 10))
        )
        self.dynamic_eq_widget.threshold_slider.setValue(
            int(round(clamp(dynamic_eq_threshold, -60.0, 0.0) * 10))
        )
        self.dynamic_eq_widget.ratio_slider.setValue(
            int(round(clamp(dynamic_eq_ratio, 1.0, 20.0) * 10))
        )

        compressor_threshold = self.settings.value("compressor/threshold", -18.0, type=float)
        compressor_ratio = self.settings.value("compressor/ratio", 4.0, type=float)
        compressor_attack = self.settings.value("compressor/attack", 10.0, type=float)
        compressor_release = self.settings.value("compressor/release", 120.0, type=float)
        compressor_makeup = self.settings.value("compressor/makeup", 0.0, type=float)

        self.compressor_widget.threshold_slider.setValue(
            int(round(clamp(compressor_threshold, -60.0, 0.0) * 10))
        )
        self.compressor_widget.ratio_slider.setValue(
            int(round(clamp(compressor_ratio, 1.0, 20.0) * 10))
        )
        self.compressor_widget.attack_slider.setValue(
            int(round(clamp(compressor_attack, 0.1, 200.0) * 10))
        )
        self.compressor_widget.release_slider.setValue(
            int(round(clamp(compressor_release, 1.0, 1000.0)))
        )
        self.compressor_widget.makeup_slider.setValue(
            int(round(clamp(compressor_makeup, 0.0, 24.0) * 10))
        )

        saturation_drive = self.settings.value("saturation/drive", 6.0, type=float)
        saturation_trim = self.settings.value("saturation/trim", 0.0, type=float)
        saturation_tone = self.settings.value("saturation/tone", 0.0, type=float)
        saturation_tone_enabled = self.settings.value("saturation/tone_enabled", False, type=bool)

        self.saturation_widget.drive_slider.setValue(
            int(round(clamp(saturation_drive, 0.0, 24.0) * 10))
        )
        self.saturation_widget.trim_slider.setValue(
            int(round(clamp(saturation_trim, -24.0, 24.0) * 10))
        )
        self.saturation_widget.tone_slider.setValue(
            int(round(clamp(saturation_tone, -1.0, 1.0) * 100))
        )
        self.saturation_widget.tone_toggle.setChecked(bool(saturation_tone_enabled))

        subharmonic_mix = self.settings.value("subharmonic/mix", 0.25, type=float)
        subharmonic_intensity = self.settings.value("subharmonic/intensity", 0.6, type=float)
        subharmonic_cutoff = self.settings.value("subharmonic/cutoff", 140.0, type=float)

        self.subharmonic_widget.mix_slider.setValue(
            int(round(clamp(subharmonic_mix, 0.0, 1.0) * 100))
        )
        self.subharmonic_widget.intensity_slider.setValue(
            int(round(clamp(subharmonic_intensity, 0.0, 1.5) * 100))
        )
        self.subharmonic_widget.cutoff_slider.setValue(
            int(round(clamp(subharmonic_cutoff, 60.0, 240.0)))
        )

        limiter_threshold = self.settings.value("limiter/threshold", -1.0, type=float)
        limiter_release = self.settings.value("limiter/release", 80.0, type=float)
        limiter_release_enabled = self.settings.value("limiter/release_enabled", True, type=bool)

        self.limiter_widget.threshold_slider.setValue(
            int(round(clamp(limiter_threshold, -60.0, 0.0) * 10))
        )
        self.limiter_widget.release_slider.setValue(
            int(round(clamp(limiter_release, 1.0, 1000.0)))
        )
        self.limiter_widget.release_toggle.setChecked(bool(limiter_release_enabled))

        reverb_decay = self.settings.value("reverb/decay", 1.4, type=float)
        reverb_predelay = self.settings.value("reverb/predelay", 20.0, type=float)
        reverb_mix = self.settings.value("reverb/mix", 0.25, type=float)

        self.reverb_widget.decay_slider.setValue(int(round(clamp(reverb_decay, 0.2, 6.0) * 100)))
        self.reverb_widget.predelay_slider.setValue(int(round(clamp(reverb_predelay, 0.0, 120.0))))
        self.reverb_widget.mix_slider.setValue(int(round(clamp(reverb_mix, 0.0, 1.0) * 100)))

        chorus_rate = self.settings.value("chorus/rate", 0.8, type=float)
        chorus_depth = self.settings.value("chorus/depth", 8.0, type=float)
        chorus_mix = self.settings.value("chorus/mix", 0.25, type=float)

        self.chorus_widget.rate_slider.setValue(int(round(clamp(chorus_rate, 0.05, 5.0) * 100)))
        self.chorus_widget.depth_slider.setValue(int(round(clamp(chorus_depth, 0.0, 20.0) * 10)))
        self.chorus_widget.mix_slider.setValue(int(round(clamp(chorus_mix, 0.0, 1.0) * 100)))

        panner_azimuth = self.settings.value("panner/azimuth", 0.0, type=float)
        panner_spread = self.settings.value("panner/spread", 1.0, type=float)

        self.stereo_panner_widget.azimuth_slider.setValue(
            int(round(clamp(panner_azimuth, -90.0, 90.0)))
        )
        self.stereo_panner_widget.spread_slider.setValue(
            int(round(clamp(panner_spread, 0.0, 1.0) * 100))
        )

        stereo_width = self.settings.value("stereo/width", 1.0, type=float)
        self.stereo_width_widget.width_slider.setValue(int(round(clamp(stereo_width, 0.0, 2.0) * 100)))

        enabled_effects = set(self.engine.get_enabled_effects())
        for name, checkbox in self.effect_toggles.items():
            enabled = self.settings.value(self._effect_setting_key(name), name in enabled_effects, type=bool)
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(enabled))
            checkbox.blockSignals(False)
            self.engine.enable_effect(name, bool(enabled))
        self._update_enabled_fx_label()

    def _apply_ui_settings(self):
        self.engine.set_volume(self.transport.volume_slider.value() / 100.0)
        self.engine.set_muted(self.transport.mute_btn.isChecked())
        tempo = self.dsp_widget.tempo_slider.value() / 100.0
        pitch = self.dsp_widget.pitch_slider.value() / 10.0
        self.engine.set_dsp_controls(
            tempo,
            pitch,
            self.dsp_widget.key_lock.isChecked(),
            self.dsp_widget.tape_mode.isChecked(),
        )
        self.engine.set_eq_gains(self.equalizer.gains())
        self.engine.set_dynamic_eq_controls(
            self.dynamic_eq_widget._slider_to_freq(self.dynamic_eq_widget.freq_slider.value()),
            self.dynamic_eq_widget.q_slider.value() / 10.0,
            self.dynamic_eq_widget.gain_slider.value() / 10.0,
            self.dynamic_eq_widget.threshold_slider.value() / 10.0,
            self.dynamic_eq_widget.ratio_slider.value() / 10.0,
        )
        self.engine.set_compressor_controls(
            self.compressor_widget.threshold_slider.value() / 10.0,
            self.compressor_widget.ratio_slider.value() / 10.0,
            self.compressor_widget.attack_slider.value() / 10.0,
            float(self.compressor_widget.release_slider.value()),
            self.compressor_widget.makeup_slider.value() / 10.0,
        )
        self.engine.set_saturation_controls(
            self.saturation_widget.drive_slider.value() / 10.0,
            self.saturation_widget.trim_slider.value() / 10.0,
            self.saturation_widget.tone_slider.value() / 100.0,
            self.saturation_widget.tone_toggle.isChecked(),
        )
        self.engine.set_subharmonic_controls(
            self.subharmonic_widget.mix_slider.value() / 100.0,
            self.subharmonic_widget.intensity_slider.value() / 100.0,
            float(self.subharmonic_widget.cutoff_slider.value()),
        )
        limiter_release = (
            float(self.limiter_widget.release_slider.value())
            if self.limiter_widget.release_toggle.isChecked()
            else None
        )
        self.engine.set_limiter_controls(
            self.limiter_widget.threshold_slider.value() / 10.0,
            limiter_release,
        )
        self.engine.set_reverb_controls(
            self.reverb_widget.decay_slider.value() / 100.0,
            float(self.reverb_widget.predelay_slider.value()),
            self.reverb_widget.mix_slider.value() / 100.0,
        )
        self.engine.set_chorus_controls(
            self.chorus_widget.rate_slider.value() / 100.0,
            self.chorus_widget.depth_slider.value() / 10.0,
            self.chorus_widget.mix_slider.value() / 100.0,
        )
        self.engine.set_stereo_panner_controls(
            float(self.stereo_panner_widget.azimuth_slider.value()),
            self.stereo_panner_widget.spread_slider.value() / 100.0,
        )
        self.engine.set_stereo_width(self.stereo_width_widget.width_slider.value() / 100.0)

    def _save_ui_settings(self):
        self.settings.setValue("audio/volume_slider", self.transport.volume_slider.value())
        self.settings.setValue("audio/muted", self.transport.mute_btn.isChecked())
        self.settings.setValue("audio/metrics_enabled", self.metrics_checkbox.isChecked())
        self.settings.setValue("dsp/tempo", self.dsp_widget.tempo_slider.value() / 100.0)
        self.settings.setValue("dsp/pitch", self.dsp_widget.pitch_slider.value() / 10.0)
        self.settings.setValue("dsp/key_lock", self.dsp_widget.key_lock.isChecked())
        self.settings.setValue("dsp/tape_mode", self.dsp_widget.tape_mode.isChecked())
        self.settings.setValue("dsp/lock_432", self.dsp_widget.lock_432.isChecked())
        self.settings.setValue("eq/gains", self.equalizer.gains())
        self.settings.setValue("eq/preset", self.equalizer.presets.currentText())
        self.settings.setValue(
            "dynamic_eq/freq",
            self.dynamic_eq_widget._slider_to_freq(self.dynamic_eq_widget.freq_slider.value()),
        )
        self.settings.setValue("dynamic_eq/q", self.dynamic_eq_widget.q_slider.value() / 10.0)
        self.settings.setValue("dynamic_eq/gain", self.dynamic_eq_widget.gain_slider.value() / 10.0)
        self.settings.setValue(
            "dynamic_eq/threshold", self.dynamic_eq_widget.threshold_slider.value() / 10.0
        )
        self.settings.setValue("dynamic_eq/ratio", self.dynamic_eq_widget.ratio_slider.value() / 10.0)
        self.settings.setValue("compressor/threshold", self.compressor_widget.threshold_slider.value() / 10.0)
        self.settings.setValue("compressor/ratio", self.compressor_widget.ratio_slider.value() / 10.0)
        self.settings.setValue("compressor/attack", self.compressor_widget.attack_slider.value() / 10.0)
        self.settings.setValue("compressor/release", float(self.compressor_widget.release_slider.value()))
        self.settings.setValue("compressor/makeup", self.compressor_widget.makeup_slider.value() / 10.0)
        self.settings.setValue("saturation/drive", self.saturation_widget.drive_slider.value() / 10.0)
        self.settings.setValue("saturation/trim", self.saturation_widget.trim_slider.value() / 10.0)
        self.settings.setValue("saturation/tone", self.saturation_widget.tone_slider.value() / 100.0)
        self.settings.setValue("saturation/tone_enabled", self.saturation_widget.tone_toggle.isChecked())
        self.settings.setValue("subharmonic/mix", self.subharmonic_widget.mix_slider.value() / 100.0)
        self.settings.setValue(
            "subharmonic/intensity", self.subharmonic_widget.intensity_slider.value() / 100.0
        )
        self.settings.setValue("subharmonic/cutoff", float(self.subharmonic_widget.cutoff_slider.value()))
        self.settings.setValue("limiter/threshold", self.limiter_widget.threshold_slider.value() / 10.0)
        self.settings.setValue("limiter/release", float(self.limiter_widget.release_slider.value()))
        self.settings.setValue("limiter/release_enabled", self.limiter_widget.release_toggle.isChecked())
        self.settings.setValue("reverb/decay", self.reverb_widget.decay_slider.value() / 100.0)
        self.settings.setValue("reverb/predelay", float(self.reverb_widget.predelay_slider.value()))
        self.settings.setValue("reverb/mix", self.reverb_widget.mix_slider.value() / 100.0)
        self.settings.setValue("chorus/rate", self.chorus_widget.rate_slider.value() / 100.0)
        self.settings.setValue("chorus/depth", self.chorus_widget.depth_slider.value() / 10.0)
        self.settings.setValue("chorus/mix", self.chorus_widget.mix_slider.value() / 100.0)
        self.settings.setValue("panner/azimuth", float(self.stereo_panner_widget.azimuth_slider.value()))
        self.settings.setValue("panner/spread", self.stereo_panner_widget.spread_slider.value() / 100.0)
        self.settings.setValue("stereo/width", self.stereo_width_widget.width_slider.value() / 100.0)
        for name, checkbox in self.effect_toggles.items():
            self.settings.setValue(self._effect_setting_key(name), checkbox.isChecked())

    @staticmethod
    def _normalize_eq_gains(values: object, band_count: int) -> list[float]:
        if isinstance(values, (tuple, list)):
            gains = [safe_float(str(v), 0.0) for v in values]
        else:
            gains = []
        if len(gains) < band_count:
            gains.extend([0.0] * (band_count - len(gains)))
        return [float(g) for g in gains[:band_count]]

    @staticmethod
    def _effect_setting_key(name: str) -> str:
        return f"effects/enabled/{name}"

    def _restore_playlist_session(self):
        saved_paths = self.settings.value("playlist/paths", [], type=list)
        if saved_paths:
            self._add_paths(saved_paths)

        saved_index = self.settings.value("playlist/current_index", -1, type=int)
        saved_pos = self.settings.value("playlist/position_sec", 0.0, type=float)

        if saved_index is None:
            return

        idx = int(saved_index)
        if 0 <= idx < self.playlist.count():
            self._current_index = idx
            self.playlist.select_index(idx)
            track = self.playlist.get_track(idx)
            if track:
                self.engine.load_track(track.path)
                if saved_pos and saved_pos > 0:
                    self.engine.seek(float(saved_pos))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
