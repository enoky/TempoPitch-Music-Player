
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
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque
from typing import Optional, List

import numpy as np

try:
    from scipy.signal import sosfilt
except Exception:
    sosfilt = None
    try:
        from numba import njit
    except Exception:
        njit = None

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    _sounddevice_import_error = e

from PySide6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)
EQ_PROFILE = False
EQ_PROFILE_LOG_EVERY = 50
EQ_PROFILE_LOW_WATERMARK_SEC = 0.25


# -----------------------------
# Utilities
# -----------------------------

def have_exe(name: str) -> bool:
    return shutil.which(name) is not None

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def semitones_to_factor(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))

def format_time(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    total = int(seconds + 0.5)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def adjust_color(color: str, *, lighter: Optional[int] = None, darker: Optional[int] = None) -> str:
    qt_color = QtGui.QColor(color)
    if lighter is not None:
        qt_color = qt_color.lighter(lighter)
    if darker is not None:
        qt_color = qt_color.darker(darker)
    return qt_color.name()

def adjust_color(color: str, *, lighter: Optional[int] = None, darker: Optional[int] = None) -> str:
    qt_color = QtGui.QColor(color)
    if lighter is not None:
        qt_color = qt_color.lighter(lighter)
    if darker is not None:
        qt_color = qt_color.darker(darker)
    return qt_color.name()


# -----------------------------
# Models
# -----------------------------

@dataclass
class Track:
    path: str
    title: str
    duration_sec: float
    artist: str = ""
    album: str = ""
    title_display: str = ""
    cover_art: Optional[bytes] = None


@dataclass
class TrackMetadata:
    duration_sec: float
    artist: str
    album: str
    title: str
    cover_art: Optional[bytes]


def format_track_title(track: Track) -> str:
    title = track.title_display or track.title or os.path.basename(track.path)
    artist = track.artist.strip()
    if artist:
        return f"{artist} — {title}"
    return title


class PlayerState(Enum):
    STOPPED = auto()
    LOADING = auto()
    PLAYING = auto()
    PAUSED = auto()
    ERROR = auto()


class RepeatMode(Enum):
    OFF = "off"
    ALL = "all"
    ONE = "one"

    @classmethod
    def from_setting(cls, value: str) -> "RepeatMode":
        for mode in cls:
            if mode.value == value:
                return mode
        return cls.OFF


# -----------------------------
# Theme
# -----------------------------

@dataclass(frozen=True)
class Theme:
    name: str
    window: str
    base: str
    text: str
    highlight: str
    accent: str
    card: str


THEMES = {
    "Ocean": Theme(
        name="Ocean",
        window="#0f172a",
        base="#0b1220",
        text="#e2e8f0",
        highlight="#38bdf8",
        accent="#22d3ee",
        card="#111c30",
    ),
    "Sunset": Theme(
        name="Sunset",
        window="#2b1d20",
        base="#201417",
        text="#fde8e8",
        highlight="#fb7185",
        accent="#f97316",
        card="#372125",
    ),
    "Forest": Theme(
        name="Forest",
        window="#0f1f17",
        base="#0b1510",
        text="#e7f6ef",
        highlight="#34d399",
        accent="#10b981",
        card="#14271d",
    ),
    "Rose": Theme(
        name="Rose",
        window="#2a1621",
        base="#1f1018",
        text="#fde7f2",
        highlight="#f472b6",
        accent="#fb7185",
        card="#331a28",
    ),
    "Slate": Theme(
        name="Slate",
        window="#1f2937",
        base="#111827",
        text="#f8fafc",
        highlight="#60a5fa",
        accent="#94a3b8",
        card="#263243",
    ),
}


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
        }}
        QSlider::groove:horizontal {{
            height: 6px;
            background: {adjust_color(theme.base, lighter=110)};
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
        QListWidget {{
            padding: 8px;
            border-radius: 10px;
            border: 1px solid {border};
            background: {theme.base};
        }}
        QLabel#now_playing {{
            font-size: 16px;
            font-weight: 700;
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
        QLabel#playlist_header {{
            font-size: 14px;
            font-weight: 600;
            color: {theme.text};
        }}
        QSplitter::handle {{
            background: {adjust_color(theme.window, lighter=110)};
        }}
    """

# -----------------------------
# Thread-safe ring buffer
# -----------------------------

class AudioRingBuffer:
    """
    Thread-safe audio buffer as deque of numpy arrays.

    push(frames): frames (n, ch) float32
    pop(n): returns exactly (n, ch) float32, zero-padded on underrun
    """
    def __init__(self, channels: int, max_seconds: float, sample_rate: int):
        self.channels = channels
        self.sample_rate = sample_rate
        self.max_frames = int(max_seconds * sample_rate)
        self._dq: deque[np.ndarray] = deque()
        self._frames = 0
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()
            self._frames = 0

    def frames_available(self) -> int:
        with self._lock:
            return self._frames

    def push(self, frames: np.ndarray) -> None:
        if frames.size == 0:
            return
        if frames.dtype != np.float32:
            frames = frames.astype(np.float32, copy=False)
        if frames.ndim != 2 or frames.shape[1] != self.channels:
            raise ValueError(f"frames must be (n,{self.channels}) float32, got {frames.shape} {frames.dtype}")

        with self._lock:
            # IMPORTANT FOR PLAYBACK QUALITY:
            # Never drop *old* audio (that causes time-jumps/garble if the decoder runs ahead).
            # If the buffer is full, we either accept a partial chunk or drop the *new* tail.
            space = self.max_frames - self._frames
            if space <= 0:
                return

            if frames.shape[0] > space:
                frames = frames[:space, :]

            self._dq.append(frames)
            self._frames += frames.shape[0]

    def pop(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        out = np.zeros((n, self.channels), dtype=np.float32)
        idx = 0
        with self._lock:
            while idx < n and self._dq:
                chunk = self._dq[0]
                take = min(n - idx, chunk.shape[0])
                out[idx:idx + take] = chunk[:take]
                idx += take
                if take == chunk.shape[0]:
                    self._dq.popleft()
                else:
                    self._dq[0] = chunk[take:, :]
                self._frames -= take
        return out


# -----------------------------
# Thread-safe visualizer buffer
# -----------------------------

class VisualizerBuffer:
    """
    Thread-safe ring buffer for recent audio snapshots.

    push(frames): frames (n, ch) float32
    get_recent(n, mono): returns most recent frames, optional mono downmix
    """
    def __init__(self, channels: int, max_seconds: float, sample_rate: int):
        self.channels = channels
        self.sample_rate = sample_rate
        self.max_frames = max(1, int(max_seconds * sample_rate))
        self._buffer = np.zeros((self.max_frames, channels), dtype=np.float32)
        self._write_index = 0
        self._filled = 0
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._write_index = 0
            self._filled = 0

    def push(self, frames: np.ndarray) -> None:
        if frames.size == 0:
            return
        if frames.dtype != np.float32:
            frames = frames.astype(np.float32, copy=False)
        if frames.ndim != 2 or frames.shape[1] != self.channels:
            raise ValueError(f"frames must be (n,{self.channels}) float32, got {frames.shape} {frames.dtype}")

        if frames.shape[0] > self.max_frames:
            frames = frames[-self.max_frames:, :]

        n = frames.shape[0]
        with self._lock:
            end = self._write_index + n
            if end <= self.max_frames:
                self._buffer[self._write_index:end, :] = frames
            else:
                first = self.max_frames - self._write_index
                self._buffer[self._write_index:, :] = frames[:first, :]
                remaining = end - self.max_frames
                self._buffer[:remaining, :] = frames[first:, :]
            self._write_index = end % self.max_frames
            self._filled = min(self.max_frames, self._filled + n)

    def get_recent(self, frames: Optional[int] = None, mono: bool = False) -> np.ndarray:
        with self._lock:
            if self._filled == 0:
                data = np.zeros((0, self.channels), dtype=np.float32)
            elif self._filled < self.max_frames:
                data = self._buffer[:self._filled, :]
            else:
                if self._write_index == 0:
                    data = self._buffer
                else:
                    data = np.vstack((self._buffer[self._write_index:, :], self._buffer[:self._write_index, :]))

            if frames is not None and frames > 0:
                data = data[-frames:, :]

            data = np.array(data, copy=True)

        if mono and data.size:
            mono_data = data.mean(axis=1, dtype=np.float32)
            return mono_data.reshape(-1, 1)
        return data


# -----------------------------
# DSP Interfaces
# -----------------------------

class DSPBase:
    name: str = "DSP"
    def set_controls(self, tempo: float, pitch_semitones: float, key_lock: bool, tape_mode: bool) -> None:
        raise NotImplementedError
    def reset(self) -> None:
        raise NotImplementedError
    def process(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def flush(self) -> np.ndarray:
        """Drain any remaining buffered audio and return it."""
        return np.zeros((0, 2), dtype=np.float32)


# -----------------------------
# Equalizer DSP (biquad peaking filters)
# -----------------------------

@dataclass(frozen=True)
class EqConfig:
    sos: np.ndarray
    zi: np.ndarray
    reset_mask: np.ndarray


def _df2_process_python(x: np.ndarray, sos: np.ndarray, zi: np.ndarray) -> np.ndarray:
    n_frames, n_ch = x.shape
    n_bands = sos.shape[0]
    for band in range(n_bands):
        b0, b1, b2 = sos[band, 0], sos[band, 1], sos[band, 2]
        a1, a2 = sos[band, 4], sos[band, 5]
        for ch in range(n_ch):
            z1 = zi[band, ch, 0]
            z2 = zi[band, ch, 1]
            for i in range(n_frames):
                x_n = x[i, ch]
                y_n = b0 * x_n + z1
                z1 = b1 * x_n - a1 * y_n + z2
                z2 = b2 * x_n - a2 * y_n
                x[i, ch] = y_n
            zi[band, ch, 0] = z1
            zi[band, ch, 1] = z2
    return x


if sosfilt is None and "njit" in globals() and njit is not None:
    @njit(cache=True)
    def _df2_process_numba(x: np.ndarray, sos: np.ndarray, zi: np.ndarray) -> np.ndarray:
        n_frames, n_ch = x.shape
        n_bands = sos.shape[0]
        for band in range(n_bands):
            b0 = sos[band, 0]
            b1 = sos[band, 1]
            b2 = sos[band, 2]
            a1 = sos[band, 4]
            a2 = sos[band, 5]
            for ch in range(n_ch):
                z1 = zi[band, ch, 0]
                z2 = zi[band, ch, 1]
                for i in range(n_frames):
                    x_n = x[i, ch]
                    y_n = b0 * x_n + z1
                    z1 = b1 * x_n - a1 * y_n + z2
                    z2 = b2 * x_n - a2 * y_n
                    x[i, ch] = y_n
                zi[band, ch, 0] = z1
                zi[band, ch, 1] = z2
        return x
else:
    _df2_process_numba = None


class EqualizerDSP:
    name = "Equalizer"

    def __init__(self, sample_rate: int, channels: int):
        self.sr = int(sample_rate)
        self.ch = int(channels)
        self.center_freqs = [31.0, 62.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
        self.q = 1.0
        self._lock = threading.Lock()
        self._pending_reset = False
        self._reset_all = False
        self._gains_db = [0.0 for _ in self.center_freqs]
        self._config = EqConfig(
            sos=np.zeros((0, 6), dtype=np.float32),
            zi=np.zeros((0, self.ch, 2), dtype=np.float32),
            reset_mask=np.zeros((0,), dtype=bool),
        )
        self._config = self._build_config(self._gains_db, reset_mask=np.zeros((0,), dtype=bool))

    def reset(self) -> None:
        self._pending_reset = True
        self._reset_all = True

    def set_eq_gains(self, gains_db: list[float]) -> None:
        if len(gains_db) != len(self.center_freqs):
            raise ValueError(f"EqualizerDSP expects {len(self.center_freqs)} gains")
        new_gains = [clamp(float(g), -12.0, 12.0) for g in gains_db]
        old_gains = self._gains_db
        if new_gains == old_gains:
            return
        config, reset_mask = self._compute_config(new_gains, old_gains)
        pending_reset = bool(reset_mask.any())
        with self._lock:
            if new_gains == self._gains_db:
                return
            self._gains_db = new_gains
            self._config = config
            self._pending_reset = pending_reset
            self._reset_all = False

    def _build_config(self, gains_db: list[float], reset_mask: np.ndarray) -> EqConfig:
        sos_rows = []
        for f0, gain_db in zip(self.center_freqs, gains_db):
            A = 10.0 ** (gain_db / 40.0)
            w0 = 2.0 * math.pi * f0 / float(self.sr)
            cos_w0 = math.cos(w0)
            sin_w0 = math.sin(w0)
            alpha = sin_w0 / (2.0 * self.q)

            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A

            b0 /= a0
            b1 /= a0
            b2 /= a0
            a1 /= a0
            a2 /= a0

            if abs(gain_db) > 1e-3:
                sos_rows.append((b0, b1, b2, 1.0, a1, a2))
        sos = np.array(sos_rows, dtype=np.float32)
        zi = np.zeros((sos.shape[0], self.ch, 2), dtype=np.float32)
        if reset_mask.shape[0] != sos.shape[0]:
            reset_mask = np.zeros((sos.shape[0],), dtype=bool)
        return EqConfig(sos=sos, zi=zi, reset_mask=reset_mask)

    def _compute_config(self, new_gains: list[float], old_gains: list[float]) -> tuple[EqConfig, np.ndarray]:
        sos_rows = []
        active_indices = []
        for i, (f0, gain_db) in enumerate(zip(self.center_freqs, new_gains)):
            A = 10.0 ** (gain_db / 40.0)
            w0 = 2.0 * math.pi * f0 / float(self.sr)
            cos_w0 = math.cos(w0)
            sin_w0 = math.sin(w0)
            alpha = sin_w0 / (2.0 * self.q)

            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A

            b0 /= a0
            b1 /= a0
            b2 /= a0
            a1 /= a0
            a2 /= a0

            if abs(gain_db) > 1e-3:
                active_indices.append(i)
                sos_rows.append((b0, b1, b2, 1.0, a1, a2))
        sos = np.array(sos_rows, dtype=np.float32)
        zi = np.zeros((sos.shape[0], self.ch, 2), dtype=np.float32)
        reset_mask = np.array(
            [abs(new_gains[i] - old_gains[i]) > 1e-6 for i in active_indices],
            dtype=bool,
        )
        return EqConfig(sos=sos, zi=zi, reset_mask=reset_mask), reset_mask

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        y = np.array(x, copy=True)
        config = self._config
        if self._pending_reset:
            if self._reset_all:
                config.zi.fill(0.0)
            elif config.reset_mask.size > 0:
                config.zi[config.reset_mask, :, :] = 0.0
            self._pending_reset = False
            self._reset_all = False
        if config.sos.shape[0] == 0:
            return y
        if sosfilt is not None:
            for ch in range(self.ch):
                y[:, ch], zi = sosfilt(config.sos, y[:, ch], zi=config.zi[:, ch, :])
                config.zi[:, ch, :] = zi
        elif _df2_process_numba is not None:
            y = _df2_process_numba(y, config.sos, config.zi)
        else:
            y = _df2_process_python(y, config.sos, config.zi)
        return y


# -----------------------------
# Modular DSP effects chain
# -----------------------------

class EffectProcessor:
    name = "Effect"

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    def reset(self) -> None:
        return None

    def process(self, x: np.ndarray) -> np.ndarray:
        return x


class GainEffect(EffectProcessor):
    name = "Gain"

    def __init__(self, gain_db: float = 0.0, enabled: bool = False):
        super().__init__(enabled=enabled)
        self._gain_db = float(gain_db)

    def set_gain_db(self, gain_db: float) -> None:
        self._gain_db = float(gain_db)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        gain = 10.0 ** (self._gain_db / 20.0)
        return x * gain


class CompressorEffect(EffectProcessor):
    name = "Compressor"

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        threshold_db: float = -18.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 120.0,
        makeup_db: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._threshold_db = float(threshold_db)
        self._ratio = float(ratio)
        self._attack_ms = float(attack_ms)
        self._release_ms = float(release_ms)
        self._makeup_db = float(makeup_db)
        self._env = 0.0
        self._last_reduction_db = 0.0
        self._attack_coeff = 0.0
        self._release_coeff = 0.0
        self._update_coeffs()

    def reset(self) -> None:
        self._env = 0.0
        self._last_reduction_db = 0.0

    def _update_coeffs(self) -> None:
        attack_ms = max(0.1, float(self._attack_ms))
        release_ms = max(1.0, float(self._release_ms))
        self._attack_coeff = math.exp(-1.0 / (self.sample_rate * (attack_ms / 1000.0)))
        self._release_coeff = math.exp(-1.0 / (self.sample_rate * (release_ms / 1000.0)))

    def set_parameters(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        makeup_db: float,
    ) -> None:
        self._threshold_db = clamp(float(threshold_db), -60.0, 0.0)
        self._ratio = clamp(float(ratio), 1.0, 20.0)
        self._attack_ms = clamp(float(attack_ms), 0.1, 200.0)
        self._release_ms = clamp(float(release_ms), 1.0, 1000.0)
        self._makeup_db = clamp(float(makeup_db), 0.0, 24.0)
        self._update_coeffs()

    def gain_reduction_db(self) -> float:
        return float(self._last_reduction_db)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        env = self._env
        threshold = self._threshold_db
        ratio = self._ratio
        makeup_db = self._makeup_db
        attack_coeff = self._attack_coeff
        release_coeff = self._release_coeff
        last_reduction = self._last_reduction_db

        out = np.empty_like(x)
        for i in range(x.shape[0]):
            peak = float(np.max(np.abs(x[i])))
            coeff = attack_coeff if peak > env else release_coeff
            env = coeff * env + (1.0 - coeff) * peak

            env_db = 20.0 * math.log10(max(env, 1e-8))
            if env_db <= threshold:
                gain_db = 0.0
            else:
                gain_db = threshold + (env_db - threshold) / ratio - env_db
            reduction_db = max(0.0, -gain_db)
            last_reduction = reduction_db

            gain = 10.0 ** ((gain_db + makeup_db) / 20.0)
            out[i] = x[i] * gain

        self._env = env
        self._last_reduction_db = last_reduction
        return out


class LimiterEffect(EffectProcessor):
    name = "Limiter"

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        threshold_db: float = -1.0,
        release_ms: Optional[float] = 80.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._threshold_db = float(threshold_db)
        self._threshold_amp = 1.0
        self._release_ms = release_ms if release_ms is None else float(release_ms)
        self._release_coeff = 0.0
        self._gain = 1.0
        self._update_params()

    def reset(self) -> None:
        self._gain = 1.0

    def _update_params(self) -> None:
        self._threshold_db = clamp(float(self._threshold_db), -60.0, 0.0)
        self._threshold_amp = 10.0 ** (self._threshold_db / 20.0)
        if self._release_ms is None:
            self._release_coeff = 0.0
            return
        release_ms = max(1.0, float(self._release_ms))
        self._release_coeff = math.exp(-1.0 / (self.sample_rate * (release_ms / 1000.0)))

    def set_parameters(self, threshold_db: float, release_ms: Optional[float]) -> None:
        self._threshold_db = threshold_db
        self._release_ms = release_ms if release_ms is None else float(release_ms)
        self._update_params()

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        threshold_amp = self._threshold_amp
        release_coeff = self._release_coeff
        use_release = self._release_ms is not None
        gain = self._gain

        out = np.empty_like(x)
        for i in range(x.shape[0]):
            peak = float(np.max(np.abs(x[i])))
            if peak > threshold_amp:
                gain = threshold_amp / max(peak, 1e-8)
            elif use_release:
                gain = release_coeff * gain + (1.0 - release_coeff)
            else:
                gain = 1.0
            out[i] = x[i] * gain

        self._gain = gain
        if threshold_amp < 1.0:
            np.clip(out, -threshold_amp, threshold_amp, out=out)
        else:
            np.clip(out, -1.0, 1.0, out=out)
        return out


class SaturationEffect(EffectProcessor):
    name = "Saturation"

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        drive_db: float = 6.0,
        trim_db: float = 0.0,
        tone: float = 0.0,
        tone_enabled: bool = False,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._drive_db = float(drive_db)
        self._trim_db = float(trim_db)
        self._tone = clamp(float(tone), -1.0, 1.0)
        self._tone_enabled = bool(tone_enabled)
        self._drive_gain = 1.0
        self._trim_gain = 1.0
        self._tone_coeff = 0.0
        self._tone_state = np.zeros(self.channels, dtype=np.float32)
        self._update_params()

    def reset(self) -> None:
        self._tone_state.fill(0.0)

    def _update_params(self) -> None:
        self._drive_db = clamp(float(self._drive_db), 0.0, 24.0)
        self._trim_db = clamp(float(self._trim_db), -24.0, 24.0)
        self._tone = clamp(float(self._tone), -1.0, 1.0)
        self._drive_gain = 10.0 ** (self._drive_db / 20.0)
        self._trim_gain = 10.0 ** (self._trim_db / 20.0)
        cutoff_hz = 2200.0
        self._tone_coeff = math.exp(-2.0 * math.pi * cutoff_hz / float(self.sample_rate))

    def set_parameters(
        self,
        drive_db: float,
        trim_db: float,
        tone: float,
        tone_enabled: bool,
    ) -> None:
        self._drive_db = float(drive_db)
        self._trim_db = float(trim_db)
        self._tone = float(tone)
        self._tone_enabled = bool(tone_enabled)
        self._update_params()

    def _apply_tone(self, x: np.ndarray) -> np.ndarray:
        coeff = self._tone_coeff
        tone = self._tone
        low_gain = 1.0 - (0.35 * tone)
        high_gain = 1.0 + (0.35 * tone)
        out = np.empty_like(x)
        state = self._tone_state
        for i in range(x.shape[0]):
            for ch in range(self.channels):
                low = (1.0 - coeff) * x[i, ch] + coeff * state[ch]
                state[ch] = low
                high = x[i, ch] - low
                out[i, ch] = (low * low_gain) + (high * high_gain)
        self._tone_state = state
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        y = np.tanh(x * self._drive_gain)
        if self._tone_enabled and abs(self._tone) > 1e-4:
            y = self._apply_tone(y)
        y *= self._trim_gain
        np.clip(y, -1.0, 1.0, out=y)
        return y


class ReverbEffect(EffectProcessor):
    name = "Reverb"

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        decay_time: float = 1.4,
        pre_delay_ms: float = 20.0,
        wet: float = 0.25,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._comb_delay_ms = [29.7, 37.1, 41.1, 43.7]
        self._allpass_delay_ms = [5.0, 1.7]
        self._allpass_gain = 0.7
        self._comb_buffers: list[np.ndarray] = []
        self._comb_indices: list[int] = []
        self._comb_feedback: list[float] = []
        self._allpass_buffers: list[np.ndarray] = []
        self._allpass_indices: list[int] = []
        self._predelay_buffer = np.zeros((1, self.channels), dtype=np.float32)
        self._predelay_index = 0
        self._decay_time = 1.4
        self._pre_delay_ms = 20.0
        self._wet = 0.25
        self._dry = 0.75
        self.set_parameters(decay_time, pre_delay_ms, wet)

    def reset(self) -> None:
        for buf in self._comb_buffers:
            buf.fill(0.0)
        for buf in self._allpass_buffers:
            buf.fill(0.0)
        self._predelay_buffer.fill(0.0)
        self._predelay_index = 0
        self._comb_indices = [0 for _ in self._comb_indices]
        self._allpass_indices = [0 for _ in self._allpass_indices]

    def set_parameters(self, decay_time: float, pre_delay_ms: float, wet: float) -> None:
        self._decay_time = clamp(float(decay_time), 0.2, 6.0)
        self._pre_delay_ms = clamp(float(pre_delay_ms), 0.0, 120.0)
        self._wet = clamp(float(wet), 0.0, 1.0)
        self._dry = 1.0 - self._wet
        self._update_predelay_buffer()
        self._update_filters()

    def _update_predelay_buffer(self) -> None:
        samples = max(1, int(round(self._pre_delay_ms * self.sample_rate / 1000.0)))
        if samples != self._predelay_buffer.shape[0]:
            self._predelay_buffer = np.zeros((samples, self.channels), dtype=np.float32)
            self._predelay_index = 0

    def _update_filters(self) -> None:
        comb_samples = [
            max(1, int(round(ms * self.sample_rate / 1000.0))) for ms in self._comb_delay_ms
        ]
        allpass_samples = [
            max(1, int(round(ms * self.sample_rate / 1000.0))) for ms in self._allpass_delay_ms
        ]

        if not self._comb_buffers or [buf.shape[0] for buf in self._comb_buffers] != comb_samples:
            self._comb_buffers = [
                np.zeros((size, self.channels), dtype=np.float32) for size in comb_samples
            ]
            self._comb_indices = [0 for _ in self._comb_buffers]

        if not self._allpass_buffers or [buf.shape[0] for buf in self._allpass_buffers] != allpass_samples:
            self._allpass_buffers = [
                np.zeros((size, self.channels), dtype=np.float32) for size in allpass_samples
            ]
            self._allpass_indices = [0 for _ in self._allpass_buffers]

        self._comb_feedback = []
        for size in comb_samples:
            delay_sec = size / self.sample_rate
            feedback = 10.0 ** (-3.0 * delay_sec / max(0.1, self._decay_time))
            self._comb_feedback.append(clamp(feedback, 0.0, 0.99))

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        out = np.empty_like(x)
        n = x.shape[0]

        for i in range(n):
            inp = x[i]
            pred = self._predelay_buffer[self._predelay_index]
            self._predelay_buffer[self._predelay_index] = inp
            self._predelay_index = (self._predelay_index + 1) % self._predelay_buffer.shape[0]

            comb_sum = np.zeros(self.channels, dtype=np.float32)
            for idx, (buf, feedback) in enumerate(zip(self._comb_buffers, self._comb_feedback)):
                pos = self._comb_indices[idx]
                y = buf[pos]
                buf[pos] = pred + y * feedback
                self._comb_indices[idx] = (pos + 1) % buf.shape[0]
                comb_sum += y

            comb_out = comb_sum / max(1, len(self._comb_buffers))
            ap = comb_out
            for idx, buf in enumerate(self._allpass_buffers):
                pos = self._allpass_indices[idx]
                buf_out = buf[pos]
                y = (-self._allpass_gain * ap) + buf_out
                buf[pos] = ap + (self._allpass_gain * y)
                ap = y
                self._allpass_indices[idx] = (pos + 1) % buf.shape[0]

            out[i] = (self._dry * inp) + (self._wet * ap)

        return out


class StereoWidenerEffect(EffectProcessor):
    name = "Stereo Width"

    def __init__(self, width: float = 1.0, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._width = clamp(float(width), 0.0, 2.0)

    def set_width(self, width: float) -> None:
        self._width = clamp(float(width), 0.0, 2.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if x.shape[1] != 2:
            return x

        width = self._width
        if abs(width - 1.0) < 1e-6:
            return x

        mid = 0.5 * (x[:, 0] + x[:, 1])
        side = 0.5 * (x[:, 0] - x[:, 1]) * width
        y = np.empty_like(x)
        y[:, 0] = mid + side
        y[:, 1] = mid - side
        return y


class StereoPannerEffect(EffectProcessor):
    name = "Stereo Panner"

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        azimuth_deg: float = 0.0,
        spread: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._azimuth_deg = clamp(float(azimuth_deg), -90.0, 90.0)
        self._spread = clamp(float(spread), 0.0, 1.0)
        self._max_delay_ms = 0.6
        self._delay_len = max(
            2,
            int(math.ceil(self._max_delay_ms * self.sample_rate / 1000.0)) + 2,
        )
        self._delay_buffer = np.zeros((self._delay_len, self.channels), dtype=np.float32)
        self._delay_index = 0
        self._lp_state = np.zeros(self.channels, dtype=np.float32)

    def reset(self) -> None:
        self._delay_buffer.fill(0.0)
        self._delay_index = 0
        self._lp_state.fill(0.0)

    def set_parameters(self, azimuth_deg: float, spread: float) -> None:
        self._azimuth_deg = clamp(float(azimuth_deg), -90.0, 90.0)
        self._spread = clamp(float(spread), 0.0, 1.0)

    def _calc_gains(self) -> tuple[float, float]:
        pan = self._azimuth_deg / 90.0
        angle = (pan + 1.0) * (math.pi / 4.0)
        left = math.cos(angle)
        right = math.sin(angle)
        norm = max(left, right, 1e-6)
        return left / norm, right / norm

    def _far_lowpass_coeff(self) -> float:
        abs_pan = abs(self._azimuth_deg) / 90.0
        cutoff_hz = 8000.0 - (6000.0 * abs_pan)
        return math.exp(-2.0 * math.pi * cutoff_hz / float(self.sample_rate))

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if x.shape[1] != 2:
            return x

        y = np.array(x, copy=True)
        spread = self._spread
        if abs(spread - 1.0) > 1e-4:
            mid = 0.5 * (y[:, 0] + y[:, 1])
            side = 0.5 * (y[:, 0] - y[:, 1]) * spread
            y[:, 0] = mid + side
            y[:, 1] = mid - side

        left_gain, right_gain = self._calc_gains()
        if abs(left_gain - 1.0) > 1e-4:
            y[:, 0] *= left_gain
        if abs(right_gain - 1.0) > 1e-4:
            y[:, 1] *= right_gain

        abs_pan = abs(self._azimuth_deg) / 90.0
        if abs_pan < 1e-4:
            return y

        delay_samples = (self._delay_len - 2) * abs_pan
        far_idx = 0 if self._azimuth_deg > 0.0 else 1
        near_idx = 1 - far_idx
        coeff = self._far_lowpass_coeff()

        out = np.empty_like(y)
        buf = self._delay_buffer
        buf_len = buf.shape[0]
        idx = self._delay_index
        lp_state = self._lp_state

        for i in range(y.shape[0]):
            buf[idx] = y[i]
            out[i, near_idx] = y[i, near_idx]

            read_pos = idx - delay_samples
            while read_pos < 0.0:
                read_pos += buf_len
            idx0 = int(read_pos) % buf_len
            idx1 = (idx0 + 1) % buf_len
            frac = read_pos - int(read_pos)
            sample = (1.0 - frac) * buf[idx0, far_idx] + frac * buf[idx1, far_idx]
            lp = (1.0 - coeff) * sample + coeff * lp_state[far_idx]
            lp_state[far_idx] = lp
            out[i, far_idx] = lp

            idx = (idx + 1) % buf_len

        self._delay_index = idx
        self._lp_state = lp_state
        return out


class ChorusEffect(EffectProcessor):
    name = "Chorus"

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        rate_hz: float = 0.8,
        depth_ms: float = 8.0,
        mix: float = 0.25,
        base_delay_ms: float = 15.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._rate_hz = 0.8
        self._depth_ms = 8.0
        self._mix = 0.25
        self._dry = 0.75
        self._base_delay_ms = 15.0
        self._phase = np.zeros(self.channels, dtype=np.float64)
        if self.channels >= 2:
            self._phase[1] = math.pi / 2.0
        self._phase_inc = 0.0
        self._delay_buffer = np.zeros((2, self.channels), dtype=np.float32)
        self._write_index = 0
        self.set_parameters(rate_hz, depth_ms, mix, base_delay_ms=base_delay_ms)

    def reset(self) -> None:
        self._delay_buffer.fill(0.0)
        self._write_index = 0
        self._phase.fill(0.0)
        if self.channels >= 2:
            self._phase[1] = math.pi / 2.0

    def set_parameters(
        self,
        rate_hz: float,
        depth_ms: float,
        mix: float,
        *,
        base_delay_ms: Optional[float] = None,
    ) -> None:
        self._rate_hz = clamp(float(rate_hz), 0.05, 5.0)
        self._depth_ms = clamp(float(depth_ms), 0.0, 20.0)
        self._mix = clamp(float(mix), 0.0, 1.0)
        self._dry = 1.0 - self._mix
        if base_delay_ms is not None:
            self._base_delay_ms = clamp(float(base_delay_ms), 5.0, 30.0)
        self._phase_inc = 2.0 * math.pi * self._rate_hz / float(self.sample_rate)
        self._update_delay_buffer()

    def _update_delay_buffer(self) -> None:
        max_delay_ms = self._base_delay_ms + self._depth_ms
        max_delay_samples = int(math.ceil(max_delay_ms * self.sample_rate / 1000.0))
        size = max(2, max_delay_samples + 2)
        if size != self._delay_buffer.shape[0]:
            self._delay_buffer = np.zeros((size, self.channels), dtype=np.float32)
            self._write_index = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        n = x.shape[0]
        out = np.empty_like(x)
        buf = self._delay_buffer
        buf_len = buf.shape[0]
        phase_inc = self._phase_inc

        for i in range(n):
            write_pos = self._write_index
            for ch in range(self.channels):
                phase = self._phase[ch]
                lfo = math.sin(phase)
                mod = 0.5 * (lfo + 1.0)
                delay_ms = self._base_delay_ms + (self._depth_ms * mod)
                delay_samples = delay_ms * self.sample_rate / 1000.0
                read_pos = write_pos - delay_samples
                while read_pos < 0.0:
                    read_pos += buf_len
                idx0 = int(read_pos) % buf_len
                idx1 = (idx0 + 1) % buf_len
                frac = read_pos - int(read_pos)
                delayed = (1.0 - frac) * buf[idx0, ch] + frac * buf[idx1, ch]

                inp = x[i, ch]
                buf[write_pos, ch] = inp
                out[i, ch] = (self._dry * inp) + (self._mix * delayed)

                phase += phase_inc
                if phase >= 2.0 * math.pi:
                    phase -= 2.0 * math.pi
                self._phase[ch] = phase
            self._write_index = (write_pos + 1) % buf_len

        return out


class EffectsChain:
    def __init__(self, sample_rate: int, channels: int, effects: Optional[list[EffectProcessor]] = None):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.effects: list[EffectProcessor] = list(effects or [])

    def set_effects(self, effects: list[EffectProcessor]) -> None:
        self.effects = list(effects)

    def set_effect_order(self, names: list[str]) -> None:
        name_map = {effect.name: effect for effect in self.effects}
        ordered = []
        for name in names:
            effect = name_map.pop(name, None)
            if effect is not None:
                ordered.append(effect)
        ordered.extend(name_map.values())
        self.effects = ordered

    def enable_effect(self, name: str, enabled: bool) -> None:
        for effect in self.effects:
            if effect.name == name:
                effect.enabled = bool(enabled)
                return

    def reset(self) -> None:
        for effect in self.effects:
            effect.reset()

    def _validate_audio(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if x.ndim != 2 or x.shape[1] != self.channels:
            raise ValueError(f"EffectsChain expects (n,{self.channels}) float32")
        if x.dtype != np.float32:
            return x.astype(np.float32, copy=False)
        return x

    def process(self, x: np.ndarray) -> np.ndarray:
        x = self._validate_audio(x)
        for effect in self.effects:
            if not effect.enabled:
                continue
            x = effect.process(x)
            x = self._validate_audio(x)
        return x


# -----------------------------
# SoundTouch DSP (ctypes)
# -----------------------------

class SoundTouchUnavailable(RuntimeError):
    pass

def _try_load_soundtouch() -> ctypes.CDLL:
    # 1) explicit env var
    explicit = os.environ.get("SOUNDTOUCH_DLL", "./SoundTouchDLL/SoundTouchDLL_x64.dll").strip()
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)

    # 2) same folder as script
    here = os.path.abspath(os.path.dirname(__file__))
    if sys.platform.startswith("win"):
        candidates += [os.path.join(here, "SoundTouchDLL_x64.dll")]
    elif sys.platform == "darwin":
        candidates += [os.path.join(here, "libSoundTouch.dylib")]
    else:
        candidates += [os.path.join(here, "libSoundTouch.so")]

    # 3) system lookup via ctypes.util.find_library
    for name in ("SoundTouch", "soundtouch", "SoundTouchDLL", "soundtouchdll", "libSoundTouch"):
        p = ctypes.util.find_library(name)
        if p:
            candidates.append(p)

    errors = []
    for c in candidates:
        try:
            if not c:
                continue
            return ctypes.CDLL(c)
        except Exception as e:
            errors.append(f"{c}: {e}")
    raise SoundTouchUnavailable("Could not load SoundTouch library. Tried:\n" + "\n".join(errors[:8]))

class SoundTouchDSP(DSPBase):
    """
    Wrap the SoundTouch DLL C API.

    Assumes float32 interleaved samples; frame count passed as "numSamples" (samples per channel).
    """
    name = "SoundTouch"

    def __init__(self, sample_rate: int, channels: int):
        self.sr = int(sample_rate)
        self.ch = int(channels)
        self._lib = _try_load_soundtouch()

        # Function bindings (common SoundTouchDLL exports)
        self._lib.soundtouch_createInstance.restype = ctypes.c_void_p
        self._lib.soundtouch_destroyInstance.argtypes = [ctypes.c_void_p]

        self._lib.soundtouch_setSampleRate.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self._lib.soundtouch_setChannels.argtypes = [ctypes.c_void_p, ctypes.c_uint]

        self._lib.soundtouch_setTempo.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.soundtouch_setPitchSemiTones.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.soundtouch_setRate.argtypes = [ctypes.c_void_p, ctypes.c_float]

        self._lib.soundtouch_putSamples.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint]
        self._lib.soundtouch_receiveSamples.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint]
        self._lib.soundtouch_receiveSamples.restype = ctypes.c_uint

        self._lib.soundtouch_numSamples.argtypes = [ctypes.c_void_p]
        self._lib.soundtouch_numSamples.restype = ctypes.c_uint

        self._lib.soundtouch_flush.argtypes = [ctypes.c_void_p]
        self._lib.soundtouch_clear.argtypes = [ctypes.c_void_p]

        self._inst = ctypes.c_void_p(self._lib.soundtouch_createInstance())
        if not self._inst:
            raise SoundTouchUnavailable("soundtouch_createInstance returned NULL")

        self._lib.soundtouch_setSampleRate(self._inst, ctypes.c_uint(self.sr))
        self._lib.soundtouch_setChannels(self._inst, ctypes.c_uint(self.ch))

        self.tempo = 1.0
        self.pitch_st = 0.0
        self.key_lock = True
        self.tape_mode = False
        self.set_controls(1.0, 0.0, True, False)

    def __del__(self):
        try:
            if getattr(self, "_inst", None):
                self._lib.soundtouch_destroyInstance(self._inst)
        except Exception:
            pass

    def reset(self) -> None:
        self._lib.soundtouch_clear(self._inst)

    def set_controls(self, tempo: float, pitch_semitones: float, key_lock: bool, tape_mode: bool) -> None:
        tempo = clamp(float(tempo), 0.5, 2.0)
        pitch_semitones = clamp(float(pitch_semitones), -12.0, 12.0)
        self.tempo = tempo
        self.pitch_st = pitch_semitones
        self.key_lock = bool(key_lock)
        self.tape_mode = bool(tape_mode)

        if self.tape_mode:
            # "Tape": rate changes pitch+tempo together; ignore pitch slider
            self._lib.soundtouch_setTempo(self._inst, ctypes.c_float(1.0))
            self._lib.soundtouch_setPitchSemiTones(self._inst, ctypes.c_float(0.0))
            self._lib.soundtouch_setRate(self._inst, ctypes.c_float(self.tempo))
            return

        if self.key_lock:
            # Independent: tempo without pitch + optional pitch shift
            self._lib.soundtouch_setRate(self._inst, ctypes.c_float(1.0))
            self._lib.soundtouch_setTempo(self._inst, ctypes.c_float(self.tempo))
            self._lib.soundtouch_setPitchSemiTones(self._inst, ctypes.c_float(self.pitch_st))
        else:
            # Tempo affects pitch via rate; pitch slider adds additional pitch shift.
            self._lib.soundtouch_setTempo(self._inst, ctypes.c_float(1.0))
            self._lib.soundtouch_setRate(self._inst, ctypes.c_float(self.tempo))
            self._lib.soundtouch_setPitchSemiTones(self._inst, ctypes.c_float(self.pitch_st))

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if x.ndim != 2 or x.shape[1] != self.ch:
            raise ValueError(f"SoundTouchDSP expects (n,{self.ch}) float32")

        x = np.ascontiguousarray(x)
        n = x.shape[0]
        ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.soundtouch_putSamples(self._inst, ptr, ctypes.c_uint(n))

        # Pull available samples
        return self._drain_available(max_frames=8192)

    def _drain_available(self, max_frames: int = 8192) -> np.ndarray:
        outs = []
        while True:
            avail = int(self._lib.soundtouch_numSamples(self._inst))
            if avail <= 0:
                break
            take = min(avail, max_frames)
            out = np.empty((take, self.ch), dtype=np.float32)
            got = int(self._lib.soundtouch_receiveSamples(
                self._inst,
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_uint(take)
            ))
            if got <= 0:
                break
            outs.append(out[:got].copy())
            if got < take:
                break
        if not outs:
            return np.zeros((0, self.ch), dtype=np.float32)
        return np.vstack(outs)

    def flush(self) -> np.ndarray:
        # Ask SoundTouch to flush internal buffers then drain.
        self._lib.soundtouch_flush(self._inst)
        return self._drain_available(max_frames=16384)


# -----------------------------
# Fallback DSP: PhaseVocoder + Resampler (from prototype)
# -----------------------------

class StreamingResampler:
    """
    Streaming linear resampler that changes speed and pitch together.

    factor > 1.0  => faster (shorter), pitch up
    factor < 1.0  => slower (longer), pitch down
    """
    def __init__(self, channels: int):
        self.channels = channels
        self.factor = 1.0
        self._pos = 0.0
        self._carry = np.zeros((1, channels), dtype=np.float32)
        self._have_carry = False

    def reset(self):
        self._pos = 0.0
        self._have_carry = False

    def set_factor(self, factor: float):
        self.factor = clamp(float(factor), 0.25, 4.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        if self._have_carry:
            x = np.vstack([self._carry, x])
            self._have_carry = False

        if x.shape[0] < 2:
            self._carry = x[-1:, :].copy()
            self._have_carry = True
            return np.zeros((0, self.channels), dtype=np.float32)

        pos = self._pos
        max_pos = x.shape[0] - 1.000001
        n_out = int(max(0, math.floor((max_pos - pos) / self.factor) + 1))
        if n_out <= 0:
            self._carry = x[-1:, :].copy()
            self._have_carry = True
            self._pos = pos - (x.shape[0] - 1)
            return np.zeros((0, self.channels), dtype=np.float32)

        out = np.empty((n_out, self.channels), dtype=np.float32)
        for i in range(n_out):
            ip = int(pos)
            frac = pos - ip
            out[i] = x[ip] * (1.0 - frac) + x[ip + 1] * frac
            pos += self.factor

        self._carry = x[-1:, :].copy()
        self._have_carry = True
        self._pos = pos - (x.shape[0] - 1)
        return out


class PhaseVocoderTimeStretch:
    """
    Streaming phase vocoder time-stretch (fallback quality).
    ratio r = output/input duration.
    """
    def __init__(self, sample_rate: int, channels: int, n_fft: int = 2048, hop_a: int = 512):
        self.sr = sample_rate
        self.channels = channels
        self.n_fft = int(n_fft)
        self.hop_a = int(hop_a)
        self.ratio = 1.0
        self.hop_s = self.hop_a
        self.window = np.hanning(self.n_fft).astype(np.float32)

        self._inbuf = np.zeros((0, channels), dtype=np.float32)
        self._phi_prev = np.zeros((channels, self.n_fft // 2 + 1), dtype=np.float32)
        self._phi_sum = np.zeros((channels, self.n_fft // 2 + 1), dtype=np.float32)

        self._outbuf = np.zeros((0, channels), dtype=np.float32)
        self._out_read = 0
        self._out_pos = 0

        k = np.arange(self.n_fft // 2 + 1, dtype=np.float32)
        self._omega = 2.0 * np.pi * k / float(self.n_fft)

    def reset(self):
        self._inbuf = np.zeros((0, self.channels), dtype=np.float32)
        self._phi_prev.fill(0.0)
        self._phi_sum.fill(0.0)
        self._outbuf = np.zeros((0, self.channels), dtype=np.float32)
        self._out_read = 0
        self._out_pos = 0

    def set_ratio(self, ratio: float):
        self.ratio = clamp(float(ratio), 0.25, 4.0)
        self.hop_s = max(1, int(round(self.hop_a * self.ratio)))

    @staticmethod
    def _principal_arg(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _process_one_frame(self, frame: np.ndarray) -> np.ndarray:
        out = np.empty_like(frame, dtype=np.float32)
        win = self.window[:, None]

        for ch in range(self.channels):
            xw = frame[:, ch].astype(np.float32, copy=False) * win[:, 0]
            X = np.fft.rfft(xw, n=self.n_fft)
            mag = np.abs(X)
            phi = np.angle(X).astype(np.float32)

            delta = phi - self._phi_prev[ch] - self._omega * self.hop_a
            delta = self._principal_arg(delta)
            true_freq = self._omega + delta / float(self.hop_a)
            self._phi_sum[ch] = self._phi_sum[ch] + true_freq * float(self.hop_s)

            Y = mag * np.exp(1j * self._phi_sum[ch])
            yw = np.fft.irfft(Y, n=self.n_fft).astype(np.float32)
            out[:, ch] = yw * win[:, 0]
            self._phi_prev[ch] = phi

        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size:
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)
            self._inbuf = np.vstack([self._inbuf, x]) if self._inbuf.size else x

        while self._inbuf.shape[0] >= self.n_fft:
            frame = self._inbuf[:self.n_fft, :]
            self._inbuf = self._inbuf[self.hop_a:, :]

            yframe = self._process_one_frame(frame)

            needed = self._out_pos + self.n_fft
            if self._outbuf.shape[0] < needed:
                grow = needed - self._outbuf.shape[0]
                if self._outbuf.size:
                    self._outbuf = np.vstack([self._outbuf, np.zeros((grow, self.channels), dtype=np.float32)])
                else:
                    self._outbuf = np.zeros((needed, self.channels), dtype=np.float32)

            self._outbuf[self._out_pos:self._out_pos + self.n_fft, :] += yframe
            self._out_pos += self.hop_s

            if self._out_read > 8192:
                self._outbuf = self._outbuf[self._out_read:, :]
                self._out_pos -= self._out_read
                self._out_read = 0

        return self._emit()

    def _emit(self) -> np.ndarray:
        # Only samples before _out_pos are finalized.
        finalized_end = min(self._out_pos, self._outbuf.shape[0])
        available = finalized_end - self._out_read
        if available <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        take = min(available, 4096)
        out = self._outbuf[self._out_read:self._out_read + take, :].copy()
        self._out_read += take
        return out


class FallbackTempoPitch(DSPBase):
    name = "PhaseVocoder"
    def __init__(self, sample_rate: int, channels: int):
        self.sr = sample_rate
        self.ch = channels
        self.tempo = 1.0
        self.pitch_st = 0.0
        self.key_lock = True
        self.tape_mode = False
        self._pv = PhaseVocoderTimeStretch(sample_rate, channels, n_fft=2048, hop_a=512)
        self._rs = StreamingResampler(channels)
        self.set_controls(1.0, 0.0, True, False)

    def reset(self) -> None:
        self._pv.reset()
        self._rs.reset()

    def set_controls(self, tempo: float, pitch_semitones: float, key_lock: bool, tape_mode: bool) -> None:
        self.tempo = clamp(float(tempo), 0.5, 2.0)
        self.pitch_st = clamp(float(pitch_semitones), -12.0, 12.0)
        self.key_lock = bool(key_lock)
        self.tape_mode = bool(tape_mode)

        if self.tape_mode:
            self._rs.set_factor(self.tempo)
            self._stretch_ratio = 1.0
        else:
            p = semitones_to_factor(self.pitch_st)
            r = p / self.tempo if self.key_lock else 1.0
            self._pv.set_ratio(r)
            self._rs.set_factor(p if self.key_lock else self.tempo)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        if self.tape_mode:
            return self._rs.process(x)

        if self.key_lock:
            y = self._pv.process(x)
            if y.size == 0:
                return y
            return self._rs.process(y)

        # key_lock False: approximate tape via resampler (tempo affects pitch)
        self._rs.set_factor(self.tempo)
        return self._rs.process(x)

    def flush(self) -> np.ndarray:
        # No internal flush in fallback beyond what process emits.
        return np.zeros((0, self.ch), dtype=np.float32)


def make_dsp(sample_rate: int, channels: int) -> tuple[DSPBase, str]:
    mode = os.environ.get("TEMPOPITCH_DSP", "auto").strip().lower()
    if mode not in ("auto", "soundtouch", "phasevocoder"):
        mode = "auto"

    if mode in ("auto", "soundtouch"):
        try:
            dsp = SoundTouchDSP(sample_rate, channels)
            return dsp, "SoundTouch"
        except Exception as e:
            if mode == "soundtouch":
                raise
            # auto -> fallback
            return FallbackTempoPitch(sample_rate, channels), f"PhaseVocoder (SoundTouch unavailable: {e})"
    else:
        return FallbackTempoPitch(sample_rate, channels), "PhaseVocoder (forced)"


# -----------------------------
# FFmpeg decoding
# -----------------------------

def probe_duration(path: str) -> float:
    return probe_metadata(path).duration_sec


def probe_metadata(path: str) -> TrackMetadata:
    if not have_exe("ffprobe"):
        return TrackMetadata(0.0, "", "", "", None)
    try:
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_entries",
                "format=duration:format_tags=artist,album,album_artist,title:stream=index,codec_type:stream_disposition=attached_pic:stream_tags=comment,title,mimetype",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            return TrackMetadata(0.0, "", "", "", None)
        data = json.loads(p.stdout or "{}")
        fmt = data.get("format", {}) or {}
        tags = fmt.get("tags", {}) or {}
        tags_lower = {str(k).lower(): str(v) for k, v in tags.items()}
        artist = tags_lower.get("artist") or tags_lower.get("album_artist") or ""
        album = tags_lower.get("album") or ""
        title = tags_lower.get("title") or ""
        duration = max(0.0, safe_float(str(fmt.get("duration", "0")), 0.0))

        cover_art = None
        streams = data.get("streams", []) or []
        attached_stream_index: Optional[int] = None

        # NOTE: ffprobe only returns stream disposition/tags if requested as
        # stream_disposition / stream_tags (see -show_entries above).
        for fallback_idx, stream in enumerate(streams):
            disp = stream.get("disposition", {}) or {}
            attached = disp.get("attached_pic")
            if attached in (1, "1", True):
                # Prefer the real ffmpeg stream index; fall back to list position.
                idx_val = stream.get("index")
                attached_stream_index = idx_val if isinstance(idx_val, int) else fallback_idx
                break

        if attached_stream_index is not None and have_exe("ffmpeg"):
            # Use the absolute stream index (0:<index>) to avoid 'video index' pitfalls.
            map_arg = f"0:{attached_stream_index}"
            art = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-map",
                    map_arg,
                    "-an",
                    "-frames:v",
                    "1",
                    "-c:v",
                    "png",
                    "-f",
                    "image2pipe",
                    "pipe:1",
                ],
                capture_output=True,
                check=False,
            )
            if art.returncode == 0 and art.stdout:
                cover_art = art.stdout

        return TrackMetadata(duration, artist, album, title, cover_art)
    except Exception:
        return TrackMetadata(0.0, "", "", "", None)


def build_track(path: str) -> Track:
    meta = probe_metadata(path)
    title = meta.title or os.path.basename(path)
    return Track(
        path=path,
        title=title,
        duration_sec=meta.duration_sec,
        artist=meta.artist,
        album=meta.album,
        title_display=title,
        cover_art=meta.cover_art,
    )


def make_ffmpeg_cmd(path: str, start_sec: float, sample_rate: int, channels: int) -> List[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, start_sec)),
        "-i", path,
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "f32le",
        "pipe:1"
    ]


# -----------------------------
# Decoder + DSP thread
# -----------------------------

class DecoderThread(threading.Thread):
    """
    Reads float32 PCM from ffmpeg, processes DSP/effects, pushes into ring buffer.
    """
    def __init__(self,
                 track_path: str,
                 start_sec: float,
                 sample_rate: int,
                 channels: int,
                 ring: AudioRingBuffer,
                 dsp: DSPBase,
                 eq_dsp: EqualizerDSP,
                 fx_chain: EffectsChain,
                 state_cb):
        super().__init__(daemon=True)
        self.track_path = track_path
        self.start_sec = float(start_sec)
        self.sample_rate = sample_rate
        self.channels = channels
        self.ring = ring
        self.dsp = dsp
        self.eq_dsp = eq_dsp
        self.fx_chain = fx_chain
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._state_cb = state_cb

        self._read_frames = 4096
        self._read_bytes = self._read_frames * channels * 4

    def stop(self):
        self._stop.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def run(self):
        cmd = make_ffmpeg_cmd(self.track_path, self.start_sec, self.sample_rate, self.channels)
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            self._state_cb("error", f"Failed to start ffmpeg: {e}")
            return

        stdout = self._proc.stdout
        if stdout is None:
            self._state_cb("error", "ffmpeg stdout not available")
            return

        self._state_cb("loading", None)
        eq_profile_enabled = EQ_PROFILE or logger.isEnabledFor(logging.DEBUG)
        profile_iter = 0
        low_watermark_frames = int(EQ_PROFILE_LOW_WATERMARK_SEC * self.sample_rate)

        try:
            # Warm-up until ~1.0s buffered
            while not self._stop.is_set():
                b = stdout.read(self._read_bytes)
                if not b:
                    break
                x = np.frombuffer(b, dtype=np.float32)
                if x.size == 0:
                    continue
                x = x.reshape((-1, self.channels))
                y = self.dsp.process(x)
                if y.size:
                    eq_start = time.perf_counter()
                    y = self.eq_dsp.process(y)
                    eq_elapsed = time.perf_counter() - eq_start
                    if eq_profile_enabled and eq_elapsed > 0:
                        n_frames = y.shape[0]
                        process_ms = eq_elapsed * 1000.0
                        expected_ms = (n_frames / self.sample_rate) * 1000.0
                        frames_per_second_processed = n_frames / eq_elapsed
                        if process_ms > 0.5 * expected_ms:
                            logger.warning(
                                "EQ too slow: %.2fms for %d frames", process_ms, n_frames
                            )
                        if (profile_iter % EQ_PROFILE_LOG_EVERY == 0
                                or self.ring.frames_available() < low_watermark_frames):
                            logger.debug(
                                "EQ warmup: %.2fms, %.1f fps, ring=%d",
                                process_ms,
                                frames_per_second_processed,
                                self.ring.frames_available(),
                            )
                        profile_iter += 1
                    y = self.fx_chain.process(y)
                if y.size:
                    self.ring.push(y)
                if self.ring.frames_available() > int(1.0 * self.sample_rate):
                    self._state_cb("ready", None)
                    break

            # Main loop
            while not self._stop.is_set():
                b = stdout.read(self._read_bytes)
                if not b:
                    break
                x = np.frombuffer(b, dtype=np.float32)
                if x.size == 0:
                    continue
                x = x.reshape((-1, self.channels))
                y = self.dsp.process(x)
                if y.size:
                    eq_start = time.perf_counter()
                    y = self.eq_dsp.process(y)
                    eq_elapsed = time.perf_counter() - eq_start
                    if eq_profile_enabled and eq_elapsed > 0:
                        n_frames = y.shape[0]
                        process_ms = eq_elapsed * 1000.0
                        expected_ms = (n_frames / self.sample_rate) * 1000.0
                        frames_per_second_processed = n_frames / eq_elapsed
                        if process_ms > 0.5 * expected_ms:
                            logger.warning(
                                "EQ too slow: %.2fms for %d frames", process_ms, n_frames
                            )
                        if (profile_iter % EQ_PROFILE_LOG_EVERY == 0
                                or self.ring.frames_available() < low_watermark_frames):
                            logger.debug(
                                "EQ main: %.2fms, %.1f fps, ring=%d",
                                process_ms,
                                frames_per_second_processed,
                                self.ring.frames_available(),
                            )
                        profile_iter += 1
                    y = self.fx_chain.process(y)
                if y.size:
                    self.ring.push(y)

                # Backpressure: keep buffer in a healthy range.
                # If the decoder gets too far ahead, it can make playback sound chaotic.
                high = int(0.625 * self.ring.max_frames)
                low = int(0.375 * self.ring.max_frames)
                if self.ring.frames_available() > high:
                    while (not self._stop.is_set()) and self.ring.frames_available() > low:
                        time.sleep(0.02)

            # EOF: flush DSP tail
            tail = self.dsp.flush()
            if tail.size:
                tail = self.eq_dsp.process(tail)
                tail = self.fx_chain.process(tail)
                self.ring.push(tail)

        except Exception as e:
            self._state_cb("error", f"Decoder/DSP error: {e}")
        finally:
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
            except Exception:
                pass
            self._state_cb("eof", None)


# -----------------------------
# Player engine
# -----------------------------

class PlayerEngine(QtCore.QObject):
    stateChanged = QtCore.Signal(object)    # PlayerState
    errorOccurred = QtCore.Signal(str)
    durationChanged = QtCore.Signal(float)
    trackChanged = QtCore.Signal(object)    # Track
    dspChanged = QtCore.Signal(str)         # name

    def __init__(self, sample_rate: int = 44100, channels: int = 2, parent=None):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.channels = channels

        self.state = PlayerState.STOPPED
        self.track: Optional[Track] = None

        self._ring = AudioRingBuffer(channels, max_seconds=7.0, sample_rate=sample_rate)
        self._viz_buffer = VisualizerBuffer(channels, max_seconds=0.75, sample_rate=sample_rate)
        self._dsp, self._dsp_name = make_dsp(sample_rate, channels)
        self._eq_dsp = EqualizerDSP(sample_rate, channels)
        self._compressor = CompressorEffect(sample_rate, channels)
        self._reverb = ReverbEffect(sample_rate, channels)
        self._chorus = ChorusEffect(sample_rate, channels)
        self._stereo_widener = StereoWidenerEffect()
        self._stereo_panner = StereoPannerEffect(sample_rate, channels)
        self._saturation = SaturationEffect(sample_rate, channels)
        self._limiter = LimiterEffect(sample_rate, channels)
        self._fx_chain = EffectsChain(
            sample_rate,
            channels,
            effects=[
                GainEffect(),
                self._compressor,
                self._chorus,
                self._stereo_panner,
                self._stereo_widener,
                self._reverb,
                self._saturation,
                self._limiter,
            ],
        )
        self._decoder: Optional[DecoderThread] = None

        self._stream: Optional[sd.OutputStream] = None if sd else None
        self._volume = 0.8
        self._muted = False

        self._tempo = 1.0
        self._pitch_st = 0.0
        self._key_lock = True
        self._tape_mode = False
        self._eq_gains = [0.0 for _ in self._eq_dsp.center_freqs]
        self._reverb_decay = 1.4
        self._reverb_predelay = 20.0
        self._reverb_wet = 0.25
        self._chorus_rate = 0.8
        self._chorus_depth = 8.0
        self._chorus_mix = 0.25
        self._stereo_width = 1.0
        self._panner_azimuth = 0.0
        self._panner_spread = 1.0
        self._compressor_threshold = -18.0
        self._compressor_ratio = 4.0
        self._compressor_attack = 10.0
        self._compressor_release = 120.0
        self._compressor_makeup = 0.0
        self._saturation_drive = 6.0
        self._saturation_trim = 0.0
        self._saturation_tone = 0.0
        self._saturation_tone_enabled = False
        self._limiter_threshold = -1.0
        self._limiter_release_ms: Optional[float] = 80.0

        self._seek_offset_sec = 0.0
        self._source_pos_sec = 0.0
        self._position_lock = threading.Lock()

        self._playing = False
        self._paused = False

        self.dspChanged.emit(self._dsp_name)

    def dsp_name(self) -> str:
        return self._dsp_name

    def set_volume(self, v: float):
        self._volume = clamp(float(v), 0.0, 1.0)

    def set_muted(self, muted: bool):
        self._muted = bool(muted)

    def set_dsp_controls(self, tempo: float, pitch_st: float, key_lock: bool, tape_mode: bool):
        self._tempo = clamp(float(tempo), 0.5, 2.0)
        self._pitch_st = clamp(float(pitch_st), -12.0, 12.0)
        self._key_lock = bool(key_lock)
        self._tape_mode = bool(tape_mode)
        self._dsp.set_controls(self._tempo, self._pitch_st, self._key_lock, self._tape_mode)

    def set_eq_gains(self, gains_db: list[float]):
        if len(gains_db) != len(self._eq_gains):
            raise ValueError(f"Expected {len(self._eq_gains)} EQ bands")
        self._eq_gains = [clamp(float(g), -12.0, 12.0) for g in gains_db]
        self._eq_dsp.set_eq_gains(self._eq_gains)

    def set_compressor_controls(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        makeup_db: float,
    ) -> None:
        self._compressor_threshold = clamp(float(threshold_db), -60.0, 0.0)
        self._compressor_ratio = clamp(float(ratio), 1.0, 20.0)
        self._compressor_attack = clamp(float(attack_ms), 0.1, 200.0)
        self._compressor_release = clamp(float(release_ms), 1.0, 1000.0)
        self._compressor_makeup = clamp(float(makeup_db), 0.0, 24.0)
        self._compressor.set_parameters(
            self._compressor_threshold,
            self._compressor_ratio,
            self._compressor_attack,
            self._compressor_release,
            self._compressor_makeup,
        )

    def get_compressor_gain_reduction_db(self) -> Optional[float]:
        if self._compressor is None:
            return None
        return self._compressor.gain_reduction_db()

    def set_limiter_controls(self, threshold_db: float, release_ms: Optional[float]) -> None:
        self._limiter_threshold = clamp(float(threshold_db), -60.0, 0.0)
        if release_ms is None:
            self._limiter_release_ms = None
        else:
            self._limiter_release_ms = clamp(float(release_ms), 1.0, 1000.0)
        self._limiter.set_parameters(self._limiter_threshold, self._limiter_release_ms)

    def set_saturation_controls(
        self,
        drive_db: float,
        trim_db: float,
        tone: float,
        tone_enabled: bool,
    ) -> None:
        self._saturation_drive = clamp(float(drive_db), 0.0, 24.0)
        self._saturation_trim = clamp(float(trim_db), -24.0, 24.0)
        self._saturation_tone = clamp(float(tone), -1.0, 1.0)
        self._saturation_tone_enabled = bool(tone_enabled)
        self._saturation.set_parameters(
            self._saturation_drive,
            self._saturation_trim,
            self._saturation_tone,
            self._saturation_tone_enabled,
        )

    def set_reverb_controls(self, decay_time: float, pre_delay_ms: float, wet: float) -> None:
        self._reverb_decay = clamp(float(decay_time), 0.2, 6.0)
        self._reverb_predelay = clamp(float(pre_delay_ms), 0.0, 120.0)
        self._reverb_wet = clamp(float(wet), 0.0, 1.0)
        self._reverb.set_parameters(self._reverb_decay, self._reverb_predelay, self._reverb_wet)

    def set_chorus_controls(self, rate_hz: float, depth_ms: float, mix: float) -> None:
        self._chorus_rate = clamp(float(rate_hz), 0.05, 5.0)
        self._chorus_depth = clamp(float(depth_ms), 0.0, 20.0)
        self._chorus_mix = clamp(float(mix), 0.0, 1.0)
        self._chorus.set_parameters(self._chorus_rate, self._chorus_depth, self._chorus_mix)

    def set_stereo_width(self, width: float) -> None:
        self._stereo_width = clamp(float(width), 0.0, 2.0)
        self._stereo_widener.set_width(self._stereo_width)

    def set_stereo_panner_controls(self, azimuth_deg: float, spread: float) -> None:
        self._panner_azimuth = clamp(float(azimuth_deg), -90.0, 90.0)
        self._panner_spread = clamp(float(spread), 0.0, 1.0)
        self._stereo_panner.set_parameters(self._panner_azimuth, self._panner_spread)

    def load_track(self, path: str):
        if not path or not os.path.exists(path):
            self._set_error(f"File not found: {path}")
            return
        self.track = build_track(path)
        self._seek_offset_sec = 0.0
        self._set_state(PlayerState.STOPPED)
        self.trackChanged.emit(self.track)
        self.durationChanged.emit(self.track.duration_sec)

        with self._position_lock:
            self._source_pos_sec = 0.0

    def play(self):
        if sd is None:
            self._set_error(f"sounddevice not available: {_sounddevice_import_error}")
            return
        if not have_exe("ffmpeg"):
            self._set_error("ffmpeg not found in PATH.")
            return
        if self.track is None:
            return

        if self.state == PlayerState.PAUSED:
            self._paused = False
            self._set_state(PlayerState.PLAYING)
            return

        self.stop()  # ensure clean slate

        self._ring.clear()
        self._viz_buffer.clear()
        self._dsp.reset()
        self._dsp.set_controls(self._tempo, self._pitch_st, self._key_lock, self._tape_mode)
        self._eq_dsp.reset()
        self._eq_dsp.set_eq_gains(self._eq_gains)
        self._compressor.set_parameters(
            self._compressor_threshold,
            self._compressor_ratio,
            self._compressor_attack,
            self._compressor_release,
            self._compressor_makeup,
        )
        self._saturation.set_parameters(
            self._saturation_drive,
            self._saturation_trim,
            self._saturation_tone,
            self._saturation_tone_enabled,
        )
        self._limiter.set_parameters(self._limiter_threshold, self._limiter_release_ms)
        self._reverb.set_parameters(self._reverb_decay, self._reverb_predelay, self._reverb_wet)
        self._chorus.set_parameters(self._chorus_rate, self._chorus_depth, self._chorus_mix)
        self._stereo_panner.set_parameters(self._panner_azimuth, self._panner_spread)
        self._fx_chain.reset()

        self._playing = True
        self._paused = False

        with self._position_lock:
            self._source_pos_sec = self._seek_offset_sec

        def state_cb(kind, msg):
            if kind == "error":
                self._set_error(msg or "Unknown error")
            elif kind == "loading":
                self._set_state(PlayerState.LOADING)
            elif kind == "ready":
                self._ensure_stream()
                self._set_state(PlayerState.PLAYING)

        self._decoder = DecoderThread(
            track_path=self.track.path,
            start_sec=self._seek_offset_sec,
            sample_rate=self.sample_rate,
            channels=self.channels,
            ring=self._ring,
            dsp=self._dsp,
            eq_dsp=self._eq_dsp,
            fx_chain=self._fx_chain,
            state_cb=state_cb
        )
        self._decoder.start()

    def pause(self):
        if self.state == PlayerState.PLAYING:
            self._paused = True
            self._set_state(PlayerState.PAUSED)

    def stop(self):
        self._playing = False
        self._paused = False

        if self._decoder:
            self._decoder.stop()
            self._decoder = None

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._ring.clear()
        self._viz_buffer.clear()
        self._set_state(PlayerState.STOPPED)

    def seek(self, target_sec: float):
        if self.track is None:
            return
        dur = self.track.duration_sec
        if dur > 0:
            target_sec = clamp(float(target_sec), 0.0, dur)
        else:
            target_sec = max(0.0, float(target_sec))

        self._seek_offset_sec = target_sec

        active = self.state in (PlayerState.PLAYING, PlayerState.PAUSED, PlayerState.LOADING)
        was_paused = (self.state == PlayerState.PAUSED)

        if active:
            self.stop()
            self.play()
            if was_paused:
                self.pause()
        else:
            with self._position_lock:
                self._source_pos_sec = target_sec

    def _ensure_stream(self):
        if self._stream is not None:
            try:
                if not self._stream.active:
                    self._stream.start()
            except Exception:
                pass
            return

        def callback(outdata, frames, time_info, status):
            if not self._playing or self._paused:
                outdata[:] = np.zeros((frames, self.channels), dtype=np.float32)
                return

            chunk = self._ring.pop(frames)
            if chunk.size:
                self._viz_buffer.push(chunk)
            vol = 0.0 if self._muted else self._volume
            outdata[:] = chunk * vol

            # Update displayed source position based on speed factor (tempo or rate).
            dt_source = (frames / self.sample_rate) * float(self._tempo)
            with self._position_lock:
                self._source_pos_sec += dt_source

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=0,
                callback=callback
            )
            self._stream.start()
        except Exception as e:
            self._set_error(f"Audio output error: {e}")
            self._stream = None

    def get_position(self) -> float:
        with self._position_lock:
            return float(self._source_pos_sec)

    def get_buffer_seconds(self) -> float:
        return self._ring.frames_available() / self.sample_rate

    def get_visualizer_frames(self, frames: Optional[int] = None, mono: bool = False) -> np.ndarray:
        return self._viz_buffer.get_recent(frames=frames, mono=mono)

    def _set_state(self, st: PlayerState):
        if self.state != st:
            self.state = st
            self.stateChanged.emit(st)

    def _set_error(self, msg: str):
        self._set_state(PlayerState.ERROR)
        self.errorOccurred.emit(msg)


# -----------------------------
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
        self.resize(980, 640)

        self.settings = QtCore.QSettings("ChatGPT", "TempoPitchPlayer")
        self._theme_name = str(self.settings.value("ui/theme", "Ocean"))

        self.engine = PlayerEngine(sample_rate=44100, channels=2, parent=self)

        self.transport = TransportWidget()
        self.visualizer = VisualizerWidget(self.engine)
        self.dsp_widget = TempoPitchWidget()
        self.compressor_widget = CompressorWidget()
        self.saturation_widget = SaturationWidget()
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

        self.artwork_label = QtWidgets.QLabel("No Artwork")
        self.artwork_label.setObjectName("artwork_label")
        self.artwork_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.artwork_label.setFixedSize(220, 220)
        self.artwork_label.setWordWrap(True)

        self.status = QtWidgets.QLabel("Ready.")
        self.status.setObjectName("status_label")

        self.header_frame = QtWidgets.QFrame()
        self.header_frame.setObjectName("header_frame")
        header_layout = QtWidgets.QVBoxLayout(self.header_frame)
        header_top_row = QtWidgets.QHBoxLayout()
        header_top_row.addWidget(self.artwork_label)
        header_top_row.addWidget(self.equalizer)
        header_layout.addLayout(header_top_row)
        header_layout.addWidget(self.now_playing)
        header_layout.addWidget(self.status)

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

        self._shuffle = bool(self.settings.value("playback/shuffle", False, type=bool))
        repeat_setting = self.settings.value("playback/repeat", RepeatMode.OFF.value)
        self._repeat_mode = RepeatMode.from_setting(repeat_setting)
        self._shuffle_history: List[int] = []
        self._shuffle_bag: List[int] = []

        app = QtWidgets.QApplication.instance()
        if app:
            self._apply_theme(self._theme_name)

        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(16, 16, 16, 16)
        left.setSpacing(12)
        left.addWidget(self.transport)
        left.addWidget(self.visualizer)
        left.addWidget(self.dsp_widget)
        left.addWidget(self.compressor_widget)
        left.addWidget(self.saturation_widget)
        left.addWidget(self.limiter_widget)
        left.addWidget(self.reverb_widget)
        left.addWidget(self.chorus_widget)
        left.addWidget(self.stereo_panner_widget)
        left.addWidget(self.stereo_width_widget)
        left.addWidget(self.appearance_group)
        left.addWidget(self.header_frame)
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

        self.engine.trackChanged.connect(self._on_track_changed)
        self.engine.stateChanged.connect(self._on_state_changed)
        self.engine.errorOccurred.connect(self._on_error)

        # Timer
        self._dur = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

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
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a *.aac);;All Files (*.*)"
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
        exts = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
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
        self._set_artwork(track.cover_art)
        self._dur = track.duration_sec

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

    def _on_error(self, msg: str):
        self.status.setText(f"❌ {msg}")
        QtWidgets.QMessageBox.warning(self, "Playback error", msg)

    def _tick(self):
        pos = self.engine.get_position()
        self.transport.set_time(pos, self._dur)

        if self.engine.state in (PlayerState.PLAYING, PlayerState.LOADING):
            buf = self.engine.get_buffer_seconds()
            self.status.setText(f"{self.engine.dsp_name()} | {'Loading…' if self.engine.state==PlayerState.LOADING else 'Playing'} | Buffer: {buf:.2f}s")
        elif self.engine.state == PlayerState.PAUSED:
            self.status.setText(f"{self.engine.dsp_name()} | Paused")
        elif self.engine.state == PlayerState.STOPPED:
            self.status.setText(f"{self.engine.dsp_name()} | Stopped")

        # Auto-advance (best-effort)
        if self._dur > 0 and pos >= self._dur - 0.25 and self.engine.state == PlayerState.PLAYING:
            self._advance_track(direction=1, auto=True)

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
        self.settings.setValue("dsp/tempo", self.dsp_widget.tempo_slider.value() / 100.0)
        self.settings.setValue("dsp/pitch", self.dsp_widget.pitch_slider.value() / 10.0)
        self.settings.setValue("dsp/key_lock", self.dsp_widget.key_lock.isChecked())
        self.settings.setValue("dsp/tape_mode", self.dsp_widget.tape_mode.isChecked())
        self.settings.setValue("dsp/lock_432", self.dsp_widget.lock_432.isChecked())
        self.settings.setValue("eq/gains", self.equalizer.gains())
        self.settings.setValue("eq/preset", self.equalizer.presets.currentText())
        self.settings.setValue("compressor/threshold", self.compressor_widget.threshold_slider.value() / 10.0)
        self.settings.setValue("compressor/ratio", self.compressor_widget.ratio_slider.value() / 10.0)
        self.settings.setValue("compressor/attack", self.compressor_widget.attack_slider.value() / 10.0)
        self.settings.setValue("compressor/release", float(self.compressor_widget.release_slider.value()))
        self.settings.setValue("compressor/makeup", self.compressor_widget.makeup_slider.value() / 10.0)
        self.settings.setValue("saturation/drive", self.saturation_widget.drive_slider.value() / 10.0)
        self.settings.setValue("saturation/trim", self.saturation_widget.trim_slider.value() / 10.0)
        self.settings.setValue("saturation/tone", self.saturation_widget.tone_slider.value() / 100.0)
        self.settings.setValue("saturation/tone_enabled", self.saturation_widget.tone_toggle.isChecked())
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

    @staticmethod
    def _normalize_eq_gains(values: object, band_count: int) -> list[float]:
        if isinstance(values, (tuple, list)):
            gains = [safe_float(str(v), 0.0) for v in values]
        else:
            gains = []
        if len(gains) < band_count:
            gains.extend([0.0] * (band_count - len(gains)))
        return [float(g) for g in gains[:band_count]]

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
