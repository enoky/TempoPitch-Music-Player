
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
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque
from typing import Optional, List

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    _sounddevice_import_error = e

from PySide6 import QtCore, QtGui, QtWidgets


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
    if not have_exe("ffprobe"):
        return 0.0
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=False
        )
        if p.returncode != 0:
            return 0.0
        return max(0.0, safe_float(p.stdout.strip(), 0.0))
    except Exception:
        return 0.0


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
    Reads float32 PCM from ffmpeg, processes DSP, pushes into ring buffer.
    """
    def __init__(self,
                 track_path: str,
                 start_sec: float,
                 sample_rate: int,
                 channels: int,
                 ring: AudioRingBuffer,
                 dsp: DSPBase,
                 state_cb):
        super().__init__(daemon=True)
        self.track_path = track_path
        self.start_sec = float(start_sec)
        self.sample_rate = sample_rate
        self.channels = channels
        self.ring = ring
        self.dsp = dsp
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

        try:
            # Warm-up until ~0.5s buffered
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
                    self.ring.push(y)
                if self.ring.frames_available() > int(0.5 * self.sample_rate):
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
                    self.ring.push(y)

                # Backpressure: keep buffer in a healthy range.
                # If the decoder gets too far ahead, it can make playback sound chaotic.
                high = int(2.5 * self.sample_rate)
                low = int(1.5 * self.sample_rate)
                if self.ring.frames_available() > high:
                    while (not self._stop.is_set()) and self.ring.frames_available() > low:
                        time.sleep(0.02)

            # EOF: flush DSP tail
            tail = self.dsp.flush()
            if tail.size:
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

        self._ring = AudioRingBuffer(channels, max_seconds=4.0, sample_rate=sample_rate)
        self._dsp, self._dsp_name = make_dsp(sample_rate, channels)
        self._decoder: Optional[DecoderThread] = None

        self._stream: Optional[sd.OutputStream] = None if sd else None
        self._volume = 0.8
        self._muted = False

        self._tempo = 1.0
        self._pitch_st = 0.0
        self._key_lock = True
        self._tape_mode = False

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

    def load_track(self, path: str):
        if not path or not os.path.exists(path):
            self._set_error(f"File not found: {path}")
            return
        dur = probe_duration(path)
        title = os.path.basename(path)
        self.track = Track(path=path, title=title, duration_sec=dur)
        self._seek_offset_sec = 0.0
        self._set_state(PlayerState.STOPPED)
        self.trackChanged.emit(self.track)
        self.durationChanged.emit(dur)

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
        self._dsp.reset()
        self._dsp.set_controls(self._tempo, self._pitch_st, self._key_lock, self._tape_mode)

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

class TempoPitchWidget(QtWidgets.QGroupBox):
    controlsChanged = QtCore.Signal(float, float, bool, bool)  # tempo, pitch_st, key_lock, tape_mode

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

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setToolTip("Reset tempo and pitch to defaults.")
        self.reset_btn.setAccessibleName("Reset tempo and pitch")

        form = QtWidgets.QFormLayout()
        form.addRow(self.tempo_label, self.tempo_slider)
        form.addRow(self.pitch_label, self.pitch_slider)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.key_lock)
        row.addWidget(self.tape_mode)
        row.addStretch(1)
        row.addWidget(self.reset_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(row)

        self.tempo_slider.valueChanged.connect(self._emit)
        self.pitch_slider.valueChanged.connect(self._emit)
        self.key_lock.toggled.connect(self._emit)
        self.tape_mode.toggled.connect(self._on_tape)
        self.reset_btn.clicked.connect(self._on_reset)

        self._emit()

    def _on_tape(self, on: bool):
        self.pitch_slider.setEnabled(not on)
        self.key_lock.setEnabled(not on)
        self._emit()

    def _on_reset(self):
        self.tempo_slider.setValue(100)
        self.pitch_slider.setValue(0)
        self.key_lock.setChecked(True)
        self.tape_mode.setChecked(False)

    def _emit(self):
        tempo = self.tempo_slider.value() / 100.0
        pitch = self.pitch_slider.value() / 10.0
        key_lock = self.key_lock.isChecked()
        tape = self.tape_mode.isChecked()

        self.tempo_label.setText(f"Tempo: {tempo:.2f}×")
        if tape:
            st = 12.0 * math.log2(max(1e-6, tempo))
            self.pitch_label.setText(f"Pitch: {st:+.2f} st (linked)")
        else:
            self.pitch_label.setText(f"Pitch: {pitch:+.1f} st")

        self.controlsChanged.emit(tempo, pitch, key_lock, tape)


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
        for t in tracks:
            item_text = f"{t.title} — {format_time(t.duration_sec)}"
            it = QtWidgets.QListWidgetItem(item_text)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, t)
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
        self.dsp_widget = TempoPitchWidget()
        self.playlist = PlaylistWidget()

        self.now_playing = QtWidgets.QLabel("No track loaded")
        self.now_playing.setObjectName("now_playing")
        self.now_playing.setWordWrap(True)
        font = self.now_playing.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.now_playing.setFont(font)

        self.status = QtWidgets.QLabel("Ready.")
        self.status.setObjectName("status_label")

        self.header_frame = QtWidgets.QFrame()
        self.header_frame.setObjectName("header_frame")
        header_layout = QtWidgets.QVBoxLayout(self.header_frame)
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
        left.addWidget(self.header_frame)
        left.addWidget(self.transport)
        left.addWidget(self.dsp_widget)
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

        self.dsp_widget.controlsChanged.connect(self.engine.set_dsp_controls)
        self.dsp_widget.controlsChanged.connect(self._on_dsp_controls_changed)

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
            dur = probe_duration(p)
            tracks.append(Track(path=p, title=os.path.basename(p), duration_sec=dur))

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

    def _on_dsp_controls_changed(self, tempo: float, pitch: float, key_lock: bool, tape_mode: bool):
        self.settings.setValue("dsp/tempo", float(tempo))
        self.settings.setValue("dsp/pitch", float(pitch))
        self.settings.setValue("dsp/key_lock", bool(key_lock))
        self.settings.setValue("dsp/tape_mode", bool(tape_mode))

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
        self.now_playing.setText(f"{track.title}\n{track.path}")
        self._dur = track.duration_sec

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

        tempo_value = int(round(clamp(float(tempo), 0.5, 2.0) * 100))
        pitch_value = int(round(clamp(float(pitch), -12.0, 12.0) * 10))

        self.dsp_widget.tempo_slider.setValue(tempo_value)
        self.dsp_widget.pitch_slider.setValue(pitch_value)
        self.dsp_widget.key_lock.setChecked(bool(key_lock))
        self.dsp_widget.tape_mode.setChecked(bool(tape_mode))

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

    def _save_ui_settings(self):
        self.settings.setValue("audio/volume_slider", self.transport.volume_slider.value())
        self.settings.setValue("audio/muted", self.transport.mute_btn.isChecked())
        self.settings.setValue("dsp/tempo", self.dsp_widget.tempo_slider.value() / 100.0)
        self.settings.setValue("dsp/pitch", self.dsp_widget.pitch_slider.value() / 10.0)
        self.settings.setValue("dsp/key_lock", self.dsp_widget.key_lock.isChecked())
        self.settings.setValue("dsp/tape_mode", self.dsp_widget.tape_mode.isChecked())

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
