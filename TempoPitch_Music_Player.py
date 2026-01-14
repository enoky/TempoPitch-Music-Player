
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
from ui.main_window import MainWindow
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

# UI widgets moved to ui/widgets.py

# Main window moved to ui/main_window.py

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
