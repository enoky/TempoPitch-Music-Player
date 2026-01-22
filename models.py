from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


@dataclass(frozen=True)
class BufferPreset:
    blocksize_frames: int
    latency: str | float
    target_sec: float
    high_sec: float
    low_sec: float
    ring_max_seconds: float


@dataclass
class Track:
    path: str
    title: str
    duration_sec: float
    artist: str = ""
    album: str = ""
    title_display: str = ""
    cover_art: Optional[bytes] = None
    has_video: bool = False
    video_fps: float = 0.0
    video_size: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class AudioParams:
    tempo: float
    pitch_st: float
    key_lock: bool
    tape_mode: bool
    eq_gains: tuple[float, ...]
    compressor_threshold: float
    compressor_ratio: float
    compressor_attack: float
    compressor_release: float
    compressor_makeup: float
    dynamic_eq_freq: float
    dynamic_eq_q: float
    dynamic_eq_gain: float
    dynamic_eq_threshold: float
    dynamic_eq_ratio: float
    saturation_drive: float
    saturation_trim: float
    saturation_tone: float
    saturation_tone_enabled: bool
    subharmonic_mix: float
    subharmonic_intensity: float
    subharmonic_cutoff: float
    reverb_decay: float
    reverb_predelay: float
    reverb_wet: float
    chorus_rate: float
    chorus_depth: float
    chorus_mix: float
    stereo_width: float
    panner_azimuth: float
    panner_spread: float
    limiter_threshold: float
    limiter_release_ms: Optional[float]
    compressor_enabled: bool
    dynamic_eq_enabled: bool
    subharmonic_enabled: bool
    reverb_enabled: bool
    chorus_enabled: bool
    stereo_panner_enabled: bool
    stereo_width_enabled: bool
    saturation_enabled: bool
    limiter_enabled: bool
    version: int


@dataclass
class TrackMetadata:
    duration_sec: float
    artist: str
    album: str
    title: str
    genre: str
    year: Optional[int]
    track_number: Optional[int]
    cover_art: Optional[bytes]
    has_video: bool
    video_fps: float
    video_size: tuple[int, int]


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
    "Daylight": Theme(
        name="Daylight",
        window="#f6f2e9",
        base="#fffaf1",
        text="#1f2933",
        highlight="#3b82f6",
        accent="#2563eb",
        card="#f0e7d8",
    ),
    "Aurora": Theme(
        name="Aurora",
        window="#0b1b22",
        base="#07141a",
        text="#e6f6f8",
        highlight="#22d3ee",
        accent="#2dd4bf",
        card="#0f2730",
    ),
    "Ember": Theme(
        name="Ember",
        window="#2a1611",
        base="#1d0f0b",
        text="#fdebd2",
        highlight="#fbbf24",
        accent="#f97316",
        card="#3a1e16",
    ),
    "Mono": Theme(
        name="Mono",
        window="#1b1b1b",
        base="#111111",
        text="#f2f2f2",
        highlight="#9ca3af",
        accent="#d1d5db",
        card="#232323",
    ),
    "Lavender": Theme(
        name="Lavender",
        window="#faf5ff",
        base="#ffffff",
        text="#581c87",
        highlight="#d8b4fe",
        accent="#a855f7",
        card="#f3e8ff",
    ),
    "Mint": Theme(
        name="Mint",
        window="#f0fdf4",
        base="#ffffff",
        text="#064e3b",
        highlight="#86efac",
        accent="#22c55e",
        card="#dcfce7",
    ),
    "Ice": Theme(
        name="Ice",
        window="#f0f9ff",
        base="#ffffff",
        text="#0c4a6e",
        highlight="#7dd3fc",
        accent="#0ea5e9",
        card="#e0f2fe",
    ),
    "Peach": Theme(
        name="Peach",
        window="#fff7ed",
        base="#ffffff",
        text="#7c2d12",
        highlight="#fdba74",
        accent="#f97316",
        card="#ffedd5",
    ),
    "Cherry": Theme(
        name="Cherry",
        window="#fff1f2",
        base="#ffffff",
        text="#881337",
        highlight="#fda4af",
        accent="#f43f5e",
        card="#ffe4e6",
    ),
    "Glacial": Theme(
        name="Glacial",
        window="#f0fdfa",
        base="#ffffff",
        text="#134e4a",
        highlight="#5eead4",
        accent="#14b8a6",
        card="#ccfbf1",
    ),
    "Cloud": Theme(
        name="Cloud",
        window="#f8fafc",
        base="#ffffff",
        text="#0f172a",
        highlight="#cbd5e1",
        accent="#64748b",
        card="#f1f5f9",
    ),
}
