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
    # Dark Themes
    "Ocean": Theme(
        name="Ocean",
        window="#0f172a",
        base="#020617",
        text="#e2e8f0",
        highlight="#38bdf8",
        accent="#0ea5e9",
        card="#1e293b",
    ),
    "Sunset": Theme(
        name="Sunset",
        window="#1c1917",
        base="#0c0a09",
        text="#f5f5f4",
        highlight="#fb7185",
        accent="#f43f5e",
        card="#292524",
    ),
    "Forest": Theme(
        name="Forest",
        window="#052e16",
        base="#020617",
        text="#dcfce7",
        highlight="#4ade80",
        accent="#22c55e",
        card="#14532d",
    ),
    "Rose": Theme(
        name="Rose",
        window="#2c0b0e",
        base="#1a0507",
        text="#ffe4e6",
        highlight="#fb7185",
        accent="#e11d48",
        card="#4c0519",
    ),
    "Slate": Theme(
        name="Slate",
        window="#111827",
        base="#030712",
        text="#f3f4f6",
        highlight="#60a5fa",
        accent="#3b82f6",
        card="#1f2937",
    ),
    "Aurora": Theme(
        name="Aurora",
        window="#042f2e",
        base="#020617",
        text="#ccfbf1",
        highlight="#2dd4bf",
        accent="#14b8a6",
        card="#115e59",
    ),
    "Ember": Theme(
        name="Ember",
        window="#2a1205",
        base="#0c0401",
        text="#ffedd5",
        highlight="#fb923c",
        accent="#f97316",
        card="#431407",
    ),
    "Mono": Theme(
        name="Mono",
        window="#171717",
        base="#0a0a0a",
        text="#e5e5e5",
        highlight="#a3a3a3",
        accent="#ffffff",
        card="#262626",
    ),

    # Medium Themes
    "Dusk": Theme(
        name="Dusk",
        window="#4c4b63",
        base="#37364d",
        text="#f3e8ff",
        highlight="#c084fc",
        accent="#a855f7",
        card="#5f5e7a",
    ),
    "Storm": Theme(
        name="Storm",
        window="#475569",
        base="#334155",
        text="#f8fafc",
        highlight="#38bdf8",
        accent="#0ea5e9",
        card="#5c6f88",
    ),
    "Copper": Theme(
        name="Copper",
        window="#573e32",
        base="#3f2c22",
        text="#ffedd5",
        highlight="#fbbf24",
        accent="#f59e0b",
        card="#6d5142",
    ),
    "Sage": Theme(
        name="Sage",
        window="#415243",
        base="#2f3d30",
        text="#ecfdf5",
        highlight="#4ade80",
        accent="#22c55e",
        card="#536655",
    ),
    "Plum": Theme(
        name="Plum",
        window="#581c38",
        base="#3b1124",
        text="#fdf2f8",
        highlight="#f472b6",
        accent="#ec4899",
        card="#70264a",
    ),
    "Steel": Theme(
        name="Steel",
        window="#4b5563",
        base="#374151",
        text="#f9fafb",
        highlight="#9ca3af",
        accent="#e5e7eb",
        card="#6b7280",
    ),
    "Terracotta": Theme(
        name="Terracotta",
        window="#7c2d12",
        base="#431407",
        text="#fff7ed",
        highlight="#fdba74",
        accent="#fb923c",
        card="#9a3412",
    ),
    "Marine": Theme(
        name="Marine",
        window="#0e7490",
        base="#083344",
        text="#ecfeff",
        highlight="#22d3ee",
        accent="#06b6d4",
        card="#155e75",
    ),

    # Light Themes
    "Daylight": Theme(
        name="Daylight",
        window="#ffffff",
        base="#f8fafc",
        text="#0f172a",
        highlight="#38bdf8",
        accent="#0284c7",
        card="#f1f5f9",
    ),
    "Lavender": Theme(
        name="Lavender",
        window="#faf5ff",
        base="#f3e8ff",
        text="#581c87",
        highlight="#c084fc",
        accent="#9333ea",
        card="#ffffff",
    ),
    "Mint": Theme(
        name="Mint",
        window="#f0fdf4",
        base="#dcfce7",
        text="#14532d",
        highlight="#4ade80",
        accent="#16a34a",
        card="#ffffff",
    ),
    "Ice": Theme(
        name="Ice",
        window="#f0f9ff",
        base="#e0f2fe",
        text="#0c4a6e",
        highlight="#38bdf8",
        accent="#0284c7",
        card="#ffffff",
    ),
    "Peach": Theme(
        name="Peach",
        window="#fff7ed",
        base="#ffedd5",
        text="#7c2d12",
        highlight="#fb923c",
        accent="#ea580c",
        card="#ffffff",
    ),
    "Cherry": Theme(
        name="Cherry",
        window="#fff1f2",
        base="#ffe4e6",
        text="#881337",
        highlight="#fb7185",
        accent="#e11d48",
        card="#ffffff",
    ),
    "Glacial": Theme(
        name="Glacial",
        window="#f0fdfa",
        base="#ccfbf1",
        text="#134e4a",
        highlight="#2dd4bf",
        accent="#0d9488",
        card="#ffffff",
    ),
    "Cloud": Theme(
        name="Cloud",
        window="#f8fafc",
        base="#f1f5f9",
        text="#020617",
        highlight="#94a3b8",
        accent="#475569",
        card="#ffffff",
    ),
}

