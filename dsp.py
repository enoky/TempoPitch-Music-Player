from __future__ import annotations

import os
import sys
import math
import threading
import subprocess
import ctypes
import json
from dataclasses import dataclass
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

from models import Track, TrackMetadata
from metadata_fetch import get_online_metadata

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
}
from utils import clamp, have_exe, safe_float, semitones_to_factor

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

    @staticmethod
    def _peaking_coeffs(f0: float, gain_db: float, q: float, sample_rate: int) -> tuple[float, float, float, float, float]:
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * f0 / float(sample_rate)
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

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
        return b0, b1, b2, a1, a2

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
        if all(abs(gain_db) <= 1e-3 for gain_db in new_gains):
            config = EqConfig(
                sos=np.zeros((0, 6), dtype=np.float32),
                zi=np.zeros((0, self.ch, 2), dtype=np.float32),
                reset_mask=np.zeros((0,), dtype=bool),
            )
            with self._lock:
                if new_gains == self._gains_db:
                    return
                self._gains_db = new_gains
                self._config = config
                self._pending_reset = False
                self._reset_all = False
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
            b0, b1, b2, a1, a2 = self._peaking_coeffs(f0, gain_db, self.q, self.sr)

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
            b0, b1, b2, a1, a2 = self._peaking_coeffs(f0, gain_db, self.q, self.sr)

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
        config = self._config
        if config.sos.shape[0] == 0 and not self._pending_reset:
            return x
        if self._pending_reset:
            if self._reset_all:
                config.zi.fill(0.0)
            elif config.reset_mask.size > 0:
                config.zi[config.reset_mask, :, :] = 0.0
            self._pending_reset = False
            self._reset_all = False
        if config.sos.shape[0] == 0:
            return x
        y = np.array(x, copy=True)
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


if njit is not None:
    @njit(cache=True, nogil=True)
    def _compressor_process(
        x: np.ndarray,
        env: float,
        threshold: float,
        ratio: float,
        makeup_db: float,
        attack_coeff: float,
        release_coeff: float,
    ) -> tuple[np.ndarray, float, float]:
        n_frames, n_channels = x.shape
        out = np.empty_like(x)
        last_reduction = 0.0
        for i in range(n_frames):
            peak = 0.0
            for ch in range(n_channels):
                value = x[i, ch]
                if value < 0.0:
                    value = -value
                if value > peak:
                    peak = value
            coeff = attack_coeff if peak > env else release_coeff
            env = coeff * env + (1.0 - coeff) * peak
            env_db = 20.0 * math.log10(env if env > 1e-8 else 1e-8)
            if env_db <= threshold:
                gain_db = 0.0
            else:
                gain_db = threshold + (env_db - threshold) / ratio - env_db
            reduction_db = 0.0
            if gain_db < 0.0:
                reduction_db = -gain_db
            last_reduction = reduction_db
            gain = 10.0 ** ((gain_db + makeup_db) / 20.0)
            for ch in range(n_channels):
                out[i, ch] = x[i, ch] * gain
        return out, env, last_reduction


    @njit(cache=True, nogil=True)
    def _limiter_process(
        x: np.ndarray,
        gain: float,
        threshold_amp: float,
        release_coeff: float,
        use_release: bool,
    ) -> tuple[np.ndarray, float]:
        n_frames, n_channels = x.shape
        out = np.empty_like(x)
        limit = threshold_amp if threshold_amp < 1.0 else 1.0
        for i in range(n_frames):
            peak = 0.0
            for ch in range(n_channels):
                value = x[i, ch]
                if value < 0.0:
                    value = -value
                if value > peak:
                    peak = value
            if peak > threshold_amp:
                gain = threshold_amp / (peak if peak > 1e-8 else 1e-8)
            elif use_release:
                gain = release_coeff * gain + (1.0 - release_coeff)
            else:
                gain = 1.0
            for ch in range(n_channels):
                value = x[i, ch] * gain
                if value > limit:
                    value = limit
                elif value < -limit:
                    value = -limit
                out[i, ch] = value
        return out, gain
else:
    def _compressor_process(
        x: np.ndarray,
        env: float,
        threshold: float,
        ratio: float,
        makeup_db: float,
        attack_coeff: float,
        release_coeff: float,
    ) -> tuple[np.ndarray, float, float]:
        n_frames, n_channels = x.shape
        out = np.empty_like(x)
        last_reduction = 0.0
        for i in range(n_frames):
            peak = 0.0
            for ch in range(n_channels):
                value = x[i, ch]
                if value < 0.0:
                    value = -value
                if value > peak:
                    peak = value
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
            for ch in range(n_channels):
                out[i, ch] = x[i, ch] * gain
        return out, env, last_reduction


    def _limiter_process(
        x: np.ndarray,
        gain: float,
        threshold_amp: float,
        release_coeff: float,
        use_release: bool,
    ) -> tuple[np.ndarray, float]:
        n_frames, n_channels = x.shape
        out = np.empty_like(x)
        limit = threshold_amp if threshold_amp < 1.0 else 1.0
        for i in range(n_frames):
            peak = 0.0
            for ch in range(n_channels):
                value = x[i, ch]
                if value < 0.0:
                    value = -value
                if value > peak:
                    peak = value
            if peak > threshold_amp:
                gain = threshold_amp / max(peak, 1e-8)
            elif use_release:
                gain = release_coeff * gain + (1.0 - release_coeff)
            else:
                gain = 1.0
            for ch in range(n_channels):
                value = x[i, ch] * gain
                if value > limit:
                    value = limit
                elif value < -limit:
                    value = -limit
                out[i, ch] = value
        return out, gain


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
        if self._ratio <= 1.0 and self._makeup_db == 0.0:
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

        out, env, last_reduction = _compressor_process(
            x,
            env,
            threshold,
            ratio,
            makeup_db,
            attack_coeff,
            release_coeff,
        )

        self._env = env
        self._last_reduction_db = last_reduction
        return out


class DynamicEqEffect(EffectProcessor):
    name = "Dynamic EQ"

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        freq_hz: float = 1000.0,
        q: float = 1.0,
        gain_db: float = 0.0,
        threshold_db: float = -24.0,
        ratio: float = 4.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._freq_hz = float(freq_hz)
        self._q = float(q)
        self._gain_db = float(gain_db)
        self._threshold_db = float(threshold_db)
        self._ratio = float(ratio)
        self._detector_z1 = 0.0
        self._detector_z2 = 0.0
        self._detector_env = 0.0
        self._filter_zi = np.zeros((self.channels, 2), dtype=np.float32)
        self._attack_coeff = math.exp(-1.0 / (self.sample_rate * 0.01))
        self._release_coeff = math.exp(-1.0 / (self.sample_rate * 0.12))
        self._bandpass_coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._peaking_coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._last_gain_db = None
        self._update_coeffs(force=True)

    def reset(self) -> None:
        self._detector_z1 = 0.0
        self._detector_z2 = 0.0
        self._detector_env = 0.0
        self._filter_zi.fill(0.0)

    def set_parameters(
        self,
        freq_hz: float,
        q: float,
        gain_db: float,
        threshold_db: float,
        ratio: float,
    ) -> None:
        self._freq_hz = clamp(float(freq_hz), 20.0, min(20000.0, self.sample_rate * 0.45))
        self._q = clamp(float(q), 0.1, 20.0)
        self._gain_db = clamp(float(gain_db), -12.0, 12.0)
        self._threshold_db = clamp(float(threshold_db), -60.0, 0.0)
        self._ratio = clamp(float(ratio), 1.0, 20.0)
        self._update_coeffs(force=True)

    def _update_coeffs(self, *, force: bool = False) -> None:
        if force or self._bandpass_coeffs == (0.0, 0.0, 0.0, 0.0, 0.0):
            self._bandpass_coeffs = self._make_bandpass_coeffs(self._freq_hz, self._q)
        if force or self._last_gain_db is None:
            self._last_gain_db = None
            self._peaking_coeffs = self._make_peaking_coeffs(self._freq_hz, self._q, self._gain_db)

    def _make_bandpass_coeffs(self, freq_hz: float, q: float) -> tuple[float, float, float, float, float]:
        w0 = 2.0 * math.pi * freq_hz / float(self.sample_rate)
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0
        return b0, b1, b2, a1, a2

    def _make_peaking_coeffs(self, freq_hz: float, q: float, gain_db: float) -> tuple[float, float, float, float, float]:
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * freq_hz / float(self.sample_rate)
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * q)

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
        return b0, b1, b2, a1, a2

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if self._gain_db == 0.0 and self._ratio <= 1.0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        mono = x.mean(axis=1)
        b0, b1, b2, a1, a2 = self._bandpass_coeffs
        z1 = self._detector_z1
        z2 = self._detector_z2
        env = self._detector_env

        for sample in mono:
            y = b0 * sample + z1
            z1 = b1 * sample - a1 * y + z2
            z2 = b2 * sample - a2 * y
            amp = abs(y)
            coeff = self._attack_coeff if amp > env else self._release_coeff
            env = coeff * env + (1.0 - coeff) * amp

        self._detector_z1 = z1
        self._detector_z2 = z2
        self._detector_env = env

        env_db = 20.0 * math.log10(max(env, 1e-8))
        if env_db <= self._threshold_db:
            dynamic_gain_db = 0.0
        else:
            dynamic_gain_db = self._threshold_db + (env_db - self._threshold_db) / self._ratio - env_db

        total_gain_db = clamp(self._gain_db + dynamic_gain_db, -18.0, 18.0)
        if self._last_gain_db is None or abs(total_gain_db - self._last_gain_db) > 1e-3:
            self._peaking_coeffs = self._make_peaking_coeffs(self._freq_hz, self._q, total_gain_db)
            self._last_gain_db = total_gain_db

        b0, b1, b2, a1, a2 = self._peaking_coeffs
        out = np.empty_like(x)
        for ch in range(self.channels):
            z1 = float(self._filter_zi[ch, 0])
            z2 = float(self._filter_zi[ch, 1])
            for i in range(x.shape[0]):
                sample = x[i, ch]
                y = b0 * sample + z1
                z1 = b1 * sample - a1 * y + z2
                z2 = b2 * sample - a2 * y
                out[i, ch] = y
            self._filter_zi[ch, 0] = z1
            self._filter_zi[ch, 1] = z2
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
        if self._threshold_db >= 0.0 and self._release_ms is None:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        threshold_amp = self._threshold_amp
        release_coeff = self._release_coeff
        use_release = self._release_ms is not None
        gain = self._gain

        out, gain = _limiter_process(
            x,
            gain,
            threshold_amp,
            release_coeff,
            use_release,
        )

        self._gain = gain
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
        if self._drive_db <= 0.0 and abs(self._trim_db) <= 1e-6:
            if not self._tone_enabled or abs(self._tone) <= 1e-4:
                return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        y = np.tanh(x * self._drive_gain)
        if self._tone_enabled and abs(self._tone) > 1e-4:
            y = self._apply_tone(y)
        y *= self._trim_gain
        np.clip(y, -1.0, 1.0, out=y)
        return y


class SubharmonicEffect(EffectProcessor):
    name = "Subharmonic"

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        mix: float = 0.25,
        intensity: float = 0.6,
        cutoff_hz: float = 140.0,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._mix = 0.25
        self._intensity = 0.6
        self._cutoff_hz = 140.0
        self._detector_cutoff_hz = 180.0
        self._lp_coeff = 0.0
        self._detector_coeff = 0.0
        self._env_coeff = 0.0
        self._lp_state = 0.0
        self._detector_state = 0.0
        self._env_state = 0.0
        self._prev_sign = 1.0
        self._cross_count = 0
        self._phase = 1.0
        self.set_parameters(mix=mix, intensity=intensity, cutoff_hz=cutoff_hz)

    def reset(self) -> None:
        self._lp_state = 0.0
        self._detector_state = 0.0
        self._env_state = 0.0
        self._prev_sign = 1.0
        self._cross_count = 0
        self._phase = 1.0

    def _update_coeffs(self) -> None:
        cutoff = clamp(float(self._cutoff_hz), 40.0, 240.0)
        detector = clamp(float(self._detector_cutoff_hz), 80.0, 280.0)
        self._cutoff_hz = cutoff
        self._detector_cutoff_hz = detector
        self._lp_coeff = math.exp(-2.0 * math.pi * cutoff / float(self.sample_rate))
        self._detector_coeff = math.exp(-2.0 * math.pi * detector / float(self.sample_rate))
        env_cutoff = 12.0
        self._env_coeff = math.exp(-2.0 * math.pi * env_cutoff / float(self.sample_rate))

    def set_parameters(self, *, mix: float, intensity: float, cutoff_hz: float) -> None:
        self._mix = clamp(float(mix), 0.0, 1.0)
        self._intensity = clamp(float(intensity), 0.0, 1.5)
        self._cutoff_hz = float(cutoff_hz)
        self._update_coeffs()

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if self._mix <= 1e-5 or self._intensity <= 1e-5:
            return x

        mono = np.mean(x, axis=1)
        out = np.array(x, copy=True)
        lp_coeff = self._lp_coeff
        det_coeff = self._detector_coeff
        env_coeff = self._env_coeff
        lp_state = self._lp_state
        det_state = self._detector_state
        env_state = self._env_state
        phase = self._phase
        prev_sign = self._prev_sign
        cross_count = self._cross_count

        for i in range(mono.shape[0]):
            det_state = (1.0 - det_coeff) * mono[i] + det_coeff * det_state
            sign = 1.0 if det_state >= 0.0 else -1.0
            if sign != prev_sign:
                cross_count += 1
                prev_sign = sign
                if cross_count >= 2:
                    phase *= -1.0
                    cross_count = 0
            env_state = (1.0 - env_coeff) * abs(det_state) + env_coeff * env_state
            sub_sample = phase * env_state * self._intensity
            lp_state = (1.0 - lp_coeff) * sub_sample + lp_coeff * lp_state
            out[i] += self._mix * lp_state

        self._lp_state = lp_state
        self._detector_state = det_state
        self._env_state = env_state
        self._phase = phase
        self._prev_sign = prev_sign
        self._cross_count = cross_count
        np.clip(out, -1.0, 1.0, out=out)
        return out


if njit is not None:
    @njit(cache=True, nogil=True)
    def _reverb_process_block(
        x: np.ndarray,
        predelay_buffer: np.ndarray,
        predelay_index: int,
        comb_buffers: np.ndarray,
        comb_indices: np.ndarray,
        comb_feedback: np.ndarray,
        comb_lengths: np.ndarray,
        allpass_buffers: np.ndarray,
        allpass_indices: np.ndarray,
        allpass_lengths: np.ndarray,
        allpass_gain: float,
        wet: float,
        dry: float,
    ) -> tuple[np.ndarray, int]:
        n_frames, n_channels = x.shape
        out = np.empty_like(x)
        predelay_len = predelay_buffer.shape[0]
        comb_count = comb_buffers.shape[0]
        allpass_count = allpass_buffers.shape[0]
        inv_comb = 1.0 / comb_count if comb_count > 0 else 0.0

        predelay = np.empty((n_channels,), dtype=np.float32)
        comb_sum = np.empty((n_channels,), dtype=np.float32)
        ap = np.empty((n_channels,), dtype=np.float32)

        for i in range(n_frames):
            for ch in range(n_channels):
                predelay[ch] = predelay_buffer[predelay_index, ch]
                predelay_buffer[predelay_index, ch] = x[i, ch]
            predelay_index += 1
            if predelay_index >= predelay_len:
                predelay_index = 0

            for ch in range(n_channels):
                comb_sum[ch] = 0.0

            for idx in range(comb_count):
                pos = comb_indices[idx]
                length = comb_lengths[idx]
                if pos >= length:
                    pos = 0
                fb = comb_feedback[idx]
                for ch in range(n_channels):
                    y = comb_buffers[idx, pos, ch]
                    comb_sum[ch] += y
                    comb_buffers[idx, pos, ch] = predelay[ch] + (y * fb)
                pos += 1
                if pos >= length:
                    pos = 0
                comb_indices[idx] = pos

            for ch in range(n_channels):
                ap[ch] = comb_sum[ch] * inv_comb

            for idx in range(allpass_count):
                pos = allpass_indices[idx]
                length = allpass_lengths[idx]
                if pos >= length:
                    pos = 0
                for ch in range(n_channels):
                    buf_out = allpass_buffers[idx, pos, ch]
                    y = (-allpass_gain * ap[ch]) + buf_out
                    allpass_buffers[idx, pos, ch] = ap[ch] + (allpass_gain * y)
                    ap[ch] = y
                pos += 1
                if pos >= length:
                    pos = 0
                allpass_indices[idx] = pos

            for ch in range(n_channels):
                out[i, ch] = (dry * x[i, ch]) + (wet * ap[ch])

        return out, predelay_index
else:
    _reverb_process_block = None


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
        self._comb_buffers = np.zeros((0, 1, self.channels), dtype=np.float32)
        self._comb_indices = np.zeros((0,), dtype=np.int32)
        self._comb_lengths = np.zeros((0,), dtype=np.int32)
        self._comb_feedback = np.zeros((0,), dtype=np.float32)
        self._allpass_buffers = np.zeros((0, 1, self.channels), dtype=np.float32)
        self._allpass_indices = np.zeros((0,), dtype=np.int32)
        self._allpass_lengths = np.zeros((0,), dtype=np.int32)
        self._predelay_buffer = np.zeros((1, self.channels), dtype=np.float32)
        self._predelay_index = 0
        self._predelay_scratch = np.zeros((self.channels,), dtype=np.float32)
        self._comb_scratch = np.zeros((self.channels,), dtype=np.float32)
        self._decay_time = 1.4
        self._pre_delay_ms = 20.0
        self._wet = 0.25
        self._dry = 0.75
        self.set_parameters(decay_time, pre_delay_ms, wet)

    def reset(self) -> None:
        self._comb_buffers.fill(0.0)
        self._allpass_buffers.fill(0.0)
        self._predelay_buffer.fill(0.0)
        self._predelay_index = 0
        if self._comb_indices.size:
            self._comb_indices.fill(0)
        if self._allpass_indices.size:
            self._allpass_indices.fill(0)

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

        comb_lengths = np.array(comb_samples, dtype=np.int32)
        comb_count = int(comb_lengths.shape[0])
        comb_max = int(comb_lengths.max()) if comb_count else 1
        if (
            self._comb_buffers.shape[0] != comb_count
            or self._comb_buffers.shape[1] != comb_max
        ):
            self._comb_buffers = np.zeros(
                (comb_count, comb_max, self.channels), dtype=np.float32
            )
            self._comb_indices = np.zeros((comb_count,), dtype=np.int32)
        self._comb_lengths = comb_lengths

        allpass_lengths = np.array(allpass_samples, dtype=np.int32)
        allpass_count = int(allpass_lengths.shape[0])
        allpass_max = int(allpass_lengths.max()) if allpass_count else 1
        if (
            self._allpass_buffers.shape[0] != allpass_count
            or self._allpass_buffers.shape[1] != allpass_max
        ):
            self._allpass_buffers = np.zeros(
                (allpass_count, allpass_max, self.channels), dtype=np.float32
            )
            self._allpass_indices = np.zeros((allpass_count,), dtype=np.int32)
        self._allpass_lengths = allpass_lengths

        comb_feedback = np.zeros((comb_count,), dtype=np.float32)
        for i in range(comb_count):
            size = comb_lengths[i]
            delay_sec = size / self.sample_rate
            feedback = 10.0 ** (-3.0 * delay_sec / max(0.1, self._decay_time))
            comb_feedback[i] = clamp(feedback, 0.0, 0.99)
        self._comb_feedback = comb_feedback

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or x.size == 0:
            return x
        if self._wet <= 0.0:
            return x
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        if _reverb_process_block is not None:
            out, predelay_index = _reverb_process_block(
                x,
                self._predelay_buffer,
                self._predelay_index,
                self._comb_buffers,
                self._comb_indices,
                self._comb_feedback,
                self._comb_lengths,
                self._allpass_buffers,
                self._allpass_indices,
                self._allpass_lengths,
                self._allpass_gain,
                self._wet,
                self._dry,
            )
            self._predelay_index = predelay_index
            return out

        out = np.empty_like(x)
        n = x.shape[0]

        predelay_buffer = self._predelay_buffer
        predelay_index = self._predelay_index
        predelay_len = predelay_buffer.shape[0]
        predelay_scratch = self._predelay_scratch

        comb_buffers = self._comb_buffers
        comb_indices = self._comb_indices
        comb_feedback = self._comb_feedback
        comb_lengths = self._comb_lengths
        comb_count = comb_buffers.shape[0]
        comb_sum = self._comb_scratch
        inv_comb_count = 1.0 / comb_count if comb_count else 0.0

        allpass_buffers = self._allpass_buffers
        allpass_indices = self._allpass_indices
        allpass_lengths = self._allpass_lengths
        allpass_gain = self._allpass_gain

        dry = self._dry
        wet = self._wet

        for i in range(n):
            inp = x[i]

            predelay_scratch[:] = predelay_buffer[predelay_index]
            predelay_buffer[predelay_index] = inp
            predelay_index += 1
            if predelay_index >= predelay_len:
                predelay_index = 0

            comb_sum.fill(0.0)
            for idx in range(comb_count):
                pos = int(comb_indices[idx])
                length = int(comb_lengths[idx])
                if pos >= length:
                    pos = 0
                y = comb_buffers[idx, pos]
                comb_sum += y
                comb_buffers[idx, pos] = predelay_scratch + (y * comb_feedback[idx])
                pos += 1
                if pos >= length:
                    pos = 0
                comb_indices[idx] = pos

            ap = comb_sum
            if comb_count:
                ap *= inv_comb_count
            for idx in range(allpass_buffers.shape[0]):
                pos = int(allpass_indices[idx])
                length = int(allpass_lengths[idx])
                if pos >= length:
                    pos = 0
                buf_out = allpass_buffers[idx, pos]
                y = (-allpass_gain * ap) + buf_out
                allpass_buffers[idx, pos] = ap + (allpass_gain * y)
                ap = y
                pos += 1
                if pos >= length:
                    pos = 0
                allpass_indices[idx] = pos

            out[i] = (dry * inp) + (wet * ap)

        self._predelay_index = predelay_index
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
        if self._mix <= 0.0:
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


def parse_ffprobe_fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = str(stream.get(key) or "").strip()
        if not rate or rate == "0/0":
            continue
        if "/" in rate:
            num_str, den_str = rate.split("/", 1)
            num = safe_float(num_str, 0.0)
            den = safe_float(den_str, 0.0)
            if den > 0:
                return num / den
        else:
            fps = safe_float(rate, 0.0)
            if fps > 0:
                return fps
    return 0.0


def probe_metadata(path: str, fetch_online: bool = True) -> TrackMetadata:
    duration = 0.0
    artist = ""
    album = ""
    title = ""
    genre = ""
    year: Optional[int] = None
    track_number: Optional[int] = None
    isrc = ""
    cover_art = None
    has_video = False
    video_fps = 0.0
    video_size = (0, 0)
    skip_online = (not fetch_online) or (os.path.splitext(path)[1].lower() in VIDEO_EXTS)

    if have_exe("ffprobe"):
        try:
            p = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_entries",
                    (
                        "format=duration:format_tags=artist,album,album_artist,title,genre,date,track,isrc:"
                        "stream=index,codec_type,width,height:stream_disposition=attached_pic:"
                        "stream_tags=comment,title,mimetype"
                    ),
                    path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if p.returncode == 0:
                data = json.loads(p.stdout or "{}")
                fmt = data.get("format", {}) or {}
                tags = fmt.get("tags", {}) or {}
                tags_lower = {str(k).lower(): str(v) for k, v in tags.items()}
                artist = tags_lower.get("artist") or tags_lower.get("album_artist") or ""
                album = tags_lower.get("album") or ""
                title = tags_lower.get("title") or ""
                genre = tags_lower.get("genre") or ""
                date_str = tags_lower.get("date") or ""
                year = None
                if date_str:
                    try:
                        # Extract just the year from ISO strings like 2023-05-20 or just 2023
                        year_str = date_str.split("-")[0].strip()
                        if year_str.isdigit():
                            year = int(year_str)
                    except ValueError:
                        pass
                
                track_str = tags_lower.get("track") or ""
                track_number = None
                if track_str:
                    try:
                        # Handle "1", "1/12", etc.
                        t_part = track_str.split("/")[0].strip()
                        if t_part.isdigit():
                            track_number = int(t_part)
                    except ValueError:
                        pass

                isrc = tags_lower.get("isrc") or ""
                duration = max(0.0, safe_float(str(fmt.get("duration", "0")), 0.0))

                streams = data.get("streams", []) or []
                attached_stream_index: Optional[int] = None

                # NOTE: ffprobe only returns stream disposition/tags if requested as
                # stream_disposition / stream_tags (see -show_entries above).
                for fallback_idx, stream in enumerate(streams):
                    disp = stream.get("disposition", {}) or {}
                    attached = disp.get("attached_pic")
                    if stream.get("codec_type") == "video" and attached not in (1, "1", True):
                        width = int(stream.get("width") or 0)
                        height = int(stream.get("height") or 0)
                        if width > 0 and height > 0:
                            has_video = True
                        if video_size == (0, 0) and width > 0 and height > 0:
                            video_size = (width, height)
                        if video_fps <= 0.0:
                            video_fps = parse_ffprobe_fps(stream)
                    if attached in (1, "1", True) and attached_stream_index is None:
                        # Prefer the real ffmpeg stream index; fall back to list position.
                        idx_val = stream.get("index")
                        attached_stream_index = idx_val if isinstance(idx_val, int) else fallback_idx

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
        except Exception:
            pass

    if has_video:
        skip_online = True

    if skip_online:
        online = None
    else:
        try:
            online = get_online_metadata(
                path,
                tag_artist=artist,
                tag_title=title,
                tag_album=album,
                tag_isrc=isrc,
                tag_duration_sec=duration,
            )
        except Exception:
            online = None

    if online:
        if not artist:
            artist = online.artist
        if not album:
            album = online.album
        if not title:
            title = online.title
        if duration <= 0.0 and online.duration_sec:
            duration = online.duration_sec
        if cover_art is None and online.cover_art:
            cover_art = online.cover_art
        if not genre and online.genre:
            genre = online.genre
        if year is None and online.year is not None:
             year = online.year

    return TrackMetadata(
        duration,
        artist,
        album,
        title,
        genre,
        year,
        track_number,
        cover_art,
        has_video,
        video_fps,
        video_size,
    )


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
        has_video=meta.has_video,
        video_fps=meta.video_fps,
        video_size=meta.video_size,
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


def make_ffmpeg_video_cmd(
    path: str,
    start_sec: float,
    fps: float,
) -> List[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(max(0.0, start_sec)),
        "-i", path,
        "-an",
        "-vf", f"fps={fps}",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "pipe:1",
    ]


# -----------------------------
