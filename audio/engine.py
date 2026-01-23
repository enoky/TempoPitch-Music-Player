from __future__ import annotations

import math
import os
import re
import threading
import time
import logging
from dataclasses import dataclass, replace
from typing import Optional, Callable
from collections import deque

import numpy as np
import subprocess

from PySide6 import QtCore, QtGui

try:
    import sounddevice as sd
    _sounddevice_import_error = None
except Exception as e:
    sd = None
    _sounddevice_import_error = e

from buffers import AudioRingBuffer, VisualizerBuffer, VideoFrameBuffer, VideoRingBuffer
from config import (
    AUTO_BUFFER_PRESET,
    AUTO_BUFFER_THRESHOLD,
    AUTO_BUFFER_WINDOW_SEC,
    BUFFER_PRESETS,
    DEFAULT_BUFFER_PRESET,
    EQ_PROFILE,
    EQ_PROFILE_LOG_EVERY,
    EQ_PROFILE_LOW_WATERMARK_SEC,
)
from dsp import (
    DSPBase,
    EqualizerDSP,
    GainEffect,
    CompressorEffect,
    SaturationEffect,
    ReverbEffect,
    ChorusEffect,
    StereoWidenerEffect,
    StereoPannerEffect,
    LimiterEffect,
    DynamicEqEffect,
    SubharmonicEffect,
    EffectsChain,
    make_dsp,
    make_ffmpeg_cmd,
    make_ffmpeg_video_cmd,
)
from metadata import build_track
from models import AudioParams, BufferPreset, PlayerState, RepeatMode, Track
from utils import clamp, env_flag, have_exe

logger = logging.getLogger(__name__)

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
                 buffer_preset: BufferPreset,
                 viz_buffer: Optional[VisualizerBuffer],
                 viz_stride: int,
                 viz_downsample: int,
                 dsp: DSPBase,
                 eq_dsp: EqualizerDSP,
                 fx_chain: EffectsChain,
                 compressor: CompressorEffect,
                 dynamic_eq: DynamicEqEffect,
                 saturation: SaturationEffect,
                 subharmonic: SubharmonicEffect,
                 reverb: ReverbEffect,
                 chorus: ChorusEffect,
                 stereo_widener: StereoWidenerEffect,
                 stereo_panner: StereoPannerEffect,
                 limiter: LimiterEffect,
                 audio_params_provider: Callable[[], AudioParams],
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
        self._compressor = compressor
        self._dynamic_eq = dynamic_eq
        self._saturation = saturation
        self._subharmonic = subharmonic
        self._reverb = reverb
        self._chorus = chorus
        self._stereo_widener = stereo_widener
        self._stereo_panner = stereo_panner
        self._limiter = limiter
        self._audio_params_provider = audio_params_provider
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._state_cb = state_cb

        self._buffer_preset = buffer_preset
        self._read_frames = max(1, buffer_preset.blocksize_frames * 2)
        self._read_bytes = self._read_frames * channels * 4
        self._frame_bytes = channels * 4
        self._byte_buffer = bytearray()
        self._fade_in_total = max(1, int(0.02 * self.sample_rate))
        self._fade_in_remaining = 0
        self._viz_buffer = viz_buffer
        self._viz_stride = max(1, int(viz_stride))
        self._viz_downsample = max(1, int(viz_downsample))
        self._viz_counter = 0
        self._metrics_lock = threading.Lock()
        self._ring_underruns_accum = 0

    def stop(self):
        self._stop.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def _apply_audio_params(self, params: AudioParams) -> None:
        self.fx_chain.enable_effect("Compressor", params.compressor_enabled)
        self.fx_chain.enable_effect("Dynamic EQ", params.dynamic_eq_enabled)
        self.fx_chain.enable_effect("Subharmonic", params.subharmonic_enabled)
        self.fx_chain.enable_effect("Reverb", params.reverb_enabled)
        self.fx_chain.enable_effect("Chorus", params.chorus_enabled)
        self.fx_chain.enable_effect("Stereo Panner", params.stereo_panner_enabled)
        self.fx_chain.enable_effect("Stereo Width", params.stereo_width_enabled)
        self.fx_chain.enable_effect("Saturation", params.saturation_enabled)
        self.fx_chain.enable_effect("Limiter", params.limiter_enabled)
        self.dsp.set_controls(params.tempo, params.pitch_st, params.key_lock, params.tape_mode)
        self.eq_dsp.set_eq_gains(list(params.eq_gains))
        self._compressor.set_parameters(
            params.compressor_threshold,
            params.compressor_ratio,
            params.compressor_attack,
            params.compressor_release,
            params.compressor_makeup,
        )
        self._dynamic_eq.set_parameters(
            params.dynamic_eq_freq,
            params.dynamic_eq_q,
            params.dynamic_eq_gain,
            params.dynamic_eq_threshold,
            params.dynamic_eq_ratio,
        )
        self._saturation.set_parameters(
            params.saturation_drive,
            params.saturation_trim,
            params.saturation_tone,
            params.saturation_tone_enabled,
        )
        self._subharmonic.set_parameters(
            mix=params.subharmonic_mix,
            intensity=params.subharmonic_intensity,
            cutoff_hz=params.subharmonic_cutoff,
        )
        self._reverb.set_parameters(params.reverb_decay, params.reverb_predelay, params.reverb_wet)
        self._chorus.set_parameters(params.chorus_rate, params.chorus_depth, params.chorus_mix)
        self._stereo_widener.set_width(params.stereo_width)
        self._stereo_panner.set_parameters(params.panner_azimuth, params.panner_spread)
        self._limiter.set_parameters(params.limiter_threshold, params.limiter_release_ms)

    def _maybe_apply_fade_in(self, y: np.ndarray) -> np.ndarray:
        if self._fade_in_remaining <= 0 or y.size == 0:
            return y
        frames = y.shape[0]
        fade_frames = min(frames, self._fade_in_remaining)
        start_index = self._fade_in_total - self._fade_in_remaining
        ramp = (np.arange(start_index, start_index + fade_frames) + 1) / float(self._fade_in_total)
        y[:fade_frames] *= ramp[:, None]
        self._fade_in_remaining -= fade_frames
        return y

    def _check_audio_params(self, last_version: int, last_params: Optional[AudioParams]) -> tuple[int, AudioParams]:
        params = self._audio_params_provider()
        if params.version != last_version:
            self._apply_audio_params(params)
            if last_params is not None and (
                params.tempo != last_params.tempo or params.pitch_st != last_params.pitch_st
            ):
                self.ring.clear()
                self._fade_in_remaining = self._fade_in_total
        return params.version, params

    def _record_ring_underruns(self, count: int) -> None:
        if count <= 0:
            return
        with self._metrics_lock:
            self._ring_underruns_accum += count

    def consume_ring_underruns(self) -> int:
        with self._metrics_lock:
            count = self._ring_underruns_accum
            self._ring_underruns_accum = 0
            return count

    def _push_visualizer(self, frames: np.ndarray) -> None:
        if self._viz_buffer is None or frames.size == 0:
            return
        self._viz_counter = (self._viz_counter + 1) % self._viz_stride
        if self._viz_counter != 0:
            return
        if self._viz_downsample > 1:
            self._viz_buffer.push(frames[::self._viz_downsample])
        else:
            self._viz_buffer.push(frames)

    def _process_audio_block(
        self,
        x: np.ndarray,
        *,
        eq_profile_enabled: bool,
        profile_iter: int,
        low_watermark_frames: int,
        profile_label: str,
    ) -> tuple[np.ndarray, int]:
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
                        "%s: %.2fms, %.1f fps, ring=%d",
                        profile_label,
                        process_ms,
                        frames_per_second_processed,
                        self.ring.frames_available(),
                    )
                profile_iter += 1
            y = self.fx_chain.process(y)
            y = self._maybe_apply_fade_in(y)
            self._push_visualizer(y)
        return y, profile_iter

    def _read_pcm_chunk(self, stdout) -> Optional[np.ndarray]:
        if self._stop.is_set():
            return None
        while len(self._byte_buffer) < self._frame_bytes:
            chunk = stdout.read(self._read_bytes)
            if not chunk:
                break
            self._byte_buffer.extend(chunk)
        if len(self._byte_buffer) < self._frame_bytes:
            return None
        available_frames = len(self._byte_buffer) // self._frame_bytes
        frames_to_take = min(available_frames, self._read_frames)
        take_bytes = frames_to_take * self._frame_bytes
        data = self._byte_buffer[:take_bytes]
        del self._byte_buffer[:take_bytes]
        if not data:
            return None
        x = np.frombuffer(data, dtype=np.float32)
        if x.size == 0:
            return None
        return x.reshape((-1, self.channels))

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
        last_version = -1
        last_params: Optional[AudioParams] = None
        PREBUFFER_SEC = min(0.6, self._buffer_preset.target_sec)
        TARGET_SEC = self._buffer_preset.target_sec
        HIGH_SEC = self._buffer_preset.high_sec
        LOW_SEC = self._buffer_preset.low_sec
        UNDERRUN_STEP_SEC = 0.05
        UNDERRUN_CAP_SEC = 0.3
        underrun_extra_sec = 0.0

        try:
            # Warm-up until prebuffered
            while not self._stop.is_set():
                last_version, last_params = self._check_audio_params(last_version, last_params)
                x = self._read_pcm_chunk(stdout)
                if x is None:
                    break
                y, profile_iter = self._process_audio_block(
                    x,
                    eq_profile_enabled=eq_profile_enabled,
                    profile_iter=profile_iter,
                    low_watermark_frames=low_watermark_frames,
                    profile_label="EQ warmup",
                )
                if y.size:
                    self.ring.push_blocking(y, stop_event=self._stop)
                if self.ring.frames_available() >= int(PREBUFFER_SEC * self.sample_rate):
                    self._state_cb("ready", None)
                    break

            # Main loop
            while not self._stop.is_set():
                last_version, last_params = self._check_audio_params(last_version, last_params)
                underruns = self.ring.consume_underruns()
                self._record_ring_underruns(underruns)
                if underruns:
                    underrun_extra_sec = min(
                        underrun_extra_sec + (UNDERRUN_STEP_SEC * underruns),
                        UNDERRUN_CAP_SEC,
                    )
                x = self._read_pcm_chunk(stdout)
                if x is None:
                    break
                y, profile_iter = self._process_audio_block(
                    x,
                    eq_profile_enabled=eq_profile_enabled,
                    profile_iter=profile_iter,
                    low_watermark_frames=low_watermark_frames,
                    profile_label="EQ main",
                )
                if y.size:
                    self.ring.push_blocking(y, stop_event=self._stop)

                # Backpressure: keep buffer in a healthy range.
                # If the decoder gets too far ahead, it can make playback sound chaotic.
                target_frames = int((TARGET_SEC + underrun_extra_sec) * self.sample_rate)
                high_frames = int((HIGH_SEC + underrun_extra_sec) * self.sample_rate)
                low_frames = int((LOW_SEC + underrun_extra_sec) * self.sample_rate)
                max_target = int(self.ring.max_frames * 0.95)
                target_frames = min(max(target_frames, low_frames), max_target)
                high_frames = min(high_frames, self.ring.max_frames)
                low_frames = min(low_frames, target_frames)
                if self.ring.frames_available() > high_frames:
                    while (not self._stop.is_set()) and self.ring.frames_available() > target_frames:
                        time.sleep(0.01)

            # EOF: flush DSP tail
            tail = self.dsp.flush()
            if tail.size:
                tail = self.eq_dsp.process(tail)
                tail = self.fx_chain.process(tail)
                self._push_visualizer(tail)
                self.ring.push_blocking(tail, stop_event=self._stop)

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
# Video decoder thread
# -----------------------------

class VideoDecoderThread(threading.Thread):
    """
    Decodes video frames from ffmpeg and pushes them into a ring buffer.
    Sync is handled externally by a Qt timer.
    """
    def __init__(
        self,
        track_path: str,
        start_sec: float,
        fps: float,
        width: int,
        height: int,
        ring_buffer: VideoRingBuffer,
        state_cb,
    ):
        super().__init__(daemon=True)
        self.track_path = track_path
        self.start_sec = float(start_sec)
        self.fps = float(max(1e-3, fps))
        self.width = int(width)
        self.height = int(height)
        self._ring_buffer = ring_buffer
        self._state_cb = state_cb
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._frame_bytes = max(0, self.width * self.height * 4)
        self._byte_buffer = bytearray()
        self._timestamp_lock = threading.Lock()
        self._timestamp_cv = threading.Condition(self._timestamp_lock)
        self._timestamps: deque[float] = deque()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()


    def _enqueue_timestamp(self, timestamp: float) -> None:
        with self._timestamp_cv:
            self._timestamps.append(timestamp)
            self._timestamp_cv.notify()

    def _pop_timestamp(self, timeout: float = 0.25) -> Optional[float]:
        with self._timestamp_cv:
            if self._timestamps:
                return self._timestamps.popleft()
            end_time = time.monotonic() + max(0.0, timeout)
            while not self._timestamps and not self._stop.is_set():
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    break
                self._timestamp_cv.wait(timeout=remaining)
            if self._timestamps:
                return self._timestamps.popleft()
        return None

    def _stderr_reader(self, stderr) -> None:
        pts_pattern = re.compile(r"pts_time:(?P<pts>[-+]?\\d*\\.\\d+|[-+]?\\d+)")
        try:
            for raw_line in iter(stderr.readline, b""):
                if self._stop.is_set():
                    break
                line = raw_line.decode("utf-8", errors="ignore")
                match = pts_pattern.search(line)
                if match:
                    pts_time = safe_float(match.group("pts"), default=math.nan)
                    if math.isfinite(pts_time):
                        self._enqueue_timestamp(pts_time)
        except Exception:
            return

    def _read_frame(self, stdout) -> Optional[bytes]:
        if self._stop.is_set() or self._frame_bytes <= 0:
            return None
        while len(self._byte_buffer) < self._frame_bytes:
            chunk = stdout.read(self._frame_bytes - len(self._byte_buffer))
            if not chunk:
                break
            self._byte_buffer.extend(chunk)
        if len(self._byte_buffer) < self._frame_bytes:
            return None
        data = bytes(self._byte_buffer[:self._frame_bytes])
        del self._byte_buffer[:self._frame_bytes]
        return data

    def run(self) -> None:
        if self._frame_bytes <= 0:
            self._state_cb("error", "Video dimensions unavailable.")
            return
        cmd = make_ffmpeg_video_cmd(self.track_path, self.start_sec, self.fps)
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            self._state_cb("error", f"Failed to start video ffmpeg: {e}")
            return

        stdout = self._proc.stdout
        if stdout is None:
            self._state_cb("error", "ffmpeg video stdout not available")
            return
        stderr = self._proc.stderr
        if stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._stderr_reader,
                args=(stderr,),
                daemon=True,
            )
            self._stderr_thread.start()

        frame_index = 0
        frame_duration = 1.0 / self.fps

        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.02)
                    continue

                data = self._read_frame(stdout)
                if data is None:
                    break

                image = QtGui.QImage(
                    data,
                    self.width,
                    self.height,
                    QtGui.QImage.Format.Format_RGBA8888,
                ).copy()
                
                timestamp = self._pop_timestamp(timeout=0.0)
                if timestamp is None:
                    timestamp = frame_index * frame_duration
                frame_timestamp = self.start_sec + timestamp
                frame_index += 1

                # Push to ring buffer; sync handled by Qt timer externally
                self._ring_buffer.push(frame_timestamp, image)

                # Throttle decode if buffer is full (backpressure)
                while self._ring_buffer.frames_buffered() >= 25 and not self._stop.is_set():
                    time.sleep(0.01)

        except Exception as e:
            self._state_cb("error", f"Video decode error: {e}")
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
    bufferPresetChanged = QtCore.Signal(str)
    effectAutoEnabled = QtCore.Signal(str)
    videoFrameReady = QtCore.Signal(QtGui.QImage, float)
    trackEnding = QtCore.Signal()           # Emitted when track is near end for gapless preload
    trackFinished = QtCore.Signal()         # Emitted when track playback completes

    _EFFECT_PARAM_KEYS = {
        "Compressor": "compressor_enabled",
        "Dynamic EQ": "dynamic_eq_enabled",
        "Subharmonic": "subharmonic_enabled",
        "Reverb": "reverb_enabled",
        "Chorus": "chorus_enabled",
        "Stereo Panner": "stereo_panner_enabled",
        "Stereo Width": "stereo_width_enabled",
        "Saturation": "saturation_enabled",
        "Limiter": "limiter_enabled",
    }

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 2,
        metrics_enabled: bool = True,
        fx_enabled: Optional[dict[str, bool]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.channels = channels

        self._buffer_preset_name = DEFAULT_BUFFER_PRESET
        self._buffer_preset = BUFFER_PRESETS[self._buffer_preset_name]
        self._blocksize_frames = self._buffer_preset.blocksize_frames
        self._latency = self._buffer_preset.latency
        self._auto_enable_effects = False

        self.state = PlayerState.STOPPED
        self.track: Optional[Track] = None

        self._ring = AudioRingBuffer(
            channels,
            max_seconds=self._buffer_preset.ring_max_seconds,
            sample_rate=sample_rate,
        )
        self._viz_buffer = VisualizerBuffer(
            channels,
            max_seconds=self._buffer_preset.ring_max_seconds,
            sample_rate=sample_rate,
        )
        self._video_ring_buffer = VideoRingBuffer(max_frames=30)
        self._video_timer: Optional[QtCore.QTimer] = None
        self._last_video_timestamp: Optional[float] = None
        self._dsp, self._dsp_name = make_dsp(sample_rate, channels)
        self._eq_dsp = EqualizerDSP(sample_rate, channels)
        self._compressor = CompressorEffect(sample_rate, channels, enabled=False)
        self._dynamic_eq = DynamicEqEffect(sample_rate, channels, enabled=False)
        self._subharmonic = SubharmonicEffect(sample_rate, channels, enabled=False)
        self._reverb = ReverbEffect(sample_rate, channels, enabled=False)
        self._chorus = ChorusEffect(sample_rate, channels, enabled=False)
        self._stereo_widener = StereoWidenerEffect()
        self._stereo_panner = StereoPannerEffect(sample_rate, channels)
        self._saturation = SaturationEffect(sample_rate, channels, enabled=False)
        self._limiter = LimiterEffect(sample_rate, channels, enabled=False)
        self._fx_chain = EffectsChain(
            sample_rate,
            channels,
            effects=[
                GainEffect(),
                self._compressor,
                self._dynamic_eq,
                self._subharmonic,
                self._chorus,
                self._stereo_panner,
                self._stereo_widener,
                self._reverb,
                self._saturation,
                self._limiter,
            ],
        )
        if fx_enabled:
            for name, enabled in fx_enabled.items():
                self._fx_chain.enable_effect(name, enabled)
        self._decoder: Optional[DecoderThread] = None
        self._video_decoder: Optional[VideoDecoderThread] = None
        self._video_fps = 30.0

        self._stream: Optional[sd.OutputStream] = None if sd else None
        self._output_device_index: Optional[int] = None
        self._volume = 0.8
        self._muted = False

        self._audio_params = AudioParams(
            tempo=1.0,
            pitch_st=0.0,
            key_lock=True,
            tape_mode=False,
            eq_gains=tuple(0.0 for _ in self._eq_dsp.center_freqs),
            compressor_threshold=0.0,
            compressor_ratio=1.0,
            compressor_attack=0.1,
            compressor_release=1.0,
            compressor_makeup=0.0,
            dynamic_eq_freq=1000.0,
            dynamic_eq_q=1.0,
            dynamic_eq_gain=0.0,
            dynamic_eq_threshold=0.0,
            dynamic_eq_ratio=1.0,
            saturation_drive=0.0,
            saturation_trim=0.0,
            saturation_tone=0.0,
            saturation_tone_enabled=False,
            subharmonic_mix=0.0,
            subharmonic_intensity=0.0,
            subharmonic_cutoff=140.0,
            reverb_decay=1.4,
            reverb_predelay=20.0,
            reverb_wet=0.0,
            chorus_rate=0.8,
            chorus_depth=8.0,
            chorus_mix=0.0,
            stereo_width=1.0,
            panner_azimuth=0.0,
            panner_spread=1.0,
            limiter_threshold=0.0,
            limiter_release_ms=None,
            compressor_enabled=self._compressor.enabled,
            dynamic_eq_enabled=self._dynamic_eq.enabled,
            subharmonic_enabled=self._subharmonic.enabled,
            reverb_enabled=self._reverb.enabled,
            chorus_enabled=self._chorus.enabled,
            stereo_panner_enabled=self._stereo_panner.enabled,
            stereo_width_enabled=self._stereo_widener.enabled,
            saturation_enabled=self._saturation.enabled,
            limiter_enabled=self._limiter.enabled,
            version=0,
        )

        self._seek_offset_sec = 0.0
        self._source_pos_sec = 0.0
        self._position_lock = threading.Lock()
        self._last_position_update = time.monotonic()

        self._playing = False
        self._paused = False
        self._callback_calls = 0
        self._callback_underflows = 0
        self._callback_overflows = 0
        self._callback_time_total = 0.0
        self._callback_time_max = 0.0
        self._metrics_force_enabled = env_flag("TEMPOPITCH_DEBUG_METRICS") or env_flag(
            "TEMPOPITCH_DEBUG_AUTOPLAY"
        )
        self._metrics_enabled = bool(metrics_enabled) or self._metrics_force_enabled
        self._metrics_last_log = time.monotonic()
        self._viz_callback_stride = 1
        self._viz_downsample = 1
        self._viz_callback_counter = 0
        self._fade_out_ramp = np.linspace(1.0, 0.0, 32, dtype=np.float32)
        self._stability_events: deque[tuple[float, int, int]] = deque()
        self._auto_buffer_last_switch = 0.0

        # Gapless playback state
        self._next_track_path: Optional[str] = None
        self._track_ending_emitted = False
        self._gapless_transition_pending = False

        self.dspChanged.emit(self._dsp_name)

    def dsp_name(self) -> str:
        return self._dsp_name

    def set_volume(self, v: float):
        self._volume = clamp(float(v), 0.0, 1.0)

    def set_muted(self, muted: bool):
        self._muted = bool(muted)

    def set_metrics_enabled(self, enabled: bool):
        self._metrics_enabled = bool(enabled) or self._metrics_force_enabled

    def set_buffer_preset(self, preset_name: str) -> None:
        if preset_name not in BUFFER_PRESETS:
            preset_name = DEFAULT_BUFFER_PRESET
        changed = preset_name != self._buffer_preset_name
        self._buffer_preset_name = preset_name
        self._buffer_preset = BUFFER_PRESETS[preset_name]
        self._blocksize_frames = self._buffer_preset.blocksize_frames
        self._latency = self._buffer_preset.latency
        if self.state == PlayerState.STOPPED:
            self._ensure_audio_buffers()
        if changed:
            self.bufferPresetChanged.emit(self._buffer_preset_name)

    def buffer_preset_name(self) -> str:
        return self._buffer_preset_name

    def get_output_devices(self) -> list[dict]:
        """Returns a list of device info dicts: {index, name, hostapi, channels}."""
        if sd is None:
            return []
        devices = []
        try:
            device_list = sd.query_devices()
            hostapis = sd.query_hostapis()
            
            for i, dev in enumerate(device_list):
                if dev['max_output_channels'] > 0:
                    api_index = dev['hostapi']
                    api_name = "Unknown"
                    if 0 <= api_index < len(hostapis):
                         api_name = hostapis[api_index]['name']
                    
                    devices.append({
                        "index": i,
                        "name": dev['name'],
                        "hostapi": api_name,
                        "channels": dev['max_output_channels']
                    })
        except Exception:
            pass
        return devices

    def set_output_device(self, index: Optional[int]) -> None:
        """Sets the output device index. Pass None for system default."""
        if self._output_device_index == index:
            return
        self._output_device_index = index
        # Restart stream if active
        if self._stream is not None:
             # If strictly playing, we might want to seamlessly restart.
             # But _ensure_stream logic is simple; simplest is to close and let main loop or play re-open.
             # Actually, if we just set _stream to None/close it, we need to re-open it.
             # The decoder thread doesn't manage the stream, the engine main thread (Qt) mostly does via _ensure_stream implicitly?
             # No, _ensure_stream is called in `play` and `_state_cb`.
             # If we are playing, we can just close current stream and open new one immediately.
             
             # Safest implementation:
             active = self.state == PlayerState.PLAYING
             self._close_stream()
             if active:
                 self._ensure_stream()
    
    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _restart_audio_for_buffer_preset(self) -> None:
        if self.track is None:
            return
        if self.state not in (PlayerState.PLAYING, PlayerState.PAUSED, PlayerState.LOADING):
            return
        was_paused = self.state == PlayerState.PAUSED
        restart_pos = self.get_position()
        self.stop()
        self._seek_offset_sec = restart_pos
        self.play()
        if was_paused:
            self.pause()

    def _update_stability_window(self, now: float, cb_underflows: int, ring_underruns: int) -> tuple[int, int]:
        self._stability_events.append((now, cb_underflows, ring_underruns))
        cutoff = now - AUTO_BUFFER_WINDOW_SEC
        while self._stability_events and self._stability_events[0][0] < cutoff:
            self._stability_events.popleft()
        total_underflows = sum(event[1] for event in self._stability_events)
        total_underruns = sum(event[2] for event in self._stability_events)
        return total_underflows, total_underruns

    def _ensure_audio_buffers(self) -> None:
        max_seconds = self._buffer_preset.ring_max_seconds
        expected_frames = int(max_seconds * self.sample_rate)
        if self._ring.max_frames != expected_frames:
            self._ring = AudioRingBuffer(
                self.channels,
                max_seconds=max_seconds,
                sample_rate=self.sample_rate,
            )
        if self._viz_buffer.max_frames != expected_frames:
            self._viz_buffer = VisualizerBuffer(
                self.channels,
                max_seconds=max_seconds,
                sample_rate=self.sample_rate,
            )

    def _video_position_provider(self) -> float:
        self.update_position_from_clock()
        # Audio heard now is delayed by hardware output latency
        pos = self.get_position()
        latency = self.get_output_latency_seconds()
        return max(0.0, pos - latency)

    def _start_video_decoder(self) -> None:
        if self.track is None or not self.track.has_video:
            return
        width, height = self.track.video_size
        if width <= 0 or height <= 0:
            logger.warning("Video stream detected but dimensions are missing.")
            return
        self._stop_video_decoder()
        self._video_ring_buffer.clear()
        self._last_video_timestamp = None

        def state_cb(kind, msg):
            if kind == "error":
                logger.warning("Video decoder error: %s", msg)

        self._video_decoder = VideoDecoderThread(
            track_path=self.track.path,
            start_sec=self._seek_offset_sec,
            fps=self._video_fps,
            width=width,
            height=height,
            ring_buffer=self._video_ring_buffer,
            state_cb=state_cb,
        )
        self._video_decoder.start()

        # Start Qt timer for frame presentation (~60fps)
        self._video_timer = QtCore.QTimer(self)
        self._video_timer.setInterval(16)  # ~60 fps
        self._video_timer.timeout.connect(self._on_video_timer_tick)
        self._video_timer.start()

    def _on_video_timer_tick(self) -> None:
        """Called by Qt timer to present the appropriate video frame."""
        if not self._playing or self._paused:
            return
        audio_pos = self._video_position_provider()
        image, timestamp = self._video_ring_buffer.get_frame_for_time(audio_pos)
        if image is not None and timestamp != self._last_video_timestamp:
            self._last_video_timestamp = timestamp
            self.videoFrameReady.emit(image, timestamp)

    def _stop_video_decoder(self) -> None:
        if self._video_timer:
            self._video_timer.stop()
            self._video_timer = None
        if self._video_decoder:
            self._video_decoder.stop()
            self._video_decoder = None
        self._video_ring_buffer.clear()
        self._last_video_timestamp = None

    def _update_audio_params(self, **changes) -> None:
        current = self._audio_params
        self._audio_params = replace(current, **changes, version=current.version + 1)

    def _effect_param_key(self, name: str) -> Optional[str]:
        return self._EFFECT_PARAM_KEYS.get(name)

    def _maybe_auto_enable_effect(self, name: str, *, should_enable: bool) -> dict[str, bool]:
        if not self._auto_enable_effects:
            return {}
        if not should_enable:
            return {}
        key = self._effect_param_key(name)
        if key is None:
            return {}
        if getattr(self._audio_params, key):
            return {}
        self.effectAutoEnabled.emit(name)
        return {key: True}

    def set_dsp_controls(self, tempo: float, pitch_st: float, key_lock: bool, tape_mode: bool):
        tempo = clamp(float(tempo), 0.5, 2.0)
        pitch_st = clamp(float(pitch_st), -12.0, 12.0)
        prev_tempo = self._audio_params.tempo
        self._update_audio_params(
            tempo=tempo,
            pitch_st=pitch_st,
            key_lock=bool(key_lock),
            tape_mode=bool(tape_mode),
        )
        if self._video_decoder and not math.isclose(tempo, prev_tempo):
            self._video_decoder.reset_sync()

    def set_eq_gains(self, gains_db: list[float]):
        if len(gains_db) != len(self._audio_params.eq_gains):
            raise ValueError(f"Expected {len(self._audio_params.eq_gains)} EQ bands")
        eq_gains = tuple(clamp(float(g), -12.0, 12.0) for g in gains_db)
        self._update_audio_params(eq_gains=eq_gains)

    def set_compressor_controls(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        makeup_db: float,
    ) -> None:
        threshold_db = clamp(float(threshold_db), -60.0, 0.0)
        ratio = clamp(float(ratio), 1.0, 20.0)
        attack_ms = clamp(float(attack_ms), 0.1, 200.0)
        release_ms = clamp(float(release_ms), 1.0, 1000.0)
        makeup_db = clamp(float(makeup_db), 0.0, 24.0)
        effect_changes = self._maybe_auto_enable_effect(
            "Compressor",
            should_enable=(ratio > 1.0 or makeup_db > 0.0),
        )
        self._update_audio_params(
            compressor_threshold=threshold_db,
            compressor_ratio=ratio,
            compressor_attack=attack_ms,
            compressor_release=release_ms,
            compressor_makeup=makeup_db,
            **effect_changes,
        )

    def set_dynamic_eq_controls(
        self,
        freq_hz: float,
        q: float,
        gain_db: float,
        threshold_db: float,
        ratio: float,
    ) -> None:
        freq_hz = clamp(float(freq_hz), 20.0, 20000.0)
        q = clamp(float(q), 0.1, 20.0)
        gain_db = clamp(float(gain_db), -12.0, 12.0)
        threshold_db = clamp(float(threshold_db), -60.0, 0.0)
        ratio = clamp(float(ratio), 1.0, 20.0)
        effect_changes = self._maybe_auto_enable_effect(
            "Dynamic EQ",
            should_enable=(abs(gain_db) > 1e-6 or ratio > 1.0),
        )
        self._update_audio_params(
            dynamic_eq_freq=freq_hz,
            dynamic_eq_q=q,
            dynamic_eq_gain=gain_db,
            dynamic_eq_threshold=threshold_db,
            dynamic_eq_ratio=ratio,
            **effect_changes,
        )

    def get_compressor_gain_reduction_db(self) -> Optional[float]:
        if self._compressor is None:
            return None
        return self._compressor.gain_reduction_db()

    def set_limiter_controls(self, threshold_db: float, release_ms: Optional[float]) -> None:
        limiter_release_ms = None if release_ms is None else clamp(float(release_ms), 1.0, 1000.0)
        threshold_db = clamp(float(threshold_db), -60.0, 0.0)
        effect_changes = self._maybe_auto_enable_effect(
            "Limiter",
            should_enable=(threshold_db < 0.0 or limiter_release_ms is not None),
        )
        self._update_audio_params(
            limiter_threshold=threshold_db,
            limiter_release_ms=limiter_release_ms,
            **effect_changes,
        )

    def set_saturation_controls(
        self,
        drive_db: float,
        trim_db: float,
        tone: float,
        tone_enabled: bool,
    ) -> None:
        drive_db = clamp(float(drive_db), 0.0, 24.0)
        trim_db = clamp(float(trim_db), -24.0, 24.0)
        tone = clamp(float(tone), -1.0, 1.0)
        tone_enabled = bool(tone_enabled)
        effect_changes = self._maybe_auto_enable_effect(
            "Saturation",
            should_enable=(
                drive_db > 0.0
                or abs(trim_db) > 1e-6
                or (tone_enabled and abs(tone) > 1e-4)
            ),
        )
        self._update_audio_params(
            saturation_drive=drive_db,
            saturation_trim=trim_db,
            saturation_tone=tone,
            saturation_tone_enabled=tone_enabled,
            **effect_changes,
        )

    def set_subharmonic_controls(self, mix: float, intensity: float, cutoff_hz: float) -> None:
        mix = clamp(float(mix), 0.0, 1.0)
        intensity = clamp(float(intensity), 0.0, 1.5)
        cutoff_hz = clamp(float(cutoff_hz), 40.0, 240.0)
        effect_changes = self._maybe_auto_enable_effect(
            "Subharmonic",
            should_enable=(mix > 1e-5 and intensity > 1e-5),
        )
        self._update_audio_params(
            subharmonic_mix=mix,
            subharmonic_intensity=intensity,
            subharmonic_cutoff=cutoff_hz,
            **effect_changes,
        )

    def set_reverb_controls(self, decay_time: float, pre_delay_ms: float, wet: float) -> None:
        decay_time = clamp(float(decay_time), 0.2, 6.0)
        pre_delay_ms = clamp(float(pre_delay_ms), 0.0, 120.0)
        wet = clamp(float(wet), 0.0, 1.0)
        effect_changes = self._maybe_auto_enable_effect("Reverb", should_enable=wet > 0.0)
        self._update_audio_params(
            reverb_decay=decay_time,
            reverb_predelay=pre_delay_ms,
            reverb_wet=wet,
            **effect_changes,
        )

    def set_chorus_controls(self, rate_hz: float, depth_ms: float, mix: float) -> None:
        rate_hz = clamp(float(rate_hz), 0.05, 5.0)
        depth_ms = clamp(float(depth_ms), 0.0, 20.0)
        mix = clamp(float(mix), 0.0, 1.0)
        effect_changes = self._maybe_auto_enable_effect("Chorus", should_enable=mix > 0.0)
        self._update_audio_params(
            chorus_rate=rate_hz,
            chorus_depth=depth_ms,
            chorus_mix=mix,
            **effect_changes,
        )

    def set_stereo_width(self, width: float) -> None:
        self._update_audio_params(stereo_width=clamp(float(width), 0.0, 2.0))

    def set_stereo_panner_controls(self, azimuth_deg: float, spread: float) -> None:
        self._update_audio_params(
            panner_azimuth=clamp(float(azimuth_deg), -90.0, 90.0),
            panner_spread=clamp(float(spread), 0.0, 1.0),
        )

    def enable_effect(self, name: str, enabled: bool) -> None:
        key = self._effect_param_key(name)
        if key is None:
            return
        self._update_audio_params(**{key: bool(enabled)})

    def get_enabled_effects(self) -> list[str]:
        params = self._audio_params
        enabled = []
        for name, key in self._EFFECT_PARAM_KEYS.items():
            if getattr(params, key):
                enabled.append(name)
        return enabled

    def load_track(self, path: str):
        if not path or not os.path.exists(path):
            self._set_error(f"File not found: {path}")
            return
        self.track = build_track(path)
        self._video_fps = self.track.video_fps if self.track.video_fps > 0 else 30.0
        self._seek_offset_sec = 0.0
        self._track_ending_emitted = False
        self._next_track_path = None
        self._gapless_transition_pending = False
        self._set_state(PlayerState.STOPPED)
        self.trackChanged.emit(self.track)
        self.durationChanged.emit(self.track.duration_sec)

        with self._position_lock:
            self._source_pos_sec = 0.0
        self._last_position_update = time.monotonic()

    def queue_next_track(self, path: str) -> None:
        """Queue a track for gapless playback transition."""
        if path and os.path.exists(path):
            self._next_track_path = path
        else:
            self._next_track_path = None

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
            self._last_position_update = time.monotonic()
            if self._video_decoder:
                self._video_decoder.set_paused(False)
            self._set_state(PlayerState.PLAYING)
            return

        self.stop()  # ensure clean slate

        self._ensure_audio_buffers()
        self._ring.clear()
        self._viz_buffer.clear()
        self._dsp.reset()
        self._eq_dsp.reset()
        self._fx_chain.reset()

        self._playing = True
        self._paused = False
        self._last_position_update = time.monotonic()

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
            elif kind == "eof":
                # Track finished - handle gapless transition
                self._handle_track_eof()

        self._decoder = DecoderThread(
            track_path=self.track.path,
            start_sec=self._seek_offset_sec,
            sample_rate=self.sample_rate,
            channels=self.channels,
            ring=self._ring,
            buffer_preset=self._buffer_preset,
            viz_buffer=None,
            viz_stride=self._viz_callback_stride,
            viz_downsample=self._viz_downsample,
            dsp=self._dsp,
            eq_dsp=self._eq_dsp,
            fx_chain=self._fx_chain,
            compressor=self._compressor,
            dynamic_eq=self._dynamic_eq,
            saturation=self._saturation,
            subharmonic=self._subharmonic,
            reverb=self._reverb,
            chorus=self._chorus,
            stereo_widener=self._stereo_widener,
            stereo_panner=self._stereo_panner,
            limiter=self._limiter,
            audio_params_provider=lambda: self._audio_params,
            state_cb=state_cb
        )
        self._decoder.start()
        self._start_video_decoder()

    def pause(self):
        if self.state == PlayerState.PLAYING:
            self._sync_position()
            self._paused = True
            if self._video_decoder:
                self._video_decoder.set_paused(True)
            self._set_state(PlayerState.PAUSED)

    def stop(self):
        self._playing = False
        self._paused = False

        if self._decoder:
            self._decoder.stop()
            self._decoder = None
        self._stop_video_decoder()

        if self._stream:
            self._close_stream()
            self._stream = None

        self._sync_position()
        self._ring.clear()
        self._viz_buffer.clear()
        self._stability_events.clear()
        self._auto_buffer_last_switch = 0.0
        self._set_state(PlayerState.STOPPED)

    def _handle_track_eof(self) -> None:
        """Handle end-of-track for gapless playback."""
        if self._next_track_path:
            # Gapless transition: start next track without stopping stream
            next_path = self._next_track_path
            self._next_track_path = None
            self._gapless_transition_pending = True
            
            # Stop current decoder but keep stream running
            if self._decoder:
                self._decoder.stop()
                self._decoder = None
            self._stop_video_decoder()
            
            # Load and play next track seamlessly
            self.track = build_track(next_path)
            self._video_fps = self.track.video_fps if self.track.video_fps > 0 else 30.0
            self._seek_offset_sec = 0.0
            self._track_ending_emitted = False
            
            # Emit track change signals
            self.trackChanged.emit(self.track)
            self.durationChanged.emit(self.track.duration_sec)
            
            with self._position_lock:
                self._source_pos_sec = 0.0
            self._last_position_update = time.monotonic()
            
            # Reset DSP state for new track (but keep effects chain warm)
            self._dsp.reset()
            
            # Start new decoder without clearing buffers (seamless transition)
            def state_cb(kind, msg):
                if kind == "error":
                    self._set_error(msg or "Unknown error")
                elif kind == "loading":
                    pass  # Don't change state - we're already playing
                elif kind == "ready":
                    self._gapless_transition_pending = False
                    self._set_state(PlayerState.PLAYING)
                elif kind == "eof":
                    self._handle_track_eof()
            
            self._decoder = DecoderThread(
                track_path=self.track.path,
                start_sec=0.0,
                sample_rate=self.sample_rate,
                channels=self.channels,
                ring=self._ring,
                buffer_preset=self._buffer_preset,
                viz_buffer=None,
                viz_stride=self._viz_callback_stride,
                viz_downsample=self._viz_downsample,
                dsp=self._dsp,
                eq_dsp=self._eq_dsp,
                fx_chain=self._fx_chain,
                compressor=self._compressor,
                dynamic_eq=self._dynamic_eq,
                saturation=self._saturation,
                subharmonic=self._subharmonic,
                reverb=self._reverb,
                chorus=self._chorus,
                stereo_widener=self._stereo_widener,
                stereo_panner=self._stereo_panner,
                limiter=self._limiter,
                audio_params_provider=lambda: self._audio_params,
                state_cb=state_cb
            )
            self._decoder.start()
            self._start_video_decoder()
        else:
            # No next track queued - emit finished signal
            self.trackFinished.emit()

    def seek(self, target_sec: float):
        if self.track is None:
            return
        dur = self.track.duration_sec
        if dur > 0:
            target_sec = clamp(float(target_sec), 0.0, dur)
        else:
            target_sec = max(0.0, float(target_sec))

        self._seek_offset_sec = target_sec
        
        # Clear gapless state when seeking - user is navigating manually
        self._next_track_path = None
        self._track_ending_emitted = False
        self._gapless_transition_pending = False

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
            self._last_position_update = time.monotonic()

    def _ensure_stream(self):
        if self._stream is not None:
            try:
                if not self._stream.active:
                    self._stream.start()
            except Exception:
                pass
            return

        def callback(outdata, frames, time_info, status):
            start = time.perf_counter()
            self._callback_calls += 1
            if not self._playing or self._paused:
                outdata.fill(0)
                elapsed = time.perf_counter() - start
                self._callback_time_total += elapsed
                if elapsed > self._callback_time_max:
                    self._callback_time_max = elapsed
                return

            outdata.fill(0)
            filled = self._ring.pop_into(outdata)
            if filled < frames:
                fade_samples = min(filled, self._fade_out_ramp.shape[0])
                if fade_samples > 1:
                    outdata[filled - fade_samples:filled] *= self._fade_out_ramp[:fade_samples, None]
            if self._viz_buffer is not None:
                self._viz_callback_counter = (self._viz_callback_counter + 1) % self._viz_callback_stride
                if self._viz_callback_counter == 0:
                    viz_frames = outdata[:filled]
                    if self._viz_downsample > 1:
                        viz_frames = viz_frames[::self._viz_downsample]
                    self._viz_buffer.push(viz_frames)
            if status and getattr(status, "output_underflow", False):
                self._callback_underflows += 1
            if status and getattr(status, "output_overflow", False):
                self._callback_overflows += 1
            vol = 0.0 if self._muted else self._volume
            if vol == 0.0:
                outdata.fill(0)
            elif vol != 1.0:
                outdata *= vol

            elapsed = time.perf_counter() - start
            self._callback_time_total += elapsed
            if elapsed > self._callback_time_max:
                self._callback_time_max = elapsed

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self._blocksize_frames,
                latency=self._latency,
                device=self._output_device_index,
                callback=callback
            )
            self._stream.start()
        except Exception as e:
            self._set_error(f"Audio output error: {e}")
            self._stream = None

    def _sync_position(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.monotonic()
        if self._playing and not self._paused and self.state == PlayerState.PLAYING:
            dt = now - self._last_position_update
            if dt > 0:
                with self._position_lock:
                    self._source_pos_sec += dt * float(self._audio_params.tempo)
        self._last_position_update = now

    def update_position_from_clock(self) -> None:
        self._sync_position()

    def log_metrics_if_needed(self) -> None:
        if not self._playing:
            self._metrics_last_log = time.monotonic()
            return
        now = time.monotonic()
        elapsed = now - self._metrics_last_log
        if elapsed < 1.0:
            return
        self._metrics_last_log = now

        ring_underruns = self._decoder.consume_ring_underruns() if self._decoder else 0
        ring_fill_frames = self._ring.frames_available()
        ring_fill_sec = ring_fill_frames / float(self.sample_rate)

        callback_calls = self._callback_calls
        callback_underflows = self._callback_underflows
        callback_overflows = self._callback_overflows
        callback_time_total = self._callback_time_total
        callback_time_max = self._callback_time_max

        self._callback_calls = 0
        self._callback_underflows = 0
        self._callback_overflows = 0
        self._callback_time_total = 0.0
        self._callback_time_max = 0.0

        callback_avg_ms = (callback_time_total / callback_calls) * 1000.0 if callback_calls else 0.0
        callback_max_ms = callback_time_max * 1000.0
        total_underflows, total_underruns = self._update_stability_window(
            now, callback_underflows, ring_underruns
        )
        if (total_underflows + total_underruns) >= AUTO_BUFFER_THRESHOLD:
            if self._buffer_preset_name != AUTO_BUFFER_PRESET and (
                now - self._auto_buffer_last_switch
            ) >= AUTO_BUFFER_WINDOW_SEC:
                self._auto_buffer_last_switch = now
                logger.warning(
                    "Auto switching buffer preset to %s (cb_underflows=%d ring_underruns=%d)",
                    AUTO_BUFFER_PRESET,
                    total_underflows,
                    total_underruns,
                )
                self.set_buffer_preset(AUTO_BUFFER_PRESET)
                self._restart_audio_for_buffer_preset()
                self._stability_events.clear()

        if self._metrics_enabled:
            logger.info(
                "Audio metrics: buffer=%.2fs (frames=%d) ring_underruns=%.2f/s "
                "cb_underflows=%.2f/s cb_overflows=%.2f/s cb_avg=%.2fms cb_max=%.2fms "
                "blocksize=%d latency=%s",
                ring_fill_sec,
                ring_fill_frames,
                ring_underruns / elapsed if elapsed > 0 else 0.0,
                callback_underflows / elapsed if elapsed > 0 else 0.0,
                callback_overflows / elapsed if elapsed > 0 else 0.0,
                callback_avg_ms,
                callback_max_ms,
                self._blocksize_frames,
                self._latency,
            )

    def get_position(self) -> float:
        with self._position_lock:
            return float(self._source_pos_sec)

    def get_buffer_seconds(self) -> float:
        return self._ring.frames_available() / self.sample_rate

    def get_output_latency_seconds(self) -> float:
        latency = None
        if self._stream is not None:
            try:
                latency = self._stream.latency
            except Exception:
                latency = None

        latency_sec = 0.0
        if latency is not None:
            latency_val = None
            if hasattr(latency, "output"):
                latency_val = getattr(latency, "output", None)
            elif isinstance(latency, dict):
                latency_val = latency.get("output")
                if latency_val is None and latency:
                    latency_val = next(iter(latency.values()))
            elif isinstance(latency, (tuple, list)):
                vals = [v for v in latency if isinstance(v, (int, float))]
                if vals:
                    latency_val = max(vals)
            else:
                latency_val = latency

            try:
                latency_sec = float(latency_val)
            except (TypeError, ValueError):
                latency_sec = 0.0

            if not math.isfinite(latency_sec):
                latency_sec = 0.0

        if latency_sec <= 0.0:
            latency_sec = self._blocksize_frames / float(self.sample_rate)

        return latency_sec

    def get_visualizer_frames(
        self,
        frames: Optional[int] = None,
        mono: bool = False,
        delay_sec: Optional[float] = None,
    ) -> np.ndarray:
        delay_frames = 0
        if delay_sec is not None:
            delay_frames = int(max(0.0, float(delay_sec)) * self.sample_rate)
        max_delay = self._viz_buffer.max_frames - 1
        if frames is not None and frames > 0:
            max_delay = max(0, self._viz_buffer.max_frames - frames)
        if delay_frames > max_delay:
            delay_frames = max_delay
        return self._viz_buffer.get_recent(
            frames=frames,
            mono=mono,
            delay_frames=delay_frames,
        )

    def get_video_frame(self) -> tuple[Optional[QtGui.QImage], Optional[float]]:
        audio_pos = self._video_position_provider()
        return self._video_ring_buffer.get_frame_for_time(audio_pos)

    def _set_state(self, st: PlayerState):
        if self.state != st:
            self.state = st
            self.stateChanged.emit(st)

    def _set_error(self, msg: str):
        self._set_state(PlayerState.ERROR)
        self.errorOccurred.emit(msg)


# -----------------------------
