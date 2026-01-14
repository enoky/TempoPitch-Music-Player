from __future__ import annotations

import os
import sys
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from audio.engine import PlayerEngine, sd, _sounddevice_import_error
from dsp import build_track
from config import BUFFER_PRESETS, DEFAULT_BUFFER_PRESET
from models import PlayerState, RepeatMode, THEMES, Track, format_track_title
from theme import build_palette, build_stylesheet
from utils import clamp, env_flag, format_time, have_exe, safe_float
from ui.widgets import (
    VisualizerWidget,
    VideoWidget,
    VideoPopoutDialog,
    TempoPitchWidget,
    EqualizerWidget,
    ReverbWidget,
    ChorusWidget,
    StereoWidthWidget,
    StereoPannerWidget,
    DynamicEqWidget,
    CompressorWidget,
    SaturationWidget,
    SubharmonicWidget,
    LimiterWidget,
    TransportWidget,
    PlaylistWidget,
)

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


