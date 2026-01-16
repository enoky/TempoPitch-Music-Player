from __future__ import annotations

import os
import sys
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from audio.engine import PlayerEngine, sd, _sounddevice_import_error
from dsp import probe_metadata
from config import BUFFER_PRESETS, DEFAULT_BUFFER_PRESET
from models import PlayerState, RepeatMode, THEMES, Track, TrackMetadata, format_track_title
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

MEDIA_EXTS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
}
TRACK_ADD_BATCH = 200


class FolderScanWorker(QtCore.QObject):
    pathsReady = QtCore.Signal(list)
    finished = QtCore.Signal(int)

    def __init__(self, folder: str, exts: set[str], parent=None):
        super().__init__(parent)
        self._folder = folder
        self._exts = exts
        self._abort = False

    @QtCore.Slot()
    def run(self) -> None:
        matches: list[str] = []
        for root, _, files in os.walk(self._folder):
            if self._abort:
                break
            for name in files:
                if self._abort:
                    break
                if os.path.splitext(name)[1].lower() in self._exts:
                    matches.append(os.path.join(root, name))
        matches.sort()
        self.pathsReady.emit(matches)
        self.finished.emit(len(matches))

    def stop(self) -> None:
        self._abort = True


class MetadataWorker(QtCore.QObject):
    metadataReady = QtCore.Signal(str, object)
    finished = QtCore.Signal(int)

    def __init__(self, paths: list[str], parent=None):
        super().__init__(parent)
        self._paths = paths
        self._abort = False

    @QtCore.Slot()
    def run(self) -> None:
        count = 0
        for path in self._paths:
            if self._abort:
                break
            meta = probe_metadata(path)
            self.metadataReady.emit(path, meta)
            count += 1
        self.finished.emit(count)

    def stop(self) -> None:
        self._abort = True

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
        self._pending_tracks: list[Track] = []
        self._pending_track_done_callbacks: list[Callable[[], None]] = []
        self._add_tracks_timer = QtCore.QTimer(self)
        self._add_tracks_timer.setSingleShot(True)
        self._add_tracks_timer.timeout.connect(self._flush_pending_tracks)
        self._pending_metadata_paths: list[str] = []
        self._metadata_thread: Optional[QtCore.QThread] = None
        self._metadata_worker: Optional[MetadataWorker] = None
        self._folder_scan_thread: Optional[QtCore.QThread] = None
        self._folder_scan_worker: Optional[FolderScanWorker] = None

        self.track_title = QtWidgets.QLabel("No track loaded")
        self.track_title.setObjectName("track_title")
        self.track_title.setWordWrap(True)
        title_font = self.track_title.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        self.track_title.setFont(title_font)

        self.track_artist = QtWidgets.QLabel("Unknown Artist")
        self.track_artist.setObjectName("track_artist")
        self.track_artist.setWordWrap(True)
        artist_font = self.track_artist.font()
        artist_font.setPointSize(artist_font.pointSize() + 1)
        self.track_artist.setFont(artist_font)

        self.track_album = QtWidgets.QLabel("Unknown Album")
        self.track_album.setObjectName("track_album")
        self.track_album.setWordWrap(True)

        self.track_meta = QtWidgets.QLabel("")
        self.track_meta.setObjectName("track_meta")
        self.track_meta.setWordWrap(True)

        self._media_size = QtCore.QSize(150, 150)
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
        self.fx_status.setWordWrap(False)
        status_bar = self.statusBar()
        status_bar.addWidget(self.status, 1)
        status_bar.addPermanentWidget(self.fx_status)

        self.header_frame = QtWidgets.QFrame()
        self.header_frame.setObjectName("header_frame")
        self.header_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum,
        )
        header_layout = QtWidgets.QVBoxLayout(self.header_frame)
        header_top_row = QtWidgets.QHBoxLayout()
        self.media_stack_widget = QtWidgets.QFrame()
        self.media_stack_widget.setObjectName("media_frame")
        self.media_stack_widget.setFixedSize(self._media_size)
        self.media_stack_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.media_stack = QtWidgets.QStackedLayout(self.media_stack_widget)
        self.media_stack.setContentsMargins(0, 0, 0, 0)
        self.media_stack.addWidget(self.artwork_label)
        self.media_stack.addWidget(self.video_widget)
        self.media_stack.setCurrentWidget(self.artwork_label)
        header_top_row.addWidget(self.media_stack_widget)
        header_text_column = QtWidgets.QVBoxLayout()
        header_text_column.addWidget(self.track_title)
        header_text_column.addWidget(self.track_artist)
        header_text_column.addWidget(self.track_album)
        header_text_column.addSpacing(6)
        header_text_column.addWidget(self.track_meta)
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
            "Stereo Panner",
            "Stereo Width",
        ]
        for index, name in enumerate(effect_names):
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setAccessibleName(f"{name} enable")
            checkbox.toggled.connect(lambda checked, effect_name=name: self._on_effect_toggled(effect_name, checked))
            row = index // 3
            col = index % 3
            effects_toggle_layout.addWidget(checkbox, row, col)
            self.effect_toggles[name] = checkbox

        self.main_tabs = QtWidgets.QTabWidget()
        self.main_tabs.setObjectName("main_tabs")
        player_tab = QtWidgets.QWidget()
        player_layout = QtWidgets.QVBoxLayout(player_tab)
        player_layout.setContentsMargins(12, 12, 12, 12)
        player_layout.setSpacing(10)
        player_layout.addWidget(self.transport)
        player_layout.addWidget(self.visualizer)
        player_layout.addWidget(self.header_frame)
        player_layout.addWidget(self.dsp_widget)
        player_layout.addStretch(1)
        self.main_tabs.addTab(player_tab, "Player")

        fx_tab = QtWidgets.QWidget()
        fx_layout = QtWidgets.QVBoxLayout(fx_tab)
        fx_layout.setContentsMargins(12, 12, 12, 12)
        fx_layout.setSpacing(10)
        fx_layout.addWidget(self.effects_toggle_group)
        fx_layout.addWidget(self.effects_tabs, 1)
        self.main_tabs.addTab(fx_tab, "FX")

        settings_tab = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(10)
        settings_layout.addWidget(self.audio_group)
        settings_layout.addWidget(self.appearance_group)
        settings_layout.addStretch(1)
        self.main_tabs.addTab(settings_tab, "Settings")

        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(12, 12, 12, 12)
        left.setSpacing(8)
        left.addWidget(self.main_tabs, 1)

        leftw = QtWidgets.QWidget()
        leftw.setLayout(left)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(leftw)
        splitter.addWidget(self.playlist)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)
        self.setCentralWidget(splitter)
        splitter.setSizes([2, 1])

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
        self.playlist.refresh_playing_highlight()
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
        self._start_folder_scan(folder)

    def _start_folder_scan(self, folder: str) -> None:
        if self._folder_scan_thread is not None:
            self._stop_folder_scan_worker(wait=False)
        worker = FolderScanWorker(folder, MEDIA_EXTS)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.pathsReady.connect(self._on_folder_scan_paths, QtCore.Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._on_folder_scan_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        thread.start()
        self._folder_scan_worker = worker
        self._folder_scan_thread = thread

    def _on_folder_scan_paths(self, paths: list[str]) -> None:
        if self.sender() is not self._folder_scan_worker:
            return
        self._add_paths(paths)

    def _on_folder_scan_finished(self, count: int) -> None:
        worker = self.sender()
        if worker is not self._folder_scan_worker:
            return
        thread = self._folder_scan_thread
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()
        self._folder_scan_worker = None
        self._folder_scan_thread = None

    def _make_placeholder_track(self, path: str) -> Track:
        title = os.path.basename(path)
        return Track(
            path=path,
            title=title,
            duration_sec=0.0,
            artist="",
            album="",
            title_display=title,
            cover_art=None,
            has_video=False,
            video_fps=0.0,
            video_size=(0, 0),
        )

    def _add_paths(
        self,
        paths: List[str],
        *,
        select_first_if_empty: bool = True,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        tracks: List[Track] = []
        for p in paths:
            if os.path.isdir(p) or not os.path.exists(p):
                continue
            tracks.append(self._make_placeholder_track(p))

        if not tracks:
            if on_done:
                on_done()
            return

        unique_paths = list(dict.fromkeys(t.path for t in tracks))

        def after_add():
            if self._shuffle:
                self._reset_shuffle_bag()
            if select_first_if_empty and self._current_index < 0 and self.playlist.count() > 0:
                self._current_index = 0
                self.playlist.select_index(0)
                t = self.playlist.get_track(0)
                if t:
                    self.engine.load_track(t.path)
            self._enqueue_metadata(unique_paths)
            if on_done:
                on_done()

        self._enqueue_tracks(tracks, on_done=after_add)

    def _enqueue_tracks(self, tracks: list[Track], *, on_done: Optional[Callable[[], None]] = None) -> None:
        if not tracks:
            if on_done:
                on_done()
            return
        self._pending_tracks.extend(tracks)
        if on_done:
            self._pending_track_done_callbacks.append(on_done)
        if not self._add_tracks_timer.isActive():
            self._add_tracks_timer.start(0)

    def _flush_pending_tracks(self) -> None:
        if not self._pending_tracks:
            return
        batch = self._pending_tracks[:TRACK_ADD_BATCH]
        del self._pending_tracks[:TRACK_ADD_BATCH]
        self.playlist.add_tracks(batch)
        if self._pending_tracks:
            self._add_tracks_timer.start(0)
            return
        callbacks = self._pending_track_done_callbacks
        self._pending_track_done_callbacks = []
        for cb in callbacks:
            cb()

    def _enqueue_metadata(self, paths: list[str]) -> None:
        if not paths:
            return
        self._pending_metadata_paths.extend(paths)
        if self._metadata_thread is None:
            self._start_metadata_worker()

    def _start_metadata_worker(self) -> None:
        if not self._pending_metadata_paths:
            return
        paths = self._pending_metadata_paths
        self._pending_metadata_paths = []
        worker = MetadataWorker(paths)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.metadataReady.connect(self._on_metadata_ready, QtCore.Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._on_metadata_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        thread.start()
        self._metadata_worker = worker
        self._metadata_thread = thread

    def _on_metadata_ready(self, path: str, metadata: TrackMetadata) -> None:
        if self.sender() is not self._metadata_worker:
            return
        self.playlist.update_track_metadata(path, metadata)

    def _on_metadata_finished(self, count: int) -> None:
        worker = self.sender()
        if worker is not self._metadata_worker:
            return
        thread = self._metadata_thread
        if thread is not None:
            thread.quit()
            thread.wait()
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()
        self._metadata_worker = None
        self._metadata_thread = None
        if self._pending_metadata_paths:
            self._start_metadata_worker()

    def _stop_metadata_worker(self, *, wait: bool = True) -> None:
        if self._metadata_worker:
            self._metadata_worker.stop()
        if self._metadata_thread:
            self._metadata_thread.quit()
            if wait:
                self._metadata_thread.wait()
        self._metadata_worker = None
        self._metadata_thread = None

    def _stop_folder_scan_worker(self, *, wait: bool = True) -> None:
        if self._folder_scan_worker:
            self._folder_scan_worker.stop()
        if self._folder_scan_thread:
            self._folder_scan_thread.quit()
            if wait:
                self._folder_scan_thread.wait()
        self._folder_scan_worker = None
        self._folder_scan_thread = None

    def _on_clear(self):
        self._stop_folder_scan_worker(wait=False)
        self._stop_metadata_worker(wait=False)
        self._pending_tracks.clear()
        self._pending_track_done_callbacks.clear()
        self._pending_metadata_paths.clear()
        self.engine.stop()
        self.engine.track = None
        self.playlist.clear()
        self._current_index = -1
        self._shuffle_history.clear()
        self._shuffle_bag = []
        self.track_title.setText("No track loaded")
        self.track_artist.setText("Unknown Artist")
        self.track_album.setText("Unknown Album")
        self.track_meta.setText("")
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
        self._set_effect_toggle(effect_name, enabled, update_checkbox=False)

    def _on_effect_auto_enabled(self, effect_name: str) -> None:
        self._set_effect_toggle(effect_name, True, update_checkbox=True)

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
        self.track_title.setText(title)
        self.track_artist.setText(artist)
        self.track_album.setText(album)
        duration_label = format_time(track.duration_sec) if track.duration_sec > 0 else "--:--"
        ext = os.path.splitext(track.path)[1].lstrip(".").upper()
        meta_parts = [duration_label]
        if ext:
            meta_parts.append(ext)
        self.track_meta.setText(" • ".join(meta_parts))
        idx = self.playlist.index_for_path(track.path)
        if idx < 0:
            idx = self.playlist.current_index()
        if idx < 0:
            idx = self._current_index
        if idx >= 0:
            self._current_index = idx
        self.playlist.set_playing_index(idx)
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
        self._stop_folder_scan_worker()
        self._stop_metadata_worker()
        self._save_ui_settings()
        paths = self.playlist.track_paths()
        if self._pending_tracks:
            paths.extend(t.path for t in self._pending_tracks)
        self.settings.setValue("playlist/paths", paths)
        current_index = self.playlist.current_index()
        if current_index < 0:
            current_index = self._current_index
        self.settings.setValue("playlist/current_index", current_index)
        self.settings.setValue("playlist/position_sec", self.engine.get_position())
        self.engine.stop()
        super().closeEvent(e)

    def _restore_ui_settings(self):
        self._set_slider_from_setting(
            "audio/volume_slider",
            self.transport.volume_slider,
            self.transport.volume_slider.value(),
            value_type=int,
        )
        self._set_checkbox_from_setting("audio/muted", self.transport.mute_btn, False)

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

        self._set_slider_from_setting(
            "dsp/tempo",
            self.dsp_widget.tempo_slider,
            1.0,
            scale=100.0,
            clamp_range=(0.5, 2.0),
        )
        self._set_slider_from_setting(
            "dsp/pitch",
            self.dsp_widget.pitch_slider,
            0.0,
            scale=10.0,
            clamp_range=(-12.0, 12.0),
        )
        self._set_checkbox_from_setting("dsp/key_lock", self.dsp_widget.key_lock, True)
        self._set_checkbox_from_setting("dsp/lock_432", self.dsp_widget.lock_432, False)
        self._set_checkbox_from_setting("dsp/tape_mode", self.dsp_widget.tape_mode, False)

        eq_preset = str(self.settings.value("eq/preset", "Flat"))
        eq_gains_raw = self.settings.value("eq/gains", [0.0] * len(self.equalizer.band_sliders))
        eq_gains = self._normalize_eq_gains(eq_gains_raw, len(self.equalizer.band_sliders))
        if eq_preset in self.equalizer.presets_map:
            self.equalizer.set_gains(eq_gains, preset=eq_preset, emit=False)
        else:
            self.equalizer.set_gains(eq_gains, preset="Custom", emit=False)

        self._set_slider_from_setting(
            "dynamic_eq/freq",
            self.dynamic_eq_widget.freq_slider,
            1000.0,
            clamp_range=(20.0, 20000.0),
            transform=self.dynamic_eq_widget._freq_to_slider,
        )
        self._set_slider_from_setting(
            "dynamic_eq/q",
            self.dynamic_eq_widget.q_slider,
            1.0,
            scale=10.0,
            clamp_range=(0.1, 20.0),
        )
        self._set_slider_from_setting(
            "dynamic_eq/gain",
            self.dynamic_eq_widget.gain_slider,
            0.0,
            scale=10.0,
            clamp_range=(-12.0, 12.0),
        )
        self._set_slider_from_setting(
            "dynamic_eq/threshold",
            self.dynamic_eq_widget.threshold_slider,
            -24.0,
            scale=10.0,
            clamp_range=(-60.0, 0.0),
        )
        self._set_slider_from_setting(
            "dynamic_eq/ratio",
            self.dynamic_eq_widget.ratio_slider,
            4.0,
            scale=10.0,
            clamp_range=(1.0, 20.0),
        )

        self._set_slider_from_setting(
            "compressor/threshold",
            self.compressor_widget.threshold_slider,
            -18.0,
            scale=10.0,
            clamp_range=(-60.0, 0.0),
        )
        self._set_slider_from_setting(
            "compressor/ratio",
            self.compressor_widget.ratio_slider,
            4.0,
            scale=10.0,
            clamp_range=(1.0, 20.0),
        )
        self._set_slider_from_setting(
            "compressor/attack",
            self.compressor_widget.attack_slider,
            10.0,
            scale=10.0,
            clamp_range=(0.1, 200.0),
        )
        self._set_slider_from_setting(
            "compressor/release",
            self.compressor_widget.release_slider,
            120.0,
            clamp_range=(1.0, 1000.0),
        )
        self._set_slider_from_setting(
            "compressor/makeup",
            self.compressor_widget.makeup_slider,
            0.0,
            scale=10.0,
            clamp_range=(0.0, 24.0),
        )

        self._set_slider_from_setting(
            "saturation/drive",
            self.saturation_widget.drive_slider,
            6.0,
            scale=10.0,
            clamp_range=(0.0, 24.0),
        )
        self._set_slider_from_setting(
            "saturation/trim",
            self.saturation_widget.trim_slider,
            0.0,
            scale=10.0,
            clamp_range=(-24.0, 24.0),
        )
        self._set_slider_from_setting(
            "saturation/tone",
            self.saturation_widget.tone_slider,
            0.0,
            scale=100.0,
            clamp_range=(-1.0, 1.0),
        )
        self._set_checkbox_from_setting("saturation/tone_enabled", self.saturation_widget.tone_toggle, False)

        self._set_slider_from_setting(
            "subharmonic/mix",
            self.subharmonic_widget.mix_slider,
            0.25,
            scale=100.0,
            clamp_range=(0.0, 1.0),
        )
        self._set_slider_from_setting(
            "subharmonic/intensity",
            self.subharmonic_widget.intensity_slider,
            0.6,
            scale=100.0,
            clamp_range=(0.0, 1.5),
        )
        self._set_slider_from_setting(
            "subharmonic/cutoff",
            self.subharmonic_widget.cutoff_slider,
            140.0,
            clamp_range=(60.0, 240.0),
        )

        self._set_slider_from_setting(
            "limiter/threshold",
            self.limiter_widget.threshold_slider,
            -1.0,
            scale=10.0,
            clamp_range=(-60.0, 0.0),
        )
        self._set_slider_from_setting(
            "limiter/release",
            self.limiter_widget.release_slider,
            80.0,
            clamp_range=(1.0, 1000.0),
        )
        self._set_checkbox_from_setting("limiter/release_enabled", self.limiter_widget.release_toggle, True)

        self._set_slider_from_setting(
            "reverb/decay",
            self.reverb_widget.decay_slider,
            1.4,
            scale=100.0,
            clamp_range=(0.2, 6.0),
        )
        self._set_slider_from_setting(
            "reverb/predelay",
            self.reverb_widget.predelay_slider,
            20.0,
            clamp_range=(0.0, 120.0),
        )
        self._set_slider_from_setting(
            "reverb/mix",
            self.reverb_widget.mix_slider,
            0.25,
            scale=100.0,
            clamp_range=(0.0, 1.0),
        )

        self._set_slider_from_setting(
            "chorus/rate",
            self.chorus_widget.rate_slider,
            0.8,
            scale=100.0,
            clamp_range=(0.05, 5.0),
        )
        self._set_slider_from_setting(
            "chorus/depth",
            self.chorus_widget.depth_slider,
            8.0,
            scale=10.0,
            clamp_range=(0.0, 20.0),
        )
        self._set_slider_from_setting(
            "chorus/mix",
            self.chorus_widget.mix_slider,
            0.25,
            scale=100.0,
            clamp_range=(0.0, 1.0),
        )

        self._set_slider_from_setting(
            "panner/azimuth",
            self.stereo_panner_widget.azimuth_slider,
            0.0,
            clamp_range=(-90.0, 90.0),
        )
        self._set_slider_from_setting(
            "panner/spread",
            self.stereo_panner_widget.spread_slider,
            1.0,
            scale=100.0,
            clamp_range=(0.0, 1.0),
        )

        self._set_slider_from_setting(
            "stereo/width",
            self.stereo_width_widget.width_slider,
            1.0,
            scale=100.0,
            clamp_range=(0.0, 2.0),
        )

        self._apply_effect_toggle_settings()

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
        self._save_slider_setting("audio/volume_slider", self.transport.volume_slider)
        self._save_checkbox_setting("audio/muted", self.transport.mute_btn)
        self._save_checkbox_setting("audio/metrics_enabled", self.metrics_checkbox)
        self._save_slider_setting("dsp/tempo", self.dsp_widget.tempo_slider, scale=100.0)
        self._save_slider_setting("dsp/pitch", self.dsp_widget.pitch_slider, scale=10.0)
        self._save_checkbox_setting("dsp/key_lock", self.dsp_widget.key_lock)
        self._save_checkbox_setting("dsp/tape_mode", self.dsp_widget.tape_mode)
        self._save_checkbox_setting("dsp/lock_432", self.dsp_widget.lock_432)
        self.settings.setValue("eq/gains", self.equalizer.gains())
        self.settings.setValue("eq/preset", self.equalizer.presets.currentText())
        self.settings.setValue(
            "dynamic_eq/freq",
            self.dynamic_eq_widget._slider_to_freq(self.dynamic_eq_widget.freq_slider.value()),
        )
        self._save_slider_setting("dynamic_eq/q", self.dynamic_eq_widget.q_slider, scale=10.0)
        self._save_slider_setting("dynamic_eq/gain", self.dynamic_eq_widget.gain_slider, scale=10.0)
        self._save_slider_setting("dynamic_eq/threshold", self.dynamic_eq_widget.threshold_slider, scale=10.0)
        self._save_slider_setting("dynamic_eq/ratio", self.dynamic_eq_widget.ratio_slider, scale=10.0)
        self._save_slider_setting("compressor/threshold", self.compressor_widget.threshold_slider, scale=10.0)
        self._save_slider_setting("compressor/ratio", self.compressor_widget.ratio_slider, scale=10.0)
        self._save_slider_setting("compressor/attack", self.compressor_widget.attack_slider, scale=10.0)
        self._save_slider_setting("compressor/release", self.compressor_widget.release_slider)
        self._save_slider_setting("compressor/makeup", self.compressor_widget.makeup_slider, scale=10.0)
        self._save_slider_setting("saturation/drive", self.saturation_widget.drive_slider, scale=10.0)
        self._save_slider_setting("saturation/trim", self.saturation_widget.trim_slider, scale=10.0)
        self._save_slider_setting("saturation/tone", self.saturation_widget.tone_slider, scale=100.0)
        self._save_checkbox_setting("saturation/tone_enabled", self.saturation_widget.tone_toggle)
        self._save_slider_setting("subharmonic/mix", self.subharmonic_widget.mix_slider, scale=100.0)
        self._save_slider_setting("subharmonic/intensity", self.subharmonic_widget.intensity_slider, scale=100.0)
        self._save_slider_setting("subharmonic/cutoff", self.subharmonic_widget.cutoff_slider)
        self._save_slider_setting("limiter/threshold", self.limiter_widget.threshold_slider, scale=10.0)
        self._save_slider_setting("limiter/release", self.limiter_widget.release_slider)
        self._save_checkbox_setting("limiter/release_enabled", self.limiter_widget.release_toggle)
        self._save_slider_setting("reverb/decay", self.reverb_widget.decay_slider, scale=100.0)
        self._save_slider_setting("reverb/predelay", self.reverb_widget.predelay_slider)
        self._save_slider_setting("reverb/mix", self.reverb_widget.mix_slider, scale=100.0)
        self._save_slider_setting("chorus/rate", self.chorus_widget.rate_slider, scale=100.0)
        self._save_slider_setting("chorus/depth", self.chorus_widget.depth_slider, scale=10.0)
        self._save_slider_setting("chorus/mix", self.chorus_widget.mix_slider, scale=100.0)
        self._save_slider_setting("panner/azimuth", self.stereo_panner_widget.azimuth_slider)
        self._save_slider_setting("panner/spread", self.stereo_panner_widget.spread_slider, scale=100.0)
        self._save_slider_setting("stereo/width", self.stereo_width_widget.width_slider, scale=100.0)
        self._save_effect_toggle_settings()

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

    def _set_checkbox_from_setting(
        self,
        key: str,
        checkbox: QtWidgets.QAbstractButton,
        default: bool = False,
    ) -> None:
        checkbox.setChecked(self.settings.value(key, default, type=bool))

    def _save_checkbox_setting(self, key: str, checkbox: QtWidgets.QAbstractButton) -> None:
        self.settings.setValue(key, checkbox.isChecked())

    def _set_slider_from_setting(
        self,
        key: str,
        slider: QtWidgets.QSlider,
        default: float,
        *,
        scale: float = 1.0,
        clamp_range: Optional[tuple[float, float]] = None,
        transform: Optional[Callable[[float], float]] = None,
        value_type: type = float,
    ) -> None:
        value = self.settings.value(key, default, type=value_type)
        value = float(value)
        if clamp_range is not None:
            value = clamp(value, clamp_range[0], clamp_range[1])
        if transform is not None:
            value = transform(value)
        slider.setValue(int(round(value * scale)))

    def _save_slider_setting(self, key: str, slider: QtWidgets.QSlider, *, scale: float = 1.0) -> None:
        self.settings.setValue(key, slider.value() / scale)

    def _set_effect_toggle(self, effect_name: str, enabled: bool, *, update_checkbox: bool) -> None:
        checkbox = self.effect_toggles.get(effect_name)
        if checkbox is None:
            return
        if update_checkbox:
            checkbox.blockSignals(True)
            checkbox.setChecked(bool(enabled))
            checkbox.blockSignals(False)
        self.engine.enable_effect(effect_name, bool(enabled))
        self._save_checkbox_setting(self._effect_setting_key(effect_name), checkbox)
        self._update_enabled_fx_label()

    def _apply_effect_toggle_settings(self) -> None:
        enabled_effects = set(self.engine.get_enabled_effects())
        for name in self.effect_toggles:
            enabled = self.settings.value(self._effect_setting_key(name), name in enabled_effects, type=bool)
            self._set_effect_toggle(name, enabled, update_checkbox=True)

    def _save_effect_toggle_settings(self) -> None:
        for name, checkbox in self.effect_toggles.items():
            self._save_checkbox_setting(self._effect_setting_key(name), checkbox)

    def _restore_playlist_session(self):
        saved_paths = self.settings.value("playlist/paths", [], type=list)
        saved_index = self.settings.value("playlist/current_index", -1, type=int)
        saved_pos = self.settings.value("playlist/position_sec", 0.0, type=float)

        def restore_position():
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

        if saved_paths:
            self._add_paths(saved_paths, select_first_if_empty=False, on_done=restore_position)
        else:
            restore_position()
