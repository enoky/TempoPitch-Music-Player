from __future__ import annotations

import os
import random
import sys
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from audio.engine import PlayerEngine, sd, _sounddevice_import_error
from metadata import probe_metadata
from ui.workers.library_scan import LibraryScanWorker, MetadataWorker

from config import BUFFER_PRESETS, DEFAULT_BUFFER_PRESET
from models import PlayerState, RepeatMode, THEMES, Track, TrackMetadata, format_track_title
from theme import build_palette, build_stylesheet
from ui.theme_selector import ThemeSelectorWidget
from utils import clamp, env_flag, format_time, have_exe, safe_float
from ui.widgets import (
    VisualizerWidget,
    VideoWidget,
    VideoPopoutDialog,
    TempoPitchWidget,
    EqualizerWidget, # Duplicate import
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
    # PlaylistWidget,  <-- Removed
)
from ui.library_widget import LibraryWidget
from ui.audio_device_dialog import AudioDeviceSelectionDialog
from library import LibraryService
from library_db import LibraryTrack

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






# Main Window
# -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # ... (rest of init) ...

# Main Window
# -----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robust FX Media Player")
        self.resize(1440, 900)

        self.settings = QtCore.QSettings("enoky", "RobustFXMediaPlayer")
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
        self.equalizer = EqualizerWidget()
        self.library = LibraryService()
        self.library_widget = LibraryWidget(self.library)
        
        self._scan_thread: Optional[QtCore.QThread] = None
        self._scan_worker: Optional[LibraryScanWorker] = None

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

        self.theme_selector = ThemeSelectorWidget(self._theme_name)
        self.theme_selector.themeChanged.connect(self._on_theme_changed)

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
        self.effects_tabs.addTab(self.equalizer, "Equalizer")
        self.effects_tabs.addTab(self.dsp_widget, "Tempo / Pitch")
        self.effects_tabs.addTab(self.dynamic_eq_widget, "Dynamic EQ")
        self.effects_tabs.addTab(self.compressor_widget, "Compressor")
        self.effects_tabs.addTab(self.limiter_widget, "Limiter")
        self.effects_tabs.addTab(self.saturation_widget, "Saturation")
        self.effects_tabs.addTab(self.subharmonic_widget, "Subharmonic")
        self.effects_tabs.addTab(self.chorus_widget, "Chorus")
        self.effects_tabs.addTab(self.reverb_widget, "Reverb")
        self.effects_tabs.addTab(self.stereo_panner_widget, "Stereo Panner")
        self.effects_tabs.addTab(self.stereo_width_widget, "Stereo Width")

        self.effects_toggle_group = QtWidgets.QGroupBox("FX Enable")
        self.effects_toggle_group.setObjectName("effects_toggle_group")
        effects_toggle_layout = QtWidgets.QVBoxLayout(self.effects_toggle_group)
        effects_toggle_layout.setContentsMargins(8, 8, 8, 8)
        effects_toggle_layout.setSpacing(6)
        effects_toggle_layout.addStretch(1) # Push checkboxes to top or center? Let's keep them at top.
        # Actually, let's put stretch at the end of loop.
        self.effect_toggles: dict[str, QtWidgets.QCheckBox] = {}
        effect_names = [
            "Dynamic EQ",
            "Compressor",
            "Limiter",
            "Saturation",
            "Subharmonic",
            "Chorus",
            "Reverb",
            "Stereo Panner",
            "Stereo Width",
        ]
        for index, name in enumerate(effect_names):
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.setAccessibleName(f"{name} enable")
            checkbox.toggled.connect(lambda checked, effect_name=name: self._on_effect_toggled(effect_name, checked))
            effects_toggle_layout.insertWidget(index, checkbox) # Insert before stretch if we added one, or just add.

            self.effect_toggles[name] = checkbox

        # Player Widget (Top Section - Always Visible)
        self.player_widget = QtWidgets.QWidget()
        self.player_widget.setObjectName("player_widget")
        player_layout = QtWidgets.QVBoxLayout(self.player_widget)
        player_layout.setContentsMargins(12, 12, 12, 12)
        player_layout.setSpacing(10)

        # Top row: Left (Transport + Viz) | Right (Metadata)
        top_row = QtWidgets.QHBoxLayout()
        
        # Left column
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(10)
        left_col.addWidget(self.transport)
        left_col.addWidget(self.visualizer)
        
        top_row.addLayout(left_col, 2)
        top_row.addWidget(self.header_frame, 1)

        player_layout.addLayout(top_row)

        # FX Tab Content
        fx_tab = QtWidgets.QWidget()
        fx_layout = QtWidgets.QHBoxLayout(fx_tab)
        fx_layout.setContentsMargins(12, 12, 12, 12)
        fx_layout.setSpacing(10)
        fx_layout.addWidget(self.effects_toggle_group)
        fx_layout.addWidget(self.effects_tabs, 1)

        # Themes Tab Content
        themes_tab = QtWidgets.QWidget()
        themes_layout = QtWidgets.QVBoxLayout(themes_tab)
        themes_layout.setContentsMargins(12, 12, 12, 12)
        themes_layout.setSpacing(10)
        themes_layout.addWidget(self.theme_selector)
        themes_layout.addStretch(1)

        # Bottom Tabs (Library, FX, Themes)
        self.content_tabs = QtWidgets.QTabWidget()
        self.content_tabs.setObjectName("content_tabs")
        self.content_tabs.addTab(self.library_widget, "Library")
        self.content_tabs.addTab(fx_tab, "FX")
        self.content_tabs.addTab(themes_tab, "Themes")

        # Main Layout Splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.player_widget)
        splitter.addWidget(self.content_tabs)
        splitter.setStretchFactor(0, 0)  # Top (Player) - let it take natural size or minimal
        splitter.setStretchFactor(1, 1)  # Bottom (Tabs) - expands
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)
        self.setCentralWidget(splitter)
        # splitter.setSizes([2, 1]) # Let layout determine initial sizes or set specifically if needed

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        open_files = QtGui.QAction("Open Files…", self)
        open_folder = QtGui.QAction("Open Folder…", self)
        quit_act = QtGui.QAction("Quit", self)
        file_menu.addAction(open_files)
        file_menu.addAction(open_folder)
        file_menu.addSeparator()
        file_menu.addAction(quit_act)

        # Audio menu with buffer preset options
        audio_menu = self.menuBar().addMenu("&Audio")
        buffer_preset_menu = audio_menu.addMenu("Buffer Preset")
        buffer_preset_menu.setToolTip("Balance output latency vs stability.")
        self._buffer_preset_actions: dict[str, QtGui.QAction] = {}
        buffer_preset_group = QtGui.QActionGroup(self)
        for preset_name in BUFFER_PRESETS.keys():
            action = QtGui.QAction(preset_name, self, checkable=True)
            buffer_preset_group.addAction(action)
            buffer_preset_menu.addAction(action)
            self._buffer_preset_actions[preset_name] = action
            action.triggered.connect(lambda checked, p=preset_name: self._on_buffer_preset_changed(p) if checked else None)

            action.triggered.connect(lambda checked, p=preset_name: self._on_buffer_preset_changed(p) if checked else None)

        # Output Device Action
        self.output_device_action = QtGui.QAction("Output Device...", self)
        self.output_device_action.setToolTip("Select audio output device.")
        self.output_device_action.triggered.connect(self._on_choose_device)
        audio_menu.addAction(self.output_device_action)
        
        # Restore last device
        self._restore_audio_device()

        audio_menu.addSeparator()
        self.metrics_action = QtGui.QAction("Enable Metrics Logging", self, checkable=True)
        self.metrics_action.setToolTip("Log audio engine metrics periodically.")
        self.metrics_action.setChecked(bool(metrics_enabled))
        audio_menu.addAction(self.metrics_action)

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



        self.library_widget.addFileRequested.connect(self._add_files_dialog)
        self.library_widget.addFolderRequested.connect(self._add_folder_dialog)
        self.library_widget.trackActivated.connect(self._on_library_track_activated)
        
        
        # Buffer preset and metrics actions are connected via the Audio menu above
        self.metrics_action.toggled.connect(self._on_metrics_toggled)
        self.popout_video_btn.clicked.connect(self._toggle_video_window)

        self.library_widget.modelLayoutChanged.connect(self._on_library_layout_changed)

        self.engine.trackChanged.connect(self._on_track_changed)
        self.engine.stateChanged.connect(self._on_state_changed)
        self.engine.errorOccurred.connect(self._on_error)
        self.engine.bufferPresetChanged.connect(self._on_engine_buffer_preset_changed)
        self.engine.effectAutoEnabled.connect(self._on_effect_auto_enabled)
        self.engine.trackFinished.connect(self._on_track_finished)

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

        # Scan queue: list of (paths, is_folder)
        self._scan_queue: list[tuple[list[str], bool]] = []
        # Metadata queue
        self._pending_metadata_paths: list[str] = []
        self._metadata_thread: Optional[QtCore.QThread] = None
        self._metadata_worker: Optional[MetadataWorker] = None
        
        self._apply_ui_settings()
        self.compressor_widget.set_meter_provider(self.engine.get_compressor_gain_reduction_db)
        self._initial_warnings()
        # self._restore_playlist_session()  # TODO: Restore session for library
        self._on_state_changed(self.engine.state)
        self._update_enabled_fx_label()
        self._schedule_debug_autoplay()

    def _schedule_debug_autoplay(self) -> None:
        pass
        # Debug autoplay disabled for now in library mode


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
        # self.playlist.refresh_playing_highlight()
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
        self._start_scan(paths, is_folder=False)

    def _add_folder_dialog(self):
        last_dir = self.settings.value("last_dir", os.path.expanduser("~"))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", last_dir)
        if not folder:
            return
        self.settings.setValue("last_dir", folder)
        self._start_scan([folder], is_folder=True)

    def _start_scan(self, paths: list[str], is_folder: bool):
        # If a scan is already running, queue this request
        if self._scan_thread is not None and self._scan_thread.isRunning():
            self._scan_queue.append((paths, is_folder))
            self.status.setText(f"Scan running. Queued {len(self._scan_queue)} item(s)...")
            return

        if self._scan_thread is not None:
             self._cleanup_scan_worker()
        
        worker = LibraryScanWorker(self.library, paths, is_folder)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_scan_finished)
        worker.preliminary_finished.connect(self.library_widget.refresh)
        worker.progress.connect(lambda _, msg: self.status.setText(msg)) # Update status bar
        
        thread.start()
        self._scan_worker = worker
        self._scan_thread = thread
        self.status.setText(f"Scanning {'folder' if is_folder else 'files'}...")

    def _on_scan_finished(self, count: int, added_paths: list[str]):
        self.status.setText(f"Scan complete. Added {count} tracks.")
        self.library_widget.refresh()
        self._cleanup_scan_worker()
        
        # Queue tracks for metadata fetching
        if added_paths:
             self._pending_metadata_paths.extend(added_paths)
             self._process_metadata_queue()

        # Process next item in scan queue immediately
        if self._scan_queue:
            next_paths, next_is_folder = self._scan_queue.pop(0)
            self._start_scan(next_paths, next_is_folder)

    def _cleanup_scan_worker(self):
        if self._scan_thread:
            self._scan_thread.quit()
            self._scan_thread.wait()
        self._scan_worker = None
        self._scan_thread = None

    def _process_metadata_queue(self):
        # If metadata thread is already running, let it finish.
        # It will restart if more items are pending in _on_metadata_finished.
        if self._metadata_thread is not None and self._metadata_thread.isRunning():
            return
            
        if not self._pending_metadata_paths:
            return

        # Take a batch of paths (or all of them)
        paths_to_process = list(self._pending_metadata_paths)
        self._pending_metadata_paths.clear()
        
        worker = MetadataWorker(self.library, paths_to_process)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_metadata_finished)
        worker.progress.connect(lambda msg: self.status.setText(msg)) # Update status bar with metadata progress
        
        thread.start()
        self._metadata_worker = worker
        self._metadata_thread = thread

    def _on_metadata_finished(self):
        if self._metadata_thread:
             self._metadata_thread.quit()
             self._metadata_thread.wait()
        self._metadata_worker = None
        self._metadata_thread = None
        self.library_widget.refresh() # Refresh to show new metadata
        self.status.setText("Metadata update complete.")
        
        # Check if more paths came in while we were working
        if self._pending_metadata_paths:
             self._process_metadata_queue()


    def _on_library_track_activated(self, track: LibraryTrack):
        self.engine.load_track(track.path)
        self.engine.play()

    def _on_clear(self):
        self._scan_queue.clear() # Clear pending scans
        self._cleanup_scan_worker()
        
        self._pending_metadata_paths.clear()
        if self._metadata_worker:
            self._metadata_worker.stop()
        if self._metadata_thread:
            self._metadata_thread.quit()
            self._metadata_thread.wait()
        self._metadata_worker = None
        self._metadata_thread = None
            
        self.engine.stop()
        self.engine.track = None
        # self.playlist.clear() # Library doesn't have partial clear yet, maybe implement later if needed
        # For now, maybe just clear current selection or stop?
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
        # Update menu action if not already checked
        action = self._buffer_preset_actions.get(preset)
        if action and not action.isChecked():
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
        self.settings.setValue("audio/buffer_preset", preset)
        self.engine.set_buffer_preset(preset)

    def _on_engine_buffer_preset_changed(self, preset: str) -> None:
        if preset not in BUFFER_PRESETS:
            return
        action = self._buffer_preset_actions.get(preset)
        if action and not action.isChecked():
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
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

    def _on_library_layout_changed(self):
        """Called when library sort order or content changes."""
        if self.engine.track:
            self._sync_current_index_from_path(self.engine.track.path)

    def _on_library_track_activated(self, track: LibraryTrack):
        # Update current index based on table selection
        indexes = self.library_widget.table.selectionModel().selectedRows()
        if indexes:
            self._current_index = indexes[0].row()
        
        self.engine.load_track(track.path)
        self.engine.play()
        self.library.record_play(track.id) if track.id else None

    def _on_clear(self):
        self._cleanup_scan_worker()
        self.engine.stop()
        self.engine.track = None
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
        self.library_widget.set_current_track_path(None)

    def _toggle_play_pause(self, _checked: Optional[bool] = None):
        if self.engine.state == PlayerState.PLAYING:
            self.engine.pause()
        else:
            self._on_play()

    def _on_play(self):
        if self.engine.track is not None:
             self.engine.play()
             return
        
        idx = self.library_widget.table.currentIndex()
        row = idx.row() if idx.isValid() else 0
        self._play_row(row)

    def _set_shuffle(self, on: bool):
        self._shuffle = bool(on)
        self.settings.setValue("playback/shuffle", self._shuffle)
        self._shuffle_history.clear()
        self._reset_shuffle_bag()

    def _set_repeat_mode(self, mode: RepeatMode):
        self._repeat_mode = mode
        self.settings.setValue("playback/repeat", mode.value)

    def _reset_shuffle_bag(self):
        model = self.library_widget.table.model()
        if not model: return
        count = model.rowCount()
        if count <= 0:
            self._shuffle_bag = []
            return
        
        indices = [i for i in range(count) if i != self._current_index]
        random.shuffle(indices)
        self._shuffle_bag = indices

    def _next_shuffle_index(self, current: int) -> Optional[int]:
        if not self._shuffle_bag:
            return None
        return self._shuffle_bag.pop()

    def _advance_track(self, direction: int, auto: bool = False):
        model = self.library_widget.table.model()
        if not model: return
        count = model.rowCount()
        if count == 0:
            return
        
        current = self._current_index

        if self._repeat_mode == RepeatMode.ONE:
            self._play_row(current, push_history=False)
            return

        if self._shuffle:
            if direction < 0 and self._shuffle_history:
                 idx = self._shuffle_history.pop()
                 self._play_row(idx, push_history=False)
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
            self._play_row(idx, push_history=False)
            return

        # Normal sequence
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
        
        self._play_row(idx, push_history=False)

    def _on_prev(self):
         self._advance_track(direction=-1)

    def _on_next(self):
        self._advance_track(direction=1)

    def _preload_next_track(self) -> None:
        """Preload the next track for gapless playback."""
        # Avoid preloading multiple times
        if self.engine._next_track_path is not None:
            return
        
        # Don't preload if we just started playing (position < 2 sec)
        # This prevents rapid transitions when seeking near end
        pos = self.engine.get_position()
        if pos < 2.0:
            return
        
        next_path = self._get_next_track_path()
        if next_path:
            self.engine.queue_next_track(next_path)

    def _validate_current_index(self, model: QtCore.QAbstractItemModel) -> None:
        """
        Verify that self._current_index actually points to the currently playing track.
        If not, attempt to find the track and update self._current_index.
        """
        if self.engine.track is None:
            return

        # Check if current index is valid and matches current track path
        if 0 <= self._current_index < model.rowCount():
            idx = model.index(self._current_index, 0)
            track = model.data(idx, QtCore.Qt.ItemDataRole.UserRole)
            if track:
                # robust comparison
                p1 = os.path.normpath(track.path).lower()
                p2 = os.path.normpath(self.engine.track.path).lower()
                if p1 == p2:
                    return # All good

        # If we got here, the index is stale or wrong. Re-sync.
        self._sync_current_index_from_path(self.engine.track.path)

    def _get_next_track_path(self) -> Optional[str]:
        """Determine the next track path based on current settings."""
        model = self.library_widget.table.model()
        if not model:
            return None
        count = model.rowCount()
        if count == 0:
            return None
        
        # Ensure we are starting from the correct index
        self._validate_current_index(model)
        
        current = self._current_index

        if self._repeat_mode == RepeatMode.ONE:
            # Repeat current track
            if current >= 0 and current < count:
                idx = model.index(current, 0)
                track = model.data(idx, QtCore.Qt.ItemDataRole.UserRole)
                return track.path if track else None
            return None

        if self._shuffle:
            # Peek at next shuffle index without consuming it
            if not self._shuffle_bag:
                if self._repeat_mode == RepeatMode.ALL:
                    # Will reset bag when needed
                    indices = [i for i in range(count) if i != current]
                    if indices:
                        random.shuffle(indices)
                        next_idx = indices[-1]
                        idx = model.index(next_idx, 0)
                        track = model.data(idx, QtCore.Qt.ItemDataRole.UserRole)
                        return track.path if track else None
                return None
            next_idx = self._shuffle_bag[-1]  # Peek, don't pop
            idx = model.index(next_idx, 0)
            track = model.data(idx, QtCore.Qt.ItemDataRole.UserRole)
            return track.path if track else None

        # Normal sequence
        next_idx = current + 1
        if next_idx >= count:
            if self._repeat_mode == RepeatMode.ALL:
                next_idx = 0
            else:
                return None
        
        if next_idx < 0 or next_idx >= count:
            return None
            
        idx = model.index(next_idx, 0)
        track = model.data(idx, QtCore.Qt.ItemDataRole.UserRole)
        return track.path if track else None

    def _on_track_finished(self) -> None:
        """Handle track finished signal when no next track was queued."""
        # Only advance if we're actually playing and not in a rapid transition state
        if self.engine.state != PlayerState.PLAYING:
            return
        # Advance to next track using the normal method
        self._advance_track(direction=1, auto=True)

    def _play_row(self, row: int, push_history: bool = True):
        model = self.library_widget.table.model()
        if not model or row < 0 or row >= model.rowCount():
            return

        idx = model.index(row, 0)
        track = model.data(idx, QtCore.Qt.ItemDataRole.UserRole)
        if not track:
            return

        if push_history and self._shuffle and self._current_index >= 0 and row != self._current_index:
             self._shuffle_history.append(self._current_index)

        self._current_index = row
        self.library_widget.table.selectRow(row)
        self.engine.load_track(track.path)
        self.engine.play()
        self.library.record_play(track.id) if track.id else None

    def _sync_current_index_from_path(self, track_path: str) -> None:
        """Sync _current_index with the library row matching the track path."""
        model = self.library_widget.table.model()
        if not model:
            return
            
        # Normalize target path for robust comparison
        norm_target = os.path.normpath(track_path).lower()
        
        count = model.rowCount()
        for row in range(count):
            idx = model.index(row, 0)
            track = model.data(idx, QtCore.Qt.ItemDataRole.UserRole)
            
            if track:
                # Normalize candidate path
                norm_candidate = os.path.normpath(track.path).lower()
                
                if norm_candidate == norm_target:
                    if self._current_index != row:
                        # Update shuffle state when transitioning
                        if self._shuffle and self._current_index >= 0:
                            self._shuffle_history.append(self._current_index)
                            # Remove the new track from shuffle bag if present
                            if row in self._shuffle_bag:
                                try:
                                    self._shuffle_bag.remove(row)
                                except ValueError:
                                    pass
                        self._current_index = row
                        self.library_widget.table.selectRow(row)
                    return

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
        # Highlight in library if present
        self.library_widget.set_current_track_path(track.path)
        
        # Sync _current_index with the library (important for gapless transitions)
        self._sync_current_index_from_path(track.path)
        
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

        # Update status text
        if st == PlayerState.STOPPED:
            self.status.setText("Stopped.")
        elif st == PlayerState.PLAYING:
            self.status.setText("Playing.")
        elif st == PlayerState.PAUSED:
            self.status.setText("Paused.")
        elif st == PlayerState.LOADING:
            self.status.setText("Loading...")
        elif st == PlayerState.ERROR:
             self.status.setText(f"Error: {self.engine.error_message if hasattr(self.engine, 'error_message') else 'Unknown Error'}")

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

        # Gapless playback: preload next track when approaching end
        if self._dur > 0 and pos >= self._dur - 3.0 and self.engine.state == PlayerState.PLAYING:
            self._preload_next_track()

    def _update_enabled_fx_label(self) -> None:
        enabled = self.engine.get_enabled_effects()
        label_text = f"Enabled FX: {', '.join(enabled)}" if enabled else "Enabled FX: None"
        if self.fx_status.text() != label_text:
            self.fx_status.setText(label_text)

    def closeEvent(self, e: QtGui.QCloseEvent):
        # self._stop_folder_scan_worker() # Legacy
        # self._stop_metadata_worker()    # Legacy
        self._save_ui_settings()
        
        # Save Library session
        self.settings.setValue("library/current_index", self._current_index)
        if self.engine.track:
             self.settings.setValue("library/current_path", self.engine.track.path)

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
        action = self._buffer_preset_actions.get(buffer_preset)
        if action:
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)
        self.engine.set_buffer_preset(buffer_preset)
        metrics_enabled = self.settings.value("audio/metrics_enabled", True, type=bool)
        self.metrics_action.blockSignals(True)
        self.metrics_action.setChecked(bool(metrics_enabled))
        self.metrics_action.blockSignals(False)
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
        self._save_slider_setting("panner/spread", self.stereo_panner_widget.spread_slider, scale=100.0)
        self._save_slider_setting("stereo/width", self.stereo_width_widget.width_slider, scale=100.0)
        self._save_effect_toggle_settings()
        
        # Save Library session
        self.settings.setValue("library/current_index", self._current_index)
        if self.engine.track:
             self.settings.setValue("library/current_path", self.engine.track.path)

    def _save_effect_toggle_settings(self) -> None:
        for name, checkbox in self.effect_toggles.items():
            self._save_checkbox_setting(self._effect_setting_key(name), checkbox)

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

    def _restore_playlist_session(self):
        saved_index = self.settings.value("library/current_index", -1, type=int)
        saved_path = self.settings.value("library/current_path", "")
        
        # If we had a path saved, try to find it in library
        if saved_path:
            track = self.library.get_track_by_path(saved_path)
            if track:
                self.engine.load_track(track.path)
            
        # Try to restore index selection
        if saved_index >= 0:
            self._current_index = saved_index
            self.library_widget.table.selectRow(saved_index)

    def _closeEvent(self, event: QtGui.QCloseEvent):
        self._save_ui_settings()
        event.accept()


    def _on_subharmonic_controls_changed(self, mix: float, intensity: float, cutoff_hz: float):
        self.settings.setValue("subharmonic/mix", float(mix))
        self.settings.setValue("subharmonic/intensity", float(intensity))
        self.settings.setValue("subharmonic/cutoff", float(cutoff_hz))

    def _on_limiter_controls_changed(self, threshold: float, release_ms: Optional[float]):
        self.settings.setValue("limiter/threshold", float(threshold))
        if release_ms is not None:
             self.settings.setValue("limiter/release", float(release_ms))

    def _on_reverb_controls_changed(self, decay: float, predelay: float, wet: float):
        self.settings.setValue("reverb/decay", float(decay))
        self.settings.setValue("reverb/predelay", float(predelay))
        self.settings.setValue("reverb/wet", float(wet))

    def _on_chorus_controls_changed(self, rate: float, depth: float, mix: float):
        self.settings.setValue("chorus/rate", float(rate))
        self.settings.setValue("chorus/depth", float(depth))
        self.settings.setValue("chorus/mix", float(mix))

    def _on_stereo_panner_changed(self, azimuth: float, spread: float):
        self.settings.setValue("stereo_panner/azimuth", float(azimuth))
        self.settings.setValue("stereo_panner/spread", float(spread))

    def _on_stereo_width_changed(self, width: float):
        self.settings.setValue("stereo_width/width", float(width))

    def _on_effect_auto_enabled(self, name: str):
        if name in self.effect_toggles:
            self.effect_toggles[name].setChecked(True)

    def _on_error(self, msg: str):
        self.status.setText(f"Error: {msg}")
        QtWidgets.QMessageBox.warning(self, "Error", msg)

    def _update_enabled_fx_label(self):
        enabled = [name for name, cb in self.effect_toggles.items() if cb.isChecked()]
        if not enabled:
            self.fx_status.setText("Enabled FX: None")
        else:
            self.fx_status.setText(f"Enabled FX: {', '.join(enabled)}")

    def _on_effect_toggled(self, name: str, enabled: bool):
        self.settings.setValue(f"fx/{name}", bool(enabled))
        self._update_enabled_fx_label()
        
        # Dispatch to engine
        if name == "Compressor":
            self.engine.set_compressor_enabled(enabled)
        elif name == "Dynamic EQ":
            self.engine.set_dynamic_eq_enabled(enabled)
        elif name == "Saturation":
            self.engine.set_saturation_enabled(enabled)
        elif name == "Subharmonic":
            self.engine.set_subharmonic_enabled(enabled)
        elif name == "Reverb":
            self.engine.set_reverb_enabled(enabled)
        elif name == "Chorus":
            self.engine.set_chorus_enabled(enabled)
        elif name == "Stereo Panner":
            self.engine.set_stereo_panner_enabled(enabled)
        elif name == "Stereo Width":
            self.engine.set_stereo_width_enabled(enabled)
        elif name == "Limiter":
            self.engine.set_limiter_enabled(enabled)

    def _restore_audio_device(self):
        # Try to restore last used device
        last_device_name = self.settings.value("audio/output_device", "Default")
        
        if last_device_name == "Default":
             self.engine.set_output_device(None)
             return

        devices = self.engine.get_output_devices()
        # devices is list of dicts {index, name, hostapi, channels}
        
        found = False
        for dev in devices:
            if dev['name'] == last_device_name:
                self.engine.set_output_device(dev['index'])
                found = True
                break
        
        if not found:
            self.engine.set_output_device(None) # Fallback

    def _on_choose_device(self):
        devices = self.engine.get_output_devices()
        
        # Find current index
        current_index = self.engine._output_device_index 
        # Using internal property since I didn't add a getter method yet, 
        # but I should ideally add one. For now this works.
        
        dialog = AudioDeviceSelectionDialog(devices, current_index, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_index = dialog.selected_device_index
            
            # Find name for storage
            name = "Default"
            if new_index is not None:
                for dev in devices:
                    if dev['index'] == new_index:
                        name = dev['name']
                        break
            
            self.settings.setValue("audio/output_device", name)
            self.engine.set_output_device(new_index)
