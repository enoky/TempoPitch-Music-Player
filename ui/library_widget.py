"""
Library widget for browsing and managing the media library.
"""

from __future__ import annotations

import os
import subprocess
import logging
from datetime import datetime
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from library_db import LibraryTrack
from library import LibraryService
from ui.widgets import SVG_ICON_TEMPLATES, render_svg_icon
from utils import format_time

logger = logging.getLogger(__name__)


class LibraryTableModel(QtCore.QAbstractTableModel):
    """
    Model for displaying library tracks in a table view.
    """

    COLUMNS = [
        "#",
        "Title",
        "Artist",
        "Album",
        "Genre",
        "Year",
        "Duration",
        "Plays",
        "Date Added",
    ]

    def __init__(self, tracks: List[LibraryTrack], parent=None):
        super().__init__(parent)
        self._tracks = tracks
        self._current_path: Optional[str] = None

    def update_tracks(self, tracks: List[LibraryTrack]):
        self.beginResetModel()
        self._tracks = tracks
        self.endResetModel()

    def set_current_path(self, path: Optional[str]):
        """Set the currently playing track path to show the indicator."""
        # Normalize path for comparison if possible
        norm_path = os.path.normpath(path) if path else None
        
        if self._current_path == norm_path:
            return
        
        self._current_path = norm_path

        # Emit dataChanged for all rows to update background color
        if self._tracks:
            tl = self.index(0, 0)
            br = self.index(len(self._tracks) - 1, len(self.COLUMNS) - 1)
            self.dataChanged.emit(tl, br, [QtCore.Qt.ItemDataRole.BackgroundRole])

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self._tracks)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self.COLUMNS)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        track = self._tracks[index.row()]
        col = index.column()

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return str(track.track_number) if track.track_number else ""
            elif col == 1:
                return track.title
            elif col == 2:
                return track.artist
            elif col == 3:
                return track.album
            elif col == 4:
                return track.genre
            elif col == 5:
                return str(track.year) if track.year else ""
            elif col == 6:
                return format_time(track.duration_sec)
            elif col == 7:
                return str(track.play_count)
            elif col == 8:
                return track.date_added.strftime("%Y-%m-%d %H:%M") if track.date_added else ""

        elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
            # Check normalized paths
            track_path = os.path.normpath(track.path) if track.path else None
            
            if self._current_path and track_path == self._current_path:
                # Use theme highlight color with transparency
                palette = QtGui.QGuiApplication.palette()
                color = palette.color(QtGui.QPalette.ColorRole.Highlight)
                color.setAlpha(100)  # Increased visibility (approx 40%)
                return QtGui.QBrush(color)

        elif role == QtCore.Qt.ItemDataRole.UserRole:
            return track

        elif role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return track.path
        
        elif role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            if col == 0: # Track number
                return QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            if col == 6 or col == 7: # Duration, Plays
                return QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            return QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter

        return None

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation == QtCore.Qt.Orientation.Horizontal and role == QtCore.Qt.ItemDataRole.DisplayRole:
            if 0 <= section < len(self.COLUMNS):
                return self.COLUMNS[section]
        return None

    def sort(self, column: int, order: QtCore.Qt.SortOrder):
        self.layoutAboutToBeChanged.emit()
        
        reverse = (order == QtCore.Qt.SortOrder.DescendingOrder)
        
        def sort_key(track: LibraryTrack):
            # Sort by Track Number
            if column == 0: 
                return (track.track_number or 0, track.album.lower(), track.title.lower())
            # Sort by Title
            if column == 1: 
                return (track.title.lower(), track.track_number or 0)
            # Sort by Artist
            if column == 2: 
                return (track.artist.lower(), track.album.lower(), track.track_number or 0)
            # Sort by Album
            if column == 3: 
                return (track.album.lower(), track.track_number or 0)
            if column == 4: return track.genre.lower()
            if column == 5: return track.year or 0
            if column == 6: return track.duration_sec
            if column == 7: return track.play_count
            if column == 8: return track.date_added or datetime.min
            return ""

        self._tracks.sort(key=sort_key, reverse=reverse)
        self.layoutChanged.emit()


class LibraryDelegate(QtWidgets.QStyledItemDelegate):
    """
    Delegate to handle custom background painting for the playing row.
    Necessary because stylesheets on QTableView often override the model's BackgroundRole.
    """
    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        # Draw custom background if present
        bg = index.data(QtCore.Qt.ItemDataRole.BackgroundRole)
        if bg and isinstance(bg, QtGui.QBrush):
            painter.save()
            painter.fillRect(option.rect, bg)
            painter.restore()
            
        super().paint(painter, option, index)


class LibraryWidget(QtWidgets.QWidget):
    trackActivated = QtCore.Signal(LibraryTrack)
    addFolderRequested = QtCore.Signal()

    def __init__(self, library_service: LibraryService, parent=None):
        super().__init__(parent)
        self._library = library_service
        self._current_track_path: Optional[str] = None
        self._setup_ui()
        self.refresh()
    
    def set_current_track_path(self, path: Optional[str]):
        """Set the currently playing track path."""
        self._current_track_path = path
        if hasattr(self, '_model'):
            self._model.set_current_path(path)

    def _setup_ui(self):
        # Layouts
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Toolbar
        toolbar_frame = QtWidgets.QFrame()
        toolbar_frame.setObjectName("library_toolbar")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        toolbar_layout.setSpacing(12)
        
        # Search
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setObjectName("search_bar")
        self.search_bar.setPlaceholderText("Search library...")
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.setFixedWidth(240)
        self.search_bar.textChanged.connect(self._on_search_changed)
        
        # Add Folder Button
        self.add_folder_btn = QtWidgets.QToolButton()
        self.add_folder_btn.setText("Add Folder")
        self.add_folder_btn.setToolTip("Add folder to library")
        self.add_folder_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.add_folder_btn.clicked.connect(self.addFolderRequested)
        self.add_folder_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        
        # Refresh Button
        self.refresh_btn = QtWidgets.QToolButton()
        self.refresh_btn.setToolTip("Refresh library")
        self.refresh_btn.clicked.connect(self.refresh)
        self.refresh_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # Clear Library Button
        self.clear_lib_btn = QtWidgets.QToolButton()
        self.clear_lib_btn.setToolTip("Clear library")
        self.clear_lib_btn.clicked.connect(self._on_clear_library)
        self.clear_lib_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        # Track Count
        self.track_count_label = QtWidgets.QLabel("0 tracks")
        self.track_count_label.setObjectName("track_count_label")

        toolbar_layout.addWidget(self.search_bar)
        toolbar_layout.addWidget(self.add_folder_btn)
        toolbar_layout.addWidget(self.refresh_btn)
        toolbar_layout.addWidget(self.clear_lib_btn)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.track_count_label)
        
        # Splitter for Sidebar + Table
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        
        # Sidebar (Artists/Albums)
        self.sidebar = QtWidgets.QTreeWidget()
        self.sidebar.setHeaderHidden(True)
        self.sidebar.setIndentation(16)
        self.sidebar.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.sidebar.itemClicked.connect(self._on_sidebar_click)
        self.sidebar.setStyleSheet("QTreeWidget { background: transparent; }")
        
        sidebar_container = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 8, 4, 0) # Top padding for aesthetics
        sidebar_layout.addWidget(self.sidebar)
        
        self.splitter.addWidget(sidebar_container)
        
        # Table
        self.table = QtWidgets.QTableView()
        self.table.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.alternatingRowColors()
        self.table.setShowGrid(False)
        self.table.setItemDelegate(LibraryDelegate(self.table))
        
        # Header Styling
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setHighlightSections(False)
        header.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        header.setFixedHeight(32)

        self.table.verticalHeader().setVisible(False)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.doubleClicked.connect(self._on_table_double_click)

        self.splitter.addWidget(self.table)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setCollapsible(0, False)

        main_layout.addWidget(toolbar_frame)
        main_layout.addWidget(self.splitter, 1)  # Ensure splitter takes all remaining vertical space

    def changeEvent(self, event: QtCore.QEvent):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.PaletteChange, QtCore.QEvent.Type.StyleChange):
            self._update_icons()

    def _update_icons(self):
        # Update icons based on current palette/theme
        text_color = self.palette().color(QtGui.QPalette.ColorRole.Text)
        self.add_folder_btn.setIcon(render_svg_icon(SVG_ICON_TEMPLATES["folder"], text_color, 16))
        self.refresh_btn.setIcon(render_svg_icon(SVG_ICON_TEMPLATES["refresh"], text_color, 16))
        self.clear_lib_btn.setIcon(render_svg_icon(SVG_ICON_TEMPLATES["trash"], text_color, 16))
        self.refresh() # To update sidebar icons

    def refresh(self):
        """Reload data from the library."""
        # Refresh sidebar
        self.sidebar.clear()
        text_color = self.palette().color(QtGui.QPalette.ColorRole.Text)
        
        # All Tracks item
        all_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["All Tracks"])
        all_item.setIcon(0, render_svg_icon(SVG_ICON_TEMPLATES["list"], text_color, 16))
        all_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, "all")
        
        # Artists
        artists_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["Artists"])
        artists_item.setIcon(0, render_svg_icon(SVG_ICON_TEMPLATES["user"], text_color, 16))
        artists = self._library.get_artists()
        for artist in artists:
            item = QtWidgets.QTreeWidgetItem(artists_item, [artist])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"artist:{artist}")
            
        self.sidebar.expandItem(artists_item)
        
        # Refresh table (default to all tracks)
        self._load_tracks(self._library.get_all_tracks())
        
        # Initial icon update just in case
        if self.add_folder_btn.icon().isNull():
             self._update_icons()

    def _load_tracks(self, tracks: List[LibraryTrack]):
        # consistently sort tracks by Artist -> Album -> Track Number -> Title
        # regardless of how they were retrieved (All, Artist, Search)
        def sort_key(t: LibraryTrack):
             # Handle None values for sorting safely
             artist = (t.artist or "").lower()
             album = (t.album or "").lower()
             track_num = t.track_number or 0
             title = (t.title or "").lower()
             return (artist, album, track_num, title)

        tracks.sort(key=sort_key)
        self._model = LibraryTableModel(tracks, self)
        self.table.setModel(self._model)
        self.table.resizeColumnsToContents()
        
        # Adjust column widths
        # Adjust column widths
        self.table.setColumnWidth(0, 40) # Track Number
        self.table.setColumnWidth(1, 240) # Title
        self.table.setColumnWidth(2, 200) # Artist
        self.table.setColumnWidth(3, 200) # Album
        
        self.track_count_label.setText(f"{len(tracks)} tracks")

    def _on_search_changed(self, text: str):
        tracks = self._library.search(text)
        self._model.update_tracks(tracks)
        self.track_count_label.setText(f"{len(tracks)} tracks")

    def _on_sidebar_click(self, item: QtWidgets.QTreeWidgetItem, column: int):
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            return
            
        if data == "all":
            self._load_tracks(self._library.get_all_tracks())
        elif data.startswith("artist:"):
            artist = data.split(":", 1)[1]
            self._load_tracks(self._library.get_tracks_by_artist(artist))

    def _on_table_double_click(self, index: QtCore.QModelIndex):
        track = self._model.data(index, QtCore.Qt.ItemDataRole.UserRole)
        if track:
            self.trackActivated.emit(track)

    def _show_context_menu(self, pos: QtCore.QPoint):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
            
        track = self._model.data(index, QtCore.Qt.ItemDataRole.UserRole)
        if not track:
            return

        menu = QtWidgets.QMenu(self)
        
        play_action = menu.addAction("Play")
        play_action.triggered.connect(lambda: self.trackActivated.emit(track))
        
        show_action = menu.addAction("Show in Explorer")
        show_action.triggered.connect(lambda: self._show_in_explorer(track.path))
        
        menu.addSeparator()
        
        remove_action = menu.addAction("Remove from Library")
        remove_action.triggered.connect(lambda: self._remove_track(track))
        
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _show_in_explorer(self, path: str):
        if not path or not os.path.exists(path):
            return
        try:
            # maintain cross-platform compatibility if possible, but prioritize Windows as per user env
            if os.name == "nt":
                subprocess.Popen(f'explorer /select,"{path}"')
        except Exception as e:
            logger.error(f"Failed to show in explorer: {e}")

    def _remove_track(self, track: LibraryTrack):
        if not track or not track.id:
            return
            
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Remove Track",
            f"Are you sure you want to remove '{track.title}' from the library?\nThe file will NOT be deleted.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            self._library.remove_track(track.id)
            self.refresh()

    def _on_clear_library(self):
        """Clear all tracks from the library."""
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Clear Library",
            "Are you sure you want to clear the entire library?\nThis will remove all tracks and playlists from the database.\nFiles on disk will NOT be deleted.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            self._library.clear_library()
            self.refresh()
