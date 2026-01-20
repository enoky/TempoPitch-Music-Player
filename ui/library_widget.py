"""
Library widget for browsing and managing the media library.
"""

from __future__ import annotations

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

    def update_tracks(self, tracks: List[LibraryTrack]):
        self.beginResetModel()
        self._tracks = tracks
        self.endResetModel()

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
                return track.title
            elif col == 1:
                return track.artist
            elif col == 2:
                return track.album
            elif col == 3:
                return track.genre
            elif col == 4:
                return str(track.year) if track.year else ""
            elif col == 5:
                return format_time(track.duration_sec)
            elif col == 6:
                return str(track.play_count)
            elif col == 7:
                return track.date_added.strftime("%Y-%m-%d %H:%M") if track.date_added else ""

        elif role == QtCore.Qt.ItemDataRole.UserRole:
            return track

        elif role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return track.path

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
            if column == 0: return track.title.lower()
            if column == 1: return track.artist.lower()
            if column == 2: return track.album.lower()
            if column == 3: return track.genre.lower()
            if column == 4: return track.year or 0
            if column == 5: return track.duration_sec
            if column == 6: return track.play_count
            if column == 7: return track.date_added or datetime.min
            return ""

        self._tracks.sort(key=sort_key, reverse=reverse)
        self.layoutChanged.emit()


class LibraryWidget(QtWidgets.QWidget):
    trackActivated = QtCore.Signal(LibraryTrack)
    addFolderRequested = QtCore.Signal()

    def __init__(self, library_service: LibraryService, parent=None):
        super().__init__(parent)
        self._library = library_service
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        # Layouts
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar_layout = QtWidgets.QHBoxLayout()
        
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Search library...")
        self.search_bar.textChanged.connect(self._on_search_changed)
        
        self.add_folder_btn = QtWidgets.QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self.addFolderRequested)
        
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)

        toolbar_layout.addWidget(self.search_bar)
        toolbar_layout.addWidget(self.add_folder_btn)
        toolbar_layout.addWidget(self.refresh_btn)
        
        # Splitter for Sidebar + Table
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        
        # Sidebar (Artists/Albums)
        self.sidebar = QtWidgets.QTreeWidget()
        self.sidebar.setHeaderLabel("Browse")
        self.sidebar.itemClicked.connect(self._on_sidebar_click)
        self.splitter.addWidget(self.sidebar)
        
        # Table
        self.table = QtWidgets.QTableView()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.alternatingRowColors()
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(self._on_table_double_click)
        
        self.splitter.addWidget(self.table)
        self.splitter.setStretchFactor(1, 3)

        main_layout.addLayout(toolbar_layout)
        main_layout.addWidget(self.splitter)

    def refresh(self):
        """Reload data from the library."""
        # Refresh sidebar
        self.sidebar.clear()
        
        # All Tracks item
        all_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["All Tracks"])
        all_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, "all")
        
        # Artists
        artists_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["Artists"])
        artists = self._library.get_artists()
        for artist in artists:
            item = QtWidgets.QTreeWidgetItem(artists_item, [artist])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"artist:{artist}")
            
        self.sidebar.expandItem(artists_item)
        
        # Refresh table (default to all tracks)
        self._load_tracks(self._library.get_all_tracks())

    def _load_tracks(self, tracks: List[LibraryTrack]):
        self._model = LibraryTableModel(tracks, self)
        self.table.setModel(self._model)
        self.table.resizeColumnsToContents()
        # Set some reasonable defaults for column widths
        if self.table.model().columnCount() > 0:
            self.table.setColumnWidth(0, 300) # Title
            self.table.setColumnWidth(1, 200) # Artist

    def _on_search_changed(self, text: str):
        tracks = self._library.search(text)
        self._model.update_tracks(tracks)

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
