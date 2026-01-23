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
            track_path = track.norm_path if track.path else None
            
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

    # -------------------------------------------------------------------------
    # Drag and Drop support for playlist reordering
    # -------------------------------------------------------------------------

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        """Return item flags, enabling drag-and-drop when in playlist mode."""
        default_flags = super().flags(index)
        if index.isValid():
            return default_flags | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled
        return default_flags | QtCore.Qt.ItemFlag.ItemIsDropEnabled

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        return QtCore.Qt.DropAction.MoveAction

    def mimeTypes(self) -> List[str]:
        return ["application/x-playlist-track-row"]

    def mimeData(self, indexes: List[QtCore.QModelIndex]) -> QtCore.QMimeData:
        """Encode row indices for drag operation."""
        mime_data = QtCore.QMimeData()
        rows = sorted(set(index.row() for index in indexes if index.isValid()))
        mime_data.setData("application/x-playlist-track-row", ",".join(str(r) for r in rows).encode())
        return mime_data

    def dropMimeData(
        self,
        data: QtCore.QMimeData,
        action: QtCore.Qt.DropAction,
        row: int,
        column: int,
        parent: QtCore.QModelIndex,
    ) -> bool:
        """Handle drop to reorder tracks. Returns True if successful."""
        if action == QtCore.Qt.DropAction.IgnoreAction:
            return True
        if not data.hasFormat("application/x-playlist-track-row"):
            return False

        # Decode source rows
        raw = data.data("application/x-playlist-track-row").data().decode()
        source_rows = [int(r) for r in raw.split(",") if r]
        if not source_rows:
            return False

        # Determine target row
        if row == -1:
            if parent.isValid():
                target_row = parent.row()
            else:
                target_row = len(self._tracks)
        else:
            target_row = row

        # Perform reorder in-memory
        self.layoutAboutToBeChanged.emit()
        
        # Extract tracks to move
        tracks_to_move = [self._tracks[r] for r in sorted(source_rows, reverse=True)]
        for r in sorted(source_rows, reverse=True):
            del self._tracks[r]
        
        # Adjust target if needed (rows above target were removed)
        for r in source_rows:
            if r < target_row:
                target_row -= 1
        
        # Insert at target
        for i, track in enumerate(reversed(tracks_to_move)):
            self._tracks.insert(target_row, track)
        
        self.layoutChanged.emit()
        return True

    def get_track_ids_in_order(self) -> List[int]:
        """Return current track IDs in display order for saving to DB."""
        return [t.id for t in self._tracks if t.id is not None]


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
    addFileRequested = QtCore.Signal()
    modelLayoutChanged = QtCore.Signal()

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
        
        # Add File Button
        self.add_file_btn = QtWidgets.QToolButton()
        self.add_file_btn.setText("Add File")
        self.add_file_btn.setToolTip("Add file(s) to library")
        self.add_file_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.add_file_btn.clicked.connect(self.addFileRequested)
        self.add_file_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

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
        toolbar_layout.addWidget(self.add_file_btn)
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
        self.sidebar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.sidebar.customContextMenuRequested.connect(self._show_sidebar_context_menu)
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
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
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
        self.add_file_btn.setIcon(render_svg_icon(SVG_ICON_TEMPLATES["file"], text_color, 16))
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

        # All Videos item
        videos_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["All Videos"])
        videos_item.setIcon(0, render_svg_icon(SVG_ICON_TEMPLATES["video"], text_color, 16))
        videos_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, "all_videos")

        # Playlists (right after All Videos for quick access)
        self.playlists_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["Playlists"])
        self.playlists_item.setIcon(0, render_svg_icon(SVG_ICON_TEMPLATES["list"], text_color, 16))
        self.playlists_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, "playlists_header")
        self._populate_playlists()
        
        # Artists
        artists_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["Artists"])
        artists_item.setIcon(0, render_svg_icon(SVG_ICON_TEMPLATES["user"], text_color, 16))
        
        artists = self._library.get_artists()
        for artist in artists:
            item = QtWidgets.QTreeWidgetItem(artists_item, [artist])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"artist:{artist}")
            
        # Albums (store item for updates)
        self.albums_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["Albums"])
        self.albums_item.setIcon(0, render_svg_icon(SVG_ICON_TEMPLATES["disc"], text_color, 16))
        self._populate_albums(None) # Initially populate with all

        # Genres
        genres_item = QtWidgets.QTreeWidgetItem(self.sidebar, ["Genres"])
        genres_item.setIcon(0, render_svg_icon(SVG_ICON_TEMPLATES["tag"], text_color, 16))
        
        genres = self._library.get_genres()
        for genre in genres:
            item = QtWidgets.QTreeWidgetItem(genres_item, [genre])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"genre:{genre}")

        # Expand items
        self.sidebar.expandItem(self.playlists_item)
        self.sidebar.expandItem(artists_item)
        self.sidebar.expandItem(self.albums_item)
        self.sidebar.expandItem(genres_item)
        
        # Refresh table
        self._load_tracks(self._library.get_all_tracks())
        
        # Initial icon update just in case
        if self.add_folder_btn.icon().isNull():
             self._update_icons()

    def _populate_albums(self, artist_filter: Optional[str]):
        """Populate the Albums tree item, optionally filtering by artist."""
        if not hasattr(self, 'albums_item'):
            return
            
        # Clear existing children
        self.albums_item.takeChildren()
        
        albums = self._library.get_albums(artist_filter)
        # Sort albums by name
        albums.sort(key=lambda x: x[0].lower())
        
        for album, artist in albums:
            # Display as "Album (Artist)" or just "Album" if artist is unknown
            # If we are filtering by artist, maybe just show Album name? 
            # Current requirement: "only show albums from that artist".
            # It's cleaner to keep the format consistent or simplify if redundant.
            # Let's keep consistent for now.
            label = f"{album} ({artist})" if artist else album
            item = QtWidgets.QTreeWidgetItem(self.albums_item, [label])
            
            safe_val = f"{album}|{artist}" if artist else album
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"album:{safe_val}")

    def _populate_playlists(self):
        """Populate the Playlists tree item."""
        if not hasattr(self, 'playlists_item'):
            return
            
        # Clear existing children
        self.playlists_item.takeChildren()
        
        playlists = self._library.get_playlists()
        for playlist in playlists:
            item = QtWidgets.QTreeWidgetItem(self.playlists_item, [playlist.name])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, f"playlist:{playlist.id}")

    def _load_tracks(self, tracks: List[LibraryTrack]):
        # Reset playlist mode
        self._current_playlist_id = None
        
        # Disable drag-and-drop for non-playlist views
        self.table.setDragEnabled(False)
        self.table.setAcceptDrops(False)
        self.table.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop)
        self.table.setSortingEnabled(True)
        
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
        self._model.layoutChanged.connect(self.modelLayoutChanged.emit)
        self.table.setModel(self._model)
        self.table.resizeColumnsToContents()
        self.modelLayoutChanged.emit()
        
        # Adjust column widths
        self.table.setColumnWidth(0, 40) # Track Number
        self.table.setColumnWidth(1, 240) # Title
        self.table.setColumnWidth(2, 200) # Artist
        self.table.setColumnWidth(3, 200) # Album
        
        self.track_count_label.setText(f"{len(tracks)} tracks")
        
        # Restore current path indicator if set
        if self._current_track_path:
            self._model.set_current_path(self._current_track_path)

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
            self._populate_albums(None) # Show all albums
        elif data == "all_videos":
            self._load_tracks(self._library.get_all_videos())
            self._populate_albums(None) # Might want to filter albums to only those with videos later, but acceptable for now
        elif data.startswith("artist:"):
            artist = data.split(":", 1)[1]
            self._load_tracks(self._library.get_tracks_by_artist(artist))
            # Filter albums list to only this artist
            self._populate_albums(artist)
            # Expand albums to show the user the filtered list
            self.sidebar.expandItem(self.albums_item)
        elif data.startswith("album:"):
            # Format is "album:Name" or "album:Name|Artist"
            val = data.split(":", 1)[1]
            if "|" in val:
                album, artist = val.rsplit("|", 1)
                self._load_tracks(self._library.get_tracks_by_album(album, artist))
            else:
                self._load_tracks(self._library.get_tracks_by_album(val))
        elif data.startswith("genre:"):
            genre = data.split(":", 1)[1]
            self._load_tracks(self._library.get_tracks_by_genre(genre))
            self._populate_albums(None)
        elif data.startswith("playlist:"):
            playlist_id = int(data.split(":", 1)[1])
            self._current_playlist_id = playlist_id
            # Don't sort playlist tracks - preserve playlist order
            tracks = self._library.get_playlist_tracks(playlist_id)
            self._model = LibraryTableModel(tracks, self)
            self.table.setModel(self._model)
            self.table.resizeColumnsToContents()
            self.table.setColumnWidth(0, 40)
            self.table.setColumnWidth(1, 240)
            self.table.setColumnWidth(2, 200)
            self.table.setColumnWidth(3, 200)
            self.track_count_label.setText(f"{len(tracks)} tracks")
            if self._current_track_path:
                self._model.set_current_path(self._current_track_path)
            # Enable drag-and-drop for playlist reordering
            self.table.setDragEnabled(True)
            self.table.setAcceptDrops(True)
            self.table.setDropIndicatorShown(True)
            self.table.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
            self.table.setSortingEnabled(False)  # Disable sorting to preserve custom order
            # Connect layoutChanged to save reorder
            self._model.layoutChanged.connect(self._on_playlist_reordered)
        elif data == "playlists_header":
            # Clicking on the header does nothing special, just ensure expanded
            self.sidebar.expandItem(self.playlists_item)
        else:
            self._current_playlist_id = None
            # Disable drag-and-drop when not in playlist mode
            self.table.setDragEnabled(False)
            self.table.setAcceptDrops(False)
            self.table.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop)
            self.table.setSortingEnabled(True)

    def _on_table_double_click(self, index: QtCore.QModelIndex):
        track = self._model.data(index, QtCore.Qt.ItemDataRole.UserRole)
        if track:
            self.trackActivated.emit(track)

    def _show_context_menu(self, pos: QtCore.QPoint):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        
        # Get all selected tracks
        selected_indexes = self.table.selectionModel().selectedRows()
        selected_tracks = []
        for idx in selected_indexes:
            track = self._model.data(idx, QtCore.Qt.ItemDataRole.UserRole)
            if track:
                selected_tracks.append(track)
        
        if not selected_tracks:
            return
        
        # Use first track for single-track operations
        first_track = selected_tracks[0]
        is_multi = len(selected_tracks) > 1

        menu = QtWidgets.QMenu(self)
        
        play_action = menu.addAction("Play")
        play_action.triggered.connect(lambda: self.trackActivated.emit(first_track))
        
        if not is_multi:
            show_action = menu.addAction("Show in Explorer")
            show_action.triggered.connect(lambda: self._show_in_explorer(first_track.path))
        
        menu.addSeparator()
        
        # Add to Playlist submenu
        playlists = self._library.get_playlists()
        if playlists or True:  # Always show submenu
            label = f"Add {len(selected_tracks)} Tracks to Playlist" if is_multi else "Add to Playlist"
            playlist_menu = menu.addMenu(label)
            
            # New Playlist option
            new_playlist_action = playlist_menu.addAction("New Playlist...")
            new_playlist_action.triggered.connect(lambda: self._add_tracks_to_new_playlist(selected_tracks))
            
            if playlists:
                playlist_menu.addSeparator()
                for playlist in playlists:
                    action = playlist_menu.addAction(playlist.name)
                    action.triggered.connect(
                        lambda checked, pl_id=playlist.id, tracks=selected_tracks: self._add_tracks_to_playlist(pl_id, tracks)
                    )
        
        # Remove from Playlist (if viewing a playlist)
        if hasattr(self, '_current_playlist_id') and self._current_playlist_id:
            label = f"Remove {len(selected_tracks)} from Playlist" if is_multi else "Remove from Playlist"
            remove_from_pl_action = menu.addAction(label)
            remove_from_pl_action.triggered.connect(
                lambda: self._remove_tracks_from_playlist(self._current_playlist_id, selected_tracks)
            )
        
        menu.addSeparator()
        
        label = f"Remove {len(selected_tracks)} from Library" if is_multi else "Remove from Library"
        remove_action = menu.addAction(label)
        remove_action.triggered.connect(lambda: self._remove_tracks(selected_tracks))
        
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

    def _show_sidebar_context_menu(self, pos: QtCore.QPoint):
        item = self.sidebar.itemAt(pos)
        if not item:
            return

        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            return

        menu = QtWidgets.QMenu(self.sidebar)

        # Handle artist context menu
        if data.startswith("artist:"):
            artist = data.split(":", 1)[1]
            remove_action = menu.addAction(f"Remove '{artist}'")
            remove_action.triggered.connect(lambda: self._remove_artist(artist))
            menu.exec(self.sidebar.viewport().mapToGlobal(pos))
            return

        # Handle playlists header context menu
        if data == "playlists_header":
            new_action = menu.addAction("New Playlist...")
            new_action.triggered.connect(self._create_new_playlist)
            menu.exec(self.sidebar.viewport().mapToGlobal(pos))
            return

        # Handle individual playlist context menu
        if data.startswith("playlist:"):
            playlist_id = int(data.split(":", 1)[1])
            playlist = self._library.get_playlist(playlist_id)
            if playlist:
                rename_action = menu.addAction(f"Rename '{playlist.name}'")
                rename_action.triggered.connect(lambda: self._rename_playlist(playlist_id, playlist.name))
                
                clear_action = menu.addAction("Clear Playlist")
                clear_action.triggered.connect(lambda: self._clear_playlist(playlist_id, playlist.name))
                
                menu.addSeparator()
                
                delete_action = menu.addAction(f"Delete '{playlist.name}'")
                delete_action.triggered.connect(lambda: self._delete_playlist(playlist_id, playlist.name))
                
            menu.exec(self.sidebar.viewport().mapToGlobal(pos))

    def _remove_artist(self, artist: str):
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Remove Artist",
            f"Are you sure you want to remove '{artist}' from the library?\nAll tracks by this artist will be removed from the database.\nFiles on disk will NOT be deleted.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )

        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            self._library.remove_tracks_by_artist(artist)
            self.refresh()

    # -------------------------------------------------------------------------
    # Playlist Management
    # -------------------------------------------------------------------------

    def _create_new_playlist(self):
        """Create a new playlist via dialog."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Playlist", "Playlist name:"
        )
        if ok and name.strip():
            try:
                self._library.create_playlist(name.strip())
                self._populate_playlists()
                self.sidebar.expandItem(self.playlists_item)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Could not create playlist: {e}"
                )

    def _rename_playlist(self, playlist_id: int, current_name: str):
        """Rename a playlist via dialog."""
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Playlist", "New name:", text=current_name
        )
        if ok and new_name.strip() and new_name.strip() != current_name:
            try:
                self._library.rename_playlist(playlist_id, new_name.strip())
                self._populate_playlists()
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Could not rename playlist: {e}"
                )

    def _delete_playlist(self, playlist_id: int, name: str):
        """Delete a playlist after confirmation."""
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete Playlist",
            f"Are you sure you want to delete the playlist '{name}'?\nTracks will remain in the library.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            self._library.delete_playlist(playlist_id)
            self._populate_playlists()
            # If we were viewing this playlist, reload all tracks
            if hasattr(self, '_current_playlist_id') and self._current_playlist_id == playlist_id:
                self._current_playlist_id = None
                self._load_tracks(self._library.get_all_tracks())

    def _clear_playlist(self, playlist_id: int, name: str):
        """Clear all tracks from a playlist after confirmation."""
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Clear Playlist",
            f"Are you sure you want to remove all tracks from '{name}'?\nThe playlist itself will remain.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            self._library.clear_playlist(playlist_id)
            # If viewing this playlist, refresh the view
            if hasattr(self, '_current_playlist_id') and self._current_playlist_id == playlist_id:
                tracks = self._library.get_playlist_tracks(playlist_id)
                self._model.update_tracks(tracks)
                self.track_count_label.setText(f"{len(tracks)} tracks")

    def _add_track_to_playlist(self, playlist_id: int, track: LibraryTrack):
        """Add a track to an existing playlist."""
        if track and track.id:
            self._library.add_to_playlist(playlist_id, track.id)

    def _add_track_to_new_playlist(self, track: LibraryTrack):
        """Create a new playlist and add a track to it."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Playlist", "Playlist name:"
        )
        if ok and name.strip() and track and track.id:
            try:
                playlist = self._library.create_playlist(name.strip())
                self._library.add_to_playlist(playlist.id, track.id)
                self._populate_playlists()
                self.sidebar.expandItem(self.playlists_item)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Could not create playlist: {e}"
                )

    def _remove_track_from_playlist(self, playlist_id: int, track: LibraryTrack):
        """Remove a track from the current playlist."""
        if track and track.id:
            self._library.remove_from_playlist(playlist_id, track.id)
            # Refresh the playlist view
            tracks = self._library.get_playlist_tracks(playlist_id)
            self._model.update_tracks(tracks)
            self.track_count_label.setText(f"{len(tracks)} tracks")

    # -------------------------------------------------------------------------
    # Multi-track operations
    # -------------------------------------------------------------------------

    def _add_tracks_to_playlist(self, playlist_id: int, tracks: List[LibraryTrack]):
        """Add multiple tracks to an existing playlist."""
        for track in tracks:
            if track and track.id:
                self._library.add_to_playlist(playlist_id, track.id)

    def _add_tracks_to_new_playlist(self, tracks: List[LibraryTrack]):
        """Create a new playlist and add multiple tracks to it."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Playlist", "Playlist name:"
        )
        if ok and name.strip():
            try:
                playlist = self._library.create_playlist(name.strip())
                for track in tracks:
                    if track and track.id:
                        self._library.add_to_playlist(playlist.id, track.id)
                self._populate_playlists()
                self.sidebar.expandItem(self.playlists_item)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Could not create playlist: {e}"
                )

    def _remove_tracks_from_playlist(self, playlist_id: int, tracks: List[LibraryTrack]):
        """Remove multiple tracks from the current playlist."""
        for track in tracks:
            if track and track.id:
                self._library.remove_from_playlist(playlist_id, track.id)
        # Refresh the playlist view
        remaining_tracks = self._library.get_playlist_tracks(playlist_id)
        self._model.update_tracks(remaining_tracks)
        self.track_count_label.setText(f"{len(remaining_tracks)} tracks")

    def _remove_tracks(self, tracks: List[LibraryTrack]):
        """Remove multiple tracks from the library."""
        count = len(tracks)
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Remove Tracks",
            f"Are you sure you want to remove {count} track(s) from the library?\nThe files will NOT be deleted.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            for track in tracks:
                if track and track.id:
                    self._library.remove_track(track.id)
            self.refresh()

    def _on_playlist_reordered(self):
        """Save the new track order after drag-and-drop reordering."""
        if not hasattr(self, '_current_playlist_id') or not self._current_playlist_id:
            return
        if not hasattr(self, '_model') or not self._model:
            return
        
        # Get the new order of track IDs from the model
        track_ids = self._model.get_track_ids_in_order()
        if track_ids:
            self._library.db.reorder_playlist_tracks(self._current_playlist_id, track_ids)
