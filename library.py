"""
Media library service for folder scanning and library synchronization.

Provides high-level operations for managing the media library,
including scanning folders for audio files and syncing with the database.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Callable, Iterator, List, Optional, Set

from library_db import LibraryDatabase, LibraryTrack

# Supported media file extensions
MEDIA_EXTENSIONS: Set[str] = {
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
    ".wma",
    ".opus",
    ".aiff",
}


def _is_media_file(path: str) -> bool:
    """Check if a file path is a supported media file."""
    ext = os.path.splitext(path)[1].lower()
    return ext in MEDIA_EXTENSIONS


def _compute_cover_hash(cover_data: Optional[bytes]) -> Optional[str]:
    """Compute a hash for cover art data."""
    if not cover_data:
        return None
    return hashlib.md5(cover_data).hexdigest()


class LibraryService:
    """
    High-level service for managing the media library.

    Provides folder scanning, library synchronization, and
    integration with metadata extraction.
    """

    def __init__(self, db: Optional[LibraryDatabase] = None):
        """
        Initialize the library service.

        Args:
            db: LibraryDatabase instance. If None, creates a new one.
        """
        self._db = db or LibraryDatabase()
        self._scan_abort = False

    @property
    def db(self) -> LibraryDatabase:
        """Get the database instance."""
        return self._db

    # -------------------------------------------------------------------------
    # Folder Scanning
    # -------------------------------------------------------------------------

    def scan_folder(
        self,
        folder: str,
        *,
        recursive: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        metadata_extractor: Optional[Callable[[str], dict]] = None,
    ) -> int:
        """
        Scan a folder for media files and add them to the library.

        Args:
            folder: Path to the folder to scan.
            recursive: If True, scan subdirectories.
            progress_callback: Optional callback(count, current_path) for progress updates.
            metadata_extractor: Optional function to extract metadata from a file path.
                               Should return a dict with keys: title, artist, album, genre,
                               year, track_number, duration_sec, cover_art.

        Returns:
            Number of tracks added/updated.
        """
        self._scan_abort = False
        count = 0

        for path in self._find_media_files(folder, recursive=recursive):
            if self._scan_abort:
                break

            try:
                track = self._create_track_from_path(path, metadata_extractor)
                self._db.add_or_update_track(track)
                count += 1

                if progress_callback:
                    progress_callback(count, path)

            except Exception:
                # Skip files that can't be processed
                continue

        return count

    def scan_files(
        self,
        paths: List[str],
        *,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        metadata_extractor: Optional[Callable[[str], dict]] = None,
    ) -> int:
        """
        Add specific files to the library.

        Args:
            paths: List of file paths to add.
            progress_callback: Optional callback for progress updates.
            metadata_extractor: Optional metadata extraction function.

        Returns:
            Number of tracks added/updated.
        """
        self._scan_abort = False
        count = 0

        for path in paths:
            if self._scan_abort:
                break

            if not os.path.isfile(path) or not _is_media_file(path):
                continue

            try:
                track = self._create_track_from_path(path, metadata_extractor)
                self._db.add_or_update_track(track)
                count += 1

                if progress_callback:
                    progress_callback(count, path)

            except Exception:
                continue

        return count

    def abort_scan(self) -> None:
        """Abort an ongoing scan operation."""
        self._scan_abort = True

    def _find_media_files(
        self, folder: str, *, recursive: bool = True
    ) -> Iterator[str]:
        """Find all media files in a folder."""
        if not os.path.isdir(folder):
            return

        if recursive:
            for root, _, files in os.walk(folder):
                for filename in sorted(files):
                    filepath = os.path.join(root, filename)
                    if _is_media_file(filepath):
                        yield filepath
        else:
            for entry in sorted(os.listdir(folder)):
                filepath = os.path.join(folder, entry)
                if os.path.isfile(filepath) and _is_media_file(filepath):
                    yield filepath

    def _create_track_from_path(
        self,
        path: str,
        metadata_extractor: Optional[Callable[[str], dict]] = None,
    ) -> LibraryTrack:
        """Create a LibraryTrack from a file path."""
        # Get basic file info
        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        filename = os.path.splitext(os.path.basename(path))[0]

        # Default metadata
        title = filename
        artist = ""
        album = ""
        genre = ""
        year = None
        track_number = None
        duration_sec = 0.0
        cover_art_hash = None

        # Extract metadata if extractor provided
        if metadata_extractor:
            try:
                meta = metadata_extractor(path)
                title = meta.get("title") or filename
                artist = meta.get("artist") or ""
                album = meta.get("album") or ""
                genre = meta.get("genre") or ""
                year = meta.get("year")
                track_number = meta.get("track_number")
                duration_sec = meta.get("duration_sec") or 0.0
                cover_art_hash = _compute_cover_hash(meta.get("cover_art"))
            except Exception:
                pass

        return LibraryTrack(
            id=None,
            path=path,
            title=title,
            artist=artist,
            album=album,
            genre=genre,
            year=year,
            track_number=track_number,
            duration_sec=duration_sec,
            file_size=file_size,
            date_added=datetime.now(),
            last_played=None,
            play_count=0,
            cover_art_hash=cover_art_hash,
        )

    # -------------------------------------------------------------------------
    # Library Synchronization
    # -------------------------------------------------------------------------

    def sync_library(self) -> dict:
        """
        Synchronize the library with the filesystem.

        Removes entries for files that no longer exist.

        Returns:
            Dict with sync statistics: {"removed": int}
        """
        removed = self._db.remove_missing_files()
        return {"removed": removed}

    def add_watched_folder(self, folder: str) -> None:
        """
        Add a folder to the watch list for automatic scanning.

        Note: Actual file watching would require additional implementation
        (e.g., using watchdog library). This is a placeholder for future.
        """
        # TODO: Implement folder watching
        pass

    # -------------------------------------------------------------------------
    # Library Operations
    # -------------------------------------------------------------------------

    def get_all_tracks(self, order_by: str = "title") -> List[LibraryTrack]:
        """Get all tracks ordered by specified column."""
        return self._db.get_all_tracks(order_by=order_by)

    def search(self, query: str) -> List[LibraryTrack]:
        """Search tracks by title, artist, or album."""
        return self._db.search_tracks(query)

    def get_artists(self) -> List[str]:
        """Get all unique artists."""
        return self._db.get_all_artists()

    def get_albums(self) -> List[tuple[str, str]]:
        """Get all albums as (album, artist) tuples."""
        return self._db.get_all_albums()

    def get_tracks_by_artist(self, artist: str) -> List[LibraryTrack]:
        """Get all tracks by an artist."""
        return self._db.get_tracks_by_artist(artist)

    def get_tracks_by_album(self, album: str, artist: str = "") -> List[LibraryTrack]:
        """Get all tracks from an album."""
        return self._db.get_tracks_by_album(album, artist)

    def get_track(self, track_id: int) -> Optional[LibraryTrack]:
        """Get a track by ID."""
        return self._db.get_track_by_id(track_id)

    def get_track_by_path(self, path: str) -> Optional[LibraryTrack]:
        """Get a track by file path."""
        return self._db.get_track_by_path(path)

    def record_play(self, track_id: int) -> None:
        """Record that a track was played."""
        self._db.increment_play_count(track_id)

    def remove_track(self, track_id: int) -> None:
        """Remove a track from the library."""
        self._db.remove_track(track_id)

    def get_stats(self) -> dict:
        """Get library statistics."""
        return self._db.get_library_stats()

    def clear_library(self) -> None:
        """Clear all tracks from the library."""
        self._db.clear_library()

    def close(self) -> None:
        """Close the library service and database connection."""
        self._db.close()
