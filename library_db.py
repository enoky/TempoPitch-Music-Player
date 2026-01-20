"""
SQLite-based media library database layer.

Provides persistent storage for track metadata with support for
search, filtering, and browsing by artist/album.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional


@dataclass
class LibraryTrack:
    """
    Represents a track stored in the media library database.
    """

    id: Optional[int]
    path: str
    title: str
    artist: str
    album: str
    genre: str
    year: Optional[int]
    track_number: Optional[int]
    duration_sec: float
    file_size: int
    date_added: datetime
    last_played: Optional[datetime]
    play_count: int
    cover_art_hash: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "year": self.year,
            "track_number": self.track_number,
            "duration_sec": self.duration_sec,
            "file_size": self.file_size,
            "date_added": self.date_added.isoformat() if self.date_added else None,
            "last_played": self.last_played.isoformat() if self.last_played else None,
            "play_count": self.play_count,
            "cover_art_hash": self.cover_art_hash,
        }


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO format datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _row_to_track(row: sqlite3.Row) -> LibraryTrack:
    """Convert a database row to a LibraryTrack object."""
    return LibraryTrack(
        id=row["id"],
        path=row["path"],
        title=row["title"] or "",
        artist=row["artist"] or "",
        album=row["album"] or "",
        genre=row["genre"] or "",
        year=row["year"],
        track_number=row["track_number"],
        duration_sec=row["duration_sec"] or 0.0,
        file_size=row["file_size"] or 0,
        date_added=_parse_datetime(row["date_added"]) or datetime.now(),
        last_played=_parse_datetime(row["last_played"]),
        play_count=row["play_count"] or 0,
        cover_art_hash=row["cover_art_hash"],
    )


class LibraryDatabase:
    """
    SQLite database for media library storage.

    Thread-safe with connection pooling per thread.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the library database.

        Args:
            db_path: Path to SQLite database file. If None, uses default
                     location in user's app data directory.
        """
        if db_path is None:
            app_data = os.environ.get("APPDATA", os.path.expanduser("~"))
            db_dir = os.path.join(app_data, "TempoPitchPlayer")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "library.db")

        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.connection = conn
        return self._local.connection

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database cursor with auto-commit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
                """
            )
            cur.execute("SELECT version FROM schema_version LIMIT 1")
            row = cur.fetchone()
            current_version = row["version"] if row else 0

            if current_version < 1:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tracks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        path TEXT UNIQUE NOT NULL,
                        title TEXT,
                        artist TEXT,
                        album TEXT,
                        genre TEXT,
                        year INTEGER,
                        track_number INTEGER,
                        duration_sec REAL,
                        file_size INTEGER,
                        date_added TEXT,
                        last_played TEXT,
                        play_count INTEGER DEFAULT 0,
                        cover_art_hash TEXT
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_artist ON tracks(artist)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_album ON tracks(album)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_genre ON tracks(genre)")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_title ON tracks(title COLLATE NOCASE)"
                )

                cur.execute("DELETE FROM schema_version")
                cur.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def add_track(self, track: LibraryTrack) -> int:
        """
        Add a track to the library.

        Returns:
            The ID of the inserted track.

        Raises:
            sqlite3.IntegrityError: If track path already exists.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO tracks (
                    path, title, artist, album, genre, year, track_number,
                    duration_sec, file_size, date_added, last_played,
                    play_count, cover_art_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    track.path,
                    track.title,
                    track.artist,
                    track.album,
                    track.genre,
                    track.year,
                    track.track_number,
                    track.duration_sec,
                    track.file_size,
                    track.date_added.isoformat() if track.date_added else None,
                    track.last_played.isoformat() if track.last_played else None,
                    track.play_count,
                    track.cover_art_hash,
                ),
            )
            return cur.lastrowid or 0

    def add_or_update_track(self, track: LibraryTrack) -> int:
        """
        Add a track or update if path already exists.

        Returns:
            The ID of the track.
        """
        existing = self.get_track_by_path(track.path)
        if existing:
            track.id = existing.id
            track.date_added = existing.date_added
            track.last_played = existing.last_played
            track.play_count = existing.play_count
            self.update_track(track)
            return existing.id
        return self.add_track(track)

    def update_track(self, track: LibraryTrack) -> None:
        """Update an existing track."""
        if track.id is None:
            raise ValueError("Track must have an ID to update")

        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE tracks SET
                    path = ?, title = ?, artist = ?, album = ?, genre = ?,
                    year = ?, track_number = ?, duration_sec = ?, file_size = ?,
                    date_added = ?, last_played = ?, play_count = ?, cover_art_hash = ?
                WHERE id = ?
                """,
                (
                    track.path,
                    track.title,
                    track.artist,
                    track.album,
                    track.genre,
                    track.year,
                    track.track_number,
                    track.duration_sec,
                    track.file_size,
                    track.date_added.isoformat() if track.date_added else None,
                    track.last_played.isoformat() if track.last_played else None,
                    track.play_count,
                    track.cover_art_hash,
                    track.id,
                ),
            )

    def remove_track(self, track_id: int) -> None:
        """Remove a track from the library by ID."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM tracks WHERE id = ?", (track_id,))

    def remove_track_by_path(self, path: str) -> None:
        """Remove a track from the library by path."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM tracks WHERE path = ?", (path,))

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get_track_by_id(self, track_id: int) -> Optional[LibraryTrack]:
        """Get a track by its ID."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
            row = cur.fetchone()
            return _row_to_track(row) if row else None

    def get_track_by_path(self, path: str) -> Optional[LibraryTrack]:
        """Get a track by its file path."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM tracks WHERE path = ?", (path,))
            row = cur.fetchone()
            return _row_to_track(row) if row else None

    def get_all_tracks(self, order_by: str = "title") -> List[LibraryTrack]:
        """
        Get all tracks in the library.

        Args:
            order_by: Column to sort by (title, artist, album, date_added, etc.)
        """
        valid_columns = {
            "title",
            "artist",
            "album",
            "genre",
            "year",
            "duration_sec",
            "date_added",
            "play_count",
        }
        if order_by not in valid_columns:
            order_by = "title"

        with self._cursor() as cur:
            cur.execute(f"SELECT * FROM tracks ORDER BY {order_by} COLLATE NOCASE")
            return [_row_to_track(row) for row in cur.fetchall()]

    def get_tracks_by_artist(self, artist: str) -> List[LibraryTrack]:
        """Get all tracks by a specific artist."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM tracks WHERE artist = ? ORDER BY album, track_number, title",
                (artist,),
            )
            return [_row_to_track(row) for row in cur.fetchall()]

    def get_tracks_by_album(self, album: str, artist: str = "") -> List[LibraryTrack]:
        """Get all tracks from a specific album."""
        with self._cursor() as cur:
            if artist:
                cur.execute(
                    "SELECT * FROM tracks WHERE album = ? AND artist = ? ORDER BY track_number, title",
                    (album, artist),
                )
            else:
                cur.execute(
                    "SELECT * FROM tracks WHERE album = ? ORDER BY track_number, title",
                    (album,),
                )
            return [_row_to_track(row) for row in cur.fetchall()]

    def search_tracks(self, query: str) -> List[LibraryTrack]:
        """
        Search tracks by title, artist, or album.

        Args:
            query: Search query (matched with LIKE %query%)
        """
        if not query.strip():
            return self.get_all_tracks()

        pattern = f"%{query}%"
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM tracks
                WHERE title LIKE ? OR artist LIKE ? OR album LIKE ?
                ORDER BY title COLLATE NOCASE
                """,
                (pattern, pattern, pattern),
            )
            return [_row_to_track(row) for row in cur.fetchall()]

    def get_all_artists(self) -> List[str]:
        """Get list of all unique artists."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT artist FROM tracks
                WHERE artist IS NOT NULL AND artist != ''
                ORDER BY artist COLLATE NOCASE
                """
            )
            return [row["artist"] for row in cur.fetchall()]

    def get_all_albums(self) -> List[tuple[str, str]]:
        """Get list of all unique albums as (album, artist) tuples."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT album, artist FROM tracks
                WHERE album IS NOT NULL AND album != ''
                ORDER BY album COLLATE NOCASE
                """
            )
            return [(row["album"], row["artist"]) for row in cur.fetchall()]

    def get_albums_by_artist(self, artist: str) -> List[str]:
        """Get all albums by a specific artist."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT album FROM tracks
                WHERE artist = ? AND album IS NOT NULL AND album != ''
                ORDER BY album COLLATE NOCASE
                """,
                (artist,),
            )
            return [row["album"] for row in cur.fetchall()]

    # -------------------------------------------------------------------------
    # Playback Tracking
    # -------------------------------------------------------------------------

    def increment_play_count(self, track_id: int) -> None:
        """Increment play count and update last_played timestamp."""
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE tracks
                SET play_count = play_count + 1, last_played = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(), track_id),
            )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_track_count(self) -> int:
        """Get total number of tracks in library."""
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM tracks")
            row = cur.fetchone()
            return row["cnt"] if row else 0

    def get_total_duration(self) -> float:
        """Get total duration of all tracks in seconds."""
        with self._cursor() as cur:
            cur.execute("SELECT SUM(duration_sec) as total FROM tracks")
            row = cur.fetchone()
            return row["total"] or 0.0

    def get_library_stats(self) -> dict:
        """Get library statistics."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as track_count,
                    COUNT(DISTINCT artist) as artist_count,
                    COUNT(DISTINCT album) as album_count,
                    SUM(duration_sec) as total_duration,
                    SUM(file_size) as total_size
                FROM tracks
                """
            )
            row = cur.fetchone()
            return {
                "track_count": row["track_count"] or 0,
                "artist_count": row["artist_count"] or 0,
                "album_count": row["album_count"] or 0,
                "total_duration": row["total_duration"] or 0.0,
                "total_size": row["total_size"] or 0,
            }

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def get_missing_files(self) -> List[LibraryTrack]:
        """Get tracks whose files no longer exist on disk."""
        all_tracks = self.get_all_tracks()
        return [t for t in all_tracks if not os.path.exists(t.path)]

    def remove_missing_files(self) -> int:
        """Remove tracks whose files no longer exist. Returns count removed."""
        missing = self.get_missing_files()
        for track in missing:
            if track.id is not None:
                self.remove_track(track.id)
        return len(missing)

    def clear_library(self) -> None:
        """Remove all tracks from the library."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM tracks")

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
