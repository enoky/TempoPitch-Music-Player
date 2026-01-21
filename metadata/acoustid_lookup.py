"""
Acoustic fingerprinting via AcoustID and MusicBrainz.

This module provides robust song identification by analyzing the audio content
rather than relying on filename or embedded tags.

Requirements:
- fpcalc binary (Chromaprint) must be available in PATH or project root
- AcoustID API key (registered at https://acoustid.org/new-application)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# MusicBrainz for full metadata
try:
    import musicbrainzngs
    musicbrainzngs.set_useragent("TempoPitch-Music-Player", "1.0", "tempopitch@example.com")
    HAS_MUSICBRAINZ = True
except ImportError:
    HAS_MUSICBRAINZ = False

# AcoustID API configuration
ACOUSTID_API_URL = "https://api.acoustid.org/v2/lookup"
ACOUSTID_API_KEY = "cSpUJKpD"  # Free API key for TempoPitch
REQUEST_TIMEOUT_SEC = 10

# fpcalc binary search paths
FPCALC_NAMES = ["fpcalc", "fpcalc.exe"]


@dataclass
class AcoustIDResult:
    """Result from AcoustID lookup."""
    recording_id: str  # MusicBrainz Recording ID
    title: str
    artist: str
    album: str
    score: float  # Confidence score (0-1)
    genre: str = ""
    year: Optional[int] = None
    duration_sec: Optional[float] = None


def _find_fpcalc() -> Optional[str]:
    """Find the fpcalc binary."""
    # Check project root first
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in FPCALC_NAMES:
        candidate = os.path.join(project_root, name)
        if os.path.isfile(candidate):
            return candidate
    
    # Check PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for directory in path_dirs:
        for name in FPCALC_NAMES:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
    
    return None


def fingerprint_file(path: str) -> Optional[tuple[str, int]]:
    """
    Generate an audio fingerprint for a file using fpcalc.
    
    Returns:
        Tuple of (fingerprint_string, duration_seconds) or None if failed.
    """
    fpcalc = _find_fpcalc()
    if not fpcalc:
        return None
    
    try:
        result = subprocess.run(
            [fpcalc, "-json", path],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        
        if result.returncode != 0:
            return None
        
        data = json.loads(result.stdout)
        fingerprint = data.get("fingerprint")
        duration = data.get("duration")
        
        if fingerprint and duration:
            return fingerprint, int(duration)
        
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass
    
    return None


def lookup_acoustid(fingerprint: str, duration: int) -> list[AcoustIDResult]:
    """
    Query the AcoustID API with a fingerprint.
    
    Returns:
        List of AcoustIDResult objects sorted by confidence score.
    """
    params = {
        "client": ACOUSTID_API_KEY,
        "duration": str(duration),
        "fingerprint": fingerprint,
        "meta": "recordings releasegroups",  # Request MusicBrainz metadata
    }
    
    url = f"{ACOUSTID_API_URL}?{urlencode(params)}"
    
    try:
        req = Request(url, headers={"User-Agent": "TempoPitch-Music-Player/1.0"})
        with urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    
    if data.get("status") != "ok":
        return []
    
    results: list[AcoustIDResult] = []
    
    for item in data.get("results", []):
        score = float(item.get("score", 0))
        
        for recording in item.get("recordings", []):
            recording_id = recording.get("id", "")
            title = recording.get("title", "")
            
            # Get artist(s)
            artists = recording.get("artists", [])
            artist = ", ".join(a.get("name", "") for a in artists) if artists else ""
            
            # Get album from release groups
            album = ""
            year = None
            release_groups = recording.get("releasegroups", [])
            if release_groups:
                rg = release_groups[0]
                album = rg.get("title", "")
                # Try to get year from release group type
                first_release = rg.get("firstreleasedate", "")
                if first_release:
                    try:
                        year = int(first_release.split("-")[0])
                    except (ValueError, IndexError):
                        pass
            
            if title and artist:
                results.append(AcoustIDResult(
                    recording_id=recording_id,
                    title=title,
                    artist=artist,
                    album=album,
                    score=score,
                    year=year,
                    duration_sec=float(duration),
                ))
    
    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def get_musicbrainz_metadata(recording_id: str) -> Optional[dict]:
    """
    Fetch full metadata from MusicBrainz for a recording ID.
    
    Returns dict with: title, artist, album, genre, year, duration_sec
    """
    if not HAS_MUSICBRAINZ or not recording_id:
        return None
    
    try:
        result = musicbrainzngs.get_recording_by_id(
            recording_id,
            includes=["artists", "releases", "tags"]
        )
        
        recording = result.get("recording", {})
        
        title = recording.get("title", "")
        
        # Artists
        artist_list = recording.get("artist-credit", [])
        artist = "".join(
            a.get("name", "") + a.get("joinphrase", "")
            for a in artist_list
            if isinstance(a, dict)
        ).strip()
        
        # Album from first release
        album = ""
        year = None
        releases = recording.get("release-list", [])
        if releases:
            release = releases[0]
            album = release.get("title", "")
            date = release.get("date", "")
            if date:
                try:
                    year = int(date.split("-")[0])
                except (ValueError, IndexError):
                    pass
        
        # Genre from tags
        genre = ""
        tags = recording.get("tag-list", [])
        if tags:
            # Use the most popular tag as genre
            sorted_tags = sorted(tags, key=lambda t: int(t.get("count", 0)), reverse=True)
            if sorted_tags:
                genre = sorted_tags[0].get("name", "").title()
        
        # Duration
        duration_sec = None
        length_ms = recording.get("length")
        if length_ms:
            try:
                duration_sec = int(length_ms) / 1000.0
            except (ValueError, TypeError):
                pass
        
        return {
            "title": title,
            "artist": artist,
            "album": album,
            "genre": genre,
            "year": year,
            "duration_sec": duration_sec,
        }
        
    except Exception:
        return None


def identify_by_fingerprint(
    path: str,
    *,
    tag_artist: str = "",
    tag_title: str = "",
    tag_album: str = "",
) -> Optional[AcoustIDResult]:
    """
    Main entry point: Identify a song by its audio fingerprint.
    
    Args:
        path: Path to audio file
        tag_artist: Hint from embedded tags (optional)
        tag_title: Hint from embedded tags (optional)
        tag_album: Hint from embedded tags (optional)
    
    Returns the best matching AcoustIDResult, or None if identification failed.
    """
    fp_result = fingerprint_file(path)
    if not fp_result:
        return None
    
    fingerprint, duration = fp_result
    results = lookup_acoustid(fingerprint, duration)
    
    if not results:
        return None
    
    # If we have tag hints, use them to select the best match
    best = _select_best_result(results, tag_artist, tag_title, tag_album)
    
    # If we have MusicBrainz, try to enrich with more metadata
    if HAS_MUSICBRAINZ and best.recording_id and best.score >= 0.8:
        mb_data = get_musicbrainz_metadata(best.recording_id)
        if mb_data:
            best.genre = mb_data.get("genre", "") or best.genre
            best.year = mb_data.get("year") or best.year
            if not best.album and mb_data.get("album"):
                best.album = mb_data["album"]
    
    return best


def _select_best_result(
    results: list[AcoustIDResult],
    tag_artist: str,
    tag_title: str,
    tag_album: str,
) -> AcoustIDResult:
    """
    Select the best AcoustID result based on tag hints.
    
    If tags are provided, prefer results that match them.
    Otherwise, return the highest-scored result.
    """
    from difflib import SequenceMatcher
    
    if not results:
        raise ValueError("No results to select from")
    
    # If no hints, just return the highest scored
    if not tag_artist and not tag_title and not tag_album:
        return results[0]
    
    def normalize(s: str) -> str:
        return s.lower().strip()
    
    def similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, normalize(a), normalize(b)).ratio()
    
    best_result = results[0]
    best_match_score = -1.0
    
    for result in results:
        match_score = 0.0
        
        # Artist match is most important
        if tag_artist:
            artist_sim = similarity(result.artist, tag_artist)
            match_score += artist_sim * 50
        
        # Title match
        if tag_title:
            title_sim = similarity(result.title, tag_title)
            match_score += title_sim * 30
        
        # Album match (bonus)
        if tag_album:
            album_sim = similarity(result.album, tag_album)
            match_score += album_sim * 20
        
        # Factor in the AcoustID confidence score
        match_score += result.score * 10
        
        if match_score > best_match_score:
            best_match_score = match_score
            best_result = result
    
    return best_result


def is_fingerprinting_available() -> bool:
    """Check if fingerprinting is available (fpcalc found)."""
    return _find_fpcalc() is not None
