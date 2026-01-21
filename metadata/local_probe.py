from __future__ import annotations

import json
import os
import subprocess
from typing import Optional, Tuple

from models import Track, TrackMetadata
from utils import have_exe, safe_float

VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
}

def parse_ffprobe_fps(stream: dict) -> float:
    """Parse FPS from ffprobe stream data."""
    fps_str = stream.get("r_frame_rate") or stream.get("avg_frame_rate") or ""
    if "/" in fps_str:
        try:
            num, den = fps_str.split("/")
            return safe_float(num, 0.0) / safe_float(den, 1.0)
        except ValueError:
            pass
    return safe_float(fps_str, 0.0)

def probe_metadata(path: str, fetch_online: bool = True) -> TrackMetadata:
    """
    Probe file metadata using ffprobe.
    """
    duration = 0.0
    artist = ""
    album = ""
    title = ""
    genre = ""
    year: Optional[int] = None
    track_number: Optional[int] = None
    isrc = ""
    cover_art = None
    has_video = False
    video_fps = 0.0
    video_size = (0, 0)
    
    # Check if file has video extension
    _, ext = os.path.splitext(path)
    is_video_ext = ext.lower() in VIDEO_EXTS
    skip_online = (not fetch_online) or is_video_ext

    if have_exe("ffprobe"):
        try:
            # We use a long command to get format tags and stream info
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_entries",
                (
                    "format=duration:format_tags=artist,album,album_artist,title,genre,date,track,isrc:"
                    "stream=index,codec_type,width,height:stream_disposition=attached_pic:"
                    "stream_tags=comment,title,mimetype"
                ),
                path,
            ]
            
            # Add creation_flags=0x08000000 for Windows to avoid popping up console window
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                startupinfo=startupinfo
            )
            
            if p.returncode == 0:
                data = json.loads(p.stdout or "{}")
                fmt = data.get("format", {}) or {}
                tags = fmt.get("tags", {}) or {}
                tags_lower = {str(k).lower(): str(v) for k, v in tags.items()}
                
                artist = tags_lower.get("artist") or tags_lower.get("album_artist") or ""
                album = tags_lower.get("album") or ""
                title = tags_lower.get("title") or ""
                genre = tags_lower.get("genre") or ""
                
                date_str = tags_lower.get("date") or ""
                if date_str:
                    try:
                        # Extract just the year from ISO strings like 2023-05-20 or just 2023
                        year_str = date_str.split("-")[0].strip()
                        if year_str.isdigit():
                            year = int(year_str)
                    except ValueError:
                        pass
                
                track_str = tags_lower.get("track") or ""
                if track_str:
                    try:
                        # Handle "1", "1/12", etc.
                        t_part = track_str.split("/")[0].strip()
                        if t_part.isdigit():
                            track_number = int(t_part)
                    except ValueError:
                        pass

                isrc = tags_lower.get("isrc") or ""
                duration = max(0.0, safe_float(str(fmt.get("duration", "0")), 0.0))

                streams = data.get("streams", []) or []
                attached_stream_index: Optional[int] = None

                # NOTE: ffprobe only returns stream disposition/tags if requested as
                # stream_disposition / stream_tags (see -show_entries above).
                for fallback_idx, stream in enumerate(streams):
                    disp = stream.get("disposition", {}) or {}
                    attached = disp.get("attached_pic")
                    
                    if stream.get("codec_type") == "video" and attached not in (1, "1", True):
                        width = int(stream.get("width") or 0)
                        height = int(stream.get("height") or 0)
                        if width > 0 and height > 0:
                            has_video = True
                        if video_size == (0, 0) and width > 0 and height > 0:
                            video_size = (width, height)
                        if video_fps <= 0.0:
                            video_fps = parse_ffprobe_fps(stream)
                            
                    if attached in (1, "1", True) and attached_stream_index is None:
                        # Prefer the real ffmpeg stream index; fall back to list position.
                        idx_val = stream.get("index")
                        attached_stream_index = idx_val if isinstance(idx_val, int) else fallback_idx

                if attached_stream_index is not None and have_exe("ffmpeg"):
                    # Use the absolute stream index (0:<index>) to avoid 'video index' pitfalls.
                    map_arg = f"0:{attached_stream_index}"
                    
                    startupinfo = None
                    if os.name == 'nt':
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                    art = subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            path,
                            "-map",
                            map_arg,
                            "-an",
                            "-frames:v",
                            "1",
                            "-c:v",
                            "png",
                            "-f",
                            "image2pipe",
                            "-",
                        ],
                        capture_output=True,
                        check=False,
                        startupinfo=startupinfo
                    )
                    if art.returncode == 0 and art.stdout:
                        cover_art = art.stdout

        except Exception:
            pass
            
    if not skip_online:
        from .online_fetch import get_online_metadata
        try:
            online = get_online_metadata(
                path,
                tag_artist=artist,
                tag_title=title,
                tag_album=album,
                tag_isrc=isrc,
                tag_duration_sec=duration
            )
            if online:
                # Merge online metadata, preferring online for most fields
                if online.artist: artist = online.artist
                if online.album: album = online.album
                if online.title: title = online.title
                if online.genre: genre = online.genre
                if online.year: year = online.year
                # For duration, we usually trust the file unless it's zero
                if duration <= 0.0 and online.duration_sec:
                    duration = online.duration_sec
                # Prefer online cover art if available
                if online.cover_art:
                    cover_art = online.cover_art
        except Exception:
            pass

    return TrackMetadata(
        duration_sec=duration,
        artist=artist,
        album=album,
        title=title,
        genre=genre,
        year=year,
        track_number=track_number,
        cover_art=cover_art,
        has_video=has_video,
        video_fps=video_fps,
        video_size=video_size,
    )

def build_track(path: str) -> Track:
    meta = probe_metadata(path)
    title = meta.title or os.path.basename(path)
    return Track(
        path=path,
        title=title,
        duration_sec=meta.duration_sec,
        artist=meta.artist,
        album=meta.album,
        title_display=title,
        cover_art=meta.cover_art,
        has_video=meta.has_video,
        video_fps=meta.video_fps,
        video_size=meta.video_size,
    )

