from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "TempoPitch-Music-Player/1.0 (local)"
MB_API_URL = "https://musicbrainz.org/ws/2/recording/"
CAA_API_URL = "https://coverartarchive.org/release/"
CAA_RG_API_URL = "https://coverartarchive.org/release-group/"
REQUEST_TIMEOUT_SEC = 7
MB_MIN_INTERVAL_SEC = 1.1
CACHE_SCHEMA_VERSION = 3
CACHE_FULL_RESPONSES = False

CACHE_DIR = os.path.join(os.path.dirname(__file__), "metadata")

_mb_lock = threading.Lock()
_last_mb_request = 0.0


@dataclass
class OnlineMetadataResult:
    artist: str
    album: str
    title: str
    duration_sec: Optional[float]
    cover_art: Optional[bytes]


def get_online_metadata(
    path: str,
    *,
    tag_artist: str = "",
    tag_title: str = "",
    tag_album: str = "",
) -> Optional[OnlineMetadataResult]:
    cache_entry = _load_cache_entry(path)
    if cache_entry:
        if cache_entry.get("status") != "ok":
            cached_version = int(cache_entry.get("schema_version") or 1)
            if cached_version >= CACHE_SCHEMA_VERSION:
                return None
        else:
            cached = _result_from_cache(cache_entry)
            if cached and cached.cover_art:
                return cached
            if _cover_art_not_found(cache_entry):
                return cached
            return _try_fetch_cover_for_cache(path, cache_entry, cached)

    query_info = _build_query_from_tags_and_filename(
        path,
        tag_artist=tag_artist,
        tag_title=tag_title,
        tag_album=tag_album,
    )
    queries = query_info.get("queries") or []
    if not queries:
        return None

    mb_data = None
    recording = None
    score = 0
    query_used = ""
    for query in queries:
        try:
            data = _fetch_musicbrainz(query)
        except Exception:
            continue
        recording, score = _select_recording(data)
        if recording:
            mb_data = data
            query_used = query
            break

    recording, score = _select_recording(mb_data)
    if not recording:
        cache_payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "not_found",
            "cached_at": _utc_timestamp(),
            "query": query_info,
        }
        if CACHE_FULL_RESPONSES:
            cache_payload["musicbrainz"] = mb_data or {}
        _write_cache_entry(path, cache_payload)
        return None
    query_info["query_used"] = query_used

    recording_id = str(recording.get("id") or "").strip()
    recording_detail = None
    if recording_id:
        try:
            recording_detail = _fetch_recording_details(recording_id)
        except Exception:
            recording_detail = None

    recording_use = recording_detail or recording

    artist = _artist_from_credit(recording_use.get("artist-credit"))
    title = str(recording_use.get("title") or "").strip()
    duration_sec = _length_ms_to_sec(recording_use.get("length"))

    release = _select_release(recording_use.get("releases"))
    album = str(release.get("title") or "").strip() if release else ""
    release_id = str(release.get("id") or "").strip() if release else ""
    release_group_id = _release_group_id_from_release(release)

    cover_bytes = None
    cover_info = None
    cover_art_archive = None
    cover_source = ""
    if release_id or release_group_id:
        cover_bytes, cover_info, cover_art_archive, cover_source = _fetch_cover_for_release_ids(
            release_id,
            release_group_id,
        )

    cover_entry = None
    cover_filename = None
    if cover_bytes:
        cache_key = _cache_key(path)
        cover_filename = _build_cover_filename(cache_key, cover_info)
        cover_path = os.path.join(CACHE_DIR, cover_filename)
        _ensure_cache_dir()
        with open(cover_path, "wb") as handle:
            handle.write(cover_bytes)
        cover_entry = {
            "release_id": release_id,
            "release_group_id": release_group_id,
            "source": cover_source,
            "url": cover_info.get("url") if cover_info else "",
            "content_type": cover_info.get("content_type") if cover_info else "",
            "file": cover_filename,
        }
    elif release_id or release_group_id:
        cover_entry = _build_cover_not_found_entry(
            release_id=release_id,
            release_group_id=release_group_id,
            source=cover_source,
        )

    cache_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "ok",
        "cached_at": _utc_timestamp(),
        "query": query_info,
        "track": {
            "artist": artist,
            "album": album,
            "title": title,
            "duration_sec": duration_sec,
        },
        "selected": {
            "recording_id": recording_id,
            "recording_score": score,
            "release_id": release_id,
            "release_group_id": release_group_id,
        },
        "cover_art": cover_entry,
        "tags": {
            "artist": tag_artist,
            "album": tag_album,
            "title": tag_title,
        },
    }
    if CACHE_FULL_RESPONSES:
        cache_payload["musicbrainz"] = mb_data
        cache_payload["cover_art_archive"] = cover_art_archive
        cache_payload["selected"]["recording_search"] = recording
        cache_payload["selected"]["recording_lookup"] = recording_detail
        cache_payload["selected"]["release"] = release
    _write_cache_entry(path, cache_payload)

    return OnlineMetadataResult(
        artist=artist,
        album=album,
        title=title,
        duration_sec=duration_sec,
        cover_art=cover_bytes,
    )


def _build_query_from_tags_and_filename(
    path: str,
    *,
    tag_artist: str = "",
    tag_title: str = "",
    tag_album: str = "",
) -> dict:
    base_name = os.path.splitext(os.path.basename(path))[0]
    cleaned = _clean_filename(base_name)
    filename_artist, filename_title = _split_artist_title(cleaned)
    folder_album = _album_from_path(path)

    artist = tag_artist.strip()
    title = tag_title.strip()
    album = tag_album.strip()

    if not artist and filename_artist:
        artist = filename_artist
    if not title and filename_title:
        title = filename_title
    if not title:
        title = cleaned
    if _looks_like_track_number(title):
        title = ""
    if not album and folder_album:
        album = folder_album

    queries = _build_query_variants(title=title, artist=artist, album=album)

    return {
        "query": queries[0] if queries else "",
        "queries": queries,
        "filename": base_name,
        "cleaned": cleaned,
        "tag_artist": tag_artist,
        "tag_title": tag_title,
        "tag_album": tag_album,
        "filename_artist": filename_artist,
        "filename_title": filename_title,
        "folder_album": folder_album,
        "artist": artist,
        "album": album,
        "title": title,
    }


def _clean_filename(name: str) -> str:
    name = name.replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^\s*\d{1,3}\s*[-._]\s*", "", name)
    name = re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", name).strip()
    return name


def _strip_parenthetical(value: str) -> str:
    return re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", value).strip()


def _album_from_path(path: str) -> str:
    parent = os.path.basename(os.path.normpath(os.path.dirname(path)))
    if not parent or parent.endswith(":"):
        return ""
    cleaned = _clean_filename(parent)
    if not cleaned:
        return ""
    if cleaned.lower() in {"music", "audio", "tracks", "songs", "albums"}:
        return ""
    return cleaned


def _split_artist_title(name: str) -> tuple[str, str]:
    for sep in (" - ", " -- ", " -"):
        if sep in name:
            left, right = name.split(sep, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right
    return "", ""


def _escape_query(value: str) -> str:
    return value.replace('"', '\\"')


def _looks_like_track_number(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", value.strip()))


def _build_query_variants(*, title: str, artist: str, album: str) -> list[str]:
    if not title:
        return []
    queries: list[str] = []

    def add_query(a: str, r: str) -> None:
        clauses = [f'recording:"{_escape_query(title)}"']
        if a:
            clauses.append(f'artist:"{_escape_query(a)}"')
        if r:
            clauses.append(f'release:"{_escape_query(r)}"')
        query = " AND ".join(clauses)
        if query not in queries:
            queries.append(query)

    album_variants: list[str] = []
    if album:
        album_variants.append(album)
        cleaned = _strip_parenthetical(album)
        if cleaned and cleaned not in album_variants:
            album_variants.append(cleaned)

    if artist and album_variants:
        for release in album_variants:
            add_query(artist, release)
    if artist:
        add_query(artist, "")
    if album_variants:
        for release in album_variants:
            add_query("", release)
    add_query("", "")

    return queries


def _fetch_musicbrainz(query: str) -> dict:
    params = {
        "query": query,
        "fmt": "json",
        "limit": 3,
        "inc": "artist-credits",
    }
    url = f"{MB_API_URL}?{urlencode(params)}"
    return _mb_get_json(url)


def _fetch_recording_details(recording_id: str) -> dict:
    params = {
        "fmt": "json",
        "inc": "artist-credits+releases",
    }
    url = f"{MB_API_URL}{recording_id}?{urlencode(params)}"
    return _mb_get_json(url)


def _mb_get_json(url: str) -> dict:
    global _last_mb_request
    with _mb_lock:
        now = time.monotonic()
        wait = MB_MIN_INTERVAL_SEC - (now - _last_mb_request)
        if wait > 0:
            time.sleep(wait)
        data = _http_get_json(url)
        _last_mb_request = time.monotonic()
        return data


def _http_get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def _http_get_bytes(url: str) -> tuple[bytes, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
    return body, content_type or ""


def _select_recording(mb_data: dict) -> tuple[Optional[dict], int]:
    recordings = mb_data.get("recordings") or []
    if not recordings:
        return None, 0

    def score_value(rec: dict) -> int:
        try:
            return int(rec.get("score") or 0)
        except Exception:
            return 0

    best = max(recordings, key=score_value)
    return best, score_value(best)


def _select_release(releases: Optional[list]) -> Optional[dict]:
    if not releases:
        return None
    for release in releases:
        if str(release.get("status") or "").lower() == "official":
            return release
    return releases[0]


def _artist_from_credit(credit: Optional[list]) -> str:
    if not credit:
        return ""
    parts = []
    for item in credit:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or ""
        if not name:
            artist = item.get("artist") or {}
            if isinstance(artist, dict):
                name = artist.get("name") or ""
        join = item.get("joinphrase") or ""
        parts.append(f"{name}{join}")
    return "".join(parts).strip()


def _length_ms_to_sec(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        ms = float(value)
    except Exception:
        return None
    if ms < 0:
        return None
    return ms / 1000.0


def _fetch_cover_art_from_url(url: str) -> tuple[Optional[bytes], Optional[dict], Optional[dict]]:
    data = _http_get_json(url)
    images = data.get("images") or []
    if not images:
        return None, None, data

    image = None
    for item in images:
        if item.get("front"):
            image = item
            break
    if image is None:
        image = images[0]

    thumbnails = image.get("thumbnails") or {}
    image_url = thumbnails.get("250") or thumbnails.get("small") or image.get("image") or ""
    if not image_url:
        return None, None, data

    body, content_type = _http_get_bytes(image_url)
    info = {
        "url": image_url,
        "content_type": content_type,
    }
    return body, info, data


def _fetch_cover_art(release_id: str) -> tuple[Optional[bytes], Optional[dict], Optional[dict]]:
    url = f"{CAA_API_URL}{release_id}"
    return _fetch_cover_art_from_url(url)


def _fetch_cover_art_release_group(release_group_id: str) -> tuple[Optional[bytes], Optional[dict], Optional[dict]]:
    url = f"{CAA_RG_API_URL}{release_group_id}"
    return _fetch_cover_art_from_url(url)


def _fetch_cover_for_release_ids(
    release_id: str,
    release_group_id: str,
) -> tuple[Optional[bytes], Optional[dict], Optional[dict], str]:
    if release_id:
        try:
            cover_bytes, cover_info, cover_art_archive = _fetch_cover_art(release_id)
        except Exception:
            cover_bytes = None
            cover_info = None
            cover_art_archive = None
        if cover_bytes:
            return cover_bytes, cover_info, cover_art_archive, "release"
    if release_group_id:
        try:
            cover_bytes, cover_info, cover_art_archive = _fetch_cover_art_release_group(release_group_id)
        except Exception:
            cover_bytes = None
            cover_info = None
            cover_art_archive = None
        if cover_bytes:
            return cover_bytes, cover_info, cover_art_archive, "release_group"
    return None, None, None, ""


def _build_cover_filename(cache_key: str, cover_info: Optional[dict]) -> str:
    content_type = ""
    url = ""
    if cover_info:
        content_type = cover_info.get("content_type") or ""
        url = cover_info.get("url") or ""
    ext = _guess_image_ext(content_type, url)
    return f"{cache_key}{ext}"


def _guess_image_ext(content_type: str, url: str) -> str:
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    if content_type in mapping:
        return mapping[content_type]
    parsed = urlparse(url or "")
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext in {".jpeg", ".jpg", ".png", ".webp", ".gif"}:
        return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def _load_cache_entry(path: str) -> Optional[dict]:
    json_path = _cache_json_path(path)
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _write_cache_entry(path: str, payload: dict) -> None:
    _ensure_cache_dir()
    json_path = _cache_json_path(path)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _try_fetch_cover_for_cache(
    path: str,
    entry: dict,
    cached: Optional[OnlineMetadataResult],
) -> Optional[OnlineMetadataResult]:
    if _cover_art_not_found(entry):
        return cached
    release_id = _release_id_from_cache(entry)
    release_group_id = _release_group_id_from_cache(entry)
    if not release_id or not release_group_id:
        recording_id = _recording_id_from_cache(entry)
        if recording_id:
            try:
                detail = _fetch_recording_details(recording_id)
                release = _select_release(detail.get("releases"))
                if release:
                    if not release_id:
                        release_id = str(release.get("id") or "").strip()
                    if not release_group_id:
                        release_group_id = _release_group_id_from_release(release)
                    selected = entry.get("selected") or {}
                    selected["recording_id"] = recording_id
                    selected["release_id"] = release_id
                    selected["release_group_id"] = release_group_id
                    if CACHE_FULL_RESPONSES:
                        if "recording_lookup" not in selected:
                            selected["recording_lookup"] = detail
                        if "recording_search" not in selected and "recording" in selected:
                            selected["recording_search"] = selected.get("recording")
                        selected["release"] = release
                    entry["selected"] = selected
            except Exception:
                pass
    if not release_id and not release_group_id:
        return cached
    cover_bytes, cover_info, cover_art_archive, cover_source = _fetch_cover_for_release_ids(
        release_id,
        release_group_id,
    )
    if not cover_bytes:
        if release_id or release_group_id:
            entry["cover_art"] = _build_cover_not_found_entry(
                release_id=release_id,
                release_group_id=release_group_id,
                source=cover_source,
            )
            entry["cached_at"] = _utc_timestamp()
            _write_cache_entry(path, entry)
        return cached

    cache_key = _cache_key(path)
    cover_filename = _build_cover_filename(cache_key, cover_info)
    cover_path = os.path.join(CACHE_DIR, cover_filename)
    _ensure_cache_dir()
    with open(cover_path, "wb") as handle:
        handle.write(cover_bytes)

    entry["cover_art"] = {
        "release_id": release_id,
        "release_group_id": release_group_id,
        "source": cover_source,
        "url": cover_info.get("url") if cover_info else "",
        "content_type": cover_info.get("content_type") if cover_info else "",
        "file": cover_filename,
    }
    if cover_art_archive is not None:
        entry["cover_art_archive"] = cover_art_archive
    entry["cached_at"] = _utc_timestamp()
    _write_cache_entry(path, entry)

    if cached:
        return OnlineMetadataResult(
            artist=cached.artist,
            album=cached.album,
            title=cached.title,
            duration_sec=cached.duration_sec,
            cover_art=cover_bytes,
        )

    track = entry.get("track") or {}
    duration_sec = track.get("duration_sec")
    if duration_sec is not None:
        try:
            duration_sec = float(duration_sec)
        except Exception:
            duration_sec = None

    return OnlineMetadataResult(
        artist=str(track.get("artist") or "").strip(),
        album=str(track.get("album") or "").strip(),
        title=str(track.get("title") or "").strip(),
        duration_sec=duration_sec,
        cover_art=cover_bytes,
    )


def _result_from_cache(entry: dict) -> Optional[OnlineMetadataResult]:
    track = entry.get("track") or {}
    artist = str(track.get("artist") or "").strip()
    album = str(track.get("album") or "").strip()
    title = str(track.get("title") or "").strip()
    duration_sec = track.get("duration_sec")
    if duration_sec is not None:
        try:
            duration_sec = float(duration_sec)
        except Exception:
            duration_sec = None

    cover_art = None
    cover_entry = entry.get("cover_art") or {}
    cover_file = cover_entry.get("file") or ""
    if cover_file:
        cover_path = os.path.join(CACHE_DIR, cover_file)
        if os.path.exists(cover_path):
            try:
                with open(cover_path, "rb") as handle:
                    cover_art = handle.read()
            except Exception:
                cover_art = None

    return OnlineMetadataResult(
        artist=artist,
        album=album,
        title=title,
        duration_sec=duration_sec,
        cover_art=cover_art,
    )


def _cover_art_not_found(entry: dict) -> bool:
    cover_entry = entry.get("cover_art") or {}
    return str(cover_entry.get("status") or "").lower() == "not_found"


def _build_cover_not_found_entry(
    *,
    release_id: str,
    release_group_id: str,
    source: str,
) -> dict:
    return {
        "status": "not_found",
        "release_id": release_id,
        "release_group_id": release_group_id,
        "source": source,
        "checked_at": _utc_timestamp(),
    }


def _release_id_from_cache(entry: dict) -> str:
    cover_entry = entry.get("cover_art") or {}
    release_id = cover_entry.get("release_id") or ""
    if release_id:
        return str(release_id)
    selected = entry.get("selected") or {}
    release_id = selected.get("release_id") or ""
    if release_id:
        return str(release_id)
    release = selected.get("release") or {}
    release_id = release.get("id") or ""
    return str(release_id) if release_id else ""


def _release_group_id_from_cache(entry: dict) -> str:
    cover_entry = entry.get("cover_art") or {}
    release_group_id = cover_entry.get("release_group_id") or ""
    if release_group_id:
        return str(release_group_id)
    selected = entry.get("selected") or {}
    release_group_id = selected.get("release_group_id") or ""
    if release_group_id:
        return str(release_group_id)
    release = selected.get("release") or {}
    return _release_group_id_from_release(release)


def _recording_id_from_cache(entry: dict) -> str:
    selected = entry.get("selected") or {}
    recording_id = selected.get("recording_id") or ""
    if recording_id:
        return str(recording_id)
    for key in ("recording_lookup", "recording_search", "recording"):
        recording = selected.get(key) or {}
        recording_id = recording.get("id") or ""
        if recording_id:
            return str(recording_id)
    return ""


def _release_group_id_from_release(release: Optional[dict]) -> str:
    if not release:
        return ""
    release_group = release.get("release-group") or {}
    release_group_id = release_group.get("id") or ""
    return str(release_group_id) if release_group_id else ""


def _cache_key(path: str) -> str:
    normalized = os.path.normcase(os.path.abspath(path))
    return hashlib.sha1(normalized.encode("utf-8", "ignore")).hexdigest()


def _cache_json_path(path: str) -> str:
    return os.path.join(CACHE_DIR, f"{_cache_key(path)}.json")


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
