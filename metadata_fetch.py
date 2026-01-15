from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

USER_AGENT = "TempoPitch-Music-Player/1.0 (local)"
MB_API_URL = "https://musicbrainz.org/ws/2/recording/"
CAA_API_URL = "https://coverartarchive.org/release/"
CAA_RG_API_URL = "https://coverartarchive.org/release-group/"
ITUNES_API_URL = "https://itunes.apple.com/search"
REQUEST_TIMEOUT_SEC = 7
MB_MIN_INTERVAL_SEC = 1.1
CACHE_SCHEMA_VERSION = 7
CACHE_FULL_RESPONSES = False
COVER_NOT_FOUND_TTL_SEC = 7 * 24 * 60 * 60

MB_NOT_FOUND_TTL_SEC = 7 * 24 * 60 * 60
CACHE_ERROR_TTL_SEC = 60 * 60
ITUNES_SEARCH_LIMIT = 5
ITUNES_MIN_SCORE = 45
COVER_FILE_EXTENSIONS = (".jpg", ".png", ".webp", ".gif")
QUALIFIER_TOKENS = {
    "live",
    "remaster",
    "remastered",
    "deluxe",
    "expanded",
    "anniversary",
    "karaoke",
    "instrumental",
    "tribute",
    "cover",
    "edit",
    "version",
    "mix",
    "mono",
    "stereo",
    "acoustic",
    "demo",
    "session",
    "radio",
    "clean",
    "explicit",
}

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
        cached_version = int(cache_entry.get("schema_version") or 1)
        status = str(cache_entry.get("status") or "")

        # If our matching logic improves, treat older schemas as a cache miss so we can refresh.
        if cached_version < CACHE_SCHEMA_VERSION:
            cache_entry = None
        elif status != "ok":
            # Negative cache: respect TTLs so we don't permanently miss covers/metadata.
            ttl = MB_NOT_FOUND_TTL_SEC if status == "not_found" else CACHE_ERROR_TTL_SEC
            if _cache_entry_is_fresh(cache_entry, ttl):
                return None
            cache_entry = None

        if cache_entry:
            cached = _result_from_cache(cache_entry)
            if cached and cached.cover_art:
                return cached

            # If MB/Cover Art Archive had no cover, try iTunes once before honoring the TTL.
            if _cover_art_not_found(cache_entry):
                cover_entry = cache_entry.get("cover_art") or {}
                if str(cover_entry.get("source") or "") != "itunes":
                    return _try_fetch_itunes_cover_for_cache(path, cache_entry, cached)
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

    best_data = None
    best_recording = None
    best_score = 0
    best_query = ""
    for query in queries:
        try:
            data = _fetch_musicbrainz(query)
        except Exception:
            continue
        rec, sc = _select_recording(data)
        if rec and sc >= best_score:
            best_data = data
            best_recording = rec
            best_score = sc
            best_query = query
            if sc >= 95:
                break

    mb_data = best_data
    recording = best_recording
    score = best_score
    query_used = best_query


    # Even if MusicBrainz fails to find a recording, we can try iTunes for metadata + cover
    # But current architecture relies on MB for metadata.
    # For now, if MB fails, we cache "not_found".
    if not recording:
        # iTunes-only fallback can still recover cover art when MB doesn't find a match.
        it_artist = str(query_info.get("artist") or "").strip()
        it_title = str(query_info.get("title") or "").strip()
        it_album = str(query_info.get("album") or "").strip()

        itunes_candidates: list[str] = []
        clean_artist = _clean_tag_value(it_artist)
        clean_title = _clean_tag_value(it_title)
        clean_album = _clean_tag_value(it_album)
        if clean_artist and clean_title:
            itunes_candidates.append(f"{clean_artist} {clean_title}")
        if it_artist and it_title:
            candidate = f"{it_artist} {it_title}"
            if candidate not in itunes_candidates:
                itunes_candidates.append(candidate)
        if clean_artist and clean_album:
            candidate = f"{clean_artist} {clean_album}"
            if candidate not in itunes_candidates:
                itunes_candidates.append(candidate)
        if it_artist and it_album:
            candidate = f"{it_artist} {it_album}"
            if candidate not in itunes_candidates:
                itunes_candidates.append(candidate)

        # Title-only queries are risky — only do these when artist is missing.
        if not clean_artist:
            if clean_title and clean_title not in itunes_candidates:
                itunes_candidates.append(clean_title)
            if it_title and it_title not in itunes_candidates:
                itunes_candidates.append(it_title)

        cover_bytes = None
        cover_entry = None
        if itunes_candidates:
            try:
                itunes_bytes, itunes_ext, itunes_cache_key, itunes_item = _fetch_cover_from_itunes(
                    itunes_candidates,
                    expected_artist=it_artist,
                    expected_title=it_title,
                    expected_album=it_album,
                    expected_duration_sec=None,
                )
                if itunes_bytes:
                    cover_bytes = itunes_bytes
                    cover_info = {
                        "url": (itunes_item.get("collectionViewUrl") or itunes_item.get("trackViewUrl") or "itunes_search") if itunes_item else "itunes_search",
                        "content_type": _content_type_for_ext(itunes_ext),
                    }
                    cover_cache_key = itunes_cache_key or _cache_key(path)
                    cover_filename = _build_cover_filename(cover_cache_key, cover_info)
                    cover_path = os.path.join(CACHE_DIR, cover_filename)
                    if not os.path.exists(cover_path):
                        _ensure_cache_dir()
                        with open(cover_path, "wb") as handle:
                            handle.write(cover_bytes)
                    cover_entry = {
                        "release_id": "",
                        "release_group_id": "",
                        "source": "itunes",
                        "url": cover_info.get("url") if cover_info else "",
                        "content_type": cover_info.get("content_type") if cover_info else "",
                        "file": cover_filename,
                        "cache_key": cover_cache_key,
                    }
            except Exception:
                pass

        if cover_bytes and cover_entry:
            cache_payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "ok",
                "cached_at": _utc_timestamp(),
                "query": query_info,
                "track": {
                    "artist": it_artist,
                    "album": it_album,
                    "title": it_title,
                    "duration_sec": None,
                },
                "selected": {
                    "recording_id": "",
                    "recording_score": 0,
                    "release_id": "",
                    "release_group_id": "",
                },
                "cover_art": cover_entry,
                "tags": {
                    "artist": tag_artist,
                    "album": tag_album,
                    "title": tag_title,
                },
            }
            if CACHE_FULL_RESPONSES:
                cache_payload["musicbrainz"] = mb_data or {}
            _write_cache_entry(path, cache_payload)
            return OnlineMetadataResult(
                artist=it_artist,
                album=it_album,
                title=it_title,
                duration_sec=None,
                cover_art=cover_bytes,
            )

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

    expected_album = str(query_info.get("album") or "").strip()
    release_candidates = _rank_releases(
        recording_use.get("releases"),
        expected_album=expected_album,
    )
    release = release_candidates[0] if release_candidates else None
    album = str(release.get("title") or "").strip() if release else ""
    release_id = str(release.get("id") or "").strip() if release else ""
    release_group_id = _release_group_id_from_release(release)

    cover_bytes = None
    cover_info = None
    cover_art_archive = None
    cover_source = ""
    cover_status = "not_found"
    cover_filename = None
    cover_cache_key = ""
    itunes_cache_key = ""

    # 1. Try Cover Art Archive (MusicBrainz) with ranked releases
    if release_candidates:
        (
            cover_bytes,
            cover_info,
            cover_source,
            cover_release,
            cover_cache_key,
            cover_filename,
        ) = _find_existing_cover_for_candidates(release_candidates)
        if cover_bytes and cover_release:
            cover_status = "ok"
            release = cover_release
            album = str(release.get("title") or "").strip()
            release_id = str(release.get("id") or "").strip()
            release_group_id = _release_group_id_from_release(release)

    if release_candidates and not cover_bytes:
        (
            cover_bytes,
            cover_info,
            cover_art_archive,
            cover_source,
            cover_status,
            cover_release,
        ) = _fetch_cover_for_release_candidates(release_candidates)
        if cover_bytes and cover_release:
            release = cover_release
            album = str(release.get("title") or "").strip()
            release_id = str(release.get("id") or "").strip()
            release_group_id = _release_group_id_from_release(release)

    # 2. Fallback: Try iTunes if CAA failed
    if not cover_bytes:
        # Construct multiple search term variants for iTunes to be robust against messy tags
        itunes_candidates = []

        # 1. Cleaned Artist + Cleaned Title (Best chance)
        clean_artist = _clean_tag_value(artist)
        clean_title = _clean_tag_value(title)
        clean_album = _clean_tag_value(album or expected_album)
        if clean_artist and clean_title:
            itunes_candidates.append(f"{clean_artist} {clean_title}")

        # 2. Original Artist + Original Title
        if artist and title:
            candidate = f"{artist} {title}"
            if candidate not in itunes_candidates:
                itunes_candidates.append(candidate)

        # 3. Artist + Album (helps when track titles collide across albums)
        if clean_artist and clean_album:
            candidate = f"{clean_artist} {clean_album}"
            if candidate not in itunes_candidates:
                itunes_candidates.append(candidate)
        if artist and (album or expected_album):
            candidate = f"{artist} {album or expected_album}"
            if candidate not in itunes_candidates:
                itunes_candidates.append(candidate)

        # 4. Title-only queries are risky — only do these when artist is missing.
        if not clean_artist:
            if clean_title and clean_title not in itunes_candidates:
                itunes_candidates.append(clean_title)
            if title and title not in itunes_candidates:
                itunes_candidates.append(title)


        if itunes_candidates:
            try:
                itunes_bytes, itunes_ext, itunes_cache_key, itunes_item = _fetch_cover_from_itunes(
                    itunes_candidates,
                    expected_artist=artist,
                    expected_title=title,
                    expected_album=album or expected_album,
                    expected_duration_sec=duration_sec,
                )
                if itunes_bytes:
                    cover_bytes = itunes_bytes
                    cover_info = {
                        "url": (itunes_item.get("collectionViewUrl") or itunes_item.get("trackViewUrl") or "itunes_search") if itunes_item else "itunes_search",
                        "content_type": _content_type_for_ext(itunes_ext),
                    }
                    cover_source = "itunes"
                    cover_status = "ok"
            except Exception:
                pass

    cover_entry = None
    if cover_bytes:
        if not cover_cache_key:
            if cover_source in {"release", "release_group"}:
                cover_cache_key = _cover_cache_key_for_mb(
                    release_id,
                    release_group_id,
                    cover_source,
                )
            elif cover_source == "itunes":
                cover_cache_key = itunes_cache_key
            if not cover_cache_key:
                cover_cache_key = _cache_key(path)
        if not cover_filename:
            cover_filename = _build_cover_filename(cover_cache_key, cover_info)
        cover_path = os.path.join(CACHE_DIR, cover_filename)
        if not os.path.exists(cover_path):
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
            "cache_key": cover_cache_key,
        }
    elif release_id or release_group_id:
        if cover_status == "not_found":
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


def _clean_tag_value(value: str) -> str:
    # Remove common noise like "ft.", "feat.", "(Remix)", "(Live)", "(... version)"
    value = re.sub(r"(?i)\b(ft\.|feat\.|featuring)\b.*", "", value)
    value = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", value)
    return value.strip()


def _build_query_variants(*, title: str, artist: str, album: str) -> list[str]:
    if not title:
        return []
    queries: list[str] = []

    def add_query(a: str, r: str, t: str) -> None:
        clauses = [f'recording:"{_escape_query(t)}"']
        if a:
            clauses.append(f'artist:"{_escape_query(a)}"')
        if r:
            clauses.append(f'release:"{_escape_query(r)}"')
        query = " AND ".join(clauses)
        if query not in queries:
            queries.append(query)

    # Clean versions of tags
    clean_title = _clean_tag_value(title)
    clean_artist = _clean_tag_value(artist)

    # 1. Strict: Original Artist + Original Title + Original Album
    album_variants: list[str] = []
    if album:
        album_variants.append(album)
        cleaned_album = _strip_parenthetical(album)
        if cleaned_album and cleaned_album not in album_variants:
            album_variants.append(cleaned_album)

    if artist and album_variants:
        for release in album_variants:
            add_query(artist, release, title)
            # Try with clean title if different
            if clean_title and clean_title != title:
                 add_query(artist, release, clean_title)

    # 2. Artist + Title (No Album)
    if artist:
        add_query(artist, "", title)
        if clean_title and clean_title != title:
            add_query(artist, "", clean_title)
        if clean_artist and clean_artist != artist:
             add_query(clean_artist, "", title)
             if clean_title and clean_title != title:
                 add_query(clean_artist, "", clean_title)

    # 3. Album + Title (No Artist)
    if album_variants:
        for release in album_variants:
            add_query("", release, title)
            if clean_title and clean_title != title:
                add_query("", release, clean_title)

    # 4. Title Only
    add_query("", "", title)
    if clean_title and clean_title != title:
        add_query("", "", clean_title)

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
        "inc": "artist-credits+releases+release-groups",
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


def _normalize_match_text(value: str) -> str:
    cleaned = _clean_tag_value(value or "")
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _score_text_match(value: str, expected: str, exact_score: int, contains_score: int) -> int:
    if not value or not expected:
        return 0
    if value == expected:
        return exact_score
    if value in expected or expected in value:
        return contains_score
    return 0


def _string_similarity(value: str, expected: str) -> float:
    if not value or not expected:
        return 0.0
    return SequenceMatcher(None, value, expected).ratio()


def _token_overlap_bonus(value: str, expected: str, max_score: int) -> int:
    if not value or not expected:
        return 0
    value_tokens = set(value.split())
    expected_tokens = set(expected.split())
    if not value_tokens or not expected_tokens:
        return 0
    overlap = len(value_tokens & expected_tokens) / len(expected_tokens)
    return int(overlap * max_score)


def _qualifier_penalty(value: str, expected: str, per_token: int) -> int:
    if not value or not expected:
        return 0
    value_tokens = set(value.split())
    expected_tokens = set(expected.split())
    extra = (QUALIFIER_TOKENS & value_tokens) - expected_tokens
    return len(extra) * per_token


def _parse_release_year(date_value: Optional[object]) -> Optional[int]:
    if not date_value:
        return None
    text = str(date_value)
    if len(text) < 4 or not text[:4].isdigit():
        return None
    try:
        return int(text[:4])
    except Exception:
        return None


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


def _score_release(release: dict, expected_album: str) -> tuple[int, Optional[int]]:
    score = 0

    status = str(release.get("status") or "").lower()
    if status == "official":
        score += 30
    elif status == "bootleg":
        score -= 20

    release_group = release.get("release-group") or {}
    primary_type = str(release_group.get("primary-type") or "").lower()
    if primary_type == "album":
        score += 10
    elif primary_type in {"single", "ep"}:
        score += 5

    expected = _normalize_match_text(expected_album)
    release_title = _normalize_match_text(str(release.get("title") or ""))
    if expected:
        score += _score_text_match(release_title, expected, 40, 20)
        score += int(_string_similarity(release_title, expected) * 15)
        score += _token_overlap_bonus(release_title, expected, 10)
        score -= _qualifier_penalty(release_title, expected, 4)

    release_group_title = _normalize_match_text(str(release_group.get("title") or ""))
    if expected and release_group_title and release_group_title != release_title:
        score += _score_text_match(release_group_title, expected, 20, 10)
        score += int(_string_similarity(release_group_title, expected) * 8)
        score += _token_overlap_bonus(release_group_title, expected, 6)
        score -= _qualifier_penalty(release_group_title, expected, 2)

    release_year = _parse_release_year(release.get("date"))
    return score, release_year


def _rank_releases(releases: Optional[list], *, expected_album: str) -> list[dict]:
    if not releases:
        return []
    scored: list[tuple[int, int, dict]] = []
    for release in releases:
        if not isinstance(release, dict):
            continue
        score, release_year = _score_release(release, expected_album)
        date_rank = -release_year if release_year is not None else -9999999
        scored.append((score, date_rank, release))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]


def _select_release(releases: Optional[list], *, expected_album: str = "") -> Optional[dict]:
    ranked = _rank_releases(releases, expected_album=expected_album)
    return ranked[0] if ranked else None


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


def _fetch_cover_art_from_url(
    url: str,
) -> tuple[Optional[bytes], Optional[dict], Optional[dict], str]:
    try:
        data = _http_get_json(url)
    except HTTPError as exc:
        if exc.code == 404:
            return None, None, None, "not_found"
        return None, None, None, "error"
    except Exception:
        return None, None, None, "error"

    images = data.get("images") or []
    if not images:
        return None, None, data, "not_found"

    image = None
    for item in images:
        if item.get("front"):
            image = item
            break
    if image is None:
        image = images[0]

    candidates = _cover_image_url_candidates(image)
    if not candidates:
        return None, None, data, "not_found"

    saw_404 = False
    for image_url in candidates:
        try:
            body, content_type = _http_get_bytes(image_url)
        except HTTPError as exc:
            if exc.code == 404:
                saw_404 = True
                continue
            return None, None, data, "error"
        except Exception:
            return None, None, data, "error"

        info = {
            "url": image_url,
            "content_type": content_type,
        }
        return body, info, data, "ok"

    return None, None, data, "not_found" if saw_404 else "error"


def _cover_image_url_candidates(image: dict) -> list[str]:
    thumbnails = image.get("thumbnails") or {}
    urls = [
        thumbnails.get("250"),
        thumbnails.get("small"),
        thumbnails.get("500"),
        thumbnails.get("large"),
        image.get("image"),
    ]
    seen = set()
    candidates: list[str] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append(url)
    return candidates


def _fetch_cover_art(
    release_id: str,
) -> tuple[Optional[bytes], Optional[dict], Optional[dict], str]:
    url = f"{CAA_API_URL}{release_id}"
    return _fetch_cover_art_from_url(url)


def _fetch_cover_art_release_group(
    release_group_id: str,
) -> tuple[Optional[bytes], Optional[dict], Optional[dict], str]:
    url = f"{CAA_RG_API_URL}{release_group_id}"
    return _fetch_cover_art_from_url(url)


def _fetch_cover_for_release_candidates(
    releases: list[dict],
) -> tuple[Optional[bytes], Optional[dict], Optional[dict], str, str, Optional[dict]]:
    cover_status = "not_found"
    cover_source = ""

    for release in releases:
        release_id = str(release.get("id") or "").strip()
        if not release_id:
            continue
        try:
            cover_bytes, cover_info, cover_art_archive, status = _fetch_cover_art(release_id)
        except Exception:
            cover_bytes = None
            cover_info = None
            cover_art_archive = None
            status = "error"
        if cover_bytes:
            return cover_bytes, cover_info, cover_art_archive, "release", "ok", release
        cover_source = "release"
        if status == "error":
            cover_status = "error"

    for release in releases:
        release_group_id = _release_group_id_from_release(release)
        if not release_group_id:
            continue
        try:
            cover_bytes, cover_info, cover_art_archive, status = _fetch_cover_art_release_group(release_group_id)
        except Exception:
            cover_bytes = None
            cover_info = None
            cover_art_archive = None
            status = "error"
        if cover_bytes:
            return cover_bytes, cover_info, cover_art_archive, "release_group", "ok", release
        cover_source = "release_group"
        if status == "error":
            cover_status = "error"

    return None, None, None, cover_source, cover_status, None


def _fetch_cover_for_release_ids(
    release_id: str,
    release_group_id: str,
) -> tuple[Optional[bytes], Optional[dict], Optional[dict], str, str]:
    cover_status = "not_found"
    cover_source = ""
    if release_id:
        try:
            cover_bytes, cover_info, cover_art_archive, status = _fetch_cover_art(release_id)
        except Exception:
            cover_bytes = None
            cover_info = None
            cover_art_archive = None
            status = "error"
        if cover_bytes:
            return cover_bytes, cover_info, cover_art_archive, "release", "ok"
        cover_source = "release"
        if status == "error":
            cover_status = "error"
    if release_group_id:
        try:
            cover_bytes, cover_info, cover_art_archive, status = _fetch_cover_art_release_group(release_group_id)
        except Exception:
            cover_bytes = None
            cover_info = None
            cover_art_archive = None
            status = "error"
        if cover_bytes:
            return cover_bytes, cover_info, cover_art_archive, "release_group", "ok"
        cover_source = "release_group"
        if status == "error":
            cover_status = "error"
    return None, None, None, cover_source, cover_status


def _itunes_search(query: str, limit: int, *, entity: str = "song") -> list[dict]:
    params = {
        "term": query,
        "media": "music",
        "entity": entity,
        "limit": str(limit),
    }
    url = f"{ITUNES_API_URL}?{urlencode(params)}"
    try:
        data = _http_get_json(url)
    except Exception:
        return []
    return data.get("results") or []


def _score_itunes_result(
    item: dict,
    *,
    expected_artist: str,
    expected_title: str,
    expected_album: str,
    expected_duration_sec: Optional[float],
) -> int:
    score = 0

    expected_title_norm = _normalize_match_text(expected_title)
    expected_artist_norm = _normalize_match_text(expected_artist)
    expected_album_norm = _normalize_match_text(expected_album)

    track_name = _normalize_match_text(str(item.get("trackName") or ""))
    artist_name = _normalize_match_text(str(item.get("artistName") or ""))
    collection_name = _normalize_match_text(str(item.get("collectionName") or ""))

    wrapper_type = str(item.get("wrapperType") or "").lower()
    collection_type = str(item.get("collectionType") or "").lower()
    is_album_result = wrapper_type == "collection" or collection_type == "album"
    if is_album_result:
        # Album-entity results tend to be more reliable for cover art.
        score += 20

    if expected_title_norm and track_name:
        score += int(_string_similarity(track_name, expected_title_norm) * 60)
        score += _token_overlap_bonus(track_name, expected_title_norm, 12)
        score -= _qualifier_penalty(track_name, expected_title_norm, 4)
        score += _score_text_match(track_name, expected_title_norm, 20, 10)

    if expected_artist_norm and artist_name:
        score += int(_string_similarity(artist_name, expected_artist_norm) * 30)
        score += _token_overlap_bonus(artist_name, expected_artist_norm, 6)
        score += _score_text_match(artist_name, expected_artist_norm, 10, 5)

        try:
            artist_sim = _string_similarity(artist_name, expected_artist_norm)
        except Exception:
            artist_sim = 0.0
        if artist_sim < 0.2:
            score -= 20

    if expected_album_norm and collection_name:
        score += int(_string_similarity(collection_name, expected_album_norm) * 35)
        score += _token_overlap_bonus(collection_name, expected_album_norm, 10)
        score -= _qualifier_penalty(collection_name, expected_album_norm, 3)
        score += _score_text_match(collection_name, expected_album_norm, 12, 6)

        try:
            album_sim = _string_similarity(collection_name, expected_album_norm)
        except Exception:
            album_sim = 0.0
        if album_sim < 0.25:
            score -= 25

    if expected_duration_sec is not None:
        track_ms = item.get("trackTimeMillis")
        if track_ms is not None:
            try:
                diff = abs((float(track_ms) / 1000.0) - float(expected_duration_sec))
            except Exception:
                diff = None
            if diff is not None:
                if diff <= 2.0:
                    score += 15
                elif diff <= 5.0:
                    score += 8
                elif diff <= 10.0:
                    score += 4

    return score


def _select_itunes_result(
    results: list[dict],
    *,
    expected_artist: str,
    expected_title: str,
    expected_album: str,
    expected_duration_sec: Optional[float],
) -> Optional[dict]:
    if not results:
        return None
    best_score = None
    best_item = None
    for item in results:
        if not isinstance(item, dict):
            continue
        score = _score_itunes_result(
            item,
            expected_artist=expected_artist,
            expected_title=expected_title,
            expected_album=expected_album,
            expected_duration_sec=expected_duration_sec,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_item = item
    if best_item is None:
        return None
    if (expected_title or expected_album or expected_artist) and (best_score or 0) < ITUNES_MIN_SCORE:
        return None
    return best_item


def _itunes_artwork_url(item: dict) -> str:
    for key in ("artworkUrl100", "artworkUrl60", "artworkUrl30"):
        url = item.get(key) or ""
        if url:
            return str(url)
    return ""


def _upgrade_itunes_artwork_url(url: str) -> str:
    return (
        url.replace("100x100", "600x600")
        .replace("60x60", "600x600")
        .replace("30x30", "600x600")
    )


def _fetch_cover_from_itunes(
    queries: list[str],
    *,
    expected_artist: str = "",
    expected_title: str = "",
    expected_album: str = "",
    expected_duration_sec: Optional[float] = None,
) -> tuple[Optional[bytes], str, str, Optional[dict]]:
    """Fetch cover art via iTunes Search API.

    Returns (cover_bytes, ext, cache_key, selected_item).
    """
    candidates: dict[str, dict] = {}

    # Track (song) search using the provided query variants.
    for query in queries:
        for item in _itunes_search(query, ITUNES_SEARCH_LIMIT, entity="song"):
            track_id = str(item.get("trackId") or "")
            if not track_id:
                track_id = f"{item.get('artistName')}|{item.get('trackName')}|{item.get('collectionName')}"
            if track_id not in candidates:
                candidates[track_id] = item

    # Album search is often a better signal for the *right cover art*.
    if expected_album:
        album_queries: list[str] = []
        clean_artist = _clean_tag_value(expected_artist)
        clean_album = _clean_tag_value(expected_album)

        if clean_artist and clean_album:
            album_queries.append(f"{clean_artist} {clean_album}")
        if expected_artist and expected_album:
            candidate = f"{expected_artist} {expected_album}"
            if candidate not in album_queries:
                album_queries.append(candidate)

        # If artist is missing, fall back to album-only queries (higher risk).
        if not expected_artist and clean_album:
            if clean_album not in album_queries:
                album_queries.append(clean_album)
            if expected_album not in album_queries:
                album_queries.append(expected_album)

        for query in album_queries:
            for item in _itunes_search(query, ITUNES_SEARCH_LIMIT, entity="album"):
                collection_id = str(item.get("collectionId") or "")
                if not collection_id:
                    collection_id = f"{item.get('artistName')}|{item.get('collectionName')}"
                if collection_id not in candidates:
                    candidates[collection_id] = item

    if not candidates:
        return None, "", "", None

    item = _select_itunes_result(
        list(candidates.values()),
        expected_artist=expected_artist,
        expected_title=expected_title,
        expected_album=expected_album,
        expected_duration_sec=expected_duration_sec,
    )
    if not item:
        return None, "", "", None

    cache_key = _cover_cache_key_for_itunes(item)
    if cache_key:
        cached_bytes, cached_info, cached_filename = _find_existing_cover_file(cache_key)
        if cached_bytes and cached_filename:
            ext = os.path.splitext(cached_filename)[1].lower() or ".jpg"
            return cached_bytes, ext, cache_key, item

    artwork_url = _itunes_artwork_url(item)
    if not artwork_url:
        return None, "", cache_key, item

    artwork_url = _upgrade_itunes_artwork_url(artwork_url)

    try:
        body, content_type = _http_get_bytes(artwork_url)
        ext = _guess_image_ext(content_type, artwork_url)
        return body, ext, cache_key, item
    except Exception:
        return None, "", cache_key, item



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


def _content_type_for_ext(ext: str) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mapping.get(ext.lower(), "image/jpeg")


def _cover_cache_key_for_mb(release_id: str, release_group_id: str, source: str) -> str:
    if source == "release" and release_id:
        return f"mb_release_{release_id}"
    if source == "release_group" and release_group_id:
        return f"mb_release_group_{release_group_id}"
    if release_id:
        return f"mb_release_{release_id}"
    if release_group_id:
        return f"mb_release_group_{release_group_id}"
    return ""


def _cover_cache_key_for_itunes(item: dict) -> str:
    collection_id = item.get("collectionId")
    if collection_id:
        return f"itunes_collection_{collection_id}"
    track_id = item.get("trackId")
    if track_id:
        return f"itunes_track_{track_id}"
    artist = str(item.get("artistName") or "")
    album = str(item.get("collectionName") or "")
    normalized = _normalize_match_text(f"{artist} {album}")
    if normalized:
        digest = hashlib.sha1(normalized.encode("utf-8", "ignore")).hexdigest()
        return f"itunes_{digest}"
    return ""


def _find_existing_cover_file(cache_key: str) -> tuple[Optional[bytes], Optional[dict], Optional[str]]:
    if not cache_key:
        return None, None, None
    for ext in COVER_FILE_EXTENSIONS:
        filename = f"{cache_key}{ext}"
        path = os.path.join(CACHE_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as handle:
                body = handle.read()
        except Exception:
            return None, None, None
        return body, {"url": "cache", "content_type": _content_type_for_ext(ext)}, filename
    return None, None, None


def _find_existing_cover_for_mb_ids(
    release_id: str,
    release_group_id: str,
) -> tuple[Optional[bytes], Optional[dict], str, Optional[str], str]:
    if release_id:
        cache_key = _cover_cache_key_for_mb(release_id, "", "release")
        body, info, filename = _find_existing_cover_file(cache_key)
        if body:
            return body, info, cache_key, filename, "release"
    if release_group_id:
        cache_key = _cover_cache_key_for_mb("", release_group_id, "release_group")
        body, info, filename = _find_existing_cover_file(cache_key)
        if body:
            return body, info, cache_key, filename, "release_group"
    return None, None, "", None, ""


def _find_existing_cover_for_candidates(
    releases: list[dict],
) -> tuple[Optional[bytes], Optional[dict], str, Optional[dict], str, Optional[str]]:
    for release in releases:
        release_id = str(release.get("id") or "").strip()
        if not release_id:
            continue
        cache_key = _cover_cache_key_for_mb(release_id, "", "release")
        body, info, filename = _find_existing_cover_file(cache_key)
        if body:
            return body, info, "release", release, cache_key, filename
    for release in releases:
        release_group_id = _release_group_id_from_release(release)
        if not release_group_id:
            continue
        cache_key = _cover_cache_key_for_mb("", release_group_id, "release_group")
        body, info, filename = _find_existing_cover_file(cache_key)
        if body:
            return body, info, "release_group", release, cache_key, filename
    return None, None, "", None, "", None


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



def _try_fetch_itunes_cover_for_cache(
    path: str,
    entry: dict,
    cached: Optional[OnlineMetadataResult],
) -> Optional[OnlineMetadataResult]:
    """Try iTunes as a secondary cover source for an existing cache entry."""
    track = entry.get("track") or {}
    artist = str(track.get("artist") or "").strip()
    title = str(track.get("title") or "").strip()
    album = str(track.get("album") or "").strip()
    duration_sec = track.get("duration_sec")
    if duration_sec is not None:
        try:
            duration_sec = float(duration_sec)
        except Exception:
            duration_sec = None

    itunes_candidates: list[str] = []

    clean_artist = _clean_tag_value(artist)
    clean_title = _clean_tag_value(title)
    clean_album = _clean_tag_value(album)
    if clean_artist and clean_title:
        itunes_candidates.append(f"{clean_artist} {clean_title}")
    if artist and title:
        candidate = f"{artist} {title}"
        if candidate not in itunes_candidates:
            itunes_candidates.append(candidate)
    if clean_artist and clean_album:
        candidate = f"{clean_artist} {clean_album}"
        if candidate not in itunes_candidates:
            itunes_candidates.append(candidate)
    if artist and album:
        candidate = f"{artist} {album}"
        if candidate not in itunes_candidates:
            itunes_candidates.append(candidate)

    # Title-only queries are risky — only do these when artist is missing.
    if not clean_artist:
        if clean_title and clean_title not in itunes_candidates:
            itunes_candidates.append(clean_title)
        if title and title not in itunes_candidates:
            itunes_candidates.append(title)

    if not itunes_candidates:
        return cached

    try:
        itunes_bytes, itunes_ext, itunes_cache_key, itunes_item = _fetch_cover_from_itunes(
            itunes_candidates,
            expected_artist=artist,
            expected_title=title,
            expected_album=album,
            expected_duration_sec=duration_sec,
        )
    except Exception:
        itunes_bytes = None
        itunes_ext = ""
        itunes_cache_key = ""
        itunes_item = None

    if not itunes_bytes:
        # Mark that iTunes was attempted so we don't retry on every run.
        release_id = _release_id_from_cache(entry)
        release_group_id = _release_group_id_from_cache(entry)
        entry["cover_art"] = _build_cover_not_found_entry(
            release_id=release_id,
            release_group_id=release_group_id,
            source="itunes",
        )
        entry["cached_at"] = _utc_timestamp()
        _write_cache_entry(path, entry)
        return cached

    cover_info = {
        "url": (itunes_item.get("collectionViewUrl") or itunes_item.get("trackViewUrl") or "itunes_search") if itunes_item else "itunes_search",
        "content_type": _content_type_for_ext(itunes_ext),
    }
    cover_cache_key = itunes_cache_key or _cache_key(path)
    cover_filename = _build_cover_filename(cover_cache_key, cover_info)
    cover_path = os.path.join(CACHE_DIR, cover_filename)
    if not os.path.exists(cover_path):
        _ensure_cache_dir()
        with open(cover_path, "wb") as handle:
            handle.write(itunes_bytes)

    release_id = _release_id_from_cache(entry)
    release_group_id = _release_group_id_from_cache(entry)
    entry["cover_art"] = {
        "release_id": release_id,
        "release_group_id": release_group_id,
        "source": "itunes",
        "url": cover_info.get("url") if cover_info else "",
        "content_type": cover_info.get("content_type") if cover_info else "",
        "file": cover_filename,
        "cache_key": cover_cache_key,
    }
    entry["cached_at"] = _utc_timestamp()
    _write_cache_entry(path, entry)

    if cached:
        return OnlineMetadataResult(
            artist=cached.artist,
            album=cached.album,
            title=cached.title,
            duration_sec=cached.duration_sec,
            cover_art=itunes_bytes,
        )

    return OnlineMetadataResult(
        artist=artist,
        album=album,
        title=title,
        duration_sec=duration_sec,
        cover_art=itunes_bytes,
    )



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
                track = entry.get("track") or {}
                expected_album = str(track.get("album") or "").strip()
                if not expected_album:
                    tags = entry.get("tags") or {}
                    expected_album = str(tags.get("album") or "").strip()
                release = _select_release(detail.get("releases"), expected_album=expected_album)
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

    (
        existing_bytes,
        existing_info,
        existing_cache_key,
        existing_filename,
        existing_source,
    ) = _find_existing_cover_for_mb_ids(release_id, release_group_id)
    if existing_bytes and existing_filename:
        entry["cover_art"] = {
            "release_id": release_id,
            "release_group_id": release_group_id,
            "source": existing_source,
            "url": existing_info.get("url") if existing_info else "",
            "content_type": existing_info.get("content_type") if existing_info else "",
            "file": existing_filename,
            "cache_key": existing_cache_key,
        }
        entry["cached_at"] = _utc_timestamp()
        _write_cache_entry(path, entry)

        if cached:
            return OnlineMetadataResult(
                artist=cached.artist,
                album=cached.album,
                title=cached.title,
                duration_sec=cached.duration_sec,
                cover_art=existing_bytes,
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
            cover_art=existing_bytes,
        )

    cover_bytes, cover_info, cover_art_archive, cover_source, cover_status = _fetch_cover_for_release_ids(
        release_id,
        release_group_id,
    )
    if not cover_bytes:
        # Try iTunes before giving up (often works even when CAA has no image).
        itunes_try = _try_fetch_itunes_cover_for_cache(path, entry, cached)
        if itunes_try and itunes_try.cover_art:
            return itunes_try

        cover_attempt = entry.get("cover_art") or {}
        if cover_attempt.get("status") == "not_found" and str(cover_attempt.get("source") or "") == "itunes":
            return cached

        if (release_id or release_group_id) and cover_status == "not_found":
            entry["cover_art"] = _build_cover_not_found_entry(
                release_id=release_id,
                release_group_id=release_group_id,
                source=cover_source,
            )
            entry["cached_at"] = _utc_timestamp()
            _write_cache_entry(path, entry)
        return cached

    cover_cache_key = _cover_cache_key_for_mb(release_id, release_group_id, cover_source)
    if not cover_cache_key:
        cover_cache_key = _cache_key(path)
    cover_filename = _build_cover_filename(cover_cache_key, cover_info)
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
        "cache_key": cover_cache_key,
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
    if str(cover_entry.get("status") or "").lower() != "not_found":
        return False
    checked_at = str(cover_entry.get("checked_at") or "")
    if not checked_at:
        return True
    checked_ts = _parse_utc_timestamp(checked_at)
    if checked_ts is None:
        return True
    return (time.time() - checked_ts) < COVER_NOT_FOUND_TTL_SEC


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


def _parse_utc_timestamp(value: str) -> Optional[float]:
    try:
        return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return None


def _cache_entry_is_fresh(entry: dict, ttl_sec: int, *, timestamp_key: str = "cached_at") -> bool:
    """Return True if cache entry timestamp is within ttl_sec from now."""
    if ttl_sec <= 0:
        return False
    ts_raw = entry.get(timestamp_key) or ""
    ts = _parse_utc_timestamp(str(ts_raw))
    if ts is None:
        return False
    return (time.time() - ts) < float(ttl_sec)

