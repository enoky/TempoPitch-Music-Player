from __future__ import annotations

import calendar
import concurrent.futures
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

# Simple in-memory cache for album details to avoid repeated network calls for same album
_DEEZER_ALBUM_CACHE = {}


USER_AGENT = "TempoPitch-Music-Player/1.0 (local)"
ITUNES_API_URL = "https://itunes.apple.com/search"
DEEZER_API_URL = "https://api.deezer.com/search"
REQUEST_TIMEOUT_SEC = 7
CACHE_SCHEMA_VERSION = 11  # Bumped for genre/year support
COVER_NOT_FOUND_TTL_SEC = 7 * 24 * 60 * 60

ONLINE_NOT_FOUND_TTL_SEC = 7 * 24 * 60 * 60
CACHE_ERROR_TTL_SEC = 60 * 60
ITUNES_SEARCH_LIMIT = 5
DEEZER_SEARCH_LIMIT = 5
ITUNES_MIN_SCORE = 45
DEEZER_MIN_SCORE = 45
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


@dataclass
class OnlineMetadataResult:
    artist: str
    album: str
    title: str
    genre: str
    year: Optional[int]
    duration_sec: Optional[float]
    cover_art: Optional[bytes]


def get_online_metadata(
    path: str,
    *,
    tag_artist: str = "",
    tag_title: str = "",
    tag_album: str = "",
    tag_isrc: str = "",
    tag_duration_sec: Optional[float] = None,
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
            ttl = ONLINE_NOT_FOUND_TTL_SEC if status == "not_found" else CACHE_ERROR_TTL_SEC
            if _cache_entry_is_fresh(cache_entry, ttl):
                return None
            cache_entry = None

        if cache_entry:
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

    expected_artist = str(query_info.get("artist") or "").strip()
    expected_title = str(query_info.get("title") or "").strip()
    expected_album = str(query_info.get("album") or "").strip()

    itunes_item = None
    deezer_item = None
    itunes_score = 0
    deezer_score = 0

    duration_hint = None
    if tag_duration_sec is not None:
        try:
            duration_hint = float(tag_duration_sec)
        except Exception:
            duration_hint = None
    if duration_hint is not None and duration_hint <= 0.0:
        duration_hint = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        ft_itunes = executor.submit(
            _fetch_itunes_metadata,
            queries,
            expected_artist=expected_artist,
            expected_title=expected_title,
            expected_album=expected_album,
            expected_duration_sec=duration_hint,
        )
        ft_deezer = executor.submit(
            _fetch_deezer_metadata,
            queries,
            expected_artist=expected_artist,
            expected_title=expected_title,
            expected_album=expected_album,
            expected_duration_sec=duration_hint,
        )

        try:
            itunes_item, itunes_score = ft_itunes.result()
        except Exception:
            pass

        try:
            deezer_item, deezer_score = ft_deezer.result()
        except Exception:
            pass

    metadata_source = ""
    metadata_item = None

    if itunes_item and deezer_item:
        if deezer_score > itunes_score:
            metadata_source = "deezer"
            metadata_item = deezer_item
        elif itunes_score > deezer_score:
            metadata_source = "itunes"
            metadata_item = itunes_item
        else:
            metadata_source = "deezer"
            metadata_item = deezer_item
    elif itunes_item:
        metadata_source = "itunes"
        metadata_item = itunes_item
    elif deezer_item:
        metadata_source = "deezer"
        metadata_item = deezer_item

    artist = expected_artist
    album = expected_album
    title = expected_title
    genre = ""
    year = None
    duration_sec = None

    if metadata_item:
        if metadata_source == "itunes":
            artist, album, title, genre, year, duration_sec = _metadata_from_itunes_item(
                metadata_item,
                fallback_artist=expected_artist,
                fallback_album=expected_album,
                fallback_title=expected_title,
            )
        else:
            metadata_item = _enrich_deezer_item(metadata_item)
            artist, album, title, genre, year, duration_sec = _metadata_from_deezer_item(
                metadata_item,
                fallback_artist=expected_artist,
                fallback_album=expected_album,
                fallback_title=expected_title,
            )

    fallback_candidates = _build_fallback_candidates(artist, title, album)

    cover_bytes = None
    cover_info = None
    cover_entry = None
    cover_source = ""
    cover_cache_key = ""
    cover_attempted = False

    if metadata_item:
        cover_attempted = True
        if metadata_source == "itunes":
            cover_bytes, cover_info, cover_cache_key = _cover_from_itunes_item(metadata_item)
            cover_source = "itunes"
        elif metadata_source == "deezer":
            cover_bytes, cover_info, cover_cache_key = _cover_from_deezer_item(metadata_item)
            cover_source = "deezer"

    if not cover_bytes and fallback_candidates:
        cover_attempted = True
        cover_duration_hint = duration_sec if duration_sec is not None else duration_hint
        itunes_bytes = None
        deezer_bytes = None
        itunes_item_cover = None
        deezer_item_cover = None
        itunes_ext = ""
        deezer_ext = ""
        itunes_key = ""
        deezer_key = ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ft_itunes = executor.submit(
                _fetch_cover_from_itunes,
                fallback_candidates,
                expected_artist=artist,
                expected_title=title,
                expected_album=album,
                expected_duration_sec=cover_duration_hint,
            )
            ft_deezer = executor.submit(
                _fetch_cover_from_deezer,
                fallback_candidates,
                expected_artist=artist,
                expected_title=title,
                expected_album=album,
                expected_duration_sec=cover_duration_hint,
            )

            try:
                itunes_bytes, itunes_ext, itunes_key, itunes_item_cover = ft_itunes.result()
            except Exception:
                pass

            try:
                deezer_bytes, deezer_ext, deezer_key, deezer_item_cover = ft_deezer.result()
            except Exception:
                pass

        preferred_source = metadata_source or "deezer"
        if preferred_source == "itunes" and itunes_bytes:
            cover_bytes = itunes_bytes
            cover_info = {
                "url": _itunes_artwork_url(itunes_item_cover),
                "content_type": _content_type_for_ext(itunes_ext),
            }
            cover_source = "itunes"
            cover_cache_key = itunes_key
        elif preferred_source == "deezer" and deezer_bytes:
            cover_bytes = deezer_bytes
            cover_info = {
                "url": deezer_item_cover.get("album", {}).get("cover_xl") or "",
                "content_type": _content_type_for_ext(deezer_ext),
            }
            cover_source = "deezer"
            cover_cache_key = deezer_key
        elif deezer_bytes:
            cover_bytes = deezer_bytes
            cover_info = {
                "url": deezer_item_cover.get("album", {}).get("cover_xl") or "",
                "content_type": _content_type_for_ext(deezer_ext),
            }
            cover_source = "deezer"
            cover_cache_key = deezer_key
        elif itunes_bytes:
            cover_bytes = itunes_bytes
            cover_info = {
                "url": _itunes_artwork_url(itunes_item_cover),
                "content_type": _content_type_for_ext(itunes_ext),
            }
            cover_source = "itunes"
            cover_cache_key = itunes_key

    if cover_bytes and cover_info:
        if not cover_cache_key:
            # Try to make a stable key from Artist + Album if we have them,
            # so multiple tracks from the same album share the same file.
            if artist and album:
                normalized = _normalize_match_text(f"{artist} {album}")
                if normalized:
                     digest = hashlib.sha1(normalized.encode("utf-8", "ignore")).hexdigest()
                     cover_cache_key = f"generic_{digest}"
        
        if not cover_cache_key:
            cover_cache_key = _cache_key(path)

        cover_filename = _build_cover_filename(cover_cache_key, cover_info)
        cover_path = os.path.join(CACHE_DIR, cover_filename)
        if not os.path.exists(cover_path):
            _save_cover_file(cover_cache_key, cover_bytes, cover_info)
        cover_entry = {
            "source": cover_source,
            "url": cover_info.get("url") if cover_info else "",
            "content_type": cover_info.get("content_type") if cover_info else "",
            "file": cover_filename,
            "cache_key": cover_cache_key,
        }
    elif cover_attempted:
        cover_entry = _build_cover_not_found_entry(
            source=cover_source or "itunes_deezer",
        )

    if not (metadata_item or cover_bytes):
        cache_payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "not_found",
            "cached_at": _utc_timestamp(),
            "query": query_info,
        }
        _write_cache_entry(path, cache_payload)
        return None

    cache_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": "ok",
        "cached_at": _utc_timestamp(),
        "query": query_info,
        "track": {
            "artist": artist,
            "album": album,
            "title": title,
            "genre": genre,
            "year": year,
            "duration_sec": duration_sec,
        },
        "cover_art": cover_entry,
        "source": metadata_source,
        "tags": {
            "artist": tag_artist,
            "album": tag_album,
            "title": tag_title,
        },
    }
    _write_cache_entry(path, cache_payload)

    return OnlineMetadataResult(
        artist=artist,
        album=album,
        title=title,
        genre=genre,
        year=year,
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

    def add_query(*parts: str) -> None:
        cleaned = [part for part in parts if part]
        if not cleaned:
            return
        query = " ".join(cleaned)
        query = re.sub(r"\s+", " ", query).strip()
        if query and query not in queries:
            queries.append(query)

    # Clean versions of tags
    clean_title = _clean_tag_value(title)
    clean_artist = _clean_tag_value(artist)
    clean_album = _clean_tag_value(album)

    # 1. Artist + Title + Album (best chance for correct release)
    album_variants: list[str] = []
    if album:
        album_variants.append(album)
        cleaned_album = _strip_parenthetical(album)
        if cleaned_album and cleaned_album not in album_variants:
            album_variants.append(cleaned_album)
    if clean_album and clean_album not in album_variants:
        album_variants.append(clean_album)

    if artist and album_variants:
        for release in album_variants:
            add_query(artist, title, release)
            # Try with clean title if different
            if clean_title and clean_title != title:
                 add_query(artist, clean_title, release)
            if clean_artist and clean_artist != artist:
                 add_query(clean_artist, title, release)
                 if clean_title and clean_title != title:
                     add_query(clean_artist, clean_title, release)

    # 2. Artist + Title (No Album)
    if artist:
        add_query(artist, title)
        if clean_title and clean_title != title:
            add_query(artist, clean_title)
        if clean_artist and clean_artist != artist:
             add_query(clean_artist, title)
             if clean_title and clean_title != title:
                 add_query(clean_artist, clean_title)

    # 3. Album + Title (No Artist)
    if album_variants:
        for release in album_variants:
            add_query(title, release)
            if clean_title and clean_title != title:
                add_query(clean_title, release)

    # 4. Title Only
    add_query(title)
    if clean_title and clean_title != title:
        add_query(clean_title)

    return queries


def _fetch_itunes_metadata(
    queries: list[str],
    *,
    expected_artist: str,
    expected_title: str,
    expected_album: str,
    expected_duration_sec: Optional[float],
) -> tuple[Optional[dict], int]:
    candidates: dict[str, dict] = {}
    for query in queries:
        for item in _itunes_search(query, ITUNES_SEARCH_LIMIT, entity="song"):
            track_id = str(item.get("trackId") or "")
            if not track_id:
                track_id = f"{item.get('artistName')}|{item.get('trackName')}|{item.get('collectionName')}"
            if track_id and track_id not in candidates:
                candidates[track_id] = item

    if not candidates:
        return None, 0

    item = _select_itunes_result(
        list(candidates.values()),
        expected_artist=expected_artist,
        expected_title=expected_title,
        expected_album=expected_album,
        expected_duration_sec=expected_duration_sec,
    )
    if not item:
        return None, 0

    score = _score_itunes_result(
        item,
        expected_artist=expected_artist,
        expected_title=expected_title,
        expected_album=expected_album,
        expected_duration_sec=expected_duration_sec,
    )
    return item, score


def _fetch_deezer_metadata(
    queries: list[str],
    *,
    expected_artist: str,
    expected_title: str,
    expected_album: str,
    expected_duration_sec: Optional[float],
) -> tuple[Optional[dict], int]:
    candidates: dict[str, dict] = {}
    for query in queries:
        cleaned_query = _clean_query_for_api(query)
        if not cleaned_query:
            continue
        for item in _deezer_search(cleaned_query, DEEZER_SEARCH_LIMIT):
            item_id = str(item.get("id") or "")
            if item_id and item_id not in candidates:
                candidates[item_id] = item

    if not candidates:
        return None, 0

    item = _select_deezer_result(
        list(candidates.values()),
        expected_artist=expected_artist,
        expected_title=expected_title,
        expected_album=expected_album,
        expected_duration_sec=expected_duration_sec,
    )
    if not item:
        return None, 0

    score = _score_deezer_result(
        item,
        expected_artist=expected_artist,
        expected_title=expected_title,
        expected_album=expected_album,
        expected_duration_sec=expected_duration_sec,
    )
    return item, score


def _enrich_deezer_item(item: dict) -> dict:
    """Fetch full album details from Deezer to get Genre and Year."""
    album_data = item.get("album") or {}
    album_id = str(album_data.get("id") or "")
    if not album_id:
        return item

    if album_id in _DEEZER_ALBUM_CACHE:
        item["enriched_album"] = _DEEZER_ALBUM_CACHE[album_id]
        return item

    url = f"https://api.deezer.com/album/{album_id}"
    try:
        full_album = _http_get_json(url)
        if full_album and "error" not in full_album:
            _DEEZER_ALBUM_CACHE[album_id] = full_album
            item["enriched_album"] = full_album
    except Exception:
        pass
    
    return item


def _metadata_from_itunes_item(
    item: dict,
    *,
    fallback_artist: str,
    fallback_album: str,
    fallback_title: str,
) -> tuple[str, str, str, str, Optional[int], Optional[float]]:
    artist = str(item.get("artistName") or "").strip() or fallback_artist
    album = str(item.get("collectionName") or "").strip() or fallback_album
    title = str(item.get("trackName") or "").strip() or fallback_title
    genre = str(item.get("primaryGenreName") or "").strip()
    year = None
    
    release_date = str(item.get("releaseDate") or "")
    if release_date:
        # ISO format: 2005-03-01T08:00:00Z
        try:
            year_str = release_date.split("-")[0]
            if year_str.isdigit():
                year = int(year_str)
        except Exception:
            pass

    duration_sec = None

    track_ms = item.get("trackTimeMillis")
    if track_ms is not None:
        try:
            duration_sec = float(track_ms) / 1000.0
        except Exception:
            duration_sec = None

    return artist, album, title, genre, year, duration_sec


def _metadata_from_deezer_item(
    item: dict,
    *,
    fallback_artist: str,
    fallback_album: str,
    fallback_title: str,
) -> tuple[str, str, str, str, Optional[int], Optional[float]]:
    artist_data = item.get("artist") or {}
    album_data = item.get("album") or {}
    artist = str(artist_data.get("name") or "").strip() or fallback_artist
    album = str(album_data.get("title") or "").strip() or fallback_album
    title = str(item.get("title") or "").strip() or fallback_title
    year = None
    
    enriched = item.get("enriched_album") or {}

    # 1. Try Genre from Enriched Album
    if enriched:
        genres_data = enriched.get("genres", {}).get("data")
        if genres_data:
            genre = str(genres_data[0].get("name") or "").strip()

    # 2. Try Year from Release Date (Enriched or basic)
    release_date = str(item.get("release_date") or album_data.get("release_date") or "")
    if not release_date and enriched:
        release_date = str(enriched.get("release_date") or "")

    if release_date:
        try:
             year_str = release_date.split("-")[0]
             if year_str.isdigit():
                 year = int(year_str)
        except Exception:
            pass

    duration_sec = None

    duration = item.get("duration")
    if duration is not None:
        try:
            duration_sec = float(duration)
        except Exception:
            duration_sec = None

    return artist, album, title, genre, year, duration_sec


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


def _cover_from_itunes_item(
    item: dict,
) -> tuple[Optional[bytes], Optional[dict], str]:
    cache_key = _cover_cache_key_for_itunes(item)
    if cache_key:
        cached_bytes, cached_info, cached_filename = _find_existing_cover_file(cache_key)
        if cached_bytes and cached_filename:
            return cached_bytes, cached_info, cache_key

    artwork_url = _itunes_artwork_url(item)
    if not artwork_url:
        return None, None, cache_key

    artwork_url = _upgrade_itunes_artwork_url(artwork_url)
    try:
        body, content_type = _http_get_bytes(artwork_url)
        info = {"url": artwork_url, "content_type": content_type}
        return body, info, cache_key
    except Exception:
        return None, None, cache_key


def _cover_from_deezer_item(
    item: dict,
) -> tuple[Optional[bytes], Optional[dict], str]:
    album_data = item.get("album") or {}
    cover_url = (
        album_data.get("cover_xl")
        or album_data.get("cover_big")
        or album_data.get("cover_medium")
    )
    cache_key = ""
    album_id = str(album_data.get("id") or "")
    if album_id:
        cache_key = f"deezer_album_{album_id}"
    elif item.get("id"):
        # Fallback to track ID if album ID is missing (rare)
        cache_key = f"deezer_track_{item.get('id')}"

    if cache_key:
        cached_bytes, cached_info, cached_filename = _find_existing_cover_file(cache_key)
        if cached_bytes and cached_filename:
            return cached_bytes, cached_info, cache_key

    if not cover_url:
        return None, None, cache_key

    try:
        body, content_type = _http_get_bytes(cover_url)
        info = {"url": cover_url, "content_type": content_type}
        return body, info, cache_key
    except Exception:
        return None, None, cache_key


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


def _deezer_search(query: str, limit: int) -> list[dict]:
    params = {
        "q": query,
        "limit": str(limit),
        "order": "RANKING",
    }
    url = f"{DEEZER_API_URL}?{urlencode(params)}"
    try:
        data = _http_get_json(url)
    except Exception:
        return []
    return data.get("data") or []


def _score_deezer_result(
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

    track_title = _normalize_match_text(str(item.get("title") or ""))
    artist_data = item.get("artist") or {}
    album_data = item.get("album") or {}
    artist_name = _normalize_match_text(str(artist_data.get("name") or ""))
    album_title = _normalize_match_text(str(album_data.get("title") or ""))

    # Deezer doesn't have explicit wrapperType like iTunes, but we are searching for tracks usually.
    
    if expected_title_norm and track_title:
        score += int(_string_similarity(track_title, expected_title_norm) * 60)
        score += _token_overlap_bonus(track_title, expected_title_norm, 12)
        score -= _qualifier_penalty(track_title, expected_title_norm, 4)
        score += _score_text_match(track_title, expected_title_norm, 20, 10)

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

    if expected_album_norm and album_title:
        score += int(_string_similarity(album_title, expected_album_norm) * 35)
        score += _token_overlap_bonus(album_title, expected_album_norm, 10)
        score -= _qualifier_penalty(album_title, expected_album_norm, 3)
        score += _score_text_match(album_title, expected_album_norm, 12, 6)

        try:
            album_sim = _string_similarity(album_title, expected_album_norm)
        except Exception:
            album_sim = 0.0
        if album_sim < 0.25:
            score -= 25

    if expected_duration_sec is not None:
        duration = item.get("duration")
        if duration is not None:
            try:
                diff = abs(float(duration) - float(expected_duration_sec))
            except Exception:
                diff = None
            if diff is not None:
                # Stricter duration matching
                if diff <= 2.0:
                    score += 15
                elif diff <= 5.0:
                    score += 8
                elif diff <= 10.0:
                    score += 4
                elif diff > 15.0:
                    score -= 10  # Penalty for significant duration mismatch

    return score


def _select_deezer_result(
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
        score = _score_deezer_result(
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
    if (expected_title or expected_album or expected_artist) and (best_score or 0) < DEEZER_MIN_SCORE:
        return None
    return best_item


def _fetch_cover_from_deezer(
    queries: list[str],
    *,
    expected_artist: str = "",
    expected_title: str = "",
    expected_album: str = "",
    expected_duration_sec: Optional[float] = None,
) -> tuple[Optional[bytes], str, str, Optional[dict]]:
    """Fetch cover art via Deezer API.

    Returns (cover_bytes, ext, cache_key, selected_item).
    """
    candidates: dict[str, dict] = {}

    for query in queries:
        # Pre-process query to remove common noise for better API hits
        cleaned_query = _clean_query_for_api(query)
        if not cleaned_query:
            continue
            
        for item in _deezer_search(cleaned_query, DEEZER_SEARCH_LIMIT):
            item_id = str(item.get("id") or "")
            if not item_id:
                continue
            if item_id not in candidates:
                candidates[item_id] = item

    if not candidates:
        return None, "", "", None

    item = _select_deezer_result(
        list(candidates.values()),
        expected_artist=expected_artist,
        expected_title=expected_title,
        expected_album=expected_album,
        expected_duration_sec=expected_duration_sec,
    )
    if not item:
        return None, "", "", None

    album_data = item.get("album") or {}
    cover_url = album_data.get("cover_xl") or album_data.get("cover_big") or album_data.get("cover_medium")
    
    if not cover_url:
        return None, "", "", item

    # Deezer cache key
    cache_key = ""
    album_id = str(album_data.get("id") or "")
    if album_id:
        cache_key = f"deezer_album_{album_id}"
    elif item.get("id"):
        cache_key = f"deezer_track_{item.get('id')}"

    # Check existing cache
    if cache_key:
        cached_bytes, cached_info, cached_filename = _find_existing_cover_file(cache_key)
        if cached_bytes and cached_filename:
            ext = os.path.splitext(cached_filename)[1].lower() or ".jpg"
            return cached_bytes, ext, cache_key, item

    try:
        body, content_type = _http_get_bytes(cover_url)
        ext = _guess_image_ext(content_type, cover_url)
        return body, ext, cache_key, item
    except Exception:
        return None, "", cache_key, item


def _clean_query_for_api(query: str) -> str:
    """Remove common noise from queries to improve API processing."""
    # Remove (Official Video), [HD], etc.
    query = re.sub(r"(?i)[\(\[]\s*(official|lyric|high definition|hd|4k|mv|music video).*?[\)\]]", "", query)
    # Remove VEVO, - Topic suffix
    query = re.sub(r"(?i)\s*(-)?\s*(vevo|topic)\s*$", "", query)
    return query.strip()



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




def _build_fallback_candidates(artist: str, title: str, album: str) -> list[str]:
    """Build a list of search queries for cover art fallback."""
    candidates: list[str] = []
    
    clean_artist = _clean_tag_value(artist)
    clean_title = _clean_tag_value(title)
    clean_album = _clean_tag_value(album)
    
    if clean_artist and clean_title:
        candidates.append(f"{clean_artist} {clean_title}")
    
    if artist and title:
        candidate = f"{artist} {title}"
        if candidate not in candidates:
            candidates.append(candidate)
            
    if clean_artist and clean_album:
        candidate = f"{clean_artist} {clean_album}"
        if candidate not in candidates:
            candidates.append(candidate)
            
    if artist and album:
        candidate = f"{artist} {album}"
        if candidate not in candidates:
            candidates.append(candidate)

    # Title-only queries are risky — only do these when artist is missing.
    if not clean_artist:
        if clean_title and clean_title not in candidates:
            candidates.append(clean_title)
        if title and title not in candidates:
            candidates.append(title)
            
    return candidates


def _save_cover_file(cache_key: str, data: bytes, info: Optional[dict]) -> None:
    _ensure_cache_dir()
    filename = _build_cover_filename(cache_key, info)
    path = os.path.join(CACHE_DIR, filename)
    try:
        with open(path, "wb") as handle:
            handle.write(data)
    except Exception:
        pass


def _try_fetch_cover_for_cache(
    path: str,
    entry: dict,
    cached: Optional[OnlineMetadataResult],
) -> Optional[OnlineMetadataResult]:
    """Try iTunes and Deezer as cover sources for an existing cache entry."""
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

    candidates = _build_fallback_candidates(artist, title, album)

    if not candidates:
        return cached

    itunes_bytes = None
    deezer_bytes = None
    itunes_item = None
    deezer_item = None
    itunes_ext = ""
    deezer_ext = ""
    itunes_key = ""
    deezer_key = ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        ft_itunes = executor.submit(
            _fetch_cover_from_itunes,
            candidates,
            expected_artist=artist,
            expected_title=title,
            expected_album=album,
            expected_duration_sec=duration_sec,
        )
        ft_deezer = executor.submit(
            _fetch_cover_from_deezer,
            candidates,
            expected_artist=artist,
            expected_title=title,
            expected_album=album,
            expected_duration_sec=duration_sec,
        )

        try:
            itunes_bytes, itunes_ext, itunes_key, itunes_item = ft_itunes.result()
        except Exception:
            pass

        try:
            deezer_bytes, deezer_ext, deezer_key, deezer_item = ft_deezer.result()
        except Exception:
            pass

    cover_entry = None
    cover_bytes = None

    preferred_source = str(entry.get("source") or "").strip().lower()
    if preferred_source not in ("itunes", "deezer"):
        preferred_source = "deezer"

    if preferred_source == "itunes" and itunes_bytes:
        cover_bytes = itunes_bytes
        cover_info = {
            "url": _itunes_artwork_url(itunes_item),
            "content_type": _content_type_for_ext(itunes_ext),
        }
        cover_entry = {
            "source": "itunes",
            "url": cover_info["url"],
            "content_type": cover_info["content_type"],
            "file": f"{itunes_key}{itunes_ext}",
            "cache_key": itunes_key,
        }
        _save_cover_file(itunes_key, itunes_bytes, cover_info)

    elif preferred_source == "deezer" and deezer_bytes:
        cover_bytes = deezer_bytes
        cover_info = {
            "url": deezer_item.get("album", {}).get("cover_xl") or "",
            "content_type": _content_type_for_ext(deezer_ext),
        }
        cover_entry = {
            "source": "deezer",
            "url": cover_info["url"],
            "content_type": cover_info["content_type"],
            "file": f"{deezer_key}{deezer_ext}",
            "cache_key": deezer_key,
        }
        _save_cover_file(deezer_key, deezer_bytes, cover_info)

    elif deezer_bytes:
        cover_bytes = deezer_bytes
        cover_info = {
            "url": deezer_item.get("album", {}).get("cover_xl") or "",
            "content_type": _content_type_for_ext(deezer_ext),
        }
        cover_entry = {
            "source": "deezer",
            "url": cover_info["url"],
            "content_type": cover_info["content_type"],
            "file": f"{deezer_key}{deezer_ext}",
            "cache_key": deezer_key,
        }
        _save_cover_file(deezer_key, deezer_bytes, cover_info)

    elif itunes_bytes:
        cover_bytes = itunes_bytes
        cover_info = {
            "url": _itunes_artwork_url(itunes_item),
            "content_type": _content_type_for_ext(itunes_ext),
        }
        cover_entry = {
            "source": "itunes",
            "url": cover_info["url"],
            "content_type": cover_info["content_type"],
            "file": f"{itunes_key}{itunes_ext}",
            "cache_key": itunes_key,
        }
        _save_cover_file(itunes_key, itunes_bytes, cover_info)

    if not cover_bytes:
        entry["cover_art"] = _build_cover_not_found_entry(
            source="itunes_deezer",
        )
        _write_cache_entry(path, entry)
        return cached

    entry["cover_art"] = cover_entry
    _write_cache_entry(path, entry)

    return OnlineMetadataResult(
        artist=cached.artist if cached else artist,
        album=cached.album if cached else album,
        title=cached.title if cached else title,
        genre=cached.genre if cached else "", # Genre/Year not fetched in this flow yet
        year=cached.year if cached else None,
        duration_sec=cached.duration_sec if cached else duration_sec,
        cover_art=cover_bytes,
    )


def _result_from_cache(entry: dict) -> Optional[OnlineMetadataResult]:
    track = entry.get("track") or {}
    artist = str(track.get("artist") or "").strip()
    album = str(track.get("album") or "").strip()
    title = str(track.get("title") or "").strip()
    genre = str(track.get("genre") or "").strip()
    year_raw = track.get("year")
    year = None
    if year_raw is not None:
        try:
             year = int(year_raw)
        except Exception:
             pass

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
        genre=genre,
        year=year,
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
    source: str,
) -> dict:
    return {
        "status": "not_found",
        "source": source,
        "checked_at": _utc_timestamp(),
    }


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
