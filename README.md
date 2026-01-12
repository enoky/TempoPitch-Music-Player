# TempoPitch Music Player 🎚️🎵

A lightweight Windows desktop music player built with **PySide6** that lets you change **tempo** and **pitch** in real time during playback. It uses the SoundTouch DSP library for high‑quality time‑stretching and pitch‑shifting.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-0078d6)

---

## Features

- **Real‑time tempo control** (0.5× → 2.0×)
- **Real‑time pitch control** (-12 → +12 semitones)
- **Key Lock**: adjust tempo without changing pitch
- **Tape Mode**: tempo changes pitch together (classic tape behavior)
- Playlist management (add files/folders, drag & drop, reorder)
- Seek bar with time display (duration via `ffprobe`)
- Volume slider + mute
- Auto‑advance

---

## Contents

- `TempoPitch_Music_Player.py` — main application
- `RUN_Player.bat` — one‑click Windows launcher
- `SoundTouchDLL/`
  - `SoundTouchDLL_x64.dll` — SoundTouch DSP library

> The app loads SoundTouch from:
>
> `./SoundTouchDLL/SoundTouchDLL_x64.dll`

---

## Requirements

### Windows

- **Python 3.10+** (64‑bit recommended)
- **FFmpeg** installed and available on PATH (`ffmpeg` + `ffprobe`)

### Python packages

```bash
pip install PySide6 numpy sounddevice
```

Verify FFmpeg:

```bat
ffmpeg -version
ffprobe -version
```

---

## Run

### Option A — One‑click launcher (recommended)

Double‑click:

- `RUN_Player.bat`

### Option B — From a terminal

```bat
python TempoPitch_Music_Player.py
```

---

## Controls & Shortcuts

- **Space**: Play / Pause
- **Ctrl+O**: Open Files…
- **Ctrl+L**: Open Folder…
- **Ctrl+Left / Ctrl+Right**: Seek ±10s

---

## Supported Formats

FFmpeg determines playable formats (e.g., **mp3**, **wav**, **flac**, **ogg**, **m4a/aac**, plus video containers like **mp4**, **mkv**, **mov**, **webm**, **avi**). If FFmpeg can decode it, the player can likely play it.

---

## Troubleshooting

### “ffmpeg not found in PATH”

Install FFmpeg and ensure these work in Command Prompt:

```bat
ffmpeg -version
ffprobe -version
```

### No audio / `sounddevice` errors

`sounddevice` uses PortAudio. Make sure:

- Your default output device is working
- You aren’t running inside an environment that blocks audio device access

### SoundTouch DLL load errors

Common causes:

- Running **32‑bit Python** with a **64‑bit DLL**
- DLL moved/renamed

Fix:

- Install **64‑bit Python**
- Keep `SoundTouchDLL/SoundTouchDLL_x64.dll` in place

Check Python architecture:

```bat
python -c "import platform; print(platform.architecture())"
```

---

## Architecture (high level)

- **Decode thread**: runs `ffmpeg` to decode audio into float32 PCM
- **DSP**: SoundTouch processes tempo/pitch
- **Audio output**: `sounddevice` OutputStream pulls from a ring buffer
- **UI**: PySide6 updates position/buffer status periodically

---

## Roadmap / Ideas

- Better metadata (artist/album/artwork)
- Repeat/shuffle modes
- Saved playlists (M3U/M3U8)
- Output device selector
- PyInstaller builds for Windows

---

## License

No license specified yet. Add one if you plan to distribute.
