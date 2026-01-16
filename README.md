# TempoPitch Music Player

A Windows desktop music player built with PySide6 that lets you change tempo and pitch in real time during playback. It uses the SoundTouch DSP library for high-quality time-stretching and pitch-shifting, with a built-in phase vocoder fallback if SoundTouch is unavailable.

---

## Features

- Real-time tempo control (0.5x to 2.0x) and pitch control (-12 to +12 semitones)
- Key lock, tape mode, and A4=432 Hz pitch lock options
- 10-band equalizer with presets
- FX chain: Dynamic EQ, Compressor, Saturation, Subharmonic, Limiter, Reverb, Chorus, Stereo Width, Stereo Panner
- Playlist with drag and drop, folder scan, reordering, and session restore
- Shuffle and repeat modes (off/all/one)
- Visualizer, album art, and video preview with pop-out window for video files
- Theme selector and audio buffer presets (latency vs stability)
- Online metadata and cover art enrichment with caching

---

## Requirements

- Windows 64-bit
- Python 3.10+
- FFmpeg installed and available on PATH
- FFprobe available on PATH (optional but recommended for duration and tags)
- SoundTouch DLL (included) or other SoundTouch library

### Python packages

```bash
pip install PySide6 numpy sounddevice scipy numba
```

Verify FFmpeg:

```bat
ffmpeg -version
ffprobe -version
```

---

## Run

### Option A - One-click launcher (recommended)

Double-click:

- `RUN_Player.bat`

### Option B - From a terminal

```bat
python app.py
```

---

## Controls and Shortcuts

- Space: Play/Pause
- Ctrl+O: Open files
- Ctrl+L: Open folder
- Ctrl+N: Next track
- Ctrl+P: Previous track
- Ctrl+Left/Right: Seek +/-10s

---

## Supported Formats

FFmpeg determines playable formats. Common ones include:

- Audio: mp3, wav, flac, ogg, m4a, aac
- Video containers: mp4, mkv, mov, webm, avi

If FFmpeg can decode it, the player can likely play it.

---

## Configuration

Environment variables:

- `SOUNDTOUCH_DLL`: Path to the SoundTouch library. Default: `./SoundTouchDLL/SoundTouchDLL_x64.dll`
- `TEMPOPITCH_DSP`: `auto` (default), `soundtouch`, or `phasevocoder`

---

## Metadata and Artwork

- Local tags and duration are read via `ffprobe`.
- Embedded artwork is extracted via `ffmpeg`.
- Online metadata and cover art are fetched from MusicBrainz, Cover Art Archive, and iTunes when available.
- Results are cached in `metadata/` to avoid repeated network calls.

Video files skip online metadata lookups by default.

---

## Troubleshooting

### FFmpeg not found in PATH

Install FFmpeg and ensure these work in Command Prompt:

```bat
ffmpeg -version
ffprobe -version
```

### No audio / `sounddevice` errors

`sounddevice` uses PortAudio. Make sure:

- Your output device is working
- Your environment allows audio device access

### SoundTouch DLL load errors

Common causes:

- Running 32-bit Python with a 64-bit DLL
- DLL moved or renamed

Fix:

- Install 64-bit Python
- Keep `SoundTouchDLL/SoundTouchDLL_x64.dll` in place or set `SOUNDTOUCH_DLL`

---

## Architecture (high level)

- Decode thread runs ffmpeg to output float32 PCM
- Tempo/pitch DSP (SoundTouch or phase vocoder fallback)
- EQ and FX chain processing before audio output
- Audio output via sounddevice OutputStream with ring buffer
- Visualizer reads recent audio frames for FFT display
- Video frames are decoded via ffmpeg when the track has video

---

## License

No license specified yet. Add one if you plan to distribute.
