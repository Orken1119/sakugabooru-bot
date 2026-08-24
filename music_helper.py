import os
import glob
import random
import subprocess

MUSIC_CACHE_DIR = "music_cache"
MAX_CACHE_SIZE = 15  # keep up to 15 tracks locally

# Royalty-free lofi search queries — targets no-copyright content
LOFI_QUERIES = [
    "lofi hip hop royalty free no copyright chill beats",
    "lofi chill music royalty free copyright free",
    "study lofi beats royalty free no copyright",
    "lofi relaxing beats no copyright free to use",
    "chill lofi music royalty free background",
]


def get_cached_tracks():
    """Returns list of locally cached audio files."""
    if not os.path.exists(MUSIC_CACHE_DIR):
        return []
    tracks = glob.glob(os.path.join(MUSIC_CACHE_DIR, "*.m4a"))
    tracks += glob.glob(os.path.join(MUSIC_CACHE_DIR, "*.mp3"))
    tracks += glob.glob(os.path.join(MUSIC_CACHE_DIR, "*.opus"))
    return tracks


def download_lofi_track():
    """
    Downloads a random royalty-free lofi track from YouTube via yt-dlp.
    Saves to music_cache/. Returns filepath or None.
    """
    os.makedirs(MUSIC_CACHE_DIR, exist_ok=True)
    query = random.choice(LOFI_QUERIES)

    print(f"[♪] Fetching lofi track: '{query}'...")
    try:
        cmd = [
            ".venv/bin/yt-dlp",
            f"ytsearch3:{query}",   # search top 3, pick first downloadable
            "--quiet",
            "--no-warnings",
            "-x",
            "--audio-format", "m4a",
            "--audio-quality", "128K",
            "--match-filter", "duration > 120",   # at least 2 min long
            "--playlist-items", "1",
            "-o", os.path.join(MUSIC_CACHE_DIR, "%(title).50s_%(id)s.%(ext)s"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Find the newly downloaded file
        tracks = get_cached_tracks()
        if tracks:
            newest = max(tracks, key=os.path.getmtime)
            print(f"[♪] Downloaded: {os.path.basename(newest)}")
            return newest

        if res.returncode != 0:
            print(f"[!] yt-dlp error: {res.stderr[-300:]}")
    except Exception as e:
        print(f"[!] Lofi download error: {e}")

    return None


def get_random_lofi_track():
    """
    Returns path to a random lofi track.
    Uses local cache if available (>=2 tracks), otherwise downloads fresh.
    """
    tracks = get_cached_tracks()

    # Maintain cache — if over limit, delete oldest
    if len(tracks) > MAX_CACHE_SIZE:
        oldest = sorted(tracks, key=os.path.getmtime)[0]
        os.remove(oldest)
        tracks.remove(oldest)

    # Use cache if we have tracks
    if len(tracks) >= 2:
        chosen = random.choice(tracks)
        print(f"[♪] Using cached track: {os.path.basename(chosen)}")
        return chosen

    # Cache empty or only 1 track — download fresh
    return download_lofi_track()


def add_lofi_to_video(video_path, output_path=None):
    """
    Merges a random lofi track into the video with fade in/out.
    - Picks a random start point within the lofi track (so it never always starts at the beginning)
    - Fades audio in over 0.5s, out over 1.0s
    - Audio trimmed to exactly match video duration
    Returns output path or None on failure.
    """
    if output_path is None:
        output_path = video_path.replace(".mp4", "_lofi.mp4")

    lofi_track = get_random_lofi_track()
    if not lofi_track:
        print("[!] No lofi track available — skipping audio.")
        return None

    # Get video duration
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True
        )
        video_duration = float(res.stdout.strip())
    except Exception:
        video_duration = 30.0

    # Get lofi track duration
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", lofi_track],
            capture_output=True, text=True, check=True
        )
        lofi_duration = float(res.stdout.strip())
    except Exception:
        lofi_duration = 180.0

    # Random start within lofi track (leave at least video_duration + 5s at end)
    max_start = max(0.0, lofi_duration - video_duration - 5.0)
    lofi_start = round(random.uniform(0, max_start), 1)

    # Fade out starts 1s before end
    fade_out_start = max(0.0, video_duration - 1.0)

    print(f"[♪] Merging lofi (start={lofi_start:.1f}s, video={video_duration:.1f}s, fade_out@{fade_out_start:.1f}s)...")

    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(lofi_start),
            "-i", lofi_track,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-af", f"afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out_start}:d=1.0",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(video_duration),
            output_path
        ]
        res_ff = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res_ff.returncode == 0 and os.path.exists(output_path):
            print(f"[♪] ✅ Lofi merged: {output_path}")
            return output_path
        else:
            print(f"[!] ffmpeg lofi merge failed: {res_ff.stderr[-300:]}")
    except Exception as e:
        print(f"[!] Lofi merge error: {e}")

    return None
