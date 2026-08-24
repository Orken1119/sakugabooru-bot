import os
import re
import glob
import time
import subprocess
import requests
from nyaa_downloader import search_and_download_nyaa

# ─────────────────────────────────────────────
TRACE_MIN_SIMILARITY = 0.70
DOWNLOAD_POLL_TIMEOUT = 7200   # 2 hours
DOWNLOAD_POLL_INTERVAL = 15    # seconds
YOUTUBE_DOMAINS = ('youtube.com', 'youtu.be', 'youtube-nocookie.com')
# ─────────────────────────────────────────────


def get_media_duration(file_path):
    """Returns duration in seconds via ffprobe."""
    try:
        cmd = ['ffprobe', '-v', 'error',
               '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return None


def parse_episode_from_source(source):
    """
    Extracts episode number from Sakugabooru source field.
    Examples: '#06 (BD)' → 6, '#24' → 24, 'Episode 14' → 14
    Returns int or None.
    """
    if not source:
        return None
    match = re.search(r'#\s*(\d+)|[Ee]p(?:isode)?\s*(\d+)|[Ss]\d+[Ee](\d+)', source)
    if match:
        num = next(g for g in match.groups() if g is not None)
        return int(num)
    return None


def is_youtube_source(source):
    """Returns True if source URL points to YouTube."""
    if not source:
        return False
    return any(d in source for d in YOUTUBE_DOMAINS)


# ─── PATH A: YouTube Source ───────────────────────────────────────────────────

def fetch_timestamp_from_tracemoe(video_path):
    """
    Queries Trace.moe and returns (start_sec, ep_num, similarity).
    Used to find WHERE in a YouTube video the clip starts.
    """
    video_duration = get_media_duration(video_path) or 10.0
    seek_time = min(1.0, max(0.2, video_duration / 4.0))
    pid = os.getpid()
    frame_path = f"temp_frame_{pid}.jpg"

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-ss', str(seek_time), '-i', video_path,
             '-vframes', '1', frame_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        if not os.path.exists(frame_path):
            return None, None, 0.0

        with open(frame_path, 'rb') as f:
            res = requests.post('https://api.trace.moe/search', files={'image': f}, timeout=10)

        if res.status_code == 200:
            results = res.json().get('result', [])
            if results:
                match = results[0]
                sim = match.get('similarity', 0)
                at_sec = match.get('at', 0)
                ep_num = match.get('episode')
                start_sec = max(0.0, at_sec - seek_time)
                print(f"[+] Trace.moe: at={at_sec:.1f}s ep={ep_num} similarity={sim:.2f}")
                return start_sec, ep_num, sim
    except Exception as e:
        print("[!] Trace.moe error:", e)
    finally:
        if os.path.exists(frame_path):
            os.remove(frame_path)

    return None, None, 0.0


def add_audio_from_youtube(video_path, youtube_url):
    """
    PATH A — YouTube Source:
    1. Downloads audio from the YouTube source URL using yt-dlp.
    2. Uses Trace.moe to find WHERE in that video our clip starts.
    3. Merges the sliced audio into the clip.
    """
    video_duration = get_media_duration(video_path) or 10.0
    pid = os.getpid()
    yt_audio_path = f"temp_yt_audio_{pid}.m4a"
    output_path = video_path.replace(".mp4", "_raw_audio.mp4")

    print(f"[A] YouTube Source detected: {youtube_url}")

    # Step 1: Try Trace.moe for timestamp within the video
    start_sec, _, similarity = fetch_timestamp_from_tracemoe(video_path)
    if start_sec is None or similarity < TRACE_MIN_SIMILARITY:
        start_sec = 0.0
        print(f"[A] Trace.moe could not pin timestamp — using 0.0s offset in YouTube audio")
    else:
        print(f"[A] Using timestamp {start_sec:.2f}s from Trace.moe")

    # Step 2: Download audio from YouTube
    print(f"[A] Downloading YouTube audio via yt-dlp...")
    try:
        yt_cmd = [
            '.venv/bin/yt-dlp',
            '--quiet',
            '-x',
            '--audio-format', 'm4a',
            '--audio-quality', '0',
            '-o', yt_audio_path,
            youtube_url
        ]
        res_yt = subprocess.run(yt_cmd, capture_output=True, text=True, timeout=120)
        if res_yt.returncode != 0 or not os.path.exists(yt_audio_path):
            print(f"[!] yt-dlp failed: {res_yt.stderr[-300:]}")
            return None
        print(f"[A] YouTube audio downloaded: {yt_audio_path}")
    except Exception as e:
        print(f"[!] yt-dlp error: {e}")
        return None

    # Step 3: Merge audio into clip
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start_sec),
            "-i", yt_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-t", str(video_duration),
            output_path
        ]
        res_ff = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res_ff.returncode == 0 and os.path.exists(output_path):
            print(f"[A] ✅ YouTube audio merged: {output_path}")
            return output_path
        else:
            print(f"[!] ffmpeg merge failed: {res_ff.stderr[-300:]}")
    except Exception as e:
        print(f"[!] ffmpeg error: {e}")
    finally:
        if os.path.exists(yt_audio_path):
            os.remove(yt_audio_path)

    return None


# ─── PATH B: Episode Source (Nyaa) ────────────────────────────────────────────

def is_download_complete(filepath):
    """ffprobe-based completion check — ignores qBittorrent sparse pre-allocated files."""
    if not filepath or not os.path.exists(filepath):
        return False
    if filepath.endswith(('.!qB', '.qB', '.part', '.crdownload', '.tmp')):
        return False
    try:
        cmd = ['ffprobe', '-v', 'error',
               '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', filepath]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip()) > 0.0
    except Exception:
        return False
    return False


def find_local_episode_file(anime_name, episode_num=None, search_dirs=None, require_complete=False):
    """
    Finds local episode file. Requires majority of title words to match (not just any one).
    """
    if search_dirs is None:
        search_dirs = ["sakugabooru-episodes", "/home/orken/Downloads"]

    clean_search = anime_name.lower().replace('_', ' ').replace('-', ' ').split(',')[0].strip()
    words = [w for w in clean_search.split() if len(w) > 2 and w not in ('the', 'and', 'for', 'series', 'sub')]
    if not words:
        return None

    ep_patterns = []
    if episode_num is not None:
        ep_str = f"{int(episode_num):02d}"
        ep_patterns = [f"e{ep_str}", f" {ep_str} ", f"_{ep_str}_", f"-{ep_str}-", f"ep{ep_str}"]

    video_extensions = ['*.mkv', '*.mp4', '*.avi']
    best_match, best_score = None, 0

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        local_files = []
        for ext in video_extensions:
            local_files.extend(glob.glob(os.path.join(d, ext)))
            local_files.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))

        for f in local_files:
            fname = os.path.basename(f).lower()
            match_count = sum(1 for w in words if w in fname)
            if match_count < max(1, (len(words) + 1) // 2):
                continue
            ep_bonus = 1 if ep_patterns and any(p in fname for p in ep_patterns) else 0
            score = match_count + ep_bonus
            if score > best_score:
                if not require_complete or is_download_complete(f):
                    best_match, best_score = f, score

    return best_match


def get_qbittorrent_progress(name_fragment):
    """Checks qBittorrent Web API for download progress. Returns 0.0–1.0 or None."""
    try:
        res = requests.get('http://localhost:8080/api/v2/torrents/info', timeout=3)
        if res.status_code == 200:
            for t in res.json():
                if name_fragment.lower() in t.get('name', '').lower():
                    return t.get('progress', 0.0)
    except Exception:
        pass
    return None


def add_audio_from_episode(video_path, anime_name, episode_num, start_sec):
    """
    PATH B — Episode Source:
    Downloads the specific episode from Nyaa, waits for completion, slices audio.
    """
    video_duration = get_media_duration(video_path) or 10.0
    episodes_dir = "sakugabooru-episodes"
    os.makedirs(episodes_dir, exist_ok=True)
    output_path = video_path.replace(".mp4", "_raw_audio.mp4")

    print(f"[B] Episode source: '{anime_name}' Episode {episode_num}")

    # Check local first
    local_ep = find_local_episode_file(anime_name, episode_num=episode_num, require_complete=True)

    if not local_ep:
        print(f"[B] Not found locally. Searching Nyaa for episode {episode_num}...")
        torrent = search_and_download_nyaa(anime_name, episode_num=episode_num,
                                           output_dir=episodes_dir, auto_open=True)
        if not torrent:
            print("[B] Nyaa search failed — no torrent found.")
            return None

        print(f"[B] Polling for download (up to {DOWNLOAD_POLL_TIMEOUT//60}min)...")
        poll_start = time.time()
        last_log = 0
        while (time.time() - poll_start) < DOWNLOAD_POLL_TIMEOUT:
            elapsed = int(time.time() - poll_start)
            qb = get_qbittorrent_progress(anime_name.split()[0])
            if qb is not None and elapsed - last_log >= 60:
                print(f"[B] qBittorrent: {qb*100:.1f}% ({elapsed//60}min)")
                last_log = elapsed

            candidate = find_local_episode_file(anime_name, episode_num=episode_num, require_complete=True)
            if candidate:
                local_ep = candidate
                print(f"[B] ✅ Download complete: {local_ep}")
                break
            time.sleep(DOWNLOAD_POLL_INTERVAL)
        else:
            print("[B] Download timed out.")
            return None

    if not local_ep:
        return None

    slice_start = start_sec if start_sec is not None else 0.0
    print(f"[B] Slicing audio from {slice_start:.2f}s for {video_duration:.1f}s...")

    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(slice_start),
            "-i", local_ep,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-t", str(video_duration),
            output_path
        ]
        res_ff = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if res_ff.returncode == 0 and os.path.exists(output_path):
            print(f"[B] ✅ Episode audio merged: {output_path}")
            return output_path
        else:
            print(f"[!] ffmpeg failed: {res_ff.stderr[-300:]}")
    except Exception as e:
        print(f"[!] ffmpeg error: {e}")

    return None


# ─── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def fetch_and_add_audio(video_path, anime_name, source_url=None):
    """
    Three-path audio engine:

    PATH A — YouTube source URL on post → yt-dlp direct download
    PATH B — Episode number in source   → Nyaa torrent search
    PATH C — No source info             → Trace.moe fallback (best effort)
    """
    print(f"\n[*] Audio Engine: '{anime_name}' | source='{source_url}'")

    output_path = video_path.replace(".mp4", "_raw_audio.mp4")

    # ── PATH A: YouTube ──────────────────────────────────────────────────────
    if is_youtube_source(source_url):
        result = add_audio_from_youtube(video_path, source_url)
        if result:
            return result
        print("[A] YouTube path failed — falling through to Trace.moe fallback")

    # ── PATH B: Episode number in source field ───────────────────────────────
    episode_num = parse_episode_from_source(source_url)
    if episode_num is not None:
        print(f"[B] Source field has episode number: {episode_num}")
        # Try Trace.moe for timestamp within episode (best effort, not required)
        start_sec, _, sim = fetch_timestamp_from_tracemoe(video_path)
        if sim < TRACE_MIN_SIMILARITY:
            start_sec = None  # Will default to 0.0 in add_audio_from_episode
        result = add_audio_from_episode(video_path, anime_name, episode_num, start_sec)
        if result:
            return result
        print("[B] Episode path failed — falling through to Trace.moe fallback")

    # ── PATH C: Trace.moe fallback ───────────────────────────────────────────
    print("[C] No source info — trying Trace.moe as last resort...")
    start_sec, trace_filename, ep_num, sim = _trace_moe_full(video_path)

    if sim < TRACE_MIN_SIMILARITY or ep_num is None:
        print(f"[C] Trace.moe could not identify clip (similarity={sim:.2f}, ep={ep_num}) — saving without audio.")
        return video_path

    search_name = trace_filename or anime_name
    result = add_audio_from_episode(video_path, search_name, ep_num, start_sec)
    if result:
        return result

    print("[!] All audio paths exhausted — saving clip without audio.")
    return video_path


def _trace_moe_full(video_path):
    """Full Trace.moe call returning (start_sec, filename, ep_num, similarity)."""
    start_sec, ep_num, sim = fetch_timestamp_from_tracemoe(video_path)
    if start_sec is None:
        return None, None, None, 0.0
    # Re-run to also get filename
    video_duration = get_media_duration(video_path) or 10.0
    seek_time = min(1.0, max(0.2, video_duration / 4.0))
    pid = os.getpid()
    frame_path = f"temp_frame2_{pid}.jpg"
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-ss', str(seek_time), '-i', video_path,
             '-vframes', '1', frame_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        with open(frame_path, 'rb') as f:
            res = requests.post('https://api.trace.moe/search', files={'image': f}, timeout=10)
        if res.status_code == 200:
            results = res.json().get('result', [])
            if results:
                match = results[0]
                return (
                    max(0.0, match.get('at', 0) - seek_time),
                    match.get('filename', ''),
                    match.get('episode'),
                    match.get('similarity', 0)
                )
    except Exception:
        pass
    finally:
        if os.path.exists(frame_path):
            os.remove(frame_path)
    return None, None, None, 0.0
