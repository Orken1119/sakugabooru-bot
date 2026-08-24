import os
import glob
import time
import subprocess
import requests
import xml.etree.ElementTree as ET
from nyaa_downloader import search_and_download_nyaa

# ─────────────────────────────────────────────
# Minimum confidence threshold for Trace.moe match
# Below this, we can't trust the episode/timestamp
TRACE_MIN_SIMILARITY = 0.85

# Max poll time in seconds when waiting for qBittorrent to finish
DOWNLOAD_POLL_TIMEOUT = 7200  # 2 hours
DOWNLOAD_POLL_INTERVAL = 15   # check every 15 seconds
# ─────────────────────────────────────────────


def get_media_duration(file_path):
    """
    Returns exact duration of video or audio file in seconds using ffprobe.
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return None


def fetch_scene_timestamp(video_path):
    """
    Queries Trace.moe API to identify exact episode start timestamp and anime info.
    Returns (start_sec, anime_title, episode_num, match_info).
    Returns (None, None, None, None) if similarity < TRACE_MIN_SIMILARITY.
    """
    video_duration = get_media_duration(video_path) or 10.0
    seek_time = min(1.0, max(0.2, video_duration / 4.0))
    pid = os.getpid()
    frame_path = f"temp_frame_{pid}.jpg"

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-ss', str(seek_time), '-i', video_path, '-vframes', '1', frame_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        if not os.path.exists(frame_path):
            return None, None, None, None

        with open(frame_path, 'rb') as f:
            res = requests.post('https://api.trace.moe/search', files={'image': f}, timeout=10)

        if res.status_code == 200:
            data = res.json()
            results = data.get('result', [])
            if results:
                match = results[0]
                similarity = match.get('similarity', 0)

                # Bug #3 Fix: reject low-confidence matches — they produce wrong episode downloads
                if similarity < TRACE_MIN_SIMILARITY:
                    print(f"[!] Trace.moe similarity too low ({similarity:.2f} < {TRACE_MIN_SIMILARITY}) — skipping audio merge")
                    return None, None, None, None

                at_sec = match.get('at', 0)
                ep_num = match.get('episode')
                filename = match.get('filename', '')

                start_sec = max(0.0, (at_sec - seek_time))
                print(f"[+] Trace.moe match: '{filename}' ep={ep_num} at={at_sec:.1f}s similarity={similarity:.2f}")
                return start_sec, filename, ep_num, match

    except Exception as e:
        print("[!] Trace.moe timestamp query notice:", e)
    finally:
        if os.path.exists(frame_path):
            os.remove(frame_path)

    return None, None, None, None


def is_download_complete(filepath):
    """
    Fail-safe download completion check using ffprobe.
    Returns True ONLY when the file has a valid media container with real duration.
    Ignores pre-allocated sparse zero-byte files created by qBittorrent.
    """
    if not filepath or not os.path.exists(filepath):
        return False

    if filepath.endswith(('.!qB', '.qB', '.part', '.crdownload', '.tmp')):
        return False

    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filepath
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            dur = float(res.stdout.strip())
            return dur > 0.0
    except Exception:
        return False

    return False


def find_local_episode_file(anime_name, episode_num=None, search_dirs=None, require_complete=False):
    """
    Searches episodes_dir and Downloads for local .mkv / .mp4 / .avi files matching anime_name.

    Bug #4 Fix: requires MAJORITY of key words to match, not just any single word.
    This prevents "Bleach" from matching "bleach_movie_hell_verse.mkv" unintentionally.
    """
    if search_dirs is None:
        search_dirs = ["sakugabooru-episodes", "/home/orken/Downloads"]

    clean_search = anime_name.lower().replace('_', ' ').replace('-', ' ').strip()
    # Remove parts after comma (series tags from Sakugabooru)
    clean_search = clean_search.split(',')[0].strip()
    words = [w for w in clean_search.split() if len(w) > 2 and w not in ('the', 'and', 'for', 'series', 'sub')]

    if not words:
        return None

    # If we know episode number, also check for it in the filename
    ep_patterns = []
    if episode_num is not None:
        ep_str = f"{int(episode_num):02d}"
        ep_patterns = [f"e{ep_str}", f" {ep_str} ", f"_{ep_str}_", f"-{ep_str}-", f"ep{ep_str}"]

    video_extensions = ['*.mkv', '*.mp4', '*.avi']
    best_match = None
    best_score = 0

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        local_files = []
        for ext in video_extensions:
            local_files.extend(glob.glob(os.path.join(d, ext)))
            local_files.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))

        for f in local_files:
            filename_lower = os.path.basename(f).lower()

            # Bug #4 Fix: count how many key words match
            match_count = sum(1 for w in words if w in filename_lower)
            required = max(1, (len(words) + 1) // 2)  # need majority of words

            if match_count < required:
                continue

            # Bonus: if episode number also appears in filename, prefer this file
            ep_bonus = 1 if ep_patterns and any(p in filename_lower for p in ep_patterns) else 0
            score = match_count + ep_bonus

            if score > best_score:
                if not require_complete or is_download_complete(f):
                    best_match = f
                    best_score = score

    return best_match


def get_qbittorrent_progress(torrent_name_fragment):
    """
    Checks qBittorrent Web API for real download progress.
    Returns float 0.0–1.0, or None if API is unavailable.
    """
    try:
        res = requests.get(
            'http://localhost:8080/api/v2/torrents/info',
            timeout=3
        )
        if res.status_code == 200:
            for t in res.json():
                if torrent_name_fragment.lower() in t.get('name', '').lower():
                    return t.get('progress', 0.0)
    except Exception:
        pass
    return None


def fetch_and_add_audio(video_path, anime_name):
    """
    Automated Local Episode Audio Engine:
    1. Gets exact episode timestamp from Trace.moe (rejects low-confidence matches).
    2. Checks for verified local episode file.
    3. If missing, launches Nyaa torrent and polls (up to 2h) for download completion.
    4. Slices 100% studio audio from correct timestamp and merges into clip.
    """
    video_duration = get_media_duration(video_path) or 10.0
    episodes_dir = "sakugabooru-episodes"
    os.makedirs(episodes_dir, exist_ok=True)

    print(f"[*] Audio Engine: '{anime_name}' (clip={video_duration:.1f}s)")

    # Step 1: Trace.moe — get timestamp + episode number
    start_sec, trace_filename, ep_num, match_info = fetch_scene_timestamp(video_path)

    # Bug #3 Fix: if no confident match, skip audio merge entirely
    if match_info is None:
        print("[!] No confident Trace.moe match — saving clip without audio merge.")
        return video_path

    # Bug #3 Fix: if episode number is unknown, do NOT fire a broad Nyaa search
    if ep_num is None:
        print("[!] Trace.moe could not determine episode number (OP/ED/Movie scene) — skipping audio merge.")
        return video_path

    search_name = trace_filename or anime_name
    raw_output_path = video_path.replace(".mp4", "_raw_audio.mp4")

    # Step 2: Check for verified complete local episode
    local_ep_file = find_local_episode_file(search_name, episode_num=ep_num, require_complete=True)

    # Step 3: If missing, launch Nyaa and poll
    if not local_ep_file:
        print(f"[!] No verified local file. Launching Nyaa search for '{search_name}' Episode {ep_num}...")
        torrent_file = search_and_download_nyaa(search_name, episode_num=ep_num, output_dir=episodes_dir, auto_open=True)

        if torrent_file:
            torrent_basename = os.path.splitext(os.path.basename(torrent_file))[0]
            print(f"[*] Polling for download completion (max {DOWNLOAD_POLL_TIMEOUT//60} min, every {DOWNLOAD_POLL_INTERVAL}s)...")
            poll_start = time.time()
            last_log = 0

            while (time.time() - poll_start) < DOWNLOAD_POLL_TIMEOUT:
                elapsed = int(time.time() - poll_start)

                # Try qBittorrent Web API first (most accurate)
                qb_progress = get_qbittorrent_progress(search_name.split()[0])
                if qb_progress is not None:
                    if elapsed - last_log >= 60:
                        print(f"[*] qBittorrent progress: {qb_progress*100:.1f}% ({elapsed//60}min elapsed)")
                        last_log = elapsed
                    if qb_progress >= 1.0:
                        print("[+] qBittorrent reports 100% complete!")
                        break
                else:
                    # Fallback: check file with ffprobe
                    if elapsed - last_log >= 60:
                        print(f"[*] Waiting for download... ({elapsed//60}min elapsed)")
                        last_log = elapsed

                # Either way, check if file is usable
                candidate = find_local_episode_file(search_name, episode_num=ep_num, require_complete=True)
                if candidate:
                    local_ep_file = candidate
                    print(f"[+] Download verified via ffprobe: {local_ep_file}")
                    break

                time.sleep(DOWNLOAD_POLL_INTERVAL)
            else:
                print(f"[!] Download poll timed out after {DOWNLOAD_POLL_TIMEOUT//60} minutes.")

    # Step 4: Merge audio
    if local_ep_file and os.path.exists(local_ep_file):
        print(f"[*] Episode file: {local_ep_file}")
        slice_start = start_sec if start_sec is not None else 0.0
        print(f"[*] Slicing audio from {slice_start:.2f}s for {video_duration:.1f}s...")

        try:
            # Bug #5 Fix: -ss must be placed BEFORE the episode input, not between inputs
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,          # input 0: the sakugabooru clip (no seek)
                "-ss", str(slice_start),   # ← seek BEFORE episode input
                "-i", local_ep_file,       # input 1: episode file (seeking applied here)
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",           # video from clip
                "-map", "1:a:0?",          # audio from episode (optional — won't crash if missing)
                "-t", str(video_duration),
                raw_output_path
            ]
            res_ff = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if res_ff.returncode == 0 and os.path.exists(raw_output_path):
                print(f"[+] Audio merged successfully: {raw_output_path}")
                return raw_output_path
            else:
                print(f"[!] ffmpeg failed (code {res_ff.returncode}):\n{res_ff.stderr[-500:]}")
        except Exception as e:
            print("[!] Local audio slicing error:", e)

    print(f"[!] Could not merge audio. Place episode MKV in '{episodes_dir}/' and retry.")
    return video_path
