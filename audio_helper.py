import os
import glob
import subprocess
import requests
import xml.etree.ElementTree as ET
from nyaa_downloader import search_and_download_nyaa

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
            if results and results[0].get('similarity', 0) >= 0.70:
                match = results[0]
                at_sec = match.get('at', 0)
                from_sec = match.get('from', 0)
                ep_num = match.get('episode')
                filename = match.get('filename', '')

                # Calculate start_sec for t=0.0s
                start_sec = max(0.0, (at_sec - seek_time))
                return start_sec, filename, ep_num, match

    except Exception as e:
        print("[!] Trace.moe timestamp query notice:", e)
    finally:
        if os.path.exists(frame_path):
            os.remove(frame_path)

    return None, None, None, None

def find_local_episode_file(anime_name, episodes_dir="sakugabooru-episodes"):
    """
    Searches episodes_dir for local .mkv / .mp4 / .avi files matching anime_name.
    """
    if not os.path.exists(episodes_dir):
        return None

    video_extensions = ['*.mkv', '*.mp4', '*.avi']
    local_files = []
    for ext in video_extensions:
        local_files.extend(glob.glob(os.path.join(episodes_dir, ext)))
        local_files.extend(glob.glob(os.path.join(episodes_dir, '**', ext), recursive=True))

    if not local_files:
        return None

    clean_search = anime_name.lower().replace('_', ' ').replace('-', ' ').strip()
    words = [w for w in clean_search.split() if len(w) > 2 and w not in ('the', 'and', 'for', 'series')]
    for f in local_files:
        filename_lower = os.path.basename(f).lower()
        if words and any(w in filename_lower for w in words):
            return f

    return None

def fetch_and_add_audio(video_path, anime_name):
    """
    100% Automated Local Episode Audio Engine:
    1. Checks sakugabooru-episodes/ for local .mkv / .mp4 file.
    2. If found, slices exact 100% studio master audio track using ffmpeg.
    3. If not found, downloads Nyaa .torrent file and auto-launches client.
    """
    video_duration = get_media_duration(video_path) or 10.0
    episodes_dir = "sakugabooru-episodes"
    os.makedirs(episodes_dir, exist_ok=True)

    print(f"[*] Audio Engine Processing Anime: '{anime_name}' (Clip Duration: {video_duration:.1f}s)")

    start_sec, trace_filename, ep_num, match_info = fetch_scene_timestamp(video_path)
    search_name = trace_filename or anime_name

    # Step 1: Check for local episode file in sakugabooru-episodes/
    local_ep_file = find_local_episode_file(search_name, episodes_dir=episodes_dir)
    raw_output_path = video_path.replace(".mp4", "_raw_audio.mp4")

    if local_ep_file and os.path.exists(local_ep_file):
        print(f"[*] Local Episode File Matched: {local_ep_file}")
        slice_start = start_sec if start_sec is not None else 0.0
        print(f"[*] Slicing 100% Studio Master Audio from {slice_start:.2f}s for {video_duration:.1f}s...")

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(slice_start),
                "-i", local_ep_file,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-t", str(video_duration),
                raw_output_path
            ]
            res_ff = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res_ff.returncode == 0 and os.path.exists(raw_output_path):
                print(f"[+] 100% Studio Audio Video Saved: {raw_output_path}")
                return raw_output_path
        except Exception as e:
            print("[!] Local audio slicing error:", e)

    # Step 2: If no local episode file, fetch exact episode Nyaa .torrent file
    print(f"[!] No local episode file found in '{episodes_dir}/'. Automated Nyaa Search for Episode {ep_num}...")
    torrent_file = search_and_download_nyaa(search_name, episode_num=ep_num, output_dir=episodes_dir, auto_open=True)

    if torrent_file:
        print(f"[+] Automated Torrent Saved & Opened: '{torrent_file}'")
    else:
        print(f"[!] Place episode MKV/MP4 into '{episodes_dir}/' and re-run main.py to slice audio.")

    return video_path
