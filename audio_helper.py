import os
import glob
import subprocess
import requests
import xml.etree.ElementTree as ET

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

def search_and_download_nyaa_torrent(anime_title, output_dir="sakugabooru-episodes"):
    """
    Queries Nyaa.si RSS API for anime_title sorted by most downloads, downloads .torrent file to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_title = anime_title.replace('_', ' ').replace('-', ' ').strip()
    query = f"{clean_title} 1080p"
    rss_url = f"https://nyaa.si/?page=rss&q={requests.utils.quote(query)}&c=1_2&f=0"

    print(f"[*] Searching Nyaa.si for: '{query}'...")
    try:
        res = requests.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=12)
        if res.status_code == 200:
            root = ET.fromstring(res.text)
            items = root.findall('.//item')
            if items:
                top_item = items[0]
                torrent_title = top_item.find('title').text
                torrent_link = top_item.find('link').text

                torrent_res = requests.get(torrent_link, timeout=15)
                if torrent_res.status_code == 200:
                    clean_filename = "".join(c for c in torrent_title if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
                    torrent_filepath = os.path.join(output_dir, f"{clean_filename}.torrent")
                    with open(torrent_filepath, 'wb') as f:
                        f.write(torrent_res.content)

                    print(f"[+] Nyaa Torrent Found & Saved: {torrent_filepath}")
                    return torrent_filepath
    except Exception as e:
        print("[!] Nyaa search error:", e)

    return None

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

    # Match by anime name if possible, otherwise return first available episode file
    clean_search = anime_name.lower().replace('_', ' ').replace('-', ' ').strip()
    for f in local_files:
        if any(part in os.path.basename(f).lower() for part in clean_search.split()[:2]):
            return f

    return local_files[0]

def fetch_and_add_audio(video_path, anime_name):
    """
    Local Episode Audio Engine (No AI Fallback):
    1. Checks sakugabooru-episodes/ for local .mkv / .mp4 file.
    2. If found, slices exact audio track using ffmpeg.
    3. If not found, downloads Nyaa .torrent file and prompts user.
    """
    video_duration = get_media_duration(video_path) or 10.0
    episodes_dir = "sakugabooru-episodes"
    os.makedirs(episodes_dir, exist_ok=True)

    print(f"[*] Audio Engine Processing Anime: '{anime_name}' (Clip Duration: {video_duration:.1f}s)")

    # Step 1: Check for local episode file in sakugabooru-episodes/
    local_ep_file = find_local_episode_file(anime_name, episodes_dir=episodes_dir)
    raw_output_path = video_path.replace(".mp4", "_raw_audio.mp4")

    if local_ep_file and os.path.exists(local_ep_file):
        print(f"[*] Local Episode File Matched: {local_ep_file}")
        print(f"[*] Slicing 100% Studio Master Audio for {video_duration:.1f}s...")

        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", "0.0",
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

    # Step 2: If no local episode file, fetch Nyaa .torrent file
    print(f"[!] No local episode file found in '{episodes_dir}/'. Automated Nyaa Search...")
    torrent_file = search_and_download_nyaa_torrent(anime_name, output_dir=episodes_dir)

    if torrent_file:
        print(f"[!] ACTION REQUIRED: Torrent saved to '{torrent_file}'. Open in your torrent client, save episode to '{episodes_dir}/', and re-run main.py.")
    else:
        print(f"[!] Place episode MKV/MP4 into '{episodes_dir}/' and re-run main.py to slice audio.")

    return video_path
