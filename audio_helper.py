import os
import re
import glob
import subprocess

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

def fetch_and_add_audio(video_path, anime_name, source_url=None):
    """
    Direct Audio Pass-Through:
    Skipping Nyaa torrent searches and episode scraping completely as requested.
    Returns video_path as-is so main.py immediately adds the Harmonic OST!
    """
    print(f"[*] Audio Engine: Passing '{anime_name}' directly to Harmonic OST Engine.")
    return video_path
