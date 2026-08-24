import os
import re
import glob
import struct
import subprocess

MUSIC_CACHE_DIR = "music_cache"
OST_CACHE_DIR   = os.path.join(MUSIC_CACHE_DIR, "ost")

# OST search queries per energy level
OST_QUERIES = {
    "high":   "{anime} OST battle intense fight theme soundtrack",
    "medium": "{anime} OST theme soundtrack",
    "low":    "{anime} OST emotional piano calm soundtrack",
}


# ─── STEP 1: Motion Level Detection ──────────────────────────────────────────

def get_motion_level(video_path):
    """
    Analyzes how much motion is in a clip using ffmpeg scene change detection.
    Returns 'high', 'medium', or 'low'.
    """
    try:
        # showinfo filter prints per-frame info including scene change score
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', 'showinfo',
            '-f', 'null', '-'
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Extract scene change scores from stderr (format: "iskey:1 type:I ... pos:0")
        # showinfo doesn't give scene scores directly — use select filter instead
        cmd2 = [
            'ffmpeg', '-i', video_path,
            '-vf', 'select=gt(scene\\,0.01)',
            '-vsync', 'vfr',
            '-f', 'null', '-'
        ]
        res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)

        # Count how many frames passed the scene-change filter
        selected_frames = 0
        for line in res2.stderr.split('\n'):
            if 'frame=' in line and 'fps=' in line:
                m = re.search(r'frame=\s*(\d+)', line)
                if m:
                    selected_frames = int(m.group(1))

        # Get total duration to compute changes-per-second rate
        dur_cmd = ['ffprobe', '-v', 'error',
                   '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        dur_res = subprocess.run(dur_cmd, capture_output=True, text=True)
        duration = float(dur_res.stdout.strip()) if dur_res.stdout.strip() else 30.0

        rate = selected_frames / max(duration, 1.0)
        print(f"[motion] {selected_frames} scene-change frames in {duration:.1f}s → rate={rate:.2f}/s")

        if rate > 3.0:
            return 'high'
        elif rate < 0.8:
            return 'low'
        else:
            return 'medium'

    except Exception as e:
        print(f"[!] Motion detection fallback (medium): {e}")
        return 'medium'


# ─── STEP 2: Download Anime OST ───────────────────────────────────────────────

def _ost_cache_path(anime_name):
    """Returns expected cache file path for this anime's OST."""
    os.makedirs(OST_CACHE_DIR, exist_ok=True)
    safe = re.sub(r'[^\w\s-]', '', anime_name.split(',')[0].strip().lower())
    safe = re.sub(r'\s+', '_', safe)[:40]
    # Check if any file already exists for this anime
    existing = glob.glob(os.path.join(OST_CACHE_DIR, f"{safe}*.m4a"))
    return existing[0] if existing else os.path.join(OST_CACHE_DIR, f"{safe}_ost.m4a")


def download_anime_ost(anime_name, motion_level):
    """
    Downloads the anime's OST from YouTube via yt-dlp.
    Caches per-anime so the same anime always reuses the same track.
    Returns filepath or None.
    """
    # Check cache first
    cached = _ost_cache_path(anime_name)
    existing = glob.glob(os.path.join(OST_CACHE_DIR, "*.m4a"))
    for f in existing:
        anime_slug = re.sub(r'[^\w\s-]', '', anime_name.split(',')[0].strip().lower())
        anime_slug = re.sub(r'\s+', '_', anime_slug)[:20]
        if anime_slug in os.path.basename(f).lower():
            print(f"[♪] Using cached OST: {os.path.basename(f)}")
            return f

    # Build search query from anime name + energy level
    clean_anime = anime_name.split(',')[0].strip().title()
    query = OST_QUERIES[motion_level].format(anime=clean_anime)
    print(f"[♪] Searching OST ({motion_level} energy): '{query}'")

    try:
        cmd = [
            '.venv/bin/yt-dlp',
            f'ytsearch1:{query}',
            '--quiet', '--no-warnings',
            '-x',
            '--audio-format', 'm4a',
            '--audio-quality', '128K',
            '-o', os.path.join(OST_CACHE_DIR, '%(title).50s_%(id)s.%(ext)s'),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Find the newest file in OST cache
        files = glob.glob(os.path.join(OST_CACHE_DIR, "*.m4a"))
        if files:
            newest = max(files, key=os.path.getmtime)
            print(f"[♪] Downloaded OST: {os.path.basename(newest)}")
            return newest

    except Exception as e:
        print(f"[!] OST download error: {e}")

    return None


# ─── STEP 3: Audio Energy Profile ────────────────────────────────────────────

def get_audio_rms_per_second(audio_path):
    """
    Extracts raw PCM from audio file and computes RMS per second in pure Python.
    Returns list of (second_index, rms_value) tuples.
    No numpy needed — uses struct + integer math.
    """
    SAMPLE_RATE = 22050   # lower rate = faster decode, good enough for energy
    CHANNELS    = 1       # mono

    try:
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-f', 's16le',              # raw 16-bit signed little-endian PCM
            '-ar', str(SAMPLE_RATE),
            '-ac', str(CHANNELS),
            '-'
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        raw = proc.stdout

        samples_per_sec = SAMPLE_RATE * CHANNELS
        bytes_per_sec   = samples_per_sec * 2  # 16-bit = 2 bytes

        rms_profile = []
        for sec in range(len(raw) // bytes_per_sec):
            chunk = raw[sec * bytes_per_sec : (sec + 1) * bytes_per_sec]
            samples = struct.unpack(f'<{len(chunk)//2}h', chunk)
            rms = (sum(s * s for s in samples) / max(len(samples), 1)) ** 0.5
            rms_profile.append((sec, rms))

        return rms_profile

    except Exception as e:
        print(f"[!] RMS profile error: {e}")
        return []


# ─── STEP 4: Find Best Matching Segment ──────────────────────────────────────

def find_best_ost_offset(rms_profile, clip_duration, motion_level):
    """
    Finds the start offset in the OST whose energy level best matches
    the clip's motion level using a sliding window over the RMS profile.

    - high motion  → aim for the 75th percentile loudest window
    - medium        → aim for the 50th percentile (median)
    - low motion   → aim for the 25th percentile quietest window
    """
    if not rms_profile:
        return 0.0

    window  = max(1, int(clip_duration))
    rms_vals = [r for _, r in rms_profile]
    n        = len(rms_vals)

    if n <= window:
        return 0.0

    # Compute per-window average RMS using a sliding sum
    window_avgs = []
    window_sum = sum(rms_vals[:window])
    window_avgs.append(window_sum / window)
    for i in range(1, n - window):
        window_sum += rms_vals[i + window - 1] - rms_vals[i - 1]
        window_avgs.append(window_sum / window)

    # Sort window averages to find target percentile
    sorted_avgs = sorted(window_avgs)
    pct_map = {'high': 0.75, 'medium': 0.50, 'low': 0.25}
    target_idx = int(len(sorted_avgs) * pct_map.get(motion_level, 0.5))
    target_rms  = sorted_avgs[min(target_idx, len(sorted_avgs) - 1)]

    # Find the window whose average is closest to target
    best_offset = 0
    best_diff   = float('inf')
    for i, avg in enumerate(window_avgs):
        diff = abs(avg - target_rms)
        if diff < best_diff:
            best_diff   = diff
            best_offset = i

    print(f"[♪] Best OST offset: {best_offset}s (motion={motion_level}, "
          f"target_rms={target_rms:.0f}, found_rms={window_avgs[best_offset]:.0f})")
    return float(best_offset)


# ─── MAIN ENTRY: Add OST to Video ────────────────────────────────────────────

def add_ost_to_video(video_path, anime_name, output_path=None):
    """
    Full harmonic OST sync pipeline:
    1. Detect clip motion level (high/medium/low)
    2. Search and download matching anime OST from YouTube
    3. Compute per-second RMS energy profile of OST
    4. Find the OST segment whose energy matches the clip's motion
    5. Merge with 0.5s fade-in and 1.0s fade-out

    Returns output path or None.
    """
    if output_path is None:
        output_path = video_path.replace(".mp4", "_ost.mp4")

    # Get video duration
    try:
        dur_res = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, check=True
        )
        video_duration = float(dur_res.stdout.strip())
    except Exception:
        video_duration = 30.0

    print(f"\n[♪] Harmonic OST Engine | clip={video_duration:.1f}s | anime='{anime_name}'")

    # Step 1: Motion level
    motion = get_motion_level(video_path)
    print(f"[♪] Clip energy level: {motion.upper()}")

    # Step 2: Download OST
    ost_path = download_anime_ost(anime_name, motion)
    if not ost_path or not os.path.exists(ost_path):
        print("[!] Could not get OST — skipping audio.")
        return None

    # Step 3: RMS profile
    print(f"[♪] Analyzing OST energy profile...")
    rms_profile = get_audio_rms_per_second(ost_path)
    if not rms_profile:
        ost_offset = 0.0
    else:
        # Step 4: Find best matching segment
        ost_offset = find_best_ost_offset(rms_profile, video_duration, motion)

    # Step 5: Merge with fade
    fade_out_start = max(0.0, video_duration - 1.0)
    print(f"[♪] Merging OST at offset {ost_offset:.1f}s with fade in/out...")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(ost_offset),
            "-i", ost_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-af", f"afade=t=in:st=0:d=0.5,afade=t=out:st={fade_out_start:.2f}:d=1.0",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(video_duration),
            output_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(output_path):
            print(f"[♪] ✅ OST merged harmonically: {output_path}")
            return output_path
        else:
            print(f"[!] ffmpeg merge failed: {res.stderr[-300:]}")
    except Exception as e:
        print(f"[!] Merge error: {e}")

    return None
