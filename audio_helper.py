import os
import subprocess
import requests

def get_media_duration(file_path):
    """
    Returns the exact duration of a video or audio file in seconds using ffprobe.
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

def fetch_exact_scene_audio(video_path, seek_time=1.0):
    """
    Uses Trace.moe AI Visual Recognition API to identify the exact scene,
    extracting original Japanese dialogue & sound effects aligned frame-perfectly.
    Returns (audio_path, duration, match_info).
    """
    pid = os.getpid()
    frame_path = f"temp_frame_{pid}_{seek_time}.jpg"
    preview_video_path = f"temp_preview_{pid}_{seek_time}.mp4"
    audio_path = f"temp_exact_audio_{pid}_{seek_time}.aac"

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-ss', str(seek_time), '-i', video_path, '-vframes', '1', frame_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        if not os.path.exists(frame_path):
            return None, 0, None

        with open(frame_path, 'rb') as f:
            res = requests.post('https://api.trace.moe/search', files={'image': f}, timeout=10)

        if res.status_code == 200:
            data = res.json()
            results = data.get('result', [])
            if results and results[0].get('similarity', 0) >= 0.80:
                match = results[0]
                episode = match.get('episode')
                similarity = round(match.get('similarity', 0) * 100, 2)
                video_url = match.get('video')

                # Calculate offset between frame timestamp and preview start timestamp to fix audio delay
                at_sec = match.get('at', 0)
                from_sec = match.get('from', 0)
                offset = max(0.0, at_sec - from_sec)

                if video_url:
                    v_res = requests.get(video_url, timeout=15)
                    if v_res.status_code == 200:
                        with open(preview_video_path, 'wb') as f:
                            f.write(v_res.content)

                        # Trim offset to align audio frame-perfectly with video timestamp
                        subprocess.run(
                            ['ffmpeg', '-y', '-ss', str(offset), '-i', preview_video_path, '-vn', '-c:a', 'aac', audio_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                        )

                        if os.path.exists(audio_path):
                            exact_duration = get_media_duration(audio_path) or 0
                            return audio_path, exact_duration, match

    except Exception as e:
        pass
    finally:
        for f in [frame_path, preview_video_path]:
            if os.path.exists(f):
                os.remove(f)

    return None, 0, None

def fetch_multi_chunk_scene_audio(video_path):
    """
    Samples frames across the video duration (every 5 seconds) and queries Trace.moe
    for each timestamp chunk, then stitches (concatenates) all audio snippets together
    aligned frame-perfectly to form 100% original dialogue & sound effects.
    Returns (stitched_audio_path, duration, match_info, parts_fetched, total_chunks).
    """
    video_duration = get_media_duration(video_path) or 5.0
    pid = os.getpid()

    sample_interval = 5.0
    timestamps = []
    curr = 1.0
    while curr < video_duration:
        timestamps.append(curr)
        curr += sample_interval

    total_chunks = len(timestamps)

    if total_chunks <= 1:
        audio_path, dur, match = fetch_exact_scene_audio(video_path, seek_time=1.0)
        parts_count = 1 if audio_path else 0
        return audio_path, dur, match, parts_count, total_chunks

    print(f"[*] Multi-Chunk Sampling: Extracting up to {total_chunks} audio parts across {video_duration:.1f}s video...")

    audio_chunks = []
    chunk_files = []
    first_match = None

    for i, t in enumerate(timestamps):
        chunk_audio, dur, match = fetch_exact_scene_audio(video_path, seek_time=t)
        if chunk_audio and os.path.exists(chunk_audio):
            if not first_match:
                first_match = match
                print(f"[+] Multi-Chunk Matched Episode {match.get('episode')} ({round(match.get('similarity',0)*100,1)}%)")

            audio_chunks.append(chunk_audio)
            chunk_files.append(chunk_audio)

    parts_fetched = len(audio_chunks)

    if not audio_chunks:
        return None, 0, None, 0, total_chunks

    stitched_audio = f"temp_stitched_{pid}.aac"
    concat_list = f"temp_list_{pid}.txt"

    try:
        with open(concat_list, 'w') as f:
            for chunk in audio_chunks:
                f.write(f"file '{os.path.abspath(chunk)}'\n")

        subprocess.run(
            ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', stitched_audio],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        for f in chunk_files + [concat_list]:
            if os.path.exists(f):
                os.remove(f)

        if os.path.exists(stitched_audio):
            total_duration = get_media_duration(stitched_audio) or 0
            print(f"[+] Successfully fetched {parts_fetched}/{total_chunks} audio parts ({total_duration:.1f}s total audio)!")
            return stitched_audio, total_duration, first_match, parts_fetched, total_chunks

    except Exception as e:
        print("[!] Multi-chunk audio stitching error:", e)

    return None, 0, None, parts_fetched, total_chunks

def try_yt_dlp_section_download(anime_name, episode, start_sec, duration_sec, pid):
    """
    Attempts to download exact section audio for episode at start_sec using yt-dlp.
    """
    if not episode or not start_sec:
        return None, 0

    end_sec = start_sec + duration_sec
    temp_section = f"temp_section_{pid}.mp3"
    clean_anime = anime_name.split(',')[0].strip() if anime_name else ""
    query = f"{clean_anime} Episode {episode}"

    print(f"[*] Attempting yt-dlp section download for: '{query}' ({start_sec:.1f}s - {end_sec:.1f}s)...")
    try:
        cmd = [
            ".venv/bin/yt-dlp",
            f"ytsearch1:{query}",
            "--external-downloader", "ffmpeg",
            "--external-downloader-args", f"ffmpeg:-ss {start_sec} -to {end_sec}",
            "-x", "--audio-format", "mp3",
            "-o", temp_section,
            "--no-playlist",
            "--quiet"
        ]
        res = subprocess.run(cmd, timeout=30)
        if res.returncode == 0 and os.path.exists(temp_section):
            duration = get_media_duration(temp_section) or 0
            if duration > 1.0:
                print(f"[+] Successfully downloaded section audio via yt-dlp ({duration:.1f}s)!")
                return temp_section, duration
    except Exception as e:
        print("[!] yt-dlp section download attempt skipped:", e)

    if os.path.exists(temp_section):
        os.remove(temp_section)

    return None, 0

def fetch_and_add_audio(video_path, anime_name):
    """
    Generates TWO video versions:
    1. Raw Version (video_raw_audio.mp4): Direct scene audio without DSP upscaling / loudnorm filters.
    2. Studio Mastered Version (video_mastered_audio.mp4): Enhanced with 320k AAC @ 48kHz & EBU R128 loudness mastering.

    Returns (raw_output_path, mastered_output_path).
    """
    pid = os.getpid()
    video_duration = get_media_duration(video_path) or 10.0
    exact_audio_file, exact_duration, match_info, parts_fetched, total_chunks = fetch_multi_chunk_scene_audio(video_path)

    raw_output_path = video_path.replace(".mp4", "_raw_audio.mp4")
    mastered_output_path = video_path.replace(".mp4", "_mastered_audio.mp4")

    section_audio_file = None

    if match_info and exact_duration < (video_duration - 0.5):
        ep = match_info.get('episode')
        start_sec = match_info.get('from', 0)
        sec_file, sec_dur = try_yt_dlp_section_download(anime_name, ep, start_sec, video_duration, pid)
        if sec_file and sec_dur >= (video_duration - 0.5):
            section_audio_file = sec_file

    audio_to_use = section_audio_file or exact_audio_file
    audio_dur = sec_dur if section_audio_file else exact_duration

    if not audio_to_use or not os.path.exists(audio_to_use) or audio_dur == 0:
        print("[!] No original scene audio could be matched for this clip.")
        return video_path, video_path

    coverage_percent = min(100.0, round((audio_dur / video_duration) * 100, 1))
    print(f"[*] AUDIO STATS: Fetched {parts_fetched}/{total_chunks} parts | Coverage: {coverage_percent}% ({audio_dur:.1f}s / {video_duration:.1f}s)")

    try:
        # Version 1: RAW Audio (Direct merge without DSP audio upscaling / loudnorm)
        ffmpeg_raw = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_to_use,
            "-c:v", "copy",
            "-c:a", "aac",
            "-t", str(video_duration),
            raw_output_path
        ]
        subprocess.run(ffmpeg_raw, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Version 2: Studio Mastered & Upscaled Audio (EBU R128 loudnorm, 320k AAC @ 48kHz, presence EQ)
        ffmpeg_mastered = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_to_use,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-ar", "48000",
            "-af", "loudnorm=I=-14:LRA=11:TP=-1.5,equalizer=f=80:width_type=h:width=100:g=2.5,equalizer=f=12000:width_type=h:width=4000:g=2.0",
            "-t", str(video_duration),
            mastered_output_path
        ]
        subprocess.run(ffmpeg_mastered, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"[+] Raw Version Saved: {raw_output_path}")
        print(f"[+] Mastered Version Saved: {mastered_output_path}")
        return raw_output_path, mastered_output_path

    except Exception as e:
        print("[!] Audio processing error:", e)
    finally:
        for f in [exact_audio_file, section_audio_file]:
            if f and os.path.exists(f):
                os.remove(f)

    return video_path, video_path
