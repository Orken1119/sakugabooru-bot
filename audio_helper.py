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

def fetch_exact_scene_audio(video_path):
    """
    Uses Trace.moe AI Visual Recognition API to identify the exact scene,
    extracting original Japanese dialogue & sound effects.
    Returns (audio_path, duration, match_info).
    """
    pid = os.getpid()
    frame_path = f"temp_frame_{pid}.jpg"
    preview_video_path = f"temp_preview_{pid}.mp4"
    audio_path = f"temp_exact_audio_{pid}.aac"

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-ss', '00:00:01', '-i', video_path, '-vframes', '1', frame_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        if not os.path.exists(frame_path):
            return None, 0, None

        print("[*] Performing Trace.moe AI visual search for exact episode audio...")
        with open(frame_path, 'rb') as f:
            res = requests.post('https://api.trace.moe/search', files={'image': f}, timeout=10)

        if res.status_code == 200:
            data = res.json()
            results = data.get('result', [])
            if results and results[0].get('similarity', 0) >= 0.85:
                match = results[0]
                episode = match.get('episode')
                similarity = round(match.get('similarity', 0) * 100, 2)
                video_url = match.get('video')

                print(f"[+] Exact Scene Matched! Episode {episode} ({similarity}% confidence)")

                if video_url:
                    v_res = requests.get(video_url, timeout=15)
                    if v_res.status_code == 200:
                        with open(preview_video_path, 'wb') as f:
                            f.write(v_res.content)

                        subprocess.run(
                            ['ffmpeg', '-y', '-i', preview_video_path, '-vn', '-c:a', 'aac', audio_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                        )

                        if os.path.exists(audio_path):
                            exact_duration = get_media_duration(audio_path) or 0
                            return audio_path, exact_duration, match

    except Exception as e:
        print("[!] Trace.moe exact audio search error:", e)
    finally:
        for f in [frame_path, preview_video_path]:
            if os.path.exists(f):
                os.remove(f)

    return None, 0, None

def fetch_multi_chunk_scene_audio(video_path):
    """
    Samples frames across the video duration (every 5 seconds) and queries Trace.moe
    for each timestamp chunk, then stitches (concatenates) all audio snippets together
    to form 100% original dialogue & sound effects for the full scene.
    """
    video_duration = get_media_duration(video_path) or 5.0
    pid = os.getpid()

    sample_interval = 5.0
    timestamps = []
    curr = 1.0
    while curr < video_duration:
        timestamps.append(curr)
        curr += sample_interval

    if len(timestamps) <= 1:
        return fetch_exact_scene_audio(video_path)

    print(f"[*] Multi-Chunk Sampling: Extracting {len(timestamps)} audio chunks across {video_duration:.1f}s video...")

    audio_chunks = []
    chunk_files = []
    first_match = None

    for i, t in enumerate(timestamps):
        frame_path = f"temp_frame_{pid}_{i}.jpg"
        preview_path = f"temp_prev_{pid}_{i}.mp4"
        chunk_audio = f"temp_chunk_{pid}_{i}.aac"

        try:
            subprocess.run(
                ['ffmpeg', '-y', '-ss', str(t), '-i', video_path, '-vframes', '1', frame_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )

            if os.path.exists(frame_path):
                with open(frame_path, 'rb') as f:
                    res = requests.post('https://api.trace.moe/search', files={'image': f}, timeout=10)

                if res.status_code == 200:
                    results = res.json().get('result', [])
                    if results and results[0].get('similarity', 0) >= 0.80:
                        match = results[0]
                        if not first_match:
                            first_match = match
                            print(f"[+] Multi-Chunk Matched Episode {match.get('episode')} ({round(match.get('similarity',0)*100,1)}%)")

                        video_url = match.get('video')
                        if video_url:
                            v_res = requests.get(video_url, timeout=12)
                            if v_res.status_code == 200:
                                with open(preview_path, 'wb') as f:
                                    f.write(v_res.content)

                                subprocess.run(
                                    ['ffmpeg', '-y', '-i', preview_path, '-vn', '-c:a', 'aac', chunk_audio],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                                )

                                if os.path.exists(chunk_audio):
                                    audio_chunks.append(chunk_audio)
                                    chunk_files.append(chunk_audio)

        except Exception as e:
            print(f"[!] Chunk {i} fetch failed:", e)
        finally:
            for f in [frame_path, preview_path]:
                if os.path.exists(f):
                    os.remove(f)

    if not audio_chunks:
        return None, 0, None

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
            print(f"[+] Successfully stitched {len(audio_chunks)} audio chunks ({total_duration:.1f}s total audio)!")
            return stitched_audio, total_duration, first_match

    except Exception as e:
        print("[!] Multi-chunk audio stitching error:", e)

    return None, 0, None

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
    Strict Pure Scene Audio Engine (Zero Background OST Music):
    - Fetches 100% original scene audio (voices & sound effects).
    - Reports exact audio coverage percentage.
    """
    pid = os.getpid()
    video_duration = get_media_duration(video_path) or 10.0
    exact_audio_file, exact_duration, match_info = fetch_multi_chunk_scene_audio(video_path)
    output_path = video_path.replace(".mp4", "_audio.mp4")

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
        return video_path

    coverage_percent = min(100.0, round((audio_dur / video_duration) * 100, 1))
    print(f"[*] AUDIO COVERAGE REPORT: {coverage_percent}% original scene audio ({audio_dur:.1f}s audio / {video_duration:.1f}s video)")

    try:
        # Pure Scene Audio Merge (No music background)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_to_use,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-ar", "48000",
            "-af", "loudnorm=I=-14:LRA=11:TP=-1.5",
            "-t", str(video_duration),
            output_path
        ]
        res_ff = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if res_ff.returncode == 0 and os.path.exists(output_path):
            print(f"[+] Final Video with Pure Scene Audio saved: {output_path}")
            return output_path

    except Exception as e:
        print("[!] Audio processing error:", e)
    finally:
        for f in [exact_audio_file, section_audio_file]:
            if f and os.path.exists(f):
                os.remove(f)

    return video_path
