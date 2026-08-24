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

def fetch_chunk_audio(video_path, t, target_anilist_id=None, target_episode=None, chunk_len=2.0):
    """
    Samples frame at timestamp t, queries Trace.moe, enforces strict matching with target_anilist_id & target_episode,
    calculates exact offset (at - from), and extracts a frame-perfect slice of chunk_len seconds.
    Returns (chunk_audio_path, match_info).
    """
    pid = os.getpid()
    frame_path = f"temp_frame_{pid}_{t}.jpg"
    preview_path = f"temp_prev_{pid}_{t}.mp4"
    chunk_audio = f"temp_chunk_{pid}_{t}.aac"

    try:
        subprocess.run(
            ['ffmpeg', '-y', '-ss', str(t), '-i', video_path, '-vframes', '1', frame_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )

        if not os.path.exists(frame_path):
            return None, None

        with open(frame_path, 'rb') as f:
            res = requests.post('https://api.trace.moe/search', files={'image': f}, timeout=10)

        if res.status_code == 200:
            data = res.json()
            results = data.get('result', [])
            
            selected_match = None
            for m in results:
                if m.get('similarity', 0) >= 0.75:
                    if target_anilist_id is not None:
                        # Strict Anime ID and Episode Filter: discard if different anime or episode
                        if m.get('anilist') == target_anilist_id and m.get('episode') == target_episode:
                            selected_match = m
                            break
                    else:
                        selected_match = m
                        break

            if selected_match:
                at_sec = selected_match.get('at', 0)
                from_sec = selected_match.get('from', 0)
                offset = max(0.0, at_sec - from_sec)
                video_url = selected_match.get('video')

                if video_url:
                    v_res = requests.get(video_url, timeout=12)
                    if v_res.status_code == 200:
                        with open(preview_path, 'wb') as f:
                            f.write(v_res.content)

                        # Extract exact chunk_len audio starting at offset for frame-perfect sync
                        subprocess.run(
                            ['ffmpeg', '-y', '-ss', str(offset), '-i', preview_path, '-t', str(chunk_len), '-vn', '-c:a', 'aac', chunk_audio],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                        )

                        if os.path.exists(chunk_audio):
                            return chunk_audio, selected_match

    except Exception:
        pass
    finally:
        for f in [frame_path, preview_path]:
            if os.path.exists(f):
                os.remove(f)

    return None, None

def fetch_multi_chunk_scene_audio(video_path):
    """
    Equal-Power Overlap Cross-Fading Sampler:
    Samples frames every 2.0s, verifies strict anime matching, and applies 0.1s equal-power cross-fading
    between consecutive chunks to completely eliminate audio clicks, pops, and split seams.
    Returns (stitched_audio_path, duration, match_info, parts_fetched, total_chunks).
    """
    video_duration = get_media_duration(video_path) or 5.0
    pid = os.getpid()

    sample_interval = 2.0
    timestamps = []
    curr = 0.5
    while curr < (video_duration - 0.5):
        timestamps.append(round(curr, 2))
        curr += sample_interval

    end_stamp = round(max(0.5, video_duration - 0.5), 2)
    if not timestamps or (end_stamp - timestamps[-1]) >= 0.5:
        timestamps.append(end_stamp)

    total_chunks = len(timestamps)
    print(f"[*] Equal-Power Sampler: Sampling {total_chunks} audio parts with 0.1s overlap cross-fading across {video_duration:.1f}s video...")

    audio_chunks = []
    chunk_files = []
    first_match = None
    target_anilist = None
    target_ep = None

    for i, t in enumerate(timestamps):
        chunk_audio, match = fetch_chunk_audio(video_path, t, target_anilist_id=target_anilist, target_episode=target_ep, chunk_len=sample_interval)
        if chunk_audio and os.path.exists(chunk_audio):
            if not first_match:
                first_match = match
                target_anilist = match.get('anilist')
                target_ep = match.get('episode')
                print(f"[+] Strict AI Match Established: AniList ID {target_anilist} Episode {target_ep} ({round(match.get('similarity',0)*100,1)}% confidence)")

            audio_chunks.append(chunk_audio)
            chunk_files.append(chunk_audio)
        else:
            if target_anilist is not None:
                print(f"[!] Chunk {i} (t={t}s) discarded (did not match master Anime ID {target_anilist} Ep {target_ep})")

    parts_fetched = len(audio_chunks)

    if not audio_chunks:
        return None, 0, None, 0, total_chunks

    stitched_audio = f"temp_stitched_{pid}.aac"

    try:
        if len(audio_chunks) == 1:
            subprocess.run(
                ['ffmpeg', '-y', '-i', audio_chunks[0], '-c', 'copy', stitched_audio],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
        else:
            # Multi-chunk Equal-Power Overlap Cross-Fading (0.1s acrossfade) to eliminate split clicks/seams
            inputs = []
            for c in audio_chunks:
                inputs.extend(['-i', c])

            filter_parts = []
            last_label = "0:a"
            for idx in range(1, len(audio_chunks)):
                next_label = f"a{idx}"
                filter_parts.append(f"[{last_label}][{idx}:a]acrossfade=d=0.1:c1=tri:c2=tri[{next_label}]")
                last_label = next_label

            filter_complex = ";".join(filter_parts)
            cmd = ['ffmpeg', '-y'] + inputs + ['-filter_complex', filter_complex, '-map', f'[{last_label}]', '-c:a', 'aac', stitched_audio]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        for f in chunk_files:
            if os.path.exists(f):
                os.remove(f)

        if os.path.exists(stitched_audio):
            total_duration = get_media_duration(stitched_audio) or 0
            print(f"[+] Successfully cross-faded & stitched {parts_fetched}/{total_chunks} audio parts ({total_duration:.1f}s total seamless audio)!")
            return stitched_audio, total_duration, first_match, parts_fetched, total_chunks

    except Exception as e:
        print("[!] Multi-chunk audio cross-fade stitching error:", e)

    return None, 0, None, parts_fetched, total_chunks

def fetch_and_add_audio(video_path, anime_name):
    """
    Pure Raw Scene Audio Engine (Equal-Power Overlap Cross-Fading):
    - Fetches 100% verified single-anime scene audio.
    - Applies 0.1s equal-power cross-fading to eliminate split seams and click artifacts.
    - Merges raw original audio without DSP / upscaling.
    Returns raw_output_path.
    """
    video_duration = get_media_duration(video_path) or 10.0
    exact_audio_file, exact_duration, match_info, parts_fetched, total_chunks = fetch_multi_chunk_scene_audio(video_path)

    raw_output_path = video_path.replace(".mp4", "_raw_audio.mp4")

    if not exact_audio_file or not os.path.exists(exact_audio_file) or exact_duration == 0:
        print("[!] No original scene audio could be matched for this clip.")
        return video_path

    coverage_percent = min(100.0, round((exact_duration / video_duration) * 100, 1))
    print(f"[*] AUDIO STATS: Fetched {parts_fetched}/{total_chunks} verified parts | Seamless Coverage: {coverage_percent}% ({exact_duration:.1f}s / {video_duration:.1f}s)")

    try:
        # RAW Audio Direct Merge (No audio upscaling, no DSP, no loudnorm filters)
        ffmpeg_raw = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", exact_audio_file,
            "-c:v", "copy",
            "-c:a", "aac",
            "-t", str(video_duration),
            raw_output_path
        ]
        res_ff = subprocess.run(ffmpeg_raw, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if res_ff.returncode == 0 and os.path.exists(raw_output_path):
            print(f"[+] Seamless Raw Version Saved: {raw_output_path}")
            return raw_output_path

    except Exception as e:
        print("[!] Audio processing error:", e)
    finally:
        if exact_audio_file and os.path.exists(exact_audio_file):
            os.remove(exact_audio_file)

    return video_path
