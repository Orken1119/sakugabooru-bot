# 🎬 aniSaku — Professional Anime Sakuga & Traditional Genga Processing Engine

> **Notice to AI Assistants & Developers**: This document is the single source of truth for the `aniSaku` codebase. Whenever any code, function signature, parameter, or module logic is created or updated in this repository, **THIS DOCUMENTATION MUST BE UPDATED IMMEDIATELY**.

---

## 📌 Project Overview

**aniSaku** is an automated Python & OpenCV computer vision pipeline designed for processing, analyzing, and compositing Japanese Key Animation (*Sakuga* / Key Drawings *Genga*). 

The repository provides non-AI algorithms that execute Matrix-level frame transformations at 15–50+ FPS with 0% GPU VRAM overhead, featuring:
1. **Clean Traditional Genga Filter Engine**: Non-AI paper texture, light-table vignetting, and distance-transform line classification simulating authentic multi-color Japanese animator pencils (Blue, Red, and Charcoal Graphite).
2. **Digital X-Sheet HUD Analyzer (v4.0 Pro)**: Real-time animation timing graph with Farnebäck Optical Flow camera pan compensation, frame hold detector (Ones, Twos, Threes), and motion velocity sparklines.
3. **Pure OpenCV XDoG Line Extraction**: Advanced Difference-of-Gaussians edge detection with Connected Component Analysis (CCA) speckle noise filter.
4. **Sakugabooru Integration & Video Compositor**: Automated post fetching, metadata scraping, audio syncing, and vertical (9:16 / 7:8) 2K video composition for TikTok/Reels/Shorts.

---

## 📂 Repository Structure

```
aniSaku/
├── README.md                      # Complete Project Documentation & API Reference
├── ERRORS_AND_PROBLEMS.md         # Technical Gotchas & Bug Fix History
├── config.py                      # Twitter / X API Authentication Config
├── main.py                        # Sakugabooru Random Fetch & Automated Bot Pipeline
├── line_extractor.py              # XDoG & ControlNet Line Art Extraction Engine
├── timing_xsheet.py               # Digital X-Sheet HUD & Animation Timing Analyzer
├── audio_helper.py                # Source Audio Extraction & Sync Helper (yt-dlp)
├── music_helper.py                # OST Background Music Mixer
├── nyaa_downloader.py             # Torrent Search & Downloader for High-Res Source Cuts
├── example_decomposite_pipeline.py# Demo Pipeline Script for Frame Decomposition
├── line_extractor_diagnostic.py   # Diagnostic Utility for Line Extraction Tuning
├── requirements.txt               # Dependencies (opencv-python, numpy, pillow, requests)
└── sakugabooru-video-files/       # Video File Output Root
    └── selected_cuts/             # Processed Sakuga Cuts & Credits Documentation
        └── CREDITS.md             # Key Animators, Social Media Handles & Source Metadata
```

---

## 🛠️ Detailed Module & Function Reference

### 1. `timing_xsheet.py` — Digital X-Sheet HUD & Timing Analyzer
Analyzes anime MP4 clips, detects frame-by-frame animation timing (Ones, Twos, Threes, Holds), compensates for camera panning using optical flow, and renders a non-overlapping HD analysis pane below the video canvas.

#### Classes & Functions:

##### `class DigitalXSheetProcessor`
* **`__init__(noise_threshold=2.2, diff_pixel_threshold=25, enable_pan_compensation=True, pane_height_ratio=0.32, timeline_window=48, border_crop_pct=0.06)`**
  * `noise_threshold` *(float)*: Base percentage of changed pixels (% of frame area) required to register a 'New Drawing'.
  * `diff_pixel_threshold` *(int)*: Pixel value difference threshold (0–255) for frame diff calculation.
  * `enable_pan_compensation` *(bool)*: Enables Farneback Optical Flow sub-pixel alignment to filter out camera panning.
  * `pane_height_ratio` *(float)*: Height ratio of analysis pane added at bottom of canvas (default `0.32`).
  * `timeline_window` *(int)*: Number of recent frames shown on rolling X-Sheet timeline graph (default `48`).
  * `border_crop_pct` *(float)*: Edge border percentage ignored during diff calculation to prevent edge pan artifacts (default `0.06`).

* **`_estimate_camera_shift(prev_gray, curr_gray) -> (shift_x, shift_y, pan_magnitude)`**
  * Estimates background translation (camera pan/tilt) using Farnebäck Optical Flow on half-scaled grayscale frames.

* **`_align_frame(prev_gray, shift_x, shift_y) -> aligned_gray`**
  * Applies sub-pixel linear interpolation translation (`cv2.warpAffine`) to align previous frame with current frame.

* **`process_video(input_path: str, output_path: str) -> str`**
  * Reads `input_path`, processes frame timing graph, and encodes native H.264 video to `output_path`.

##### `process_timing_graph(input_path: str, output_path: str, noise_threshold=2.2, enable_pan_compensation=True) -> str`
* Helper function that initializes `DigitalXSheetProcessor` and runs `process_video`.

---

### 2. `line_extractor.py` — XDoG Line Extraction Engine
Extracts clean, production-ready black pencil line art from colored anime frames using Difference of Gaussians (XDoG) and Connected Component Analysis (CCA) speckle noise filtering.

#### Classes & Functions:

##### `class LineExtractor`
* **`__init__(method="xdog", device=None, denoise_strength=40.0, line_threshold=0.5, clean_speckles=True, min_speckle_area=4.0, high_chaos_mode=False, auto_crop=True, crop_threshold=15, debug_mode=False, debug_dir=".")`**
  * `method` *(str)*: `'xdog'` (Pure OpenCV XDoG) or `'controlnet_lineart'` (ControlNet Aux AI detector).
  * `denoise_strength` *(float)*: Bilateral pre-filter intensity (default `40.0`, or `60.0` in High-Chaos Mode).
  * `line_threshold` *(float)*: Sensitivity to faint lines (default `0.5`, or `0.8` in High-Chaos Mode).
  * `clean_speckles` *(bool)*: Enables CCA contour area filtering to erase isolated noise dots without eroding lines.
  * `min_speckle_area` *(float)*: Minimum contour area in pixels for speckle removal.
  * `high_chaos_mode` *(bool)*: Aggressive preset for heavy background grain & explosion particles.
  * `auto_crop` *(bool)*: Automatically crops letterbox black bars.

* **`process_frame(frame_bgr: np.ndarray) -> np.ndarray`**
  * Takes a single BGR color frame and returns a 3-channel BGR line art image (black lines on pure white background).

* **`process_stream(input_path: str) -> Generator[np.ndarray]`**
  * Generator that yields processed line-art frames one-by-one for downstream video pipelines.

* **`process_to_file(input_path: str, output_path: str) -> str`**
  * Reads `input_path`, extracts line art for all frames, and saves H.264 encoded video to `output_path`.

##### `extract_lines_opencv(frame_bgr: np.ndarray, denoise_strength=40.0, line_threshold=0.5, clean_speckles=True, min_speckle_area=4.0, high_chaos_mode=False, auto_crop=True, crop_threshold=15, debug_mode=False, debug_dir=".") -> np.ndarray`
* Standalone pure OpenCV function executing Bilateral Filtering -> Difference of Gaussians (XDoG) -> CCA Speckle Denoising.

---

### 3. Clean Traditional Genga Filter Engine (Algorithm Breakdown)
A non-AI matrix transformation engine that turns line art into authentic Japanese key animation pencil drawings (*Genga*).

#### Color Palette & Math Specification:
* **Paper Base**: Warm Off-White Cream (`#F8F2E4` -> BGR `[228, 242, 248]`).
* **Paper Grain Noise**: Gaussian Noise ($\mu=0, \sigma=3.5$) clipped to $[0, 255]$.
* **Light-Table Vignette**: 2D Gaussian kernel ($0.7 \cdot W \times 0.7 \cdot H$) scaled to $[0.88, 1.0]$ brightness modulation.
* **Line Classification via Distance Transform**:
  $$\text{dist} = \text{cv2.distanceTransform}(\text{line\_mask}, \text{DIST\_L2}, 3)$$
  * **Blue Pencil** (`#2D82BE` / BGR `[190, 130, 45]`): $0.5 < \text{dist} \le 1.8$ (shadow boundaries & fine detail).
  * **Red Pencil** (`#D74646` / BGR `[70, 70, 215]`): $1.8 < \text{dist} \le 3.2$ (highlight boundaries & secondary forms).
  * **Charcoal Graphite** (`#28282D` / BGR `[45, 40, 40]`): $\text{dist} > 3.2$ (main structural outlines).

---

### 4. `main.py` — Sakugabooru Fetching & Bot Pipeline
Automates random Sakugabooru post fetching, metadata scraping, and posting.

#### Functions:
* **`grab_post_metadata(posturl: str) -> (artist_str, anime_str)`**
  * Scrapes Sakugabooru HTML page to extract animator (`tag-type-artist`) and anime title (`tag-type-copyright`).
* **`filetypechecker(boorurl: str) -> bool`**
  * Validates whether a Sakugabooru post file URL is a valid `.mp4` video clip.
* **`boorurandom(retries=0) -> dict`**
  * Fetches random Sakugabooru posts matching `order:random -western score:>100`, filtering out non-MP4 posts. (Includes max 10 retries safety limit).

---

### 5. `audio_helper.py` & `music_helper.py` — Audio Synchronization
* **`fetch_and_add_audio(source_url: str, input_video_path: str, output_video_path: str) -> bool`**
  * Uses `yt-dlp` to extract original audio from source URL (YouTube/X/Vimeo) and merges it into `input_video_path`.
* **`add_ost_to_video(input_video_path: str, output_video_path: str, ost_audio_path: str) -> bool`**
  * Mixes background OST audio track into processed video.

---

### 7. `generate_tweet_text.py` — Social Media Post Copy Generator
* **`make_tweet_text(animator: str, anime_title: str, commentary: str = "", custom_tags: list = None) -> str`**
  * Generates formatted Twitter/X post text matching template `Animator: [Artist] / [Anime]` with mandatory hashtags `#falsememory #原画 #xsheet #コマ打ち`.
* **`generate_tweets_for_selected_cuts() -> list[dict]`**
  * Batch generates post copy for all processed cuts in `sakugabooru-video-files/selected_cuts/`.

---

## 📝 Code Update Maintenance Protocol

> ⚠️ **MANDATORY RULE FOR DEVELOPERS & AGENTS**:
> Whenever any file in this project is added, renamed, deleted, or updated (including changes to arguments, default values, algorithms, or output directory structures):
> 1. Open this `README.md` file.
> 2. Update the corresponding section and function signature.
> 3. Verify that all line links and directory maps match the current repository state.
