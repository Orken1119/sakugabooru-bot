# System Technical Audit: Errors, Bottlenecks & Known Problems

This document summarizes all technical errors, architectural bottlenecks, and unresolved failure points encountered during the implementation of the Sakugabooru automated audio slicing & Nyaa.si torrent integration pipeline.

---

## 1. Web Scraping & Direct Streaming Failure Points

### A. Cloudflare & WAF Challenges (AnimePahe / Anime3rb / Gogoanime)
- **Symptom**: HTTP 403 Forbidden / Cloudflare Turnstile JavaScript verification screens.
- **Root Cause**: Streaming platforms utilize active Web Application Firewalls (WAF) and Turnstile JavaScript challenges. Direct HTTP requests (`requests`, `urllib`, `curl`) cannot evaluate the required JS tokens, blocking automated scrapers.

### B. BitTorrent CLI Port & NAT Blockade (`peerflix` / `aria2c`)
- **Symptom**: CLI BitTorrent downloaders stall at `0%` progress or time out.
- **Root Cause**: BitTorrent P2P ports (`TCP/UDP 6881–6889`) are blocked or firewalled on non-GUI terminal interfaces. System torrent clients (qBittorrent) bypass this via UPnP / NAT-PMP port forwarding.

---

## 2. Nyaa.si Search & Download Bottlenecks

### A. Title String Formatting Failures
- **Symptom**: Nyaa returns `0` search results for valid anime.
- **Root Cause**: Sakugabooru metadata includes commas, tags, and series specifiers (e.g., `yamato series, yamato yo towa ni`). Submitting uncleaned tags directly to Nyaa's search bar causes 0 matches.

### B. Season Batch & Full BluRay Pack Hijacking
- **Symptom**: Downloader triggers 20GB–68GB full-season or BD batch downloads instead of single episodes.
- **Root Cause**: Nyaa RSS results sorted by *Most Downloaded* (`s=downloads&o=desc`) place full-season batch packs at the top of search rankings.

### C. Missing Episode Metadata (`ep_num = None`)
- **Symptom**: Trace.moe returns `episode_num = None` for creditless openings, OVAs, or movie cuts, falling back to full-series queries that select season packs.

---

## 3. Storage, File System & Detection Failures

### A. Sparse Pre-Allocated File False Positives
- **Symptom**: `ffmpeg` fails with `Invalid data found when processing input` / `EBML header parsing failed`.
- **Root Cause**: qBittorrent pre-allocates disk space immediately by creating a 0-filled sparse `.mkv` file. Naive file detection (`os.path.exists()` or `os.path.getsize()`) falsely reports completion while the file contains only empty zero bytes.

### B. Directory Misalignment Between GUI & Pipeline
- **Symptom**: qBittorrent completes downloads to `~/Downloads` while the pipeline searches `sakugabooru-episodes/`, leaving the script hanging.

---

## 4. Audio Slicing & FFmpeg Integration Issues

### A. Wrong Audio Stream Mapping & Stream Index Errors
- **Symptom**: `ffmpeg` fails with exit code `183` or produces silent output.
- **Root Cause**: Hardcoded stream mapping (`-map 1:a:0`) fails when an `.mkv` container contains multi-audio tracks (Dual-Audio, commentary, or DTS/Opus streams on different indexes).

### B. Incorrect Timestamp Seeking
- **Symptom**: Sliced video plays audio from second `0.0` (episode prologue/OP) instead of the actual scene timestamp.
- **Root Cause**: Applying `-ss` seeking after input flags or using uncalibrated `Trace.moe` relative timestamps cuts audio from the beginning of the file.

---

## Summary Matrix

| Category | Problem | Technical Impact | Current Status |
| :--- | :--- | :--- | :--- |
| **Scraping** | Cloudflare WAF 403 | Direct stream scrapers fail completely | Abandoned in favor of Nyaa P2P |
| **Download** | Torrent Season Batches | 20GB–68GB downloads fill disk | Filtered via regex & `<1.2GB` size cap |
| **Detection** | Sparse Zero File Pre-allocation | `ffmpeg` crashes on partial downloads | Solved via `ffprobe` duration check |
| **Slicing** | Stream Index & Timestamp Mismatch | Silent / offset audio outputs | Solved via `-ss` pre-input & `-map 1:a:0?` |
