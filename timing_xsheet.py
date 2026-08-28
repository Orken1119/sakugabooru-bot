"""
Digital X-Sheet / Animation Timing Graph Generator (Version 6.0 — Genga Paper Theme)
---------------------------------------------------------------------------------------------
Analyzes anime/sakuga MP4 clips, detects frame-by-frame animation timing (Ones, Twos, Threes, Holds),
performs automatic scene cut detection with a 12-frame cooldown, computes Japanese Timesheet Page/Seconds,
and renders an authentic "Genga Paper" (Animation Tracing Paper) HUD Pane with 85% opacity blend.
"""

import os
import sys
import time
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DigitalXSheetProcessor:
    def __init__(
        self,
        noise_threshold: float = 2.2,
        diff_pixel_threshold: int = 25,
        enable_pan_compensation: bool = True,
        pane_height: int = 170,
        timeline_window: int = 48,
        border_crop_pct: float = 0.06,
        scene_cut_threshold: float = 2.0,
        min_frames_between_cuts: int = 12
    ):
        self.noise_threshold = noise_threshold
        self.diff_pixel_threshold = diff_pixel_threshold
        self.enable_pan_compensation = enable_pan_compensation
        self.pane_height = pane_height
        self.timeline_window = timeline_window
        self.border_crop_pct = border_crop_pct
        self.scene_cut_threshold = scene_cut_threshold
        self.min_frames_between_cuts = min_frames_between_cuts

        # Load crisp TrueType fonts including CJK Japanese support (NotoSansCJK)
        self.font_bold_lg = self._load_ttf_font(18, bold=True)
        self.font_bold_md = self._load_ttf_font(15, bold=True)
        self.font_bold_sm = self._load_ttf_font(13, bold=True)

    def _load_ttf_font(self, size: int, bold: bool = True):
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
        ]
        for fpath in font_candidates:
            if os.path.exists(fpath):
                try:
                    return ImageFont.truetype(fpath, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def _estimate_camera_shift(self, prev_gray, curr_gray):
        try:
            small_prev = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
            small_curr = cv2.resize(curr_gray, (0, 0), fx=0.5, fy=0.5)

            flow = cv2.calcOpticalFlowFarneback(
                small_prev, small_curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            shift_x = float(np.median(flow[..., 0])) * 2.0
            shift_y = float(np.median(flow[..., 1])) * 2.0
            pan_mag = float(np.sqrt(shift_x**2 + shift_y**2))
            return shift_x, shift_y, pan_mag
        except Exception:
            return 0.0, 0.0, 0.0

    def _align_frame(self, prev_gray, shift_x, shift_y):
        h, w = prev_gray.shape
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(prev_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _detect_scene_cut(self, prev_hsv, curr_hsv, frame_idx: int, last_cut_frame: int) -> bool:
        """
        1. Scene Detection (Cut Counter)
        Hard cut detector using cv2.compareHist (cv2.HISTCMP_CHISQR).
        Includes a 12-frame cooldown after a cut is detected to avoid false positives.
        """
        if prev_hsv is None or curr_hsv is None:
            return False

        if (frame_idx - last_cut_frame) < self.min_frames_between_cuts:
            return False

        try:
            hist_prev = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist_curr = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist_prev, hist_prev, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_curr, hist_curr, 0, 1, cv2.NORM_MINMAX)
            dist = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CHISQR)
            return dist > self.scene_cut_threshold
        except Exception:
            return False

    def _classify_timing_type(self, history):
        """
        3. Slow-Motion Fallback
        If a held drawing lasts longer than 4 frames, change timing status to HOLD (SLOW-MO).
        Do not break standard 1s/2s/3s detection logic for the rest of the clip.
        """
        if not history:
            return "HOLD", 0

        drawing_indices = [i for i, val in enumerate(history) if val == 1]
        if not drawing_indices:
            return "HOLD", 0

        if len(drawing_indices) < 2:
            last_draw = drawing_indices[-1]
            dist = len(history) - 1 - last_draw
            if dist == 0:
                return "KEY DRAWING", 1
            elif dist > 4:
                return f"HOLD (SLOW-MO {dist}f)", dist
            return f"HOLD ({dist}f)", dist

        diffs = [drawing_indices[i] - drawing_indices[i - 1] for i in range(1, len(drawing_indices))]
        recent_spacing = diffs[-1]

        if recent_spacing == 1:
            return "ON 1s (24 FPS)", 1
        elif recent_spacing == 2:
            return "ON 2s (12 FPS)", 2
        elif recent_spacing == 3:
            return "ON 3s (8 FPS)", 3
        elif recent_spacing == 4:
            return "ON 4s (6 FPS)", 4
        elif recent_spacing > 4:
            return f"HOLD (SLOW-MO {recent_spacing}f)", recent_spacing
        else:
            return f"ON {recent_spacing}s", recent_spacing

    def process_video(self, input_path: str, output_path: str, initial_cut_num: int = 1, show_progress: bool = True):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or np.isnan(fps):
            fps = 24.0

        pane_h = self.pane_height
        output_height = height + pane_h

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))

        prev_gray = None
        prev_hsv = None
        drawing_history = []
        timing_labels = []

        cut_idx = initial_cut_num
        last_cut_frame = -999
        frame_idx = 0
        start_time = time.time()

        crop_y = int(height * self.border_crop_pct)
        crop_x = int(width * self.border_crop_pct)
        valid_area = max(1, (height - 2 * crop_y) * (width - 2 * crop_x))

        print(f"[X-Sheet Engine V6.0 Genga Paper Theme] Processing '{input_path}'...")
        print(f" Art Resolution: {width}x{height} | Expanded Canvas: {width}x{output_height} (+{pane_h}px Genga Paper pane)")
        print(f" FPS: {fps:.2f} | Total Frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Scene cut detection with 12-frame cooldown
            if self._detect_scene_cut(prev_hsv, hsv, frame_idx, last_cut_frame):
                cut_idx += 1
                last_cut_frame = frame_idx

            prev_hsv = hsv.copy()

            shift_x, shift_y, pan_mag = 0.0, 0.0, 0.0
            is_new_drawing = False

            if prev_gray is not None:
                if self.enable_pan_compensation:
                    shift_x, shift_y, pan_mag = self._estimate_camera_shift(prev_gray, gray_blur)
                    aligned_prev = self._align_frame(prev_gray, shift_x, shift_y)
                else:
                    aligned_prev = prev_gray

                frame_diff = cv2.absdiff(gray_blur, aligned_prev)
                _, thresh = cv2.threshold(frame_diff, self.diff_pixel_threshold, 255, cv2.THRESH_BINARY)

                thresh_cropped = thresh[crop_y:height - crop_y, crop_x:width - crop_x]
                changed_pixels = np.count_nonzero(thresh_cropped)
                diff_percentage = (changed_pixels / valid_area) * 100.0

                effective_threshold = self.noise_threshold + (0.8 * min(pan_mag, 5.0) if self.enable_pan_compensation else 0.0)
                if diff_percentage > effective_threshold:
                    is_new_drawing = True
            else:
                is_new_drawing = True

            prev_gray = gray_blur.copy()
            drawing_history.append(1 if is_new_drawing else 0)

            current_timing, spacing = self._classify_timing_type(drawing_history[-24:])
            timing_labels.append(current_timing)

            canvas = np.zeros((output_height, width, 3), dtype=np.uint8)
            canvas[0:height, 0:width] = frame

            self._render_genga_paper_pane(
                canvas=canvas,
                frame=frame,
                art_height=height,
                pane_height=pane_h,
                width=width,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=fps,
                cut_idx=cut_idx,
                current_timing=current_timing,
                shift_x=shift_x,
                shift_y=shift_y,
                pan_mag=pan_mag,
                drawing_history=drawing_history
            )

            out.write(canvas)
            frame_idx += 1

            if show_progress and total_frames > 0 and frame_idx % 100 == 0:
                pct = (frame_idx / total_frames) * 100.0
                elapsed = time.time() - start_time
                print(f" Progress: {frame_idx}/{total_frames} frames ({pct:.1f}%) - {frame_idx / elapsed:.1f} fps")

        cap.release()
        out.release()
        elapsed_total = time.time() - start_time
        print(f"[X-Sheet Engine] Render completed in {elapsed_total:.2f}s ({frame_idx / elapsed_total:.1f} fps)")

        # Save metrics JSON
        json_output_path = output_path.rsplit('.', 1)[0] + ".json"
        try:
            import json
            metrics = {
                "total_frames": frame_idx,
                "fps": round(fps, 2),
                "total_cuts": cut_idx,
                "on_1s_count": sum(1 for t in timing_labels if "ON 1s" in t),
                "on_2s_count": sum(1 for t in timing_labels if "ON 2s" in t),
                "on_3s_count": sum(1 for t in timing_labels if "ON 3s" in t),
                "hold_count": sum(1 for t in timing_labels if "HOLD" in t),
            }
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass

        # Convert output to H.264 (yuv420p)
        h264_tmp = output_path.rsplit('.', 1)[0] + "_h264.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            h264_tmp
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(h264_tmp, output_path)
            print(f"[X-Sheet Engine] H.264 video saved to '{output_path}'")
        except Exception as err:
            print(f"[X-Sheet Engine] H.264 warning: {err}")

        return output_path

    def _render_genga_paper_pane(
        self,
        canvas,
        frame,
        art_height,
        pane_height,
        width,
        frame_idx,
        total_frames,
        fps,
        cut_idx,
        current_timing,
        shift_x,
        shift_y,
        pan_mag,
        drawing_history
    ):
        """
        4. "Genga Paper" UI Redesign (OpenCV BGR & PIL TrueType)
        - Paper Background (BGR: 225, 238, 242)
        - Graphite Text/Lines (BGR: 65, 65, 65)
        - Animation Blue (BGR: 210, 150, 70)
        - Animation Red (BGR: 80, 80, 220)
        - Tracing Paper Blend: cv2.addWeighted (85% paper + 15% original frame)
        """
        pane_y1 = art_height
        pane_y2 = art_height + pane_height

        # Extract bottom ROI of current animation frame for tracing paper blend
        bottom_art = cv2.resize(frame, (width, pane_height))

        # Create Solid Genga Paper Off-White Background (BGR: 225, 238, 242)
        paper_bgr = np.full((pane_height, width, 3), (225, 238, 242), dtype=np.uint8)

        # Tracing Paper Blending: 85% Paper + 15% Original Art
        pane_bgr = cv2.addWeighted(paper_bgr, 0.85, bottom_art, 0.15, 0)

        # Top Red Accent Pencil Border
        cv2.line(pane_bgr, (0, 0), (width, 0), (80, 80, 220), 3)

        # Convert to PIL RGB for TrueType CJK rendering
        pane_rgb = cv2.cvtColor(pane_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pane_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Colors in RGB format for PIL
        RGB_GRAPHITE = (65, 65, 65)
        RGB_ANIMATION_BLUE = (70, 150, 210)
        RGB_ANIMATION_RED = (220, 80, 80)
        RGB_GOLD = (230, 180, 0)
        RGB_LIME = (40, 190, 80)
        RGB_GRAY_HOLD = (110, 115, 125)

        # -------------------------------------------------------------
        # 2. Japanese Timesheet Math & Header
        # -------------------------------------------------------------
        sheet_page = (frame_idx // 144) + 1
        sheet_sec = ((frame_idx % 144) // 24) + 1
        curr_frame_num = frame_idx + 1

        margin_x = 20
        start_y = 14

        header_text = f"SHEET P.{sheet_page} ({sheet_sec:02d}/06s)   |   CUT: #{cut_idx:02d}   |   FRAME: {curr_frame_num:04d} / {total_frames:04d}"
        draw.text((margin_x, start_y), header_text, fill=RGB_GRAPHITE, font=self.font_bold_md)

        # -------------------------------------------------------------
        # Dynamic Timing Badge (Genga Paper Style)
        # -------------------------------------------------------------
        badge_y = start_y + 28
        badge_w, badge_h = 240, 32

        if "1s" in current_timing:
            badge_bg = RGB_ANIMATION_RED
            badge_fg = (255, 255, 255)
            badge_text = "TIMING: ON 1s (24 FPS)"
        elif "2s" in current_timing:
            badge_bg = RGB_GOLD
            badge_fg = (10, 10, 10)
            badge_text = "TIMING: ON 2s (12 FPS)"
        elif "3s" in current_timing:
            badge_bg = RGB_LIME
            badge_fg = (255, 255, 255)
            badge_text = "TIMING: ON 3s (8 FPS)"
        elif "KEY" in current_timing:
            badge_bg = (240, 130, 20)
            badge_fg = (255, 255, 255)
            badge_text = "KEY DRAWING DETECTED"
        else:
            badge_bg = RGB_GRAY_HOLD
            badge_fg = (255, 255, 255)
            badge_text = f"TIMING: {current_timing}"

        draw.rectangle([margin_x, badge_y, margin_x + badge_w, badge_y + badge_h], fill=badge_bg)
        draw.text((margin_x + 10, badge_y + 6), badge_text, fill=badge_fg, font=self.font_bold_sm)

        # Camera Pan Status
        if pan_mag > 0.8:
            if abs(shift_x) > abs(shift_y):
                pan_dir = "PAN RIGHT" if shift_x > 0 else "PAN LEFT"
            else:
                pan_dir = "TILT DOWN" if shift_y > 0 else "TILT UP"
            pan_text = f"CAM PAN: {pan_dir} ({pan_mag:.1f}px)"
        else:
            pan_text = "CAM PAN: STATIC"

        draw.text((margin_x, badge_y + badge_h + 12), pan_text, fill=RGB_GRAPHITE, font=self.font_bold_sm)

        # -------------------------------------------------------------
        # Right Panel: Genga Paper Koma-uchi Timeline (コマ打ち)
        # -------------------------------------------------------------
        timeline_x1 = int(width * 0.48)
        timeline_w = int(width * 0.48)
        timeline_y = 32
        cell_h = 42

        recent_drawings = drawing_history[-self.timeline_window:]
        num_cells = self.timeline_window
        cell_w = max(5, timeline_w // num_cells)

        # Header Title with Japanese Katakana (コマ打ち)
        draw.text((timeline_x1, timeline_y - 20), "DIGITAL X-SHEET TIMELINE  •  コマ打ち (KOMA-UCHI)", fill=RGB_GRAPHITE, font=self.font_bold_sm)

        # Outer Graphite Grid Frame
        track_x2 = timeline_x1 + num_cells * cell_w
        draw.rectangle([timeline_x1, timeline_y, track_x2, timeline_y + cell_h], fill=(238, 245, 248), outline=RGB_GRAPHITE, width=2)

        # Draw Grid Cells & Holds in Animation Blue with graphite grid margin
        for i in range(len(recent_drawings)):
            cx = timeline_x1 + i * cell_w
            is_draw = recent_drawings[i] == 1

            # Grid Cell Divider
            draw.line([(cx, timeline_y), (cx, timeline_y + cell_h)], fill=(180, 185, 195), width=1)

            if is_draw:
                fill_color = RGB_ANIMATION_RED if i == len(recent_drawings) - 1 else RGB_ANIMATION_BLUE
                # Leave a 2px margin so underlying graphite grid remains visible
                draw.rectangle([cx + 2, timeline_y + 3, cx + cell_w - 2, timeline_y + cell_h - 3], fill=fill_color)
            else:
                # Hold dot in graphite
                draw.line([(cx + cell_w // 2, timeline_y + cell_h // 2 - 2), (cx + cell_w // 2, timeline_y + cell_h // 2 + 2)], fill=RGB_GRAPHITE, width=2)

        # Highlight Current Frame Outline in Animation Red
        curr_cx = timeline_x1 + (len(recent_drawings) - 1) * cell_w
        draw.rectangle([curr_cx - 2, timeline_y - 3, curr_cx + cell_w + 2, timeline_y + cell_h + 3], outline=RGB_ANIMATION_RED, width=3)

        # Place composite pane back onto canvas
        res_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        canvas[pane_y1:pane_y2, 0:width] = res_bgr


def process_timing_graph(input_path: str, output_path: str, cut_num: int = 1):
    processor = DigitalXSheetProcessor()
    return processor.process_video(input_path, output_path, initial_cut_num=cut_num)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Digital X-Sheet Processor — Genga Paper Edition")
    parser.add_argument("input", help="Path to input video MP4")
    parser.add_argument("output", nargs="?", default="output_genga_paper.mp4", help="Path to output video MP4")
    parser.add_argument("--cut-num", type=int, default=1, help="Starting cut number (e.g. 1 for CUT: #01)")
    args = parser.parse_args()

    process_timing_graph(args.input, args.output, cut_num=args.cut_num)
