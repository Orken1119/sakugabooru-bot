"""
Digital X-Sheet / Animation Timing Graph Generator (Version 5.1 Animator Edition)
---------------------------------------------------------------------------------------------
Analyzes anime/sakuga MP4 clips, detects frame-by-frame animation timing (Ones, Twos, Threes, Holds),
performs automatic scene cut detection (cv2.compareHist with cooldown), computes Japanese Timesheet Page/Seconds,
and renders a professional, ultra-crisp "Animator Edition Analysis Pane" below 100% of original animation art.
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
        scene_cut_threshold: float = 2.5,
        min_frames_between_cuts: int = 15
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
        self.font_title = self._load_ttf_font(20, bold=True, cjk=True)
        self.font_bold_lg = self._load_ttf_font(18, bold=True, cjk=True)
        self.font_bold_md = self._load_ttf_font(15, bold=True, cjk=True)
        self.font_bold_sm = self._load_ttf_font(13, bold=True, cjk=True)

    def _load_ttf_font(self, size: int, bold: bool = True, cjk: bool = False):
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
        Automatic Scene Cut Detection using Chi-Square HSV Histogram Distance with Cooldown.
        """
        if prev_hsv is None or curr_hsv is None:
            return False

        # Cooldown guard: require at least 15 frames (~0.6s) between cuts
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
            elif dist > 5:
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
        elif recent_spacing > 5:
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

        print(f"[X-Sheet Engine V5.1 Animator Edition] Processing '{input_path}'...")
        print(f" Resolution: {width}x{height} | Expanded Canvas: {width}x{output_height} (+{pane_h}px pane)")
        print(f" FPS: {fps:.2f} | Total Frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Check for scene cut with cooldown guard
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

            # Build Expanded Canvas: Top = 100% Unobscured Art, Bottom = Animator Analysis Pane
            canvas = np.zeros((output_height, width, 3), dtype=np.uint8)
            canvas[0:height, 0:width] = frame

            self._render_animator_analysis_pane(
                canvas=canvas,
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

    def _render_animator_analysis_pane(
        self,
        canvas,
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
        Renders a Clean, Non-Overlapping Animator Edition HUD Pane with CJK Japanese Font Support.
        """
        pane_y1 = art_height
        pane_y2 = art_height + pane_height

        pane_bgr = np.zeros((pane_height, width, 3), dtype=np.uint8)
        pane_bgr[:] = (10, 12, 18)  # OLED Deep Dark Slate

        # Top Gold Accent Line
        cv2.line(pane_bgr, (0, 0), (width, 0), (0, 210, 255), 3)

        pane_rgb = cv2.cvtColor(pane_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pane_rgb)
        draw = ImageDraw.Draw(pil_img)

        # -------------------------------------------------------------
        # Left Panel: Japanese Timesheet Page/Sec & Frame Counter
        # -------------------------------------------------------------
        sheet_page = (frame_idx // 144) + 1
        sheet_sec = ((frame_idx % 144) // 24) + 1
        curr_frame_num = frame_idx + 1

        margin_x = 20
        start_y = 16

        # 1. Header: Compact non-overlapping text string
        header_str = f"SHEET P.{sheet_page} ({sheet_sec:02d}/06s)   |   CUT: #{cut_idx:02d}   |   FRM: {curr_frame_num:04d}/{total_frames:04d}"
        draw.text((margin_x, start_y), header_str, fill=(240, 245, 250), font=self.font_bold_sm)

        # 2. Dynamic Timing Badge (High-Alarm Color Rules)
        badge_y = start_y + 28
        badge_w, badge_h = 240, 32

        if "1s" in current_timing:
            badge_bg = (255, 30, 50)       # GLARED RED for 1s (24 FPS)
            badge_fg = (255, 255, 255)
            badge_text = "TIMING: ON 1s (24 FPS)"
        elif "2s" in current_timing:
            badge_bg = (255, 210, 0)       # BRIGHT GOLD for 2s (12 FPS)
            badge_fg = (10, 10, 10)
            badge_text = "TIMING: ON 2s (12 FPS)"
        elif "3s" in current_timing:
            badge_bg = (0, 230, 120)       # NEON LIME for 3s (8 FPS)
            badge_fg = (10, 10, 10)
            badge_text = "TIMING: ON 3s (8 FPS)"
        elif "KEY" in current_timing:
            badge_bg = (255, 140, 0)       # BRIGHT ORANGE for Key Drawing
            badge_fg = (255, 255, 255)
            badge_text = "KEY DRAWING DETECTED"
        else:
            badge_bg = (70, 80, 95)        # Slate Gray for Hold
            badge_fg = (240, 240, 240)
            badge_text = f"TIMING: {current_timing}"

        draw.rectangle([margin_x, badge_y, margin_x + badge_w, badge_y + badge_h], fill=badge_bg)
        draw.text((margin_x + 10, badge_y + 6), badge_text, fill=badge_fg, font=self.font_bold_sm)

        # 3. Clean Layout Camera Pan Tracker
        if pan_mag > 0.8:
            if abs(shift_x) > abs(shift_y):
                pan_dir = "PAN RIGHT" if shift_x > 0 else "PAN LEFT"
            else:
                pan_dir = "TILT DOWN" if shift_y > 0 else "TILT UP"
            pan_text = f"CAM PAN: {pan_dir} ({pan_mag:.1f}px)"
        else:
            pan_text = "CAM PAN: STATIC"

        draw.text((margin_x, badge_y + badge_h + 12), pan_text, fill=(170, 185, 205), font=self.font_bold_sm)

        # -------------------------------------------------------------
        # Right Panel: DIGITAL X-SHEET TIMELINE (Koma-uchi コマ打ち)
        # Positioned with generous left offset to prevent text overlap!
        # -------------------------------------------------------------
        timeline_x1 = int(width * 0.48)
        timeline_w = int(width * 0.48)
        timeline_y = 32
        cell_h = 42

        recent_drawings = drawing_history[-self.timeline_window:]
        num_cells = self.timeline_window
        cell_w = max(5, timeline_w // num_cells)

        # Timeline Header with rendering Japanese Katakana (コマ打ち)
        draw.text((timeline_x1, timeline_y - 20), "DIGITAL X-SHEET TIMELINE  •  コマ打ち (KOMA-UCHI)", fill=(200, 210, 225), font=self.font_bold_sm)

        # Outer Track Box
        track_x2 = timeline_x1 + num_cells * cell_w
        draw.rectangle([timeline_x1, timeline_y, track_x2, timeline_y + cell_h], fill=(22, 28, 42), outline=(70, 85, 110), width=2)

        # Render Timeline Block Holds
        for i in range(len(recent_drawings)):
            cx = timeline_x1 + i * cell_w
            is_draw = recent_drawings[i] == 1

            if is_draw:
                bar_color = (255, 210, 0) if i == len(recent_drawings) - 1 else (230, 180, 0)
                draw.rectangle([cx + 1, timeline_y + 3, cx + cell_w - 1, timeline_y + cell_h - 3], fill=bar_color)
            else:
                draw.line([(cx + cell_w // 2, timeline_y + cell_h // 2 - 2), (cx + cell_w // 2, timeline_y + cell_h // 2 + 2)], fill=(90, 105, 125), width=2)

        # Active Frame Cursor Highlight
        curr_cx = timeline_x1 + (len(recent_drawings) - 1) * cell_w
        draw.rectangle([curr_cx - 2, timeline_y - 3, curr_cx + cell_w + 2, timeline_y + cell_h + 3], outline=(0, 255, 120), width=3)

        # Place pane on bottom canvas
        res_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        canvas[pane_y1:pane_y2, 0:width] = res_bgr


def process_timing_graph(input_path: str, output_path: str):
    processor = DigitalXSheetProcessor()
    return processor.process_video(input_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        outp = sys.argv[2] if len(sys.argv) > 2 else "output_xsheet_v5_1.mp4"
        process_timing_graph(inp, outp)
    else:
        print("Usage: python3 timing_xsheet.py <input_video.mp4> [output_video.mp4]")
