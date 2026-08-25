"""
Digital X-Sheet / Animation Timing Graph Generator (Version 4.0 Pro - HD Typography & Layout)
---------------------------------------------------------------------------------------------
Analyzes anime/sakuga MP4 clips, detects frame-by-frame animation timing (Ones, Twos, Threes, Holds),
compensates for camera panning using optical flow, and renders a professional, ultra-crisp, non-overlapping
"Analysis Pane" BELOW 100% of the original animation art on an expanded canvas.
"""

import cv2
import numpy as np
import time
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont


class DigitalXSheetProcessor:
    def __init__(
        self,
        noise_threshold: float = 2.2,
        diff_pixel_threshold: int = 25,
        enable_pan_compensation: bool = True,
        pane_height_ratio: float = 0.32,      # 32% added at the bottom for spacious analysis pane
        timeline_window: int = 48,
        border_crop_pct: float = 0.06
    ):
        """
        Parameters:
        -----------
        noise_threshold : float
            Base percentage of changed pixels (% of frame area) required to register a 'New Drawing'.
        diff_pixel_threshold : int
            Pixel value difference threshold (0-255) for cv2.threshold on frame diffs.
        enable_pan_compensation : bool
            If True, uses Farneback Optical Flow with sub-pixel alignment and dynamic adaptive thresholding.
        pane_height_ratio : float
            Ratio of original video height added at the bottom as a dedicated non-overlapping analysis pane.
        timeline_window : int
            Number of recent frames to display on the rolling X-Sheet timeline graph.
        border_crop_pct : float
            Percentage of edge border to ignore during diff calculation to prevent camera pan edge artifacts.
        """
        self.noise_threshold = noise_threshold
        self.diff_pixel_threshold = diff_pixel_threshold
        self.enable_pan_compensation = enable_pan_compensation
        self.pane_height_ratio = pane_height_ratio
        self.timeline_window = timeline_window
        self.border_crop_pct = border_crop_pct

        # Load crisp TrueType fonts with fallbacks
        self.font_bold_lg = self._load_ttf_font(18, bold=True)
        self.font_bold_md = self._load_ttf_font(14, bold=True)
        self.font_bold_sm = self._load_ttf_font(12, bold=True)
        self.font_mono = self._load_ttf_font(13, mono=True)

    def _load_ttf_font(self, size: int, bold: bool = False, mono: bool = False):
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if mono else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
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
        """
        Estimates global background translation (camera pan/tilt) using Farneback Optical Flow.
        Returns (shift_x, shift_y, pan_magnitude) in pixels.
        """
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
        """
        Applies sub-pixel linear interpolation translation to align previous frame with current frame.
        """
        h, w = prev_gray.shape
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        aligned = cv2.warpAffine(
            prev_gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return aligned

    def _classify_timing_type(self, history):
        """
        Analyzes recent history of drawing events (1 = New Drawing, 0 = Hold)
        to determine current timing pattern (On 1s, On 2s, On 3s, On 4s+, Hold).
        """
        if not history:
            return "HOLD", 0

        drawing_indices = [i for i, val in enumerate(history) if val == 1]
        if not drawing_indices:
            return "HOLD (0s)", 0

        if len(drawing_indices) < 2:
            last_draw = drawing_indices[-1]
            dist = len(history) - 1 - last_draw
            if dist == 0:
                return "NEW DRAWING", 1
            return f"HOLD ({dist}f)", dist

        diffs = [drawing_indices[i] - drawing_indices[i - 1] for i in range(1, len(drawing_indices))]
        recent_spacing = diffs[-1]

        if recent_spacing == 1:
            return "ON 1s (24fps)", 1
        elif recent_spacing == 2:
            return "ON 2s (12fps)", 2
        elif recent_spacing == 3:
            return "ON 3s (8fps)", 3
        elif recent_spacing == 4:
            return "ON 4s (6fps)", 4
        else:
            return f"ON {recent_spacing}s", recent_spacing

    def process_video(self, input_path: str, output_path: str, show_progress: bool = True):
        """
        Processes the input video, performs timing analysis, and writes the output MP4 with expanded canvas.
        """
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

        # Calculate expanded canvas height with generous bottom padding for player controls
        pane_height = max(160, int(height * self.pane_height_ratio))
        output_height = height + pane_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))

        prev_gray = None
        drawing_history = []
        diff_score_history = []
        timing_labels = []

        frame_idx = 0
        start_time = time.time()

        # Border crop mask to prevent edge artifacts from camera movement
        crop_y = int(height * self.border_crop_pct)
        crop_x = int(width * self.border_crop_pct)
        valid_area = max(1, (height - 2 * crop_y) * (width - 2 * crop_x))

        print(f"[X-Sheet Engine V4.0 Pro] Processing '{input_path}'...")
        print(f" Input Art Resolution: {width}x{height} | Output Canvas: {width}x{output_height} (+{pane_height}px pane)")
        print(f" FPS: {fps:.2f} | Total Frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

            shift_x, shift_y, pan_mag = 0.0, 0.0, 0.0
            is_new_drawing = False
            diff_percentage = 0.0

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
                diff_percentage = 10.0

            prev_gray = gray_blur.copy()

            drawing_history.append(1 if is_new_drawing else 0)
            diff_score_history.append(diff_percentage)

            current_timing, spacing = self._classify_timing_type(drawing_history[-24:])
            timing_labels.append(current_timing)

            # Build Expanded Canvas: Top = 100% Unobscured Art, Bottom = HD Analysis Pane
            canvas = np.zeros((output_height, width, 3), dtype=np.uint8)
            canvas[0:height, 0:width] = frame  # Pure animation art with 0% overlays!

            # Render HD Analysis Pane at the bottom using PIL TrueType engine
            self._render_hd_analysis_pane(
                canvas=canvas,
                art_height=height,
                pane_height=pane_height,
                width=width,
                frame_idx=frame_idx,
                total_frames=total_frames,
                fps=fps,
                diff_percentage=diff_percentage,
                is_new_drawing=is_new_drawing,
                current_timing=current_timing,
                shift_x=shift_x,
                shift_y=shift_y,
                pan_mag=pan_mag,
                drawing_history=drawing_history,
                diff_history=diff_score_history
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
        print(f"[X-Sheet Engine] Raw render finished in {elapsed_total:.2f}s ({frame_idx / elapsed_total:.1f} fps)")

        # Convert output to H.264 (yuv420p) for full IDE and web player compatibility
        h264_tmp = output_path.rsplit('.', 1)[0] + "_h264.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            h264_tmp
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(h264_tmp, output_path)
            print(f"[X-Sheet Engine] Encoded H.264 video saved to '{output_path}'")
        except Exception as err:
            print(f"[X-Sheet Engine] H.264 post-processing warning: {err}")

        return output_path

    def _render_hd_analysis_pane(
        self,
        canvas,
        art_height,
        pane_height,
        width,
        frame_idx,
        total_frames,
        fps,
        diff_percentage,
        is_new_drawing,
        current_timing,
        shift_x,
        shift_y,
        pan_mag,
        drawing_history,
        diff_history
    ):
        """
        Renders an Ultra-Crisp HD Analysis Pane using PIL TrueType anti-aliased text rendering.
        """
        # Convert pane ROI to PIL Image
        pane_y1 = art_height
        pane_y2 = art_height + pane_height

        pane_bgr = np.zeros((pane_height, width, 3), dtype=np.uint8)
        pane_bgr[:] = (10, 12, 18)  # Deep OLED dark slate background

        # Draw glowing Cyan top divider accent line
        cv2.line(pane_bgr, (0, 0), (width, 0), (0, 220, 255), 3)

        pane_rgb = cv2.cvtColor(pane_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pane_rgb)
        draw = ImageDraw.Draw(pil_img)

        # -------------------------------------------------------------
        # Left Panel: Crisp Text & High-Alarm Timing Badge
        # -------------------------------------------------------------
        timestamp_sec = frame_idx / fps
        mins = int(timestamp_sec // 60)
        secs = timestamp_sec % 60
        time_str = f"{mins:02d}:{secs:05.2f}"

        margin_x = 24
        start_y = 18

        # 1. Header Frame Count & Timestamp
        info_text = f"FRAME: {frame_idx:04d} / {total_frames:04d}  |  TIME: {time_str}"
        draw.text((margin_x, start_y), info_text, fill=(240, 245, 250), font=self.font_bold_md)

        # 2. Timing Badge Color Rules (Preserving User Glaring Alarms)
        if "1s" in current_timing:
            badge_bg = (255, 30, 50)       # GLARED RED / Hot Magenta for 1s (24 FPS)
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
        elif "NEW" in current_timing:
            badge_bg = (255, 140, 0)       # BRIGHT ORANGE for Key Drawing
            badge_fg = (255, 255, 255)
            badge_text = "KEY DRAWING DETECTED"
        else:
            badge_bg = (90, 100, 115)      # Dim Gray for Hold
            badge_fg = (240, 240, 240)
            badge_text = f"TIMING: {current_timing}"

        # Render Pill/Badge Container
        badge_y = start_y + 26
        badge_w, badge_h = 240, 28
        draw.rectangle([margin_x, badge_y, margin_x + badge_w, badge_y + badge_h], fill=badge_bg)
        draw.text((margin_x + 12, badge_y + 5), badge_text, fill=badge_fg, font=self.font_bold_sm)

        # 3. Camera Pan Tracker & Motion Percentage
        pan_status = f"CAM PAN: {pan_mag:.1f}px (dx={shift_x:+.1f}, dy={shift_y:+.1f})" if pan_mag > 0.4 else "CAM PAN: STATIC"
        pan_text = f"{pan_status}  |  CHG: {diff_percentage:.1f}%"
        draw.text((margin_x, badge_y + badge_h + 10), pan_text, fill=(160, 175, 195), font=self.font_bold_sm)

        # -------------------------------------------------------------
        # Right Top Panel: Rolling X-Sheet Timeline Bar Graph
        # -------------------------------------------------------------
        timeline_x1 = int(width * 0.40)
        timeline_w = int(width * 0.56)
        timeline_y = 28
        cell_h = 32

        recent_drawings = drawing_history[-self.timeline_window:]
        num_cells = self.timeline_window
        cell_w = max(4, timeline_w // num_cells)

        # Title Label
        draw.text((timeline_x1, timeline_y - 18), "DIGITAL X-SHEET TIMELINE  •  LAST 48 FRAMES", fill=(190, 200, 215), font=self.font_bold_sm)

        # Track Outer Frame
        track_x2 = timeline_x1 + num_cells * cell_w
        draw.rectangle([timeline_x1, timeline_y, track_x2, timeline_y + cell_h], fill=(20, 26, 38), outline=(60, 75, 95), width=1)

        # Draw Cells
        for i in range(len(recent_drawings)):
            cx = timeline_x1 + i * cell_w
            is_draw = recent_drawings[i] == 1

            if is_draw:
                bar_color = (255, 210, 0) if i == len(recent_drawings) - 1 else (230, 180, 0)
                draw.rectangle([cx + 1, timeline_y + 2, cx + cell_w - 1, timeline_y + cell_h - 2], fill=bar_color)
            else:
                # Hold dot
                draw.line([(cx + cell_w // 2, timeline_y + cell_h // 2 - 1), (cx + cell_w // 2, timeline_y + cell_h // 2 + 1)], fill=(90, 105, 125), width=2)

        # Playback Cursor
        curr_cx = timeline_x1 + (len(recent_drawings) - 1) * cell_w
        draw.rectangle([curr_cx - 1, timeline_y - 2, curr_cx + cell_w + 1, timeline_y + cell_h + 2], outline=(0, 255, 120), width=2)

        # -------------------------------------------------------------
        # Right Bottom Panel: Motion Velocity Sparkline Graph
        # -------------------------------------------------------------
        spark_y1 = timeline_y + cell_h + 12
        spark_h = 28
        recent_diffs = diff_history[-num_cells:]

        draw.text((timeline_x1, spark_y1 + spark_h + 4), "MOTION VELOCITY (CHG %)", fill=(130, 145, 165), font=self.font_bold_sm)

        if len(recent_diffs) > 1:
            max_diff = max(10.0, max(recent_diffs))
            points = []
            for idx_val, val in enumerate(recent_diffs):
                px = timeline_x1 + idx_val * cell_w + cell_w // 2
                py = int(spark_y1 + spark_h - (val / max_diff) * spark_h)
                points.append((px, py))

            # Draw sparkline vector line
            for i_pt in range(1, len(points)):
                draw.line([points[i_pt - 1], points[i_pt]], fill=(0, 230, 160), width=2)

        # Convert back to BGR and place into canvas bottom pane
        res_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        canvas[pane_y1:pane_y2, 0:width] = res_bgr


def process_timing_graph(
    input_path: str,
    output_path: str,
    noise_threshold: float = 2.2,
    enable_pan_compensation: bool = True
):
    processor = DigitalXSheetProcessor(
        noise_threshold=noise_threshold,
        enable_pan_compensation=enable_pan_compensation
    )
    return processor.process_video(input_path, output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        outp = sys.argv[2] if len(sys.argv) > 2 else "output_xsheet_v4.mp4"
        process_timing_graph(inp, outp)
    else:
        print("Usage: python3 timing_xsheet.py <input_video.mp4> [output_video.mp4]")
