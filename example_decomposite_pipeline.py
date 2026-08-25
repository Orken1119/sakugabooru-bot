"""
Example Pipeline Integration: Automatic De-Compositing & X-Sheet Generation
-----------------------------------------------------------------------------
1. Extracts clean pencil line art from a colored final composite sakuga clip using LineExtractor.
2. Feeds the extracted line art stream directly into DigitalXSheetProcessor.
3. Outputs an expanded canvas video with zero overlays on the line art and a dedicated X-Sheet HUD below.
"""

import os
import subprocess
import cv2
import numpy as np

from line_extractor import LineExtractor
from timing_xsheet import DigitalXSheetProcessor


def run_decomposite_xsheet_pipeline(input_path: str, output_path: str):
    print(f"--- [Pipeline Step 1] Initializing LineExtractor for '{input_path}' ---")
    line_extractor = LineExtractor(batch_size=2, use_fp16=True)

    print(f"--- [Pipeline Step 2] Initializing DigitalXSheetProcessor ---")
    xsheet_processor = DigitalXSheetProcessor(
        noise_threshold=2.2,
        enable_pan_compensation=True
    )

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or np.isnan(fps):
        fps = 24.0

    pane_height = max(160, int(height * 0.32))
    output_height = height + pane_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, output_height))

    prev_gray = None
    drawing_history = []
    diff_score_history = []

    crop_y = int(height * 0.06)
    crop_x = int(width * 0.06)
    valid_area = max(1, (height - 2 * crop_y) * (width - 2 * crop_x))

    frame_idx = 0
    print(f"--- [Pipeline Step 3] De-Compositing & Burning X-Sheet HUD to '{output_path}' ---")

    # Mode B: Stream extracted sketch frames directly without writing temporary video files!
    for sketch_frame in line_extractor.process_stream(input_path):
        gray = cv2.cvtColor(sketch_frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        shift_x, shift_y, pan_mag = 0.0, 0.0, 0.0
        is_new_drawing = False
        diff_percentage = 0.0

        if prev_gray is not None:
            shift_x, shift_y, pan_mag = xsheet_processor._estimate_camera_shift(prev_gray, gray_blur)
            aligned_prev = xsheet_processor._align_frame(prev_gray, shift_x, shift_y)

            frame_diff = cv2.absdiff(gray_blur, aligned_prev)
            _, thresh = cv2.threshold(frame_diff, xsheet_processor.diff_pixel_threshold, 255, cv2.THRESH_BINARY)

            thresh_cropped = thresh[crop_y:height - crop_y, crop_x:width - crop_x]
            changed_pixels = np.count_nonzero(thresh_cropped)
            diff_percentage = (changed_pixels / valid_area) * 100.0

            effective_threshold = xsheet_processor.noise_threshold + (0.8 * min(pan_mag, 5.0))
            if diff_percentage > effective_threshold:
                is_new_drawing = True
        else:
            is_new_drawing = True
            diff_percentage = 10.0

        prev_gray = gray_blur.copy()
        drawing_history.append(1 if is_new_drawing else 0)
        diff_score_history.append(diff_percentage)

        current_timing, spacing = xsheet_processor._classify_timing_type(drawing_history[-24:])

        # Build Expanded Canvas: Top = Clean Extracted Pencil Art, Bottom = HD Analysis Pane
        canvas = np.zeros((output_height, width, 3), dtype=np.uint8)
        canvas[0:height, 0:width] = sketch_frame

        xsheet_processor._render_hd_analysis_pane(
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

    out.release()

    # Convert to web-compatible H.264 format
    h264_tmp = output_path.rsplit('.', 1)[0] + "_h264.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", output_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        h264_tmp
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(h264_tmp, output_path)

    print(f"--- [Pipeline Complete] Output saved: '{output_path}' ---")
    return output_path


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "sakugabooru-video-files/2026-08-25/finals/5_7.00s.mp4"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sakugabooru-video-files/2026-08-25/x_sheets/decomposite_xsheet_demo.mp4"
    run_decomposite_xsheet_pipeline(input_file, output_file)
