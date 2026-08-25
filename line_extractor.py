"""
LineExtractor: Robust Anime Line Art & Sketch Extraction Engine
----------------------------------------------------------------
Provides high-performance, deterministic line art extraction for anime/sakuga footage.

Supports two robust extraction pipelines:
1. Method 1 (Default): Pure OpenCV XDoG (Extended Difference of Gaussians) + Color Dodge.
   - Ultra-fast (100+ FPS), zero VRAM footprint, deterministic pencil line output.
2. Method 2 (Optional AI Toggle): ControlNet Aux LineartAnimeDetector.
   - High-end AI-based line art detector via controlnet-aux library.
"""

import os
import sys
import time
import subprocess
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

# -----------------------------------------------------------------------------
# Helper: Automatic Letterbox & Black Bar Cropping
# -----------------------------------------------------------------------------
def auto_crop_black_bars(frame_bgr: np.ndarray, threshold: int = 15) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Detects black letterboxing/pillarboxing bars and crops the active video content area.

    Parameters:
    -----------
    frame_bgr : np.ndarray
        Input 3-channel BGR image [H, W, 3].
    threshold : int
        Luminance threshold (0-255, default 15) to separate black letterbox bars from active content.

    Returns:
    --------
    tuple[np.ndarray, tuple[int, int, int, int]]
        - Cropped active frame content [h_crop, w_crop, 3]
        - Active bounding box coordinates (x, y, w, h) relative to original frame
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr, (0, 0, 0, 0)

    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold near-black pixels: active video content becomes white (255), letterbox bars stay black (0)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    non_zero = cv2.findNonZero(thresh)
    if non_zero is not None and len(non_zero) > 0:
        x, y, w, h = cv2.boundingRect(non_zero)
        # Ensure bounding box is valid and not collapsed
        if w > 20 and h > 20 and (w < W or h < H):
            cropped = frame_bgr[y:y+h, x:x+w]
            return cropped, (x, y, w, h)

    return frame_bgr, (0, 0, W, H)


# -----------------------------------------------------------------------------
# Method 1: Pure OpenCV XDoG (Extended Difference of Gaussians)
# -----------------------------------------------------------------------------
def extract_lines_opencv(
    frame_bgr: np.ndarray,
    denoise_strength: float = None,
    line_threshold: float = None,
    clean_speckles: bool = True,
    min_speckle_area: float = None,
    high_chaos_mode: bool = False,
    auto_crop: bool = True,
    crop_threshold: int = 15,
    sigma1: float = 0.6,
    sigma2: float = 1.2,
    gamma: float = 0.98,
    phi: float = 200.0,
    use_color_dodge: bool = True,
    debug_mode: bool = False,
    debug_dir: str = "."
) -> np.ndarray:
    """
    Optimized OpenCV line art extractor supporting automatic letterbox cropping
    and a High-Chaos preset for grainy scenes (explosions, impact frames, speed lines).

    Parameters:
    -----------
    frame_bgr : np.ndarray
        Input 3-channel BGR image [H, W, 3].
    denoise_strength : float
        Controls bilateral pre-filter intensity (default 40.0, or 60.0 in High-Chaos Mode).
    line_threshold : float
        Controls sensitivity to faint edges (default 0.5, or 0.80 in High-Chaos Mode).
    clean_speckles : bool
        Enables contour-based area filtering to remove isolated noise specks without eroding thin lines.
    min_speckle_area : float
        Minimum pixel area threshold for contour filtering (default 4.0, or 15.0 in High-Chaos Mode).
    high_chaos_mode : bool
        Enables aggressive preset against background grain & explosion particles (default False).
    auto_crop : bool
        Automatically crops letterbox black bars to prevent edge noise tracing (default True).
    crop_threshold : int
        Near-black pixel luminance threshold for letterbox detection (default 15).
    sigma1, sigma2, gamma, phi : float
        Calibrated XDoG parameters for sharp anime line art.
    use_color_dodge : bool
        If True, applies adaptive color dodge blend for crisp manga genga lines.
    debug_mode : bool
        If True, saves intermediate frames at 3 key stages for visual inspection.
    debug_dir : str
        Directory to save diagnostic debug images.

    Returns:
    --------
    np.ndarray
        Clean 3-channel BGR uint8 image [H, W, 3] (black ink on pure white background).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Invalid or empty input frame provided to extract_lines_opencv.")

    # High-Chaos Preset Defaults
    if high_chaos_mode:
        if denoise_strength is None: denoise_strength = 60.0
        if line_threshold is None: line_threshold = 0.80
        if min_speckle_area is None: min_speckle_area = 15.0
    else:
        if denoise_strength is None: denoise_strength = 40.0
        if line_threshold is None: line_threshold = 0.50
        if min_speckle_area is None: min_speckle_area = 4.0

    orig_H, orig_W = frame_bgr.shape[:2]

    # 1. Automatic Letterbox Cropping (Pre-Processing)
    if auto_crop:
        active_frame, (crop_x, crop_y, crop_w, crop_h) = auto_crop_black_bars(frame_bgr, threshold=crop_threshold)
    else:
        active_frame, (crop_x, crop_y, crop_w, crop_h) = frame_bgr, (0, 0, orig_W, orig_H)

    # 2. Convert BGR to Grayscale
    gray = cv2.cvtColor(active_frame, cv2.COLOR_BGR2GRAY)

    # 3. Edge-Preserving Pre-Smoothing (Bilateral Filtering)
    sig = float(denoise_strength)
    filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=sig, sigmaSpace=sig)

    # DEBUG STAGE 1: Saved immediately after Bilateral Filter
    if debug_mode:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "debug_1_blurred.png"), filtered)

    # 4. XDoG / Color Dodge Edge Extraction
    if use_color_dodge:
        inv_gray = 255 - filtered
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        dodge = cv2.divide(filtered, np.maximum(255 - blur, 1), scale=256.0)
        
        c_val = max(1, int(2.0 * line_threshold + 1.0))
        sketch = cv2.adaptiveThreshold(
            dodge, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, c_val
        )
    else:
        g1 = cv2.GaussianBlur(filtered.astype(np.float32), (0, 0), sigma1)
        g2 = cv2.GaussianBlur(filtered.astype(np.float32), (0, 0), sigma2)
        dog = g1 - gamma * g2

        epsilon = -0.1 * (1.0 / max(0.1, line_threshold))
        u = dog - epsilon
        val = np.where(u < 0, 1.0, 1.0 + np.tanh(phi * u))
        sketch = (val * 255.0).clip(0, 255).astype(np.uint8)

    # Save raw sketch before cleanup
    sketch_raw = sketch.copy()

    # DEBUG STAGE 2: Saved immediately after XDoG / Thresholding (before cleanup)
    if debug_mode:
        cv2.imwrite(os.path.join(debug_dir, "debug_2_xdog_raw.png"), sketch_raw)

    # 5. Contour Area Speckle Cleanup (No Line Erosion)
    if clean_speckles:
        # Invert so black ink lines/speckles are white (255) on black background (0)
        sketch_inv = 255 - sketch
        
        # Find contours of all black regions
        contours, _ = cv2.findContours(sketch_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill isolated background speckles smaller than min_speckle_area with black in sketch_inv (white in final sketch)
        for c in contours:
            if cv2.contourArea(c) < min_speckle_area:
                cv2.drawContours(sketch_inv, [c], -1, 0, -1)

        # Invert back to black ink lines on solid white background
        sketch = 255 - sketch_inv

    # 6. Re-pad to Original Canvas Dimensions (Replaces letterbox black bars with pure white margins)
    if auto_crop and (crop_w < orig_W or crop_h < orig_H):
        full_sketch = np.full((orig_H, orig_W), 255, dtype=np.uint8)
        full_sketch[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = sketch
        sketch = full_sketch

    # DEBUG STAGE 3: Final cleaned image
    if debug_mode:
        path3 = os.path.join(debug_dir, "debug_3_final.png")
        cv2.imwrite(path3, sketch)
        print(f"[LineExtractor Debug] Saved diagnostic frames to '{debug_dir}':")
        print(f"  1. {os.path.join(debug_dir, 'debug_1_blurred.png')}")
        print(f"  2. {os.path.join(debug_dir, 'debug_2_xdog_raw.png')}")
        print(f"  3. {path3}")

    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# -----------------------------------------------------------------------------
# LineExtractor Main Class (Supports Method 1 and Method 2)
# -----------------------------------------------------------------------------
class LineExtractor:
    def __init__(
        self,
        method: str = "xdog",
        device: str = None,
        denoise_strength: float = None,
        line_threshold: float = None,
        clean_speckles: bool = True,
        min_speckle_area: float = None,
        high_chaos_mode: bool = False,
        auto_crop: bool = True,
        crop_threshold: int = 15,
        debug_mode: bool = False,
        debug_dir: str = "."
    ):
        """
        Parameters:
        -----------
        method : str
            'xdog' (Method 1: Pure OpenCV, Default) or
            'controlnet_lineart' (Method 2: High-End ControlNet Aux AI Detector)
        device : str
            Execution device for AI model ('cuda' or 'cpu').
        denoise_strength : float
            Controls bilateral pre-filter intensity (default 40.0, or 60.0 in High-Chaos Mode).
        line_threshold : float
            Controls sensitivity to faint edges (default 0.5, or 0.80 in High-Chaos Mode).
        clean_speckles : bool
            Enables contour area filtering to erase isolated noise dots without eroding lines (default True).
        min_speckle_area : float
            Minimum contour area threshold for speckle removal (default 4.0, or 15.0 in High-Chaos Mode).
        high_chaos_mode : bool
            Enables aggressive preset against background grain & explosion particles (default False).
        auto_crop : bool
            Automatically crops letterbox black bars (default True).
        crop_threshold : int
            Luminance threshold for letterbox detection (default 15).
        debug_mode : bool
            Saves diagnostic intermediate frames if True (default False).
        debug_dir : str
            Directory for debug frame exports (default '.').
        """
        self.method = method.lower()
        self.device = device
        self.high_chaos_mode = high_chaos_mode
        self.auto_crop = auto_crop
        self.crop_threshold = crop_threshold
        self.clean_speckles = clean_speckles
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        self.detector = None

        if self.high_chaos_mode:
            self.denoise_strength = 60.0 if denoise_strength is None else denoise_strength
            self.line_threshold = 0.80 if line_threshold is None else line_threshold
            self.min_speckle_area = 15.0 if min_speckle_area is None else min_speckle_area
        else:
            self.denoise_strength = 40.0 if denoise_strength is None else denoise_strength
            self.line_threshold = 0.50 if line_threshold is None else line_threshold
            self.min_speckle_area = 4.0 if min_speckle_area is None else min_speckle_area

        if self.method == "controlnet_lineart":
            self._init_controlnet_detector()
        else:
            print(f"[LineExtractor] Initialized Engine: Method 1 (Pure OpenCV XDoG) - Contour Denoising (ChaosMode={self.high_chaos_mode}, Denoise={self.denoise_strength}, Threshold={self.line_threshold}, MinArea={self.min_speckle_area}px, AutoCrop={self.auto_crop})")

    def _init_controlnet_detector(self):
        """Initializes Method 2 (ControlNet Aux LineartAnimeDetector) with graceful fallback."""
        try:
            print("[LineExtractor] Initializing Method 2: ControlNet Aux LineartAnimeDetector...")
            from controlnet_aux import LineartAnimeDetector
            self.detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
            print("[LineExtractor] ControlNet Aux LineartAnimeDetector loaded successfully!")
        except ImportError:
            print("[LineExtractor Warning] 'controlnet-aux' is not installed!")
            print("[LineExtractor] To install AI detector: pip install controlnet-aux")
            print("[LineExtractor] Falling back to Method 1 (Pure OpenCV XDoG)...")
            self.method = "xdog"
        except Exception as err:
            print(f"[LineExtractor Warning] Failed to load ControlNet Aux detector: {err}")
            print("[LineExtractor] Falling back to Method 1 (Pure OpenCV XDoG)...")
            self.method = "xdog"

    def process_frame(self, frame_bgr: np.ndarray, debug_mode: bool = None) -> np.ndarray:
        """
        Processes a single BGR uint8 frame [H, W, 3] and returns line art [H, W, 3].
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        is_debug = self.debug_mode if debug_mode is None else debug_mode

        if self.method == "controlnet_lineart" and self.detector is not None:
            try:
                # Convert BGR -> PIL RGB
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                
                # Inference via controlnet_aux LineartAnimeDetector
                res_pil = self.detector(pil_img)
                
                # Convert PIL RGB -> BGR numpy array
                res_np = np.array(res_pil)
                if res_np.ndim == 2:
                    return cv2.cvtColor(res_np, cv2.COLOR_GRAY2BGR)
                return cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)
            except Exception as err:
                print(f"[LineExtractor Warning] ControlNet Aux frame error: {err}. Using XDoG fallback.")
                return extract_lines_opencv(
                    frame_bgr,
                    denoise_strength=self.denoise_strength,
                    line_threshold=self.line_threshold,
                    clean_speckles=self.clean_speckles,
                    min_speckle_area=self.min_speckle_area,
                    high_chaos_mode=self.high_chaos_mode,
                    auto_crop=self.auto_crop,
                    crop_threshold=self.crop_threshold,
                    debug_mode=is_debug,
                    debug_dir=self.debug_dir
                )
        else:
            # Method 1 Default
            return extract_lines_opencv(
                frame_bgr,
                denoise_strength=self.denoise_strength,
                line_threshold=self.line_threshold,
                clean_speckles=self.clean_speckles,
                min_speckle_area=self.min_speckle_area,
                high_chaos_mode=self.high_chaos_mode,
                auto_crop=self.auto_crop,
                crop_threshold=self.crop_threshold,
                debug_mode=is_debug,
                debug_dir=self.debug_dir
            )

    # -------------------------------------------------------------------------
    # Mode B: Streaming Generator
    # -------------------------------------------------------------------------
    def process_stream(self, input_path: str):
        """
        Yields processed line-art frames directly as np.ndarray (Mode B).
        Ideal for streaming into downstream processors (e.g., DigitalXSheetProcessor).
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {input_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield self.process_frame(frame)

        cap.release()

    # -------------------------------------------------------------------------
    # Mode A: Standalone Video Export
    # -------------------------------------------------------------------------
    def process_to_file(self, input_path: str, output_path: str) -> str:
        """
        Processes input video file and writes line-art video to output_path (Mode A).
        Enforces original resolution, framerate, and native H.264 web compatibility.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0 or np.isnan(fps):
            fps = 24.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        pbar = tqdm(total=total_frames, desc=f"[LineExtractor ({self.method.upper()})]", unit="frame")

        for line_frame in self.process_stream(input_path):
            out.write(line_frame)
            pbar.update(1)

        pbar.close()
        out.release()

        # Re-encode with FFmpeg H.264 for native browser/IDE playback
        h264_tmp = output_path.rsplit('.', 1)[0] + "_h264.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            h264_tmp
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(h264_tmp, output_path)
            print(f"[LineExtractor] Saved H.264 Line Art video: '{output_path}'")
        except Exception as err:
            print(f"[LineExtractor] H.264 encoding notice: {err}")

        return output_path

    # -------------------------------------------------------------------------
    # Mode C: Side-by-Side Split View
    # -------------------------------------------------------------------------
    def process_side_by_side(self, input_path: str, output_path: str, orientation: str = "horizontal") -> str:
        """
        Composites original composite footage alongside extracted line art (Mode C).
        """
        cap_raw = cv2.VideoCapture(input_path)
        width = int(cap_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap_raw.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or np.isnan(fps):
            fps = 24.0

        if orientation == "horizontal":
            out_w, out_h = width * 2, height
        else:
            out_w, out_h = width, height * 2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        pbar = tqdm(total=total_frames, desc=f"[LineExtractor Mode C ({self.method.upper()})]", unit="frame")

        for line_frame in self.process_stream(input_path):
            ret, raw_frame = cap_raw.read()
            if not ret:
                break

            if orientation == "horizontal":
                canvas = np.hstack([raw_frame, line_frame])
            else:
                canvas = np.vstack([raw_frame, line_frame])

            out.write(canvas)
            pbar.update(1)

        pbar.close()
        cap_raw.release()
        out.release()

        # Convert to H.264
        h264_tmp = output_path.rsplit('.', 1)[0] + "_h264.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            h264_tmp
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(h264_tmp, output_path)
            print(f"[LineExtractor] Saved Side-by-Side Video: '{output_path}'")
        except Exception as err:
            print(f"[LineExtractor] H.264 notice: {err}")

        return output_path


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        outp = sys.argv[2] if len(sys.argv) > 2 else "output_line_art.mp4"
        method_arg = sys.argv[3] if len(sys.argv) > 3 else "xdog"
        extractor = LineExtractor(method=method_arg)
        extractor.process_to_file(inp, outp)
    else:
        print("Usage: python3 line_extractor.py <input_video.mp4> [output_line_art.mp4] [method: xdog|controlnet_lineart]")
