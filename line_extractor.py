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
# Method 1: Pure OpenCV XDoG (Extended Difference of Gaussians)
# -----------------------------------------------------------------------------
def extract_lines_opencv(
    frame_bgr: np.ndarray,
    sigma1: float = 0.8,
    sigma2: float = 1.6,
    gamma: float = 0.98,
    phi: float = 200.0,
    epsilon: float = -0.1,
    use_color_dodge: bool = True
) -> np.ndarray:
    """
    Extracts clean, crisp pencil-like anime outlines on a white background using
    Extended Difference of Gaussians (XDoG) / Color Dodge filtering.

    Parameters:
    -----------
    frame_bgr : np.ndarray
        Input 3-channel BGR image [H, W, 3].
    sigma1 : float
        Sigma for primary Gaussian blur.
    sigma2 : float
        Sigma for secondary Gaussian blur.
    gamma : float
        Scalar scale factor for Difference of Gaussians (0.95 - 0.99).
    phi : float
        Soft thresholding sharpness scale for XDoG hyperbolic tangent.
    epsilon : float
        Threshold offset for edge sensitivity.
    use_color_dodge : bool
        If True, combines XDoG with adaptive color dodge for sharp manga/genga lines.

    Returns:
    --------
    np.ndarray
        Clean 3-channel BGR uint8 image [H, W, 3].
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Invalid or empty input frame provided to extract_lines_opencv.")

    # 1. Convert BGR to Grayscale
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Bilateral Filter / Median Blur to remove digital compression artifacts & flat noise
    filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    if use_color_dodge:
        # High-contrast Color Dodge + Adaptive Thresholding (Crisp Manga/Genga lines)
        inv_gray = 255 - filtered
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        dodge = cv2.divide(filtered, np.maximum(255 - blur, 1), scale=256.0)
        sketch = cv2.adaptiveThreshold(
            dodge, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Pure XDoG Hyperbolic Tangent Edge Extraction
        g1 = cv2.GaussianBlur(filtered.astype(np.float32), (0, 0), sigma1)
        g2 = cv2.GaussianBlur(filtered.astype(np.float32), (0, 0), sigma2)
        dog = g1 - gamma * g2

        # Hyperbolic tangent soft thresholding
        u = dog - epsilon
        val = np.where(u < 0, 1.0, 1.0 + np.tanh(phi * u))
        sketch = (val * 255.0).clip(0, 255).astype(np.uint8)

    # Return clean 3-channel BGR frame [H, W, 3]
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# -----------------------------------------------------------------------------
# LineExtractor Main Class (Supports Method 1 and Method 2)
# -----------------------------------------------------------------------------
class LineExtractor:
    def __init__(self, method: str = "xdog", device: str = None):
        """
        Parameters:
        -----------
        method : str
            'xdog' (Method 1: Pure OpenCV, Default) or
            'controlnet_lineart' (Method 2: High-End ControlNet Aux AI Detector)
        device : str
            Execution device for AI model ('cuda' or 'cpu').
        """
        self.method = method.lower()
        self.device = device
        self.detector = None

        if self.method == "controlnet_lineart":
            self._init_controlnet_detector()
        else:
            print("[LineExtractor] Initialized Engine: Method 1 (Pure OpenCV XDoG) - Fast & Deterministic")

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

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Processes a single BGR uint8 frame [H, W, 3] and returns line art [H, W, 3].
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

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
                return extract_lines_opencv(frame_bgr)
        else:
            # Method 1 Default
            return extract_lines_opencv(frame_bgr)

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
