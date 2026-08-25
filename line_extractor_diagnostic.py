"""
LineExtractor Diagnostic Tool & Tensor Pipeline Inspector
----------------------------------------------------------
Performs multi-stage shape, dtype, stride, and range inspection at every stage of the
Anime2Sketch inference and post-processing pipeline to detect memory misalignment,
channel mismatches, or U-Net skip-connection checkerboard artifacts.
"""

import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from line_extractor import SketchGenerator


def print_stage_info(stage_name: str, data, extra_info: str = ""):
    """Helper to print uniform diagnostic information for tensors and numpy arrays."""
    print(f"\n=======================================================")
    print(f"STAGE: {stage_name}")
    print(f"=======================================================")
    if isinstance(data, torch.Tensor):
        print(f"  - Type         : torch.Tensor ({data.device})")
        print(f"  - Shape        : {tuple(data.shape)}  (dim={data.dim()})")
        print(f"  - Dtype        : {data.dtype}")
        print(f"  - Stride       : {data.stride()}")
        print(f"  - Contiguous   : {data.is_contiguous()}")
        data_flt = data.detach().cpu().float()
        print(f"  - Range        : min = {data_flt.min().item():.4f}, max = {data_flt.max().item():.4f}")
    elif isinstance(data, np.ndarray):
        print(f"  - Type         : np.ndarray")
        print(f"  - Shape        : {data.shape}  (ndim={data.ndim})")
        print(f"  - Dtype        : {data.dtype}")
        print(f"  - Strides      : {data.strides}")
        print(f"  - Contiguous   : {data.flags['C_CONTIGUOUS']}")
        print(f"  - Range        : min = {data.min()}, max = {data.max()}")
    else:
        print(f"  - Type         : {type(data)}")
        print(f"  - Value        : {data}")
    
    if extra_info:
        print(f"  - Notes        : {extra_info}")


def run_diagnostic_pipeline(
    input_frame_or_path,
    input_channels: int = 1,   # 1 for Grayscale [1, 1, H, W], 3 for RGB [1, 3, H, W]
    align_modulo: int = 32,    # U-Net dimension constraint (multiples of 32)
    device_str: str = "cuda"
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Diagnostic Mode] Using Device: {device}")

    # -------------------------------------------------------------------------
    # STAGE 1: Initial Raw OpenCV BGR Array
    # -------------------------------------------------------------------------
    if isinstance(input_frame_or_path, str):
        if not os.path.exists(input_frame_or_path):
            raise FileNotFoundError(f"Input video file not found: {input_frame_or_path}")
        cap = cv2.VideoCapture(input_frame_or_path)
        ret, raw_bgr = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Failed to read video frame.")
    else:
        raw_bgr = input_frame_or_path

    print_stage_info("1. Initial Raw OpenCV BGR Frame", raw_bgr, "Original video frame directly from cv2.VideoCapture")

    orig_h, orig_w = raw_bgr.shape[:2]

    # -------------------------------------------------------------------------
    # STAGE 2: Architecture Constraints & Preprocessing
    # -------------------------------------------------------------------------
    # Rule 1: Enforce U-Net dimension divisibility by modulo (e.g. 32)
    aligned_h = ((orig_h + align_modulo - 1) // align_modulo) * align_modulo
    aligned_w = ((orig_w + align_modulo - 1) // align_modulo) * align_modulo
    
    print(f"\n[Architecture Constraint Verification]")
    print(f"  - Original Dimensions : {orig_w}x{orig_h}")
    print(f"  - Aligned Dimensions  : {aligned_w}x{aligned_h} (Divisible by {align_modulo})")

    # Rule 2: Channel Count Verification (1-Channel Grayscale vs 3-Channel RGB)
    if input_channels == 1:
        gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (aligned_w, aligned_h), interpolation=cv2.INTER_AREA)
        # Normalize [0, 255] -> [-1.0, 1.0]
        norm_arr = (resized.astype(np.float32) / 127.5) - 1.0
        # Shape: [1, H, W]
        input_tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).to(device)
        chan_note = "1-Channel Grayscale [B=1, C=1, H, W] for Anime2Sketch"
    else:
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (aligned_w, aligned_h), interpolation=cv2.INTER_AREA)
        norm_arr = (resized.astype(np.float32) / 127.5) - 1.0
        # Shape: [1, 3, H, W]
        input_tensor = torch.from_numpy(norm_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        chan_note = "3-Channel RGB [B=1, C=3, H, W]"

    print_stage_info("2. Preprocessed Tensor (Input to Model)", input_tensor, chan_note)

    # -------------------------------------------------------------------------
    # STAGE 3: Model Forward Pass & Raw Output Tensor
    # -------------------------------------------------------------------------
    model = SketchGenerator(in_channels=input_channels, out_channels=1).to(device)
    model.eval()

    with torch.no_grad():
        raw_output = model(input_tensor)

    print_stage_info("3. Raw Output Tensor (From Model)", raw_output, "Output tensor directly returned by forward pass")

    # -------------------------------------------------------------------------
    # STAGE 4: Squeeze & Tensor Dimension Verification (Preventing 4D Permute Bug)
    # -------------------------------------------------------------------------
    # CRITICAL VERIFICATION: Ensure Squeeze is called to get a 3D Tensor [C, H, W] BEFORE Permute!
    tensor_3d = raw_output.detach().cpu().float()
    
    if tensor_3d.dim() == 4:
        # Squeeze batch dimension [1, C, H, W] -> [C, H, W]
        tensor_3d = tensor_3d.squeeze(0)

    print_stage_info("4. Squeezed 3D Tensor [C, H, W]", tensor_3d, "Batch dimension removed. Must be 3D [C, H, W] before permute!")

    # -------------------------------------------------------------------------
    # STAGE 5: Denormalization & Clamping
    # -------------------------------------------------------------------------
    # Map [-1.0, 1.0] -> [0.0, 255.0] with strict torch.clamp() to prevent uint8 underflow
    denorm_tensor = (tensor_3d + 1.0) * 127.5
    clamped_tensor = torch.clamp(denorm_tensor, 0.0, 255.0).to(torch.uint8)

    print_stage_info("5. Denormalized & Clamped uint8 Tensor", clamped_tensor, "Mapped from [-1, 1] to [0, 255] uint8")

    # -------------------------------------------------------------------------
    # STAGE 6: Dimension Permutation [C, H, W] -> [H, W, C]
    # -------------------------------------------------------------------------
    # Correct permutation swapping memory strides
    permuted_tensor = clamped_tensor.permute(1, 2, 0)
    
    # Ensure memory is C-contiguous before passing to NumPy / OpenCV
    if not permuted_tensor.is_contiguous():
        permuted_tensor = permuted_tensor.contiguous()

    output_np = permuted_tensor.numpy()

    print_stage_info("6. Permuted NumPy Array [H, W, C]", output_np, "Proper permute(1,2,0) converted to C-contiguous NumPy")

    # -------------------------------------------------------------------------
    # STAGE 7: Final Color Formatting & Resizing to Original Dimensions
    # -------------------------------------------------------------------------
    if output_np.shape[2] == 1:
        final_bgr = cv2.cvtColor(output_np[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        final_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

    if (final_bgr.shape[1], final_bgr.shape[0]) != (orig_w, orig_h):
        final_bgr = cv2.resize(final_bgr, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

    print_stage_info("7. Final Output 3-Channel BGR Frame", final_bgr, f"Resized back to original video dimensions ({orig_w}x{orig_h})")

    return final_bgr


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "sakugabooru-video-files/2026-08-25/finals/5_7.00s.mp4"
    print("=" * 60)
    print(" ANIME2SKETCH / LINE EXTRACTION DIAGNOSTIC RUNNER")
    print("=" * 60)
    res = run_diagnostic_pipeline(video_path, input_channels=3, align_modulo=32)
    print("\n[Diagnostic Mode Complete] All stages inspected successfully!")
