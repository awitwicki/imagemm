#!/usr/bin/env python3
"""
GPU-accelerated image processing for Apple Silicon (M1/M2/M3) using TensorFlow + Metal.
Performs Richardson–Lucy deconvolution and basic post-processing.

Folders:
  - input:  aligned .fit, .fits or .xisf files (monochrome or color)
  - output: results

Examples:
  uv run --with numpy,tensorflow-macos,xisf,tensorflow-metal,astropy,matplotlib,scipy,tqdm,scikit-image main.py --iters 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Try to import TensorFlow GPU stack
try:
    import tensorflow as tf
except Exception as e:
    print("ERROR: TensorFlow is required. On macOS with Apple Silicon, install:", file=sys.stderr)
    print("  uv add tensorflow-macos tensorflow-metal", file=sys.stderr)
    raise

# Try to import XISF support
try:
    import xisf
except ImportError:
    xisf = None
    print("[WARN] xisf package not available. Install with: uv add xisf")

# Try to import FITS support
try:
    from astropy.io import fits
except ImportError:
    fits = None
    print("[WARN] astropy not available. Install with: uv add astropy")


# ---------------------------
# Device and precision config
# ---------------------------

def configure_device(mixed_precision: bool) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Using GPU: {[d.name for d in gpus]}")
        except Exception as e:
            print(f"[WARN] Could not set GPU memory growth: {e}")
    else:
        print("[INFO] No GPU found. Running on CPU.")

    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled (float16 compute, float32 vars).")
        except Exception as e:
            print(f"[WARN] Mixed precision not enabled: {e}")


# ---------------------------
# File I/O with XISF and FITS support
# ---------------------------

def read_xisf(path: Path) -> np.ndarray:
    """Read XISF file and return image data as numpy array."""
    if xisf is None:
        raise ImportError("xisf package required for XISF support")

    try:
        xisf_file = xisf.XISF(str(path))
        image_data = xisf_file.read_image(0)  # Read first image
        return image_data.astype(np.float32)
    except Exception as e:
        raise IOError(f"Failed to read XISF file {path}: {e}")


def read_fits(path: Path) -> np.ndarray:
    """Read FITS file and return image data as numpy array."""
    if fits is None:
        raise ImportError("astropy required for FITS support")

    try:
        with fits.open(path, memmap=False) as hdul:
            # Get the first HDU with data
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break
            else:
                raise ValueError(f"No image data found in FITS file {path}")

            # Handle different data types
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            return data
    except Exception as e:
        raise IOError(f"Failed to read FITS file {path}: {e}")


def read_image(path: Path) -> np.ndarray:
    """Read image file (XISF, FIT, or FITS) and return normalized numpy array."""
    suffix = path.suffix.lower()

    if suffix == '.xisf':
        data = read_xisf(path)
    elif suffix in ['.fit', '.fits']:
        data = read_fits(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Handle different data types and shapes
    if data.ndim == 3:
        # Color image - ensure channel order is correct
        if data.shape[0] in [3, 4]:  # [channels, height, width]
            data = np.transpose(data, (1, 2, 0))  # [height, width, channels]
        elif data.shape[2] in [3, 4]:  # [height, width, channels]
            pass  # Already in correct format
        # If it's 3D but not color (e.g., [depth, height, width]), take first slice
        elif data.shape[0] not in [3, 4] and data.shape[2] not in [3, 4]:
            data = data[0]  # Take first frame
            data = data[..., np.newaxis]  # Add channel dimension
    elif data.ndim == 2:
        # Monochrome - add channel dimension
        data = data[..., np.newaxis]
    elif data.ndim > 3:
        # Higher dimensional data - take first frame and squeeze
        data = data[0]
        while data.ndim > 2:
            data = data[0]
        data = data[..., np.newaxis]
    else:
        raise ValueError(f"Unsupported image dimensions: {data.ndim}")

    # Remove alpha channel if present
    if data.shape[-1] == 4:
        data = data[..., :3]

    return data


def write_fits(path: Path, image: np.ndarray) -> None:
    """Write image data to FITS file."""
    if fits is None:
        raise ImportError("astropy required for FITS output")

    # Remove channel dimension for monochrome, preserve for color
    if image.ndim == 3 and image.shape[-1] == 1:
        output_data = image[..., 0]
    else:
        output_data = image

    hdu = fits.PrimaryHDU(output_data.astype(np.float32))
    hdul = fits.HDUList([hdu])
    path.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(path, overwrite=True)


def load_input_stack(input_dir: Path) -> Tuple[List[np.ndarray], str]:
    """Load all images from input directory and return stack + color mode."""
    # Find all supported image files
    supported_extensions = ['.xisf', '.fit', '.fits']
    files = []
    for ext in supported_extensions:
        files.extend(input_dir.glob(f"*{ext}"))
        # Also check uppercase extensions
        files.extend(input_dir.glob(f"*{ext.upper()}"))

    files = sorted([p for p in files if p.is_file()])

    if not files:
        available_files = list(input_dir.glob("*"))
        available_extensions = {f.suffix for f in available_files if f.is_file()}
        raise FileNotFoundError(
            f"No supported image files (.xisf, .fit, .fits) found in {input_dir}. "
            f"Available files: {list(available_files)[:10]}... "  # Show first 10 files
            f"Available extensions: {available_extensions}"
        )

    print(f"[INFO] Found {len(files)} image files:")
    for f in files[:5]:  # Show first 5 files
        print(f"       {f.name}")
    if len(files) > 5:
        print(f"       ... and {len(files) - 5} more")

    # Load images
    stack = []
    failed_files = []




    ################################################################################################################################################################################################################################################################################################################################################
    # files = files[:5]




    for p in tqdm(files, desc="Loading images"):
        try:
            img = read_image(p)
            stack.append(img)
        except Exception as e:
            print(f"[WARN] Failed to load {p.name}: {e}")
            failed_files.append(p.name)

    if not stack:
        raise ValueError(f"Could not load any images from {input_dir}")

    if failed_files:
        print(f"[WARN] Failed to load {len(failed_files)}")

    # Determine color mode from first image
    first_img = stack[0]
    if first_img.ndim == 3 and first_img.shape[-1] == 3:
        color_mode = "color"
        print("[INFO] Processing as color images (3 channels)")
    else:
        color_mode = "monochrome"
        print("[INFO] Processing as monochrome images")

    # Validate shapes
    shapes = {img.shape for img in stack}
    if len(shapes) != 1:
        print(f"[WARN] Input images have different shapes: {shapes}")
        # Find the most common shape and resize others
        from collections import Counter
        shape_counter = Counter(img.shape for img in stack)
        most_common_shape = shape_counter.most_common(1)[0][0]
        print(f"[INFO] Resizing all images to most common shape: {most_common_shape}")

        resized_stack = []
        for i, img in enumerate(stack):
            if img.shape != most_common_shape:
                from skimage.transform import resize
                print(f"[INFO] Resizing image {i} from {img.shape} to {most_common_shape}")
                if img.ndim == 3:
                    # Color image
                    resized_img = np.zeros(most_common_shape, dtype=img.dtype)
                    for c in range(min(img.shape[2], most_common_shape[2])):
                        resized_img[..., c] = resize(img[..., c], most_common_shape[:2])
                else:
                    # Monochrome
                    resized_img = resize(img, most_common_shape)
                resized_stack.append(resized_img)
            else:
                resized_stack.append(img)
        stack = resized_stack

    return stack, color_mode


# ---------------------------
# PSF utilities
# ---------------------------

def make_gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Create a normalized 2D Gaussian PSF."""
    if size % 2 == 0:
        size += 1  # ensure odd for proper centering
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    psf = psf / np.clip(psf.sum(), 1e-8, np.inf)
    return psf


def load_or_build_psf(psf_path: Path | None, psf_size: int, psf_sigma: float) -> np.ndarray:
    if psf_path and psf_path.exists():
        print(f"[INFO] Loading PSF from {psf_path}")
        try:
            psf = read_image(psf_path)

            # Ensure PSF is 2D
            if psf.ndim == 3:
                psf = psf.mean(axis=-1)  # Convert to 2D by averaging channels

            psf_sum = psf.sum()
            if not np.isfinite(psf_sum) or psf_sum <= 0:
                raise ValueError("Loaded PSF is invalid or non-positive.")
            psf = psf.astype(np.float32)
            psf /= psf_sum
            return psf
        except Exception as e:
            print(f"[WARN] Failed to load PSF from {psf_path}: {e}. Building Gaussian PSF instead.")

    print(f"[INFO] Building Gaussian PSF (size={psf_size}, sigma={psf_sigma})")
    return make_gaussian_psf(psf_size, psf_sigma)


# ---------------------------
# TensorFlow RL Deconvolution
# ---------------------------

def rl_deconvolve_single_channel(
        observed: np.ndarray,
        psf: np.ndarray,
        iterations: int,
) -> np.ndarray:
    """
    Richardson-Lucy deconvolution for single channel images.
    Uses proper TensorFlow convolution with correct kernel shapes.
    """
    # Ensure inputs are 2D
    observed_2d = observed.squeeze().astype(np.float32)
    psf_2d = psf.squeeze().astype(np.float32)

    # Normalize PSF
    psf_2d = psf_2d / np.sum(psf_2d)

    # Convert to TensorFlow tensors with proper shapes
    # Input: [batch, height, width, channels] = [1, H, W, 1]
    # Kernel: [filter_height, filter_width, in_channels, out_channels] = [psf_h, psf_w, 1, 1]
    observed_tf = tf.constant(observed_2d[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    psf_tf = tf.constant(psf_2d[:, :, np.newaxis, np.newaxis], dtype=tf.float32)
    psf_flip_tf = tf.constant(np.flip(psf_2d)[:, :, np.newaxis, np.newaxis], dtype=tf.float32)

    # Initial estimate
    x = tf.clip_by_value(observed_tf, 0.0, tf.float32.max)

    eps = tf.constant(1e-7, dtype=tf.float32)

    @tf.function(jit_compile=False)
    def rl_iteration(x_current):
        # Convolution: x ⊗ PSF
        conv = tf.nn.conv2d(x_current, psf_tf, strides=[1, 1, 1, 1], padding="SAME")
        # Ratio: observed / (x ⊗ PSF)
        ratio = observed_tf / (conv + eps)
        # Correlation: ratio ⊗ flipped_PSF
        corr = tf.nn.conv2d(ratio, psf_flip_tf, strides=[1, 1, 1, 1], padding="SAME")
        # Update: x * corr
        x_new = x_current * corr
        # Clamp to positive values
        return tf.clip_by_value(x_new, 0.0, tf.float32.max)

    # Perform iterations
    for i in tqdm(range(iterations), desc="RL Deconvolution", unit="iter"):
        x = rl_iteration(x)

    # Convert back to numpy
    result = x.numpy()[0, :, :, 0]
    return result.astype(np.float32)


def normalize_image(img: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.0) -> np.ndarray:
    """Normalize image using percentiles, handling multi-channel images."""
    if img.ndim == 3 and img.shape[-1] > 1:
        # Normalize each channel independently
        result = np.zeros_like(img)
        for c in range(img.shape[-1]):
            channel = img[..., c]
            # Avoid normalizing if the channel is empty
            if np.ptp(channel) > 0:  # peak-to-peak > 0
                lo = np.percentile(channel, percentile_low)
                hi = np.percentile(channel, percentile_high)
                if hi <= lo:
                    hi = lo + 1e-6
                result[..., c] = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
            else:
                result[..., c] = channel
        return result.astype(np.float32)
    else:
        # Monochrome
        img_2d = img.squeeze()
        if np.ptp(img_2d) > 0:
            lo = np.percentile(img_2d, percentile_low)
            hi = np.percentile(img_2d, percentile_high)
            if hi <= lo:
                hi = lo + 1e-6
            img_2d = np.clip((img_2d - lo) / (hi - lo), 0.0, 1.0)
        return img_2d.astype(np.float32)


# ---------------------------
# Channel Processing
# ---------------------------

def process_channels_separately(
        observed: np.ndarray,
        psf: np.ndarray,
        iterations: int,
) -> np.ndarray:
    """Process each channel separately to avoid convolution dimension issues."""
    if observed.ndim == 2 or observed.shape[-1] == 1:
        # Monochrome case
        print("[INFO] Processing monochrome image")
        deconv = rl_deconvolve_single_channel(
            observed=observed,
            psf=psf,
            iterations=iterations,
        )
        return deconv[..., np.newaxis]  # Add channel dimension back

    else:
        # Color case - process each channel separately
        channels = observed.shape[-1]
        results = []

        for c in range(channels):
            print(f"[INFO] Processing channel {c + 1}/{channels}")
            # Extract single channel
            observed_channel = observed[..., c]

            deconv_channel = rl_deconvolve_single_channel(
                observed=observed_channel,
                psf=psf,  # Use the same 2D PSF for all channels
                iterations=iterations,
            )
            results.append(deconv_channel)

        # Stack channels back together
        return np.stack(results, axis=-1)


# ---------------------------
# Pipeline
# ---------------------------

def process_stack(
        input_dir: Path,
        output_dir: Path,
        iterations: int,
        psf_size: int,
        psf_sigma: float,
        mixed_precision: bool,
        save_intermediate: bool,
) -> None:
    configure_device(mixed_precision=mixed_precision)

    stack, color_mode = load_input_stack(input_dir)

    if color_mode == "color":
        H, W, C = stack[0].shape
        print(f"[INFO] Color image size: {H}x{W}x{C}")
    else:
        H, W = stack[0].shape[:2]
        print(f"[INFO] Monochrome image size: {H}x{W}")

    # Average the aligned stack to boost SNR before deconvolution
    print("[INFO] Averaging stack...")
    if color_mode == "color":
        acc = np.zeros((H, W, 3), dtype=np.float64)
    else:
        acc = np.zeros((H, W), dtype=np.float64)

    for img in tqdm(stack, desc="Stacking", unit="img"):
        # print(img[:, :, 0].shape)
        # print(acc.shape)
        acc += img[:, :, 0].astype(np.float64)

    mean_img = (acc / len(stack)).astype(np.float32)
    mean_img = normalize_image(mean_img, percentile_low=1.0, percentile_high=99.5)

    print(f"[INFO] Stack mean - shape: {mean_img.shape}, range: [{mean_img.min():.3f}, {mean_img.max():.3f}]")

    # Build or load PSF - always use 2D PSF
    psf = load_or_build_psf(psf_path=None, psf_size=psf_size, psf_sigma=psf_sigma)
    print(f"[INFO] PSF - shape: {psf.shape}, sum: {psf.sum():.6f}")

    print(f"[INFO] Starting RL deconvolution for {iterations} iterations...")

    # Use the channel-separated approach
    deconv = process_channels_separately(
        observed=mean_img,
        psf=psf,
        iterations=iterations,
    )

    # Normalize the deconvolved result
    deconv = normalize_image(deconv, percentile_low=0.5, percentile_high=99.8)
    print(f"[INFO] Deconvolved - shape: {deconv.shape}, range: [{deconv.min():.3f}, {deconv.max():.3f}]")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use .fits extension for output files
    write_fits(output_dir / "deconvolved.fits", deconv)
    write_fits(output_dir / "stack_mean.fits", mean_img)
    if save_intermediate:
        write_fits(output_dir / "psf.fits", psf)

    print(f"[INFO] Saved results to {output_dir}")

    # Save preview images
    try:
        import matplotlib.pyplot as plt
        if color_mode == "color":
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            axes[0].imshow(mean_img)
            axes[0].set_title("Input (stack mean)")
            axes[0].axis("off")
            axes[1].imshow(deconv)
            axes[1].set_title(f"Deconvolved (RL x{iterations})")
            axes[1].axis("off")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            im0 = axes[0].imshow(mean_img[..., 0] if mean_img.ndim == 3 else mean_img, cmap="gray")
            axes[0].set_title("Input (stack mean)")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(deconv[..., 0] if deconv.ndim == 3 else deconv, cmap="gray")
            axes[1].set_title(f"Deconvolved (RL x{iterations})")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        png_path = output_dir / "deconvolved.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved preview: {png_path}")
    except Exception as e:
        print(f"[WARN] Could not save PNG preview: {e}")

    print(f"[INFO] Processing complete! Results saved to: {output_dir}")


# ---------------------------
# CLI
# ---------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU-accelerated deconvolution on Apple Silicon using TensorFlow Metal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --iters 20
  python main.py --input data/ --output results/ --iters 30 --psf-size 25 --psf-sigma 2.5
        """
    )
    p.add_argument("--input", type=Path, default=Path("input"),
                   help="Input folder with aligned .fit, .fits or .xisf files (default: input)")
    p.add_argument("--output", type=Path, default=Path("output"),
                   help="Output folder (default: output)")
    p.add_argument("--iters", type=int, default=20,
                   help="RL iterations (default: 20)")
    p.add_argument("--psf-size", type=int, default=21,
                   help="PSF kernel size (odd, default: 21)")
    p.add_argument("--psf-sigma", type=float, default=3.0,
                   help="PSF Gaussian sigma in pixels (default: 3.0)")
    p.add_argument("--psf-file", type=Path, default=None,
                   help="Custom PSF file (optional)")
    p.add_argument("--no-mixed-precision", action="store_true",
                   help="Disable float16 mixed precision")
    p.add_argument("--save-intermediate", action="store_true",
                   help="Save PSF and intermediate files")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])

        # Check if input directory exists
        if not args.input.exists():
            print(f"ERROR: Input directory {args.input} does not exist")
            return 1

        # Check if we have required packages
        if xisf is None and fits is None:
            print("ERROR: No image I/O packages available.", file=sys.stderr)
            print("Install at least one of:", file=sys.stderr)
            print("  uv add xisf  (for XISF support)", file=sys.stderr)
            print("  uv add astropy  (for FITS support)", file=sys.stderr)
            return 1

        process_stack(
            input_dir=args.input,
            output_dir=args.output,
            iterations=args.iters,
            psf_size=args.psf_size,
            psf_sigma=args.psf_sigma,
            mixed_precision=not args.no_mixed_precision,
            save_intermediate=args.save_intermediate,
        )
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
