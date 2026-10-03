import os
import re
import pydicom
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import torch

def order_slices_3d(dcm_paths: List[Path]) -> List[Path]:
    """
    Physically orders DICOM slices along the 3D anatomical volume axis.
    Computes projection scalar k = IPP · (IOP_row × IOP_col).
    Falls back to InstanceNumber or natural numeric sorting if tags are absent.
    """
    if len(dcm_paths) <= 1:
        return dcm_paths

    ORDER_TAGS = [
        (0x0020, 0x0032),  # ImagePositionPatient
        (0x0020, 0x0037),  # ImageOrientationPatient
        (0x0020, 0x0013),  # InstanceNumber
    ]

    keyed = []
    for p in dcm_paths:
        k = None
        try:
            ds = pydicom.dcmread(str(p), force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
            iop = getattr(ds, "ImageOrientationPatient", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) >= 6 and len(ipp) >= 3:
                r = np.array(iop[:3], dtype=float)
                c = np.array(iop[3:6], dtype=float)
                n = np.cross(r, c)
                pos = np.array(ipp[:3], dtype=float)
                k = float(np.dot(pos, n))
            if k is None:
                inst = getattr(ds, "InstanceNumber", None)
                if inst is not None:
                    k = float(inst)
        except Exception:
            k = None
        keyed.append((k, p))

    if any(k is None for k, _ in keyed):
        # Fallback to InstanceNumber or natural sort
        try:
            return sorted(dcm_paths, key=lambda x: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', x.name)])
        except Exception:
            return sorted(dcm_paths)

    return [p for _, p in sorted(keyed, key=lambda t: t[0])]


def infer_laterality_from_geometry(dcm_path: Union[str, Path], lat_offset_threshold: float = 10.0) -> Optional[str]:
    """
    Infers knee laterality ('L' or 'R') from patient coordinate X in ImagePositionPatient.
    Negative X in DICOM HFS coordinate system indicates Right side, positive indicates Left side.
    """
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True, stop_before_pixels=True, specific_tags=[(0x0020, 0x0032), (0x0020, 0x0060)])
        # Direct tag first
        lat_tag = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
        if lat_tag and str(lat_tag).strip().upper() in ("L", "R"):
            return str(lat_tag).strip().upper()

        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 1:
            x = float(ipp[0])
            if abs(x) >= lat_offset_threshold:
                return "R" if x < 0 else "L"
    except Exception:
        pass
    return None


def normalise_laterality(tensor: torch.Tensor, plane: str, lat: Optional[str] = None) -> torch.Tensor:
    """
    Standardizes medial/lateral spatial orientation between left and right knees.
    If right knee ('R'), horizontally flips Coronal and Axial slices so that the medial
    compartment is consistently positioned on the same side across all patients.
    For Sagittal views, flips the slice orientation to canonical anterior-posterior direction.

    Args:
        tensor: Tensor of shape (..., H, W).
        plane: 'Coronal', 'Axial', or 'Sagittal'.
        lat: 'R', 'L', or None/unknown.

    Returns:
        Standardized tensor with consistent spatial laterality.
    """
    if lat != 'R':
        return tensor
    if plane.capitalize() in ('Coronal', 'Axial'):
        # Flip along width dimension (last dimension)
        return torch.flip(tensor, dims=[-1])
    # Sagittal view: flip anterior-posterior (height dimension)
    return torch.flip(tensor, dims=[-2])


def mm_crop_and_resize(
    pixel_array: np.ndarray,
    pixel_spacing: Optional[float] = None,
    crop_mm: float = 135.0,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Crops a square region of physical size `crop_mm` around the image center,
    scaling all knees to a standard anatomical millimeter field of view.
    """
    h, w = pixel_array.shape
    if pixel_spacing and np.isfinite(pixel_spacing) and pixel_spacing > 0:
        cpx = int(round(crop_mm / pixel_spacing))
        if 16 < cpx < min(h, w):
            cy, cx = h // 2, w // 2
            half = cpx // 2
            pixel_array = pixel_array[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]

    if pixel_array.shape != target_size:
        pixel_array = cv2.resize(pixel_array, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    return pixel_array


def load_dicom_slice(
    dcm_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = (224, 224),
    crop_mm: Optional[float] = 135.0
) -> np.ndarray:
    """
    Loads a single DICOM slice, normalizes intensity to [0, 1],
    optionally crops to canonical mm field of view, and resizes to target_size.
    """
    dcm = pydicom.dcmread(str(dcm_path), force=True)
    pixel_array = dcm.pixel_array.astype(np.float32)

    # Invert if MONOCHROME1 (0 is white, max is black)
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    # Rescale slope and intercept if present
    slope = float(getattr(dcm, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept

    # Robust intensity normalization via 1st and 99th percentiles
    p1 = np.percentile(pixel_array, 1)
    p99 = np.percentile(pixel_array, 99)
    if p99 > p1:
        pixel_array = np.clip(pixel_array, p1, p99)
        pixel_array = (pixel_array - p1) / (p99 - p1)
    else:
        max_val = np.max(pixel_array)
        if max_val > 0:
            pixel_array = pixel_array / max_val
        else:
            pixel_array = np.zeros_like(pixel_array)

    # Physical spacing
    ps_val = None
    ps = getattr(dcm, "PixelSpacing", None)
    if ps is not None and len(ps) > 0:
        try:
            ps_val = float(ps[0])
        except Exception:
            ps_val = None

    if target_size is not None:
        pixel_array = mm_crop_and_resize(pixel_array, pixel_spacing=ps_val, crop_mm=crop_mm or 135.0, target_size=target_size)

    return pixel_array.astype(np.float32)


def sample_uniform_slices(slice_paths: List[Path], num_slices: int = 16, band: Tuple[float, float] = (0.15, 0.85)) -> List[Path]:
    """
    Samples `num_slices` paths across the active anatomical band of the ordered slice series.
    Band filtering (e.g. 15% to 85%) trims empty or out-of-joint peripheral slices.
    """
    n = len(slice_paths)
    if n == 0:
        return []
    if n == 1:
        return [slice_paths[0]] * num_slices

    lo = int(round(band[0] * (n - 1)))
    hi = int(round(band[1] * (n - 1)))
    if hi <= lo:
        lo, hi = 0, n - 1

    sub_paths = slice_paths[lo:hi + 1]
    if len(sub_paths) == num_slices:
        return sub_paths

    indices = np.linspace(0, len(sub_paths) - 1, num_slices).round().astype(int)
    return [sub_paths[i] for i in indices]

