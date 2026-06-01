"""
Gain calibration and image calibration functions for PANOSETI data processing.

This module contains functions for:
- Gain calibration loading from CSV files
- Per-pixel gain correction
- Pedestal subtraction
- Full calibration pipeline (pedestal + gain)

Last modified: 22 April 2026
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..plugin_src.instrument import subarray

__all__ = [
    "load_gain_file",
    "subtract_pedestal",
    "correct_gain",
    "calibrate_image",
]

logger = logging.getLogger(__name__)


def load_gain_file(tel_id, gain_file_path=None):
    """
    Load per-pixel gain calibration from CSV file.

    Parameters
    ----------
    tel_id : int
        Telescope ID
    gain_file_path : str, Path, or None
        Path to CSV file with 32x32 gain matrix.
        If None, uses default packaged gain file for the telescope.

    Returns
    -------
    np.ndarray
        32x32 array of per-pixel gain values
    """
    # If no path provided, use default packaged file
    if gain_file_path is None:
        data_dir = Path(__file__).parent.parent / "default_data"
        tel_description = subarray.tel.get(tel_id)
        tel_name = tel_description.name if tel_description is not None else f"tel{tel_id}"
        gain_file_path = data_dir / f"gains_tel{tel_id}_{tel_name}.csv"

        # Keep compatibility with the existing tel4 filename while PTI becomes the
        # canonical telescope name in instrument metadata.
        if not gain_file_path.exists() and tel_name == "PTI":
            gain_file_path = data_dir / "gains_tel4_PTI-Heli.csv"
    
    try:
        df = pd.read_csv(gain_file_path, header=None)
        gains = df.values.astype(np.float32)
        if gains.shape != (32, 32):
            logger.warning(
                f"Gain file for tel {tel_id} has shape {gains.shape}, expected (32, 32)"
            )
        return gains
    except Exception as e:
        logger.error(f"Failed to load gain file for tel {tel_id} ({gain_file_path}): {e}")
        raise


def subtract_pedestal(image, pedestal):
    """
    Subtract pedestal from raw camera image.

    Parameters
    ----------
    image : np.ndarray
        Raw pulse height image, shape (1024,) or (32, 32)
    pedestal : np.ndarray
        Pedestal values, same shape as image

    Returns
    -------
    np.ndarray
        Pedestal-subtracted image
    """
    image_array = np.asarray(image)
    pedestal_array = np.asarray(pedestal)

    if image_array.shape == pedestal_array.shape:
        return image_array - pedestal_array

    if image_array.ndim == 1 and image_array.size == 1024 and pedestal_array.shape == (32, 32):
        return (image_array.reshape(32, 32) - pedestal_array).flatten()

    if image_array.ndim == 2 and image_array.shape[1] == 1024 and pedestal_array.shape == (32, 32):
        return image_array - pedestal_array.reshape(1, -1)

    if image_array.ndim == 2 and image_array.shape[1] == 1024 and pedestal_array.shape == (1024,):
        return image_array - pedestal_array.reshape(1, -1)

    logger.warning(
        f"Image shape {image_array.shape} != pedestal shape {pedestal_array.shape}"
    )
    return image_array - pedestal_array


def correct_gain(image, gains):
    """
    Apply per-pixel gain correction to calibrate raw ADC → physical units.

    Parameters
    ----------
    image : np.ndarray
        Pedestal-subtracted image, shape (1024,) or (32, 32)
    gains : np.ndarray
        Gain values (typically 1.0 for identity), shape (32, 32) or (1024,)

    Returns
    -------
    np.ndarray
        Gain-corrected image
    """
    image_array = np.asarray(image)
    gains_array = np.asarray(gains)

    if image_array.shape == gains_array.shape:
        return image_array * gains_array

    if image_array.ndim == 1 and image_array.size == 1024 and gains_array.shape == (32, 32):
        return (image_array.reshape(32, 32) * gains_array).flatten()

    if image_array.ndim == 2 and image_array.shape[1] == 1024 and gains_array.shape == (32, 32):
        return image_array * gains_array.reshape(1, -1)

    if image_array.ndim == 2 and image_array.shape[1] == 1024 and gains_array.shape == (1024,):
        return image_array * gains_array.reshape(1, -1)

    logger.warning(f"Image shape {image_array.shape} != gains shape {gains_array.shape}")
    return image_array * gains_array


def calibrate_image(image, pedestal=None, gains=None):
    """
    Apply full calibration: pedestal subtraction + gain correction.

    Parameters
    ----------
    image : np.ndarray
        Raw pulse height image, shape (1024,) or (32, 32)
    pedestal : np.ndarray, optional
        Pedestal array, shape (32, 32) or (1024,)
        If None, no pedestal subtraction is applied
    gains : np.ndarray, optional
        Gain correction array, shape (32, 32) or (1024,)
        If None, no gain correction is applied

    Returns
    -------
    np.ndarray
        Fully calibrated image
    """
    calibrated = image.copy()
    
    if pedestal is not None:
        calibrated = subtract_pedestal(calibrated, pedestal)
    
    if gains is not None:
        calibrated = correct_gain(calibrated, gains)
    
    return calibrated
