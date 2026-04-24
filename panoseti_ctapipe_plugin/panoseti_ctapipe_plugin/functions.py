"""
Utility functions for PANOSETI data processing.

This module contains functions for:
- Gain calibration loading
- Timestamp conversion (White Rabbit to Unix)
- Pre-filtering (packet loss, rate spikes)
- Pedestal computation with outlier removal
- Pointing offset correction (pixel to sky coordinates)
- Time interval selection

Author: Sruthi Ravikularaman
Last modified: 22 April 2026
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

__all__ = [
    "load_gain_file",
    "subtract_pedestal",
    "apply_gain_correction",
    "calibrate_image",
    "wr_to_unix",
    "apply_packet_loss_filter",
    "apply_rate_spike_filter",
    "calculate_pedestal_and_pedvar_robust",
    "load_pointing_offset_csv",
    "pixel_to_skycoord",
    "get_pointing_offset_for_observation",
    "rotate_images_after_meridian_flip",
]

logger = logging.getLogger(__name__)


# ==============================================================================
# GAIN CALIBRATION
# ==============================================================================


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
        from pathlib import Path
        data_dir = Path(__file__).parent.parent / "data"
        # Map telescope IDs to gain file names
        telescope_names = {
            1: "Gattini",
            2: "Winter",
            3: "Fern",
            4: "PTI-Heli"
        }
        tel_name = telescope_names.get(tel_id, f"tel{tel_id}")
        gain_file_path = data_dir / f"gains_tel{tel_id}_{tel_name}.csv"
    
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


# ==============================================================================
# TIMESTAMP CONVERSION
# ==============================================================================


def wr_to_unix(pkt_nsec, tv_sec, tv_usec, ignore_clock_desync=False):
    """
    Convert White Rabbit timestamps to unix time (datetime64[ns]).

    Handles clock desynchronization between White Rabbit and Unix clocks
    using heuristics to determine the correct second boundary.

    Parameters
    ----------
    pkt_nsec : int or array-like
        Packet nanoseconds
    tv_sec : int or array-like
        Unix seconds
    tv_usec : int or array-like
        Unix microseconds
    ignore_clock_desync : bool
        If True, ignore clock desynchronization errors

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] timestamps
    """
    # Cast to arrays
    pkt_nsec = np.asarray(pkt_nsec, dtype=np.int64)
    tv_sec = np.asarray(tv_sec, dtype=np.int64)
    tv_usec = np.asarray(tv_usec, dtype=np.int64)

    # Convert tv_usec to ns
    usec_ns = tv_usec * 1000

    # Difference in ns
    diff = usec_ns - pkt_nsec

    # 25ms threshold in ns
    TH = 25_000_000

    # Determine which case applies
    mask0 = np.abs(diff) < TH  # Normal case: same second
    mask1023 = diff > TH  # tv_usec much larger: second +1
    mask1 = diff < -TH  # pkt_nsec much larger: second -1

    # Preallocate output as ns integer
    out_ns = np.empty_like(tv_sec, dtype=np.int64)

    # Normal case: same second
    out_ns[mask0] = (
        tv_sec[mask0] * 1_000_000_000 + pkt_nsec[mask0] + usec_ns[mask0]
    )

    # Case 1: pkt_nsec much larger → second -1
    out_ns[mask1] = (
        (tv_sec[mask1] - 1) * 1_000_000_000 + pkt_nsec[mask1] + usec_ns[mask1]
    )

    # Case 1023: tv_usec much larger → second +1
    out_ns[mask1023] = (
        (tv_sec[mask1023] + 1) * 1_000_000_000 + pkt_nsec[mask1023] + usec_ns[mask1023]
    )

    # Handle bad cases
    mask_bad = ~(mask0 | mask1 | mask1023)

    if np.any(mask_bad):
        if ignore_clock_desync:
            out_ns[mask_bad] = (
                tv_sec[mask_bad] * 1_000_000_000
                + pkt_nsec[mask_bad]
                + usec_ns[mask_bad]
            )
        else:
            i = np.flatnonzero(mask_bad)[0]
            raise Exception(
                f"Clock mismatch: tv_sec={tv_sec[i]} tv_usec={tv_usec[i]} "
                f"pkt_nsec={pkt_nsec[i]} diff={diff[i]}"
            )

    # Convert to datetime64[ns]
    return out_ns.astype("datetime64[ns]")


# ==============================================================================
# DATA FILTERING
# ==============================================================================


def apply_packet_loss_filter(metadata, data=None):
    """
    Filter out events with packet loss (pkt_num == 0 in any QUABO).

    When pkt_num == 0 for any QUABO, it indicates missing data for that
    QUABO. This function creates a mask to exclude such events.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary from PFF file containing quabo_0, quabo_1, quabo_2, quabo_3 entries
    data : np.ndarray, optional
        Event data array. If provided, will be filtered and returned along with mask.

    Returns
    -------
    loss_mask : np.ndarray
        Boolean mask of events to keep (True = keep, False = packet loss)
    pkt_loss_count : int
        Number of events removed due to packet loss
    """
    # Get packet numbers from all 4 QUABOs
    pkt_num_0 = np.asarray(metadata["quabo_0"]["pkt_num"])
    pkt_num_1 = np.asarray(metadata["quabo_1"]["pkt_num"])
    pkt_num_2 = np.asarray(metadata["quabo_2"]["pkt_num"])
    pkt_num_3 = np.asarray(metadata["quabo_3"]["pkt_num"])

    # Create mask: keep events where ALL QUABOs have pkt_num != 0
    loss_mask = (pkt_num_0 != 0) & (pkt_num_1 != 0) & (pkt_num_2 != 0) & (pkt_num_3 != 0)

    pkt_loss_count = np.sum(~loss_mask)
    pct = 100 * pkt_loss_count / len(loss_mask) if len(loss_mask) > 0 else 0
    logger.info(
        f"Packet loss filter: removed {pkt_loss_count} events ({pct:.2f}%)"
    )

    if data is not None:
        data_filtered = data[loss_mask]
        return loss_mask, pkt_loss_count, data_filtered

    return loss_mask, pkt_loss_count


def apply_rate_spike_filter(timestamps, bin_width=30, rate_threshold=2.0):
    """
    Filter out trigger rate spikes (e.g., from planes, cosmic rays).

    Divides data into time bins and removes events in bins exceeding
    the rate threshold.

    Parameters
    ----------
    timestamps : np.ndarray
        Unix timestamps (float or datetime64)
    bin_width : float
        Time width for rate calculation in seconds
    rate_threshold : float
        Rate threshold in Hz; bins exceeding this are removed

    Returns
    -------
    spike_mask : np.ndarray
        Boolean mask of events to keep (True = keep, False = spike)
    spike_count : int
        Number of events removed
    """
    # Convert datetime64 to float seconds if needed
    if np.issubdtype(timestamps.dtype, np.datetime64):
        timestamps_float = (
            timestamps.astype("datetime64[ns]").astype("int64") * 1e-9
        )
    else:
        timestamps_float = np.asarray(timestamps, dtype=np.float64)

    # Create bins and compute rate
    bins = np.arange(
        timestamps_float.min(), timestamps_float.max() + bin_width, bin_width
    )
    counts, _ = np.histogram(timestamps_float, bins=bins)
    rate = counts / bin_width  # Hz

    # Find bad bins (exceeding threshold)
    bad_bins = rate > rate_threshold

    # Assign each event to a bin
    bin_indices = np.digitize(timestamps_float, bins) - 1

    # Clip indices to valid range [0, len(bad_bins)-1]
    bin_indices = np.clip(bin_indices, 0, len(bad_bins) - 1)

    # Create mask: keep events NOT in bad bins
    spike_mask = ~bad_bins[bin_indices]

    spike_count = np.sum(~spike_mask)
    pct = 100 * spike_count / len(timestamps_float) if len(timestamps_float) > 0 else 0
    logger.info(
        f"Rate spike filter (threshold={rate_threshold} Hz): "
        f"removed {spike_count} events ({pct:.2f}%)"
    )

    return spike_mask, spike_count


# ==============================================================================
# PEDESTAL COMPUTATION
# ==============================================================================




def _gaussian(x, A, mu, sigma):
    """Gaussian function for pedestal variance fitting."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def calculate_pedestal_and_pedvar_robust(
    data, nsig=5.0, fit_gaussian=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate pedestal and pedestal variance with outlier removal.

    Returns pedestal (mean) and pedvar (std or Gaussian-fitted sigma) per pixel.
    Bright outliers (> nsig * sigma above mean) are masked and excluded.

    Parameters
    ----------
    data : np.ndarray
        Data frames in format (n_frames, 1024) or (n_frames, 32, 32)
    nsig : float
        Number of sigmas above mean to define outlier threshold (default = 5.0)
    fit_gaussian : bool
        If True, fit Gaussian to masked data for pedvar.
        If False, use std of masked data.

    Returns
    -------
    mean_pixels : np.ndarray
        Shape (1024,) or (32, 32) pedestal (mean) per pixel
    sigma_pixels : np.ndarray
        Shape (1024,) or (32, 32) pedestal variance (Gaussian sigma or std)
    """
    # Flatten to (n_frames, n_pixels) if needed
    original_shape = data.shape
    if len(data.shape) == 3:
        n_frames, nx, ny = data.shape
        data_flat = data.reshape((n_frames, nx * ny))
    else:
        data_flat = data
        n_frames, n_pixels = data_flat.shape
        nx, ny = 32, 32

    n_pixels = data_flat.shape[1]

    # Initial estimates (robust to outliers)
    mean_pixels_initial = np.nanmean(data_flat, axis=0)
    sigma_pixels_initial = np.abs(np.nanstd(data_flat, axis=0))

    # Mask outliers: keep only data < mean + nsig*sigma
    threshold = mean_pixels_initial + nsig * sigma_pixels_initial
    data_masked = np.where(data_flat < threshold[None, :], data_flat, np.nan)

    # Recompute mean on masked data
    mean_pixels = np.nanmean(data_masked, axis=0)

    # If no Gaussian fit, return std of masked data
    if not fit_gaussian:
        sigma_pixels = np.abs(np.nanstd(data_masked, axis=0))
        return mean_pixels.reshape(original_shape[1:]), sigma_pixels.reshape(
            original_shape[1:]
        )

    # Gaussian fitting per pixel
    sigma_pixels = np.zeros(n_pixels)

    for i in range(n_pixels):
        x = data_masked[:, i]
        x_clean = x[np.isfinite(x)]  # Remove NaN values

        mu0 = mean_pixels[i]
        sigma0 = sigma_pixels_initial[i]

        if x_clean.size < 5:
            sigma_pixels[i] = sigma0
            continue

        # Histogram of clean data
        hmin, hmax, hbins = -500, 500, 1000
        hist, edges = np.histogram(x_clean, bins=hbins, range=(hmin, hmax))
        centers = 0.5 * (edges[1:] + edges[:-1])

        try:
            p0 = [hist.max(), mu0, sigma0]
            popt, _ = curve_fit(_gaussian, centers, hist, p0=p0, maxfev=2000)
            sigma_pixels[i] = abs(popt[2])
        except RuntimeError:
            logger.debug(f"Gaussian fit failed for pixel {i}, using initial sigma")
            sigma_pixels[i] = sigma0

    # Reshape back to original spatial dimensions
    return mean_pixels.reshape(original_shape[1:]), sigma_pixels.reshape(
        original_shape[1:]
    )


# ==============================================================================
# CALIBRATION PIPELINE
# ==============================================================================


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
    if image.shape != pedestal.shape:
        # Try to reshape
        if image.size == 1024 and pedestal.shape == (32, 32):
            image_reshaped = image.reshape(32, 32)
            result = image_reshaped - pedestal
            return result.flatten()
        else:
            logger.warning(
                f"Image shape {image.shape} != pedestal shape {pedestal.shape}"
            )
    return image - pedestal


def apply_gain_correction(image, gains):
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
    if image.shape != gains.shape:
        # Try to reshape
        if image.size == 1024 and gains.shape == (32, 32):
            image_reshaped = image.reshape(32, 32)
            result = image_reshaped * gains
            return result.flatten()
        else:
            logger.warning(f"Image shape {image.shape} != gains shape {gains.shape}")
    return image * gains


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
        calibrated = apply_gain_correction(calibrated, gains)
    
    return calibrated


# ==============================================================================
# POINTING OFFSET CORRECTION
# ==============================================================================


def load_pointing_offset_csv(csv_path=None):
    """
    Load pointing offset CSV file with source pixel coordinates.

    CSV format: date, tel, pixel_x, pixel_y
    where pixel_x, pixel_y are in range [0, 32) for 32x32 camera.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to pointing offset CSV file. If None, uses default packaged file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, tel, pixel_x, pixel_y
        Date is converted to datetime64
    """
    if csv_path is None:
        data_dir = Path(__file__).parent.parent / "data"
        csv_path = data_dir / "pointing_offsets.csv"

    try:
        df = pd.read_csv(csv_path)
        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])
        logger.info(f"Loaded pointing offsets from {csv_path}: {len(df)} entries")
        return df
    except Exception as e:
        logger.error(f"Failed to load pointing offset CSV {csv_path}: {e}")
        raise


def pixel_to_skycoord(
    pixel_x: float,
    pixel_y: float,
    tel_pointing_ra_deg: float,
    tel_pointing_dec_deg: float,
    obs_time: Time,
    focal_length_m: float = 0.46,
    pixel_size_mm: float = 3.0,
) -> SkyCoord:
    """
    Convert pixel coordinates to sky coordinates (RA/Dec).

    Converts pixel position in camera frame to focal plane angle,
    then combines with telescope pointing to get final sky position.

    Parameters
    ----------
    pixel_x : float
        Pixel X coordinate (0-32, center is 16)
    pixel_y : float
        Pixel Y coordinate (0-32, center is 16)
    tel_pointing_ra_deg : float
        Telescope pointing RA in degrees
    tel_pointing_dec_deg : float
        Telescope pointing Dec in degrees
    obs_time : astropy Time
        Observation time for coordinate transformation
    focal_length_m : float
        Focal length of optics in meters (default 0.46 m)
    pixel_size_mm : float
        Physical size of pixel in mm (default 3.0 mm)

    Returns
    -------
    SkyCoord
        Detected source position in ICRS frame
    """
    # Convert pixel coordinates to mm on detector
    # Pixel (16, 16) is center, pixel (0, 0) is corner
    pixel_offset_mm = np.array([(pixel_x - 16) * pixel_size_mm, (pixel_y - 16) * pixel_size_mm])

    # Convert mm to radians on focal plane
    focal_plane_angle = pixel_offset_mm / 1000.0 / focal_length_m  # radians

    # Create offset SkyCoord in Alt/Az (focal plane is tangent to sky)
    # Offset in Alt is positive toward +Y (pixel Y direction)
    # Offset in Az is positive toward +X (pixel X direction, but AZ is opposite)
    offset_alt_deg = np.degrees(focal_plane_angle[1])
    offset_az_deg = -np.degrees(focal_plane_angle[0])  # Negative because AZ convention

    # Create base pointing coordinate
    base_coord = SkyCoord(
        ra=tel_pointing_ra_deg * u.deg,
        dec=tel_pointing_dec_deg * u.deg,
        frame="icrs",
        obstime=obs_time
    )

    # Apply offset by shifting in celestial frame
    # For small angles, can use small angle approximation
    # ΔRA = ΔAz / cos(Dec)
    # ΔDec = ΔAlt
    dec_rad = np.radians(tel_pointing_dec_deg)
    offset_ra = offset_az_deg / np.cos(dec_rad)
    offset_dec = offset_alt_deg

    source_ra_deg = tel_pointing_ra_deg + offset_ra
    source_dec_deg = tel_pointing_dec_deg + offset_dec

    source_coord = SkyCoord(
        ra=source_ra_deg * u.deg,
        dec=source_dec_deg * u.deg,
        frame="icrs",
        obstime=obs_time
    )

    return source_coord


def get_pointing_offset_for_observation(
    obs_date, tel_id, pointing_offset_df: Optional[pd.DataFrame] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get pointing offset (pixel coordinates) for a specific observation.

    Matches observation date (YYYYMMDD) to the pointing offset CSV.
    Returns None if no matching entry found.

    Parameters
    ----------
    obs_date : str or pd.Timestamp
        Observation date ('YYYYMMDD' format or Timestamp)
    tel_id : int
        Telescope ID (1, 2, 3, or 4)
    pointing_offset_df : pd.DataFrame, optional
        Pre-loaded pointing offset DataFrame. If None, will load from default file.

    Returns
    -------
    pixel_x : float or None
        Pixel X coordinate of source (or None if not found)
    pixel_y : float or None
        Pixel Y coordinate of source (or None if not found)
    """
    if pointing_offset_df is None:
        pointing_offset_df = load_pointing_offset_csv()

    # Convert obs_date to datetime
    if isinstance(obs_date, str):
        obs_date_dt = pd.to_datetime(obs_date)
    else:
        obs_date_dt = pd.to_datetime(obs_date)

    # Extract date part
    obs_date_only = obs_date_dt.date()

    # Filter by date and telescope
    matching = pointing_offset_df[
        (pointing_offset_df["date"].dt.date == obs_date_only)
        & (pointing_offset_df["tel"] == tel_id)
    ]

    if len(matching) == 0:
        logger.warning(
            f"No pointing offset found for {obs_date_only}, tel {tel_id}. "
            f"Using default center (16, 16)."
        )
        return 16.0, 16.0

    if len(matching) > 1:
        logger.warning(
            f"Multiple matching offsets for {obs_date_only}, tel {tel_id}. "
            f"Using first entry."
        )

    row = matching.iloc[0]
    return row["pixel_x"], row["pixel_y"]


# ==============================================================================
# MERIDIAN FLIP CORRECTION
# ==============================================================================


def rotate_images_after_meridian_flip(
    data: np.ndarray,
    meridian_flip_phase: str = "pre",
    n_pix: int = 32,
) -> np.ndarray:
    """
    Rotate images by 180° if observation is post-meridian-flip.

    When a telescope crosses the meridian (post-flip), the entire field rotates 180°.
    This function rotates post-flip images to align with pre-flip coordinate system
    for consistent analysis across multiple observations.

    Parameters
    ----------
    data : np.ndarray
        Event data, shape (n_events, n_pix*n_pix)
    meridian_flip_phase : str, optional
        Phase of observation: "pre" or "post" (default: "pre").
        If "pre", data is returned unchanged. If "post", data is rotated 180°.
    n_pix : int, optional
        Pixel grid dimension (default 32 for 32×32 camera)

    Returns
    -------
    data_corrected : np.ndarray
        Data rotated by 180° if post-flip, otherwise unchanged.
        Shape: (n_events, n_pix*n_pix)
    """
    if meridian_flip_phase == "pre":
        logger.debug("Pre-meridian-flip observation: no rotation needed")
        return data

    if meridian_flip_phase != "post":
        logger.warning(
            f"Unknown meridian_flip_phase '{meridian_flip_phase}'. "
            f"Expected 'pre' or 'post'. Returning data unchanged."
        )
        return data

    # Reshape from (n, n_pix*n_pix) → (n, n_pix, n_pix)
    reshaped_data = np.reshape(data, (len(data), n_pix, n_pix))

    # Rotate each frame by 180° (k=2 means 2 × 90° = 180°)
    rotated_data = np.rot90(reshaped_data, k=2, axes=(1, 2))

    # Reshape back from (n, n_pix, n_pix) → (n, n_pix*n_pix)
    reshaped_rotated_data = np.reshape(rotated_data, (len(data), n_pix * n_pix))

    logger.info(f"Post-meridian-flip rotation applied: rotated {len(data)} images by 180°")

    return reshaped_rotated_data

