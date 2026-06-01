"""
Pointing offset correction and pixel-to-sky coordinate conversion for PANOSETI.

This module contains functions for:
- Loading pointing offset calibration from CSV
- Converting pixel coordinates to sky coordinates (RA/Dec)
- Retrieving observation-specific pointing offsets

Last modified: 22 April 2026
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

__all__ = [
    "load_pointing_offset_csv",
    "pixel_to_skycoord",
    "get_pointing_offset_for_observation",
]

logger = logging.getLogger(__name__)


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
        data_dir = Path(__file__).parent.parent / "default_data"
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
