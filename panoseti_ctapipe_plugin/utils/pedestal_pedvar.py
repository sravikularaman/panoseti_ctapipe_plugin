"""
Pedestal and pedestal variance computation functions for PANOSETI data processing.

This module contains functions for:
- Robust pedestal (mean) and pedvar (variance) calculation with outlier removal
- Gaussian fitting for pedestal variance estimation

Last modified: 22 April 2026
"""

import logging
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "_gaussian",
    "calculate_pedestal_and_pedvar_robust",
]

logger = logging.getLogger(__name__)


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
    sigma_pixels_initial = np.nanstd(data_flat, axis=0)

    # Mask outliers: keep only data < mean + nsig*sigma
    threshold = mean_pixels_initial + nsig * sigma_pixels_initial
    data_masked = np.where(data_flat < threshold[None, :], data_flat, np.nan)

    # Recompute mean on masked data
    mean_pixels = np.nanmean(data_masked, axis=0)

    # If no Gaussian fit, return std of masked data
    if not fit_gaussian:
        sigma_pixels = np.nanstd(data_masked, axis=0)
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
