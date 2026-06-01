"""
Meridian flip image rotation for PANOSETI data processing.

This module contains functions for:
- Image rotation correction after meridian flip during observations

Last modified: 22 April 2026
"""

import logging

import numpy as np

__all__ = [
    "rotate_images_after_meridian_flip",
]

logger = logging.getLogger(__name__)


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
