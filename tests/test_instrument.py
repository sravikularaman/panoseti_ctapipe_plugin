"""
Unit tests for PANOSETI instrument description module.

Tests the camera geometry, optics, telescope, and subarray definitions.

Following ctapipe guidelines:
https://ctapipe.readthedocs.io/en/stable/developer-guide/code-guidelines.html#unit-tests

Author: Sruthi Ravikularaman
Last modified: 17 April 2026
"""

import numpy as np
import pytest
import astropy.units as u

from panoseti_ctapipe_plugin.instrument import (
    camera,
    optics,
    subarray,
    telescope_1,
    telescope_2,
    telescope_3,
    telescope_4,
    MODULE_TO_TEL_ID,
)


# Camera tests


def test_camera_exists():
    """Test that camera is defined."""
    assert camera is not None
    assert camera.name == "Panoseti"


def test_camera_n_pixels():
    """Test that camera has correct number of pixels."""
    assert camera.readout.n_pixels == 1024


def test_camera_geometry():
    """Test that camera geometry is properly configured."""
    assert camera.geometry is not None
    # 32x32 = 1024 pixels
    assert camera.geometry.n_pixels == 1024


def test_camera_geometry_dimensions():
    """Test camera pixel coordinates are within reasonable bounds."""
    pix_x = camera.geometry.pix_x
    pix_y = camera.geometry.pix_y
    # Should be roughly a square grid
    assert len(pix_x) == 1024
    assert len(pix_y) == 1024


def test_camera_unique_pixels():
    """Test that pixel coordinates are unique."""
    pix_x = np.unique(camera.geometry.pix_x)
    pix_y = np.unique(camera.geometry.pix_y)
    assert len(pix_x) == 32
    assert len(pix_y) == 32


# Optics tests


def test_optics_exists():
    """Test that optics is defined."""
    assert optics is not None
    assert optics.name == "Panoseti_Fresnel"


def test_optics_aperture():
    """Test that aperture is reasonable (0.46m for PANOSETI)."""
    aperture_m = optics.mirror_area.to(u.m ** 2).value
    # ~0.46m diameter → ~0.166 m² area
    expected_area = 3.1416 * (0.46 / 2) ** 2
    assert np.isclose(aperture_m, expected_area, rtol=0.01)


def test_optics_focal_length():
    """Test that focal length is defined."""
    assert optics.effective_focal_length is not None
    assert optics.equivalent_focal_length is not None


# Telescope tests


def test_telescope_1_config():
    """Test telescope 1 (Gattini) configuration."""
    assert telescope_1 is not None
    assert telescope_1.name == "Gattini"
    assert telescope_1.camera is camera
    assert telescope_1.optics is optics


def test_telescope_2_config():
    """Test telescope 2 (Winter) configuration."""
    assert telescope_2 is not None
    assert telescope_2.name == "Winter"
    assert telescope_2.camera is camera
    assert telescope_2.optics is optics


def test_telescope_3_config():
    """Test telescope 3 (Fern) configuration."""
    assert telescope_3 is not None
    assert telescope_3.name == "Fern"
    assert telescope_3.camera is camera
    assert telescope_3.optics is optics


def test_telescope_4_config():
    """Test telescope 4 (PTI-Heli) configuration."""
    assert telescope_4 is not None
    assert telescope_4.name == "PTI-Heli"
    assert telescope_4.camera is camera
    assert telescope_4.optics is optics


# Subarray tests


def test_subarray_exists():
    """Test that subarray is defined."""
    assert subarray is not None


def test_subarray_name():
    """Test subarray name."""
    assert subarray.name == "Panoseti-Palomar"


def test_subarray_n_telescopes():
    """Test that all 4 telescopes are included."""
    assert len(subarray.tel_ids) == 4
    assert set(subarray.tel_ids) == {1, 2, 3, 4}


def test_subarray_tel_descriptions():
    """Test that telescope descriptions are correctly configured."""
    assert 1 in subarray.tel
    assert 2 in subarray.tel
    assert 3 in subarray.tel
    assert 4 in subarray.tel


def test_subarray_tel_positions():
    """Test that telescope positions are defined."""
    positions = subarray.positions
    assert len(positions) == 4

    for tel_id, pos in positions.items():
        assert tel_id in {1, 2, 3, 4}
        assert pos.shape == (3,)  # x, y, z coordinates
        assert pos.unit == u.m  # Should be in meters


def test_subarray_reference_location():
    """Test that reference location is Palomar Observatory."""
    ref_loc = subarray.reference_location
    assert ref_loc is not None
    # Palomar coordinates
    assert np.isclose(ref_loc.lat.deg, 33.3564, atol=0.01)
    assert np.isclose(ref_loc.lon.deg, -116.865, atol=0.01)
    assert np.isclose(ref_loc.height.to(u.m).value, 1712, atol=1)


def test_subarray_camera_consistency():
    """Test that all telescopes have the same camera."""
    for tel_id, tel_desc in subarray.tel.items():
        assert tel_desc.camera is camera


def test_subarray_optics_consistency():
    """Test that all telescopes have the same optics."""
    for tel_id, tel_desc in subarray.tel.items():
        assert tel_desc.optics is optics


# Module mapping tests


def test_mapping_exists():
    """Test that MODULE_TO_TEL_ID mapping is defined."""
    assert MODULE_TO_TEL_ID is not None
    assert len(MODULE_TO_TEL_ID) == 4


def test_mapping_values():
    """Test that mapping contains correct telescope IDs."""
    tel_ids = set(MODULE_TO_TEL_ID.values())
    assert tel_ids == {1, 2, 3, 4}


def test_mapping_modules():
    """Test specific module to telescope mapping."""
    assert MODULE_TO_TEL_ID[254] == 1  # Gattini
    assert MODULE_TO_TEL_ID[253] == 2  # Winter
    assert MODULE_TO_TEL_ID[252] == 3  # Fern
    assert MODULE_TO_TEL_ID[250] == 4  # PTI-Heli


def test_mapping_is_consistent():
    """Test that mapping covers all telescopes exactly once."""
    mapped_tels = set(MODULE_TO_TEL_ID.values())
    assert len(mapped_tels) == len(MODULE_TO_TEL_ID)  # No duplicates
    assert mapped_tels == {1, 2, 3, 4}  # Covers all 4 telescopes
