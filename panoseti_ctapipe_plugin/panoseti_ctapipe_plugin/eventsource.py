"""
PanoEventSource: ctapipe EventSource for PANOSETI PFF files.

This module provides the main EventSource class that reads PANOSETI pulse height
data from PFF files and yields calibrated DL1 images compatible with ctapipe.

Author: Sruthi Ravikularaman
Last modified: 17 April 2026
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pypff
from astropy.time import Time
import astropy.units as u

from ctapipe.containers import (
    ArrayEventContainer,
    CoordinateFrameType,
    ObservationBlockContainer,
    SchedulingBlockContainer,
    SimulationConfigContainer,
    TriggerContainer,
)
from ctapipe.io import DataLevel, EventSource

from .instrument import (
    MODULE_TO_TEL_ID,
    subarray,
)
from .functions import (
    apply_packet_loss_filter,
    apply_rate_spike_filter,
    calculate_pedestal_and_pedvar_robust,
    compute_pedestals_from_data,
    load_gain_file,
    select_time_interval,
    wr_to_unix,
)

__all__ = ["PanoEventSource"]

logger = logging.getLogger(__name__)


class PanoEventSource(EventSource):
    """
    EventSource for PANOSETI PFF pulse height data.

    Reads raw pulse heights from PFF module files, applies calibration
    (pedestal subtraction, gain correction), and yields DL1 calibrated images.

    Pre-filtering is applied to remove bad events:
    - Packet loss detection (QUABOs with pkt_num == 0)
    - Rate spike filtering (removes high-rate trigger events)
    """

    is_simulation = False
    datalevels = (DataLevel.DL1_IMAGES,)  # Pulse heights calibrated to DL1 images

    def __init__(
        self,
        input_url=None,
        subarray_desc=None,
        sb_type=None,
        observing_mode=None,
        pointing_mode=None,
        pedestal_file=None,
        gain_file=None,
        **kwargs
    ):
        """
        Initialize PanoEventSource for PFF pulse height data.

        Parameters
        ----------
        input_url : str or Path, optional
            Path to the observation run folder containing module_*.pff files and hk.pff
        subarray_desc : SubarrayDescription, optional
            Subarray to use. If None, defaults to the full Panoseti array
        sb_type : SchedulingBlockType, optional
            Type of scheduling block. Defaults to OBSERVATION
        observing_mode : ObservingMode, optional
            Observing mode. Defaults to ON_OFF
        pointing_mode : PointingMode, optional
            Pointing mode. Defaults to TRACK
        pedestal_file : str or Path, optional
            Path to pedestal calibration file (ROOT .pedvars or similar)
        gain_file : str or Path, optional
            Path to gain calibration file (per-pixel gains for ADC → calibrated units).
            Currently all gains are set to 1.0 (identity). Will be measured and
            applied once available.
        **kwargs
            Additional arguments passed to EventSource
        """
        if subarray_desc is None:
            subarray_desc = subarray

        self._subarray = subarray_desc

        # Set scheduling block parameters with defaults
        from ctapipe.containers import SchedulingBlockType, ObservingMode, PointingMode

        self.sb_type = (
            sb_type if sb_type is not None else SchedulingBlockType.OBSERVATION
        )
        self.observing_mode = (
            observing_mode if observing_mode is not None else ObservingMode.ON_OFF
        )
        self.pointing_mode = (
            pointing_mode if pointing_mode is not None else PointingMode.TRACK
        )

        # Load pedestal calibration (TODO: implement loading from file)
        self._pedestals = {}
        self._pedvars = {}  # Pedestal variances
        self._pedestals_precomputed = False  # Track if pedestals were loaded from file

        # Load gain calibration from file (default or user-provided)
        self._gains = {}

        for tel_id in self._subarray.tel_ids:
            # Pedestals: Initialize to zeros
            # Will be computed from data if not provided via pedestal_file
            self._pedestals[tel_id] = np.zeros((32, 32))
            self._pedvars[tel_id] = np.zeros((32, 32))

            # TODO: Load actual pedestals from pedestal_file when available
            # if pedestal_file is not None:
            #     self._pedestals[tel_id] = load_pedestal_file(pedestal_file, tel_id)
            #     self._pedestals_precomputed = True

        for tel_id in self._subarray.tel_ids:
            # Pedestals: for now, no pedestal subtraction (zeros)
            # TODO: Load actual pedestals from pedestal_file when available
            self._pedestals[tel_id] = np.zeros((32, 32))

            # Gains: load from user file or default packaged file
            if gain_file is not None:
                # User-provided gain file
                gain_path = Path(gain_file)
                logger.info(f"Loading gains for tel {tel_id} from {gain_path}")
                self._gains[tel_id] = load_gain_file(tel_id, gain_path)
            else:
                # Use default packaged gain file
                data_dir = Path(__file__).parent.parent / "data"
                telescope_names = {1: "Gattini", 2: "Winter", 3: "Fern", 4: "PTI-Heli"}
                default_gain_file = data_dir / f"gains_tel{tel_id}_{telescope_names[tel_id]}.csv"
                if default_gain_file.exists():
                    self._gains[tel_id] = load_gain_file(tel_id, default_gain_file)
                    logger.info(
                        f"Loaded default gains for tel {tel_id} from {default_gain_file}"
                    )
                else:
                    logger.warning(
                        f"Default gain file not found: {default_gain_file}. "
                        f"Using identity gains."
                    )
                    self._gains[tel_id] = np.ones((32, 32))

        super().__init__(input_url=input_url, **kwargs)
        self._pff_files = []

    @property
    def subarray(self):
        """Obtain the subarray from the EventSource."""
        return self._subarray

    @property
    def pedestals(self):
        """
        Get computed pedestals for all telescopes.

        Returns
        -------
        dict
            Dictionary mapping tel_id to 32x32 pedestal array
        """
        return self._pedestals

    def get_pedestal(self, tel_id):
        """Get pedestal for a specific telescope."""
        return self._pedestals.get(tel_id, None)

    @property
    def pedvars(self):
        """
        Get pedestal variances for all telescopes.

        Returns
        -------
        dict
            Dictionary mapping tel_id to 32x32 variance array
        """
        return self._pedvars

    def get_pedvar(self, tel_id):
        """Get pedestal variance for a specific telescope."""
        return self._pedvars.get(tel_id, None)

    def save_pedestals(self, output_dir):
        """
        Save computed pedestals and variances to CSV files.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save files to (one per telescope)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for tel_id in self._pedestals.keys():
            # Save pedestals
            ped_file = output_dir / f"pedestal_tel{tel_id}.csv"
            np.savetxt(ped_file, self._pedestals[tel_id], delimiter=",", fmt="%.4f")
            logger.info(f"Saved pedestal to {ped_file}")

            # Save variances
            var_file = output_dir / f"pedvar_tel{tel_id}.csv"
            np.savetxt(var_file, self._pedvars[tel_id], delimiter=",", fmt="%.4f")
            logger.info(f"Saved pedvar to {var_file}")

    @staticmethod
    def compute_pedestals_robust_on_data(
        data: np.ndarray, nsig: float = 5.0, fit_gaussian: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robust pedestals and pedvars on given data with outlier removal.

        Useful for analyzing pedestals on specific datasets or time intervals.

        Parameters
        ----------
        data : np.ndarray
            Data frames shape (n_frames, 1024) or (n_frames, 32, 32)
        nsig : float
            Sigma threshold for outlier removal (default = 5.0)
        fit_gaussian : bool
            If True, fit Gaussian to pedvar; else use std (default = True)

        Returns
        -------
        pedestal : np.ndarray
            Per-pixel mean values
        pedvar : np.ndarray
            Per-pixel variance (Gaussian sigma or std)
        """
        return calculate_pedestal_and_pedvar_robust(data, nsig=nsig, fit_gaussian=fit_gaussian)

    @staticmethod
    def compute_pedestals_on_interval(
        timestamps: np.ndarray,
        data: np.ndarray,
        start_time,
        end_time,
        nsig: float = 5.0,
        fit_gaussian: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pedestals and pedvars for a specific time interval.

        Parameters
        ----------
        timestamps : np.ndarray
            Unix timestamps (seconds) for each frame
        data : np.ndarray
            Data frames shape (n_frames, 1024) or (n_frames, 32, 32)
        start_time : str or pd.Timestamp
            Start time 'YYYY-MM-DDTHH:MM:SSZ' or Timestamp
        end_time : str or pd.Timestamp
            End time 'YYYY-MM-DDTHH:MM:SSZ' or Timestamp
        nsig : float
            Sigma threshold for outlier removal (default = 5.0)
        fit_gaussian : bool
            If True, fit Gaussian to pedvar (default = True)

        Returns
        -------
        pedestal : np.ndarray
            Per-pixel pedestal on interval
        pedvar : np.ndarray
            Per-pixel pedvar on interval
        data_cut : np.ndarray
            Data within the interval
        timestamps_cut : np.ndarray
            Timestamps within the interval
        """
        data_cut, timestamps_cut, _ = select_time_interval(
            timestamps, data, start_time, end_time
        )
        pedestal, pedvar = calculate_pedestal_and_pedvar_robust(
            data_cut, nsig=nsig, fit_gaussian=fit_gaussian
        )
        return pedestal, pedvar, data_cut, timestamps_cut

    @classmethod
    def is_compatible(cls, path):
        """Check if path is an observation run folder containing .pff files."""
        path = Path(path)
        if not path.is_dir():
            return False
        # Check if there are any module_*.pff files in the directory
        pff_files = list(path.glob("start*ph1024*module_*.*.pff"))
        return len(pff_files) > 0

    def _generator(self):
        """
        Generator that yields DL1 calibrated images from PFF pulse height data.

        Applies pre-cleaning filters:
        - Packet loss removal
        - Rate spike filtering

        Reads raw pulse heights from PFF files, applies pedestal subtraction and
        gain calibration, and yields DL1 images.

        Yields
        ------
        event : ArrayEventContainer
            Event container with DL1 images populated
        """

        # Discover all .pff files in the observation folder
        obs_dir = Path(self.input_url)
        module_files_list = sorted(obs_dir.glob("start*ph1024*module_*.*.pff"))

        if not module_files_list:
            raise FileNotFoundError(f"No module_*.pff files found in {obs_dir}")

        # Map discovered module files to telescope IDs
        module_files = {}
        for file_path in module_files_list:
            # Extract module number from filename (e.g., "module_254" from filename)
            filename = file_path.name
            parts = filename.split("module_")
            if len(parts) > 1:
                module_num = int(parts[1].split(".")[0])

                # Map module number to telescope ID
                if module_num in MODULE_TO_TEL_ID:
                    tel_id = MODULE_TO_TEL_ID[module_num]
                    module_files[tel_id] = str(file_path)

        if not module_files:
            raise FileNotFoundError(f"No recognized modules found in {obs_dir}")

        # Get sorted telescope IDs
        tel_ids = sorted(module_files.keys())

        # Filter by allowed_tels if specified
        if self.allowed_tels is not None:
            module_files = {
                tid: path for tid, path in module_files.items() if tid in self.allowed_tels
            }
            tel_ids = [tid for tid in tel_ids if tid in self.allowed_tels]

        # Read all telescopes' data and metadata
        telescope_data = {}
        telescope_metadata = {}

        for tel_id, file_path in module_files.items():
            pff_file = pypff.io.datapff(file_path)
            data, metadata = pff_file.readpff(metadata=True)
            telescope_data[tel_id] = data
            telescope_metadata[tel_id] = metadata
            self._pff_files.append(pff_file)
            logger.info(f"Read {len(data)} events from telescope {tel_id}")

        # ===== PRE-FILTERING: PACKET LOSS =====
        for tel_id in tel_ids:
            data_filtered, valid_mask, pkt_loss_count = apply_packet_loss_filter(
                telescope_data[tel_id], telescope_metadata[tel_id]
            )
            telescope_data[tel_id] = data_filtered

            # Also filter metadata arrays to match filtered data
            for i in range(4):
                for key in telescope_metadata[tel_id][f"quabo_{i}"]:
                    arr = telescope_metadata[tel_id][f"quabo_{i}"][key]
                    if hasattr(arr, "__len__") and len(arr) == len(valid_mask):
                        telescope_metadata[tel_id][f"quabo_{i}"][key] = arr[valid_mask]

        # ===== TIMESTAMP CONVERSION AND VALID EVENT MASK =====
        all_timestamps_by_telescope = {}
        valid_event_mask_by_telescope = {}

        for tel_id in tel_ids:
            metadata = telescope_metadata[tel_id]

            # Convert White Rabbit timestamps to unix time
            # Use min across QUABOs as the event time (packet arrival time)
            timestamps_qubao = []
            for i in range(4):
                ts = wr_to_unix(
                    metadata[f"quabo_{i}"]["pkt_nsec"],
                    metadata[f"quabo_{i}"]["tv_sec"],
                    metadata[f"quabo_{i}"]["tv_usec"],
                )
                timestamps_qubao.append(ts)

            # Take minimum timestamp across QUABOs
            timestamps = np.min(np.array(timestamps_qubao), axis=0)
            all_timestamps_by_telescope[tel_id] = timestamps

        # ===== PRE-FILTERING: RATE SPIKES =====
        # Apply spike filter independently to each telescope's timestamps
        # (timestamps may not be perfectly synchronized across telescopes)
        for tel_id in tel_ids:
            spike_mask, spike_count = apply_rate_spike_filter(
                all_timestamps_by_telescope[tel_id],
                bin_width=30,  # 30 second bins
                rate_threshold=2.0,  # 2 Hz threshold
            )

            # Apply spike mask to this telescope's data
            telescope_data[tel_id] = telescope_data[tel_id][spike_mask]
            all_timestamps_by_telescope[tel_id] = all_timestamps_by_telescope[tel_id][
                spike_mask
            ]

        # ===== COMPUTE PEDESTALS FROM FILTERED DATA =====
        # If pedestals were not provided at initialization, compute them from the data
        for tel_id in tel_ids:
            if not self._pedestals_precomputed:
                # Pedestals were not loaded from file, compute from filtered data
                logger.info(
                    f"Computing pedestals for tel {tel_id} "
                    f"from {len(telescope_data[tel_id])} events"
                )
                ped, pedvar = compute_pedestals_from_data(telescope_data[tel_id])
                self._pedestals[tel_id] = ped
                self._pedvars[tel_id] = pedvar
            else:
                logger.info(f"Using pre-loaded pedestals for tel {tel_id}")

        # Use first telescope as reference for event iteration
        ref_tel_id = tel_ids[0]

        # Loop through events - use minimum count across all telescopes
        # (since spike filtering is now independent per telescope)
        num_events = min(len(telescope_data[tel_id]) for tel_id in tel_ids)

        event_count = 0
        for i in range(num_events):
            # Check if we've reached max_events limit
            if self.max_events is not None and event_count >= self.max_events:
                break

            # Get timestamp from reference telescope
            event_time = all_timestamps_by_telescope[ref_tel_id][i]
            trigger = TriggerContainer(
                time=Time(event_time, scale="utc", format="datetime64")
            )
            event = ArrayEventContainer(trigger=trigger)
            event.count = event_count
            event.trigger.tels_with_trigger = []

            for tel_id in tel_ids:
                raw_pulse_height = np.array(
                    telescope_data[tel_id][i], dtype=np.float32
                )

                # Reshape to 32x32 camera image
                # TODO: Verify this is the correct shape from PFF data
                image = raw_pulse_height.reshape((32, 32))

                # Apply calibrations to convert raw ADC → physical units
                # Step 1: Subtract pedestals
                calibrated_image = image - self._pedestals[tel_id]

                # Step 2: Apply gain normalization (ADC → calibrated units)
                # Currently gains are all 1.0 (identity); will update when measured
                calibrated_image *= self._gains[tel_id]

                # Store as DL1 calibrated image (flattened to 1D for ctapipe compatibility)
                event.dl1.tel[tel_id].image = calibrated_image.flatten()

                # Set trigger time for this telescope
                event.trigger.tel[tel_id].time = Time(
                    all_timestamps_by_telescope[tel_id][i],
                    scale="utc",
                    format="datetime64",
                )

                event.trigger.tels_with_trigger.append(tel_id)

            event.index.obs_id = list(self.obs_ids)[0] if self.obs_ids else 0

            yield event
            event_count += 1

    def close(self):
        """Close all open pypff file handles."""
        for pff_file in self._pff_files:
            if hasattr(pff_file, "close"):
                pff_file.close()
        self._pff_files.clear()

    @property
    def observation_blocks(self):
        """Extract observation metadata from input folder and housekeeping."""
        try:
            # input_url is now the observation run folder
            data_dir = Path(self.input_url)
            hk_file = data_dir / "hk.pff"

            if not hk_file.exists():
                return {}

            # Load housekeeping data using hkpff
            hkpff = pypff.io.hkpff(str(hk_file))
            hk = hkpff.readhk()

            # Extract pointing info from first available mount
            mount_key = None
            for key in ["MOUNT_GATTINI", "MOUNT_WINTER", "MOUNT_FERN"]:
                if key in hk:
                    mount_key = key
                    break

            if mount_key is None:
                return {}

            ra_hours = hk[mount_key]["ra_hours"]
            dec_deg = hk[mount_key]["dec_deg"]
            ra_deg = ra_hours * 15  # Convert hours to degrees

            # Extract start time from any module file in the folder
            module_files = list(data_dir.glob("start*ph1024*module_*.*.pff"))
            if not module_files:
                return {}

            # Parse filename to get start time
            filename = module_files[0].name
            start_str = filename.split("start_")[1].split(".")[0]
            start_time = Time(start_str, format="isot")

            obs_id = 0
            obs_block = ObservationBlockContainer(
                obs_id=obs_id,
                producer_id="Panoseti",
                actual_start_time=start_time,
                subarray_pointing_lon=ra_deg * u.deg,
                subarray_pointing_lat=dec_deg * u.deg,
                subarray_pointing_frame=CoordinateFrameType.ICRS,
            )
            return {obs_id: obs_block}
        except Exception as e:
            # Fallback if parsing fails
            return {}

    @property
    def scheduling_blocks(self):
        """Extract scheduling block metadata from observation date."""
        try:
            # Extract start time from any module file in the folder
            data_dir = Path(self.input_url)
            module_files = list(data_dir.glob("start*ph1024*module_*.*.pff"))
            if not module_files:
                return {}

            # Parse filename to get start time
            filename = module_files[0].name
            start_str = filename.split("start_")[1].split(".")[0]
            start_time = Time(start_str, format="isot")

            # Use date as sb_id (YYYYMMDD format)
            sb_id = np.uint64(int(start_time.strftime("%Y%m%d")))

            sb_block = SchedulingBlockContainer(
                sb_id=sb_id,
                producer_id="Panoseti",
                sb_type=self.sb_type,
                observing_mode=self.observing_mode,
                pointing_mode=self.pointing_mode,
            )
            return {sb_id: sb_block}
        except Exception as e:
            # Fallback if parsing fails
            return {}

    @property
    def simulation_block(self):
        """Return simulation block (None for real data)."""
        return None

    @property
    def simulation_config(self) -> Dict[int, SimulationConfigContainer]:
        """Return simulation configuration (empty for real data)."""
        return {}

    @property
    def obs_ids(self) -> Iterable[int]:
        """Return observation IDs from observation blocks."""
        return self.observation_blocks.keys()
