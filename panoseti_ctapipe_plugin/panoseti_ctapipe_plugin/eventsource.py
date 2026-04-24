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
    apply_gain_correction,
    apply_packet_loss_filter,
    apply_rate_spike_filter,
    calculate_pedestal_and_pedvar_robust,
    calibrate_image,
    get_pointing_offset_for_observation,
    load_gain_file,
    load_pointing_offset_csv,
    pixel_to_skycoord,
    rotate_images_after_meridian_flip,
    subtract_pedestal,
    wr_to_unix,
)

__all__ = ["PanoEventSource", "CalibrationPipeline"]

logger = logging.getLogger(__name__)


class PanoEventSource(EventSource):
    """
    EventSource for PANOSETI PFF pulse height data.

    Reads raw pulse heights from PFF module files and yields DL1 images.
    For a complete calibration workflow with pedestal computation and filtering,
    use CalibrationPipeline.

    If pedestals and gains are provided via files, they will be applied during
    event iteration. Otherwise, identity operations are used.
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
        pointing_offset_csv=None,
        meridian_flip_phase=None,
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
        pointing_offset_csv : str or Path, optional
            Path to CSV file with source pixel coordinates for pointing correction.
            CSV should contain: date, tel, phase, pixel_x, pixel_y.
            When provided, actual source position will be computed from pixel coords.
        meridian_flip_phase : str, optional
            Observation phase: "pre" or "post" (relative to meridian flip).
            Default is "pre". Used to select correct pointing offset from CSV.
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
            observing_mode if observing_mode is not None else ObservingMode.ON_OFF #WOBBLE
        )
        self.pointing_mode = (
            pointing_mode if pointing_mode is not None else PointingMode.TRACK
        )

        # Load pedestal calibration from file (if provided)
        self._pedestals = {}
        for tel_id in self._subarray.tel_ids:
            # TODO: implement load_pedestal_file when pedestal format is finalized
            # For now, initialize to zeros (identity operation)
            self._pedestals[tel_id] = np.zeros((32, 32))

        # Load gain calibration from file (default or user-provided)
        self._gains = {}
        for tel_id in self._subarray.tel_ids:
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
        self._metadata = {}  # Store metadata for filtering use

        # Load pointing offset corrections (pixel coordinates of source)
        self._pointing_offset_df = None
        if pointing_offset_csv is not None:
            try:
                self._pointing_offset_df = load_pointing_offset_csv(pointing_offset_csv)
                logger.info(f"Loaded pointing offsets from {pointing_offset_csv}")
            except Exception as e:
                logger.warning(f"Failed to load pointing offsets: {e}. Will use housekeeping pointing.")

        # Store meridian flip phase ("pre" or "post")
        self._meridian_flip_phase = meridian_flip_phase if meridian_flip_phase is not None else "pre"
        if self._meridian_flip_phase not in ("pre", "post"):
            logger.warning(f"Invalid meridian_flip_phase '{self._meridian_flip_phase}'. Using 'pre'.")
            self._meridian_flip_phase = "pre"

    @property
    def subarray(self):
        """Obtain the subarray from the EventSource."""
        return self._subarray

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
        Generator that yields DL1 images from PFF pulse height data.

        Reads raw pulse heights from PFF files and applies pre-computed calibrations
        (pedestal subtraction and gain correction).

        For a complete calibration workflow including pedestal computation from data
        and filtering, use CalibrationPipeline.

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
            self._metadata[tel_id] = metadata  # Store for filtering
            self._pff_files.append(pff_file)
            logger.info(f"Read {len(data)} events from telescope {tel_id}")

        # ===== TIMESTAMP CONVERSION =====
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

        # Use first telescope as reference for event iteration
        ref_tel_id = tel_ids[0]

        # Loop through events
        # Note: Raw data is used here without filtering.
        # Filtering (packet loss, rate spikes) and pedestal computation
        # should be done using CalibrationPipeline for standard workflows.
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
            
            # Store source file paths for this event (for debugging/comparison)
            event.meta = getattr(event, 'meta', {})
            event.meta['source_files'] = {tel_id: module_files[tel_id] for tel_id in tel_ids}

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
        """Extract observation metadata from input folder and housekeeping, with pointing offset correction if available."""
        try:
            # input_url is now the observation run folder
            data_dir = Path(self.input_url)
            hk_file = data_dir / "hk.pff"

            if not hk_file.exists():
                return {}

            # Load housekeeping data using hkpff
            hkpff = pypff.io.hkpff(str(hk_file))
            hk = hkpff.readhk()

            # Extract start time from any module file in the folder first (needed for matching CSV)
            module_files = list(data_dir.glob("start*ph1024*module_*.*.pff"))
            if not module_files:
                return {}

            # Parse filename to get start time
            filename = module_files[0].name
            start_str = filename.split("start_")[1].split(".")[0]
            start_time = Time(start_str, format="isot")
            obs_date = start_time.datetime.date()

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

            # Check if pointing offset correction is available
            meridian_flip = self._meridian_flip_phase  # Use the phase specified at init
            corrected_pointing = False
            
            if self._pointing_offset_df is not None:
                # Look for matching entry in CSV (by date, telescope, and phase)
                mask = (self._pointing_offset_df["date"].dt.date == obs_date)
                if mask.any():
                    # Try to find matching phase entry
                    phase_mask = mask & (self._pointing_offset_df["phase"] == meridian_flip)
                    if phase_mask.any():
                        offset_row = self._pointing_offset_df[phase_mask].iloc[0]
                    else:
                        # Fallback to any matching date entry
                        logger.warning(
                            f"No matching phase '{meridian_flip}' for {obs_date}. Using first available entry."
                        )
                        offset_row = self._pointing_offset_df[mask].iloc[0]
                    
                    pixel_x = offset_row["pixel_x"]
                    pixel_y = offset_row["pixel_y"]
                    
                    # Convert pixel coords to sky coords
                    source_skycoord = pixel_to_skycoord(
                        pixel_x=pixel_x,
                        pixel_y=pixel_y,
                        tel_pointing_ra_deg=ra_deg,
                        tel_pointing_dec_deg=dec_deg,
                        obs_time=start_time,
                        focal_length_m=0.46,
                        pixel_size_mm=3.0
                    )
                    
                    ra_deg = source_skycoord.ra.deg
                    dec_deg = source_skycoord.dec.deg
                    corrected_pointing = True
                    logger.info(
                        f"Applied pointing correction ({meridian_flip} meridian flip): "
                        f"pixel ({pixel_x}, {pixel_y}) → RA={ra_deg:.4f}°, Dec={dec_deg:.4f}°"
                    )

            obs_id = 0
            obs_block = ObservationBlockContainer(
                obs_id=obs_id,
                producer_id="Panoseti",
                actual_start_time=start_time,
                subarray_pointing_lon=ra_deg * u.deg,
                subarray_pointing_lat=dec_deg * u.deg,
                subarray_pointing_frame=CoordinateFrameType.ICRS,
            )
            
            # Add meridian flip phase to metadata
            obs_block.meta["meridian_flip"] = meridian_flip
            
            return {obs_id: obs_block}
        except Exception as e:
            # Fallback if parsing fails
            logger.error(f"Error extracting observation blocks: {e}")
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
                observing_mode=self.observing_mode, #(POINT)
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


class CalibrationPipeline:
    """
    Standard calibration workflow for PANOSETI data.

    Collects all events, computes pedestals from packet-loss and rate-spike filtered
    events, then applies calibration (pedestal subtraction + gain correction) to all events.

    Parameters
    ----------
    source : PanoEventSource
        Event source to calibrate
    spike_threshold : float, optional
        Rate threshold in Hz for spike detection (default: 2.0)
    bin_width : int, optional
        Time bin width in seconds for rate calculation (default: 10)
    gain_file : str or Path, optional
        Path to custom gain calibration file. If None, uses default packaged files.
    nsig : float, optional
        Sigma threshold for outlier removal in pedestal calculation (default: 5.0)

    Examples
    --------
    >>> source = PanoEventSource(input_url)
    >>> pipeline = CalibrationPipeline(source, spike_threshold=2.0)
    >>> calibrated_events = list(pipeline.calibrate_all())
    """

    def __init__(
        self,
        source,
        spike_threshold=2.0,
        bin_width=10,
        gain_file=None,
        nsig=5.0,
    ):
        self.source = source
        self.spike_threshold = spike_threshold
        self.bin_width = bin_width
        self.gain_file = gain_file
        self.nsig = nsig

        self.pedestals = {}  # {tel_id: pedestal array}
        self.gains = {}  # {tel_id: gain array}
        self._all_events = None  # Cached all events
        self._pedestals_computed = False

        logger.info(
            f"CalibrationPipeline initialized: "
            f"spike_threshold={spike_threshold} Hz, bin_width={bin_width}s"
        )

    def _collect_all_events(self, verbose=True):
        """Collect all events from source into cache. Idempotent."""
        if self._all_events is not None:
            return self._all_events

        if verbose:
            logger.info("Collecting all events from source...")

        self._all_events = []
        for i, event in enumerate(self.source):
            if verbose and i % 100 == 0:
                logger.info(f"  Collected {i} events...")
            self._all_events.append(event)

        if verbose:
            logger.info(f"Collected {len(self._all_events)} total events")
        return self._all_events

    def _organize_event_data(self, verbose=True):
        """
        Organize raw event data by telescope.

        Returns
        -------
        all_data : dict
            {tel_id: np.array of images}
        all_timestamps : dict
            {tel_id: np.array of timestamps}
        """
        all_events = self._collect_all_events(verbose=verbose)

        all_data = {tel_id: [] for tel_id in self.source.subarray.tel_ids}
        all_timestamps = {tel_id: [] for tel_id in self.source.subarray.tel_ids}

        for event in all_events:
            for tel_id in self.source.subarray.tel_ids:
                if tel_id in event.trigger.tels_with_trigger:
                    all_data[tel_id].append(event.dl1.tel[tel_id].image)
                    all_timestamps[tel_id].append(event.trigger.tel[tel_id].time.unix)

        # Convert to numpy arrays
        all_data_np = {}
        all_timestamps_np = {}
        for tel_id in self.source.subarray.tel_ids:
            all_data_np[tel_id] = np.array(all_data[tel_id]) if all_data[tel_id] else np.array([])
            all_timestamps_np[tel_id] = np.array(all_timestamps[tel_id]) if all_timestamps[tel_id] else np.array([])

        return all_data_np, all_timestamps_np

    def _apply_packet_loss_filtering(self, all_data, all_timestamps, verbose=True):
        """
        Apply packet loss filtering to all telescopes.

        Parameters
        ----------
        all_data : dict
            {tel_id: data array}
        all_timestamps : dict
            {tel_id: timestamps array}
        verbose : bool

        Returns
        -------
        all_data : dict
            Filtered data
        all_timestamps : dict
            Filtered timestamps
        """
        if verbose:
            logger.info("Applying packet loss filtering...")

        for tel_id in self.source.subarray.tel_ids:
            if not all_data[tel_id].size:
                continue

            if tel_id in self.source._metadata:
                loss_mask, loss_count, data_filtered = apply_packet_loss_filter(
                    self.source._metadata[tel_id], data=all_data[tel_id]
                )
                all_data[tel_id] = data_filtered
                all_timestamps[tel_id] = all_timestamps[tel_id][loss_mask]

                if verbose:
                    logger.info(f"  Tel {tel_id}: Removed {loss_count} events (packet loss)")
            else:
                if verbose:
                    logger.warning(f"  Tel {tel_id}: No metadata for packet loss filtering (skipping)")

        return all_data, all_timestamps

    def _apply_rate_spike_filtering(self, all_data, all_timestamps, verbose=True):
        """
        Apply rate spike filtering to all telescopes.

        Parameters
        ----------
        all_data : dict
            {tel_id: data array}
        all_timestamps : dict
            {tel_id: timestamps array}
        verbose : bool

        Returns
        -------
        all_data : dict
            Filtered data
        """
        if verbose:
            logger.info("Applying rate spike filtering...")

        for tel_id in self.source.subarray.tel_ids:
            if not all_timestamps[tel_id].size:
                continue

            spike_mask, spike_count = apply_rate_spike_filter(
                all_timestamps[tel_id],
                bin_width=self.bin_width,
                rate_threshold=self.spike_threshold,
            )
            all_data[tel_id] = all_data[tel_id][spike_mask]

            if verbose:
                logger.info(f"  Tel {tel_id}: Removed {spike_count} events (rate spikes)")

        return all_data

    def compute_pedestals(self, all_data, verbose=True):
        """
        Compute pedestals from filtered event data.

        Parameters
        ----------
        all_data : dict
            {tel_id: filtered data array}
        verbose : bool

        Returns
        -------
        dict
            {tel_id: pedestal array shape (32, 32)}
        """
        if verbose:
            logger.info("Computing pedestals from filtered data...")

        for tel_id in self.source.subarray.tel_ids:
            if not all_data[tel_id].size:
                logger.warning(f"Tel {tel_id}: No data available, using zero pedestals")
                self.pedestals[tel_id] = np.zeros((32, 32))
                continue

            # Compute pedestals on filtered data
            ped_flat, _ = calculate_pedestal_and_pedvar_robust(
                all_data[tel_id], nsig=self.nsig, fit_gaussian=True
            )
            self.pedestals[tel_id] = ped_flat.reshape(32, 32)

            if verbose:
                logger.info(f"  Tel {tel_id}: Pedestal from {len(all_data[tel_id])} events")

        self._pedestals_computed = True
        return self.pedestals

    def load_gains(self, verbose=True):
        """
        Load per-pixel gain calibration for all telescopes.

        Parameters
        ----------
        verbose : bool

        Returns
        -------
        dict
            {tel_id: gain array shape (32, 32)}
        """
        if verbose:
            logger.info("Loading gain calibration...")

        for tel_id in self.source.subarray.tel_ids:
            self.gains[tel_id] = load_gain_file(tel_id, self.gain_file)
            if verbose:
                logger.info(f"  Tel {tel_id}: Loaded gains")

        return self.gains

    def calibrate_all(self, verbose=True):
        """
        Apply full calibration pipeline to all events.

        Orchestrates: event collection → packet loss filtering → rate spike filtering →
        pedestal computation → gain loading → calibration application.

        Parameters
        ----------
        verbose : bool, optional
            If True, print progress information

        Yields
        ------
        ArrayEventContainer
            Calibrated events with pedestal and gain corrections applied
        """
        if not self._pedestals_computed:
            # Organize data by telescope
            all_data, all_timestamps = self._organize_event_data(verbose=verbose)

            # Apply filters
            all_data, all_timestamps = self._apply_packet_loss_filtering(
                all_data, all_timestamps, verbose=verbose
            )
            all_data = self._apply_rate_spike_filtering(
                all_data, all_timestamps, verbose=verbose
            )

            # Compute pedestals and load gains
            self.compute_pedestals(all_data, verbose=verbose)
            self.load_gains(verbose=verbose)

        # Apply calibration to all cached events
        all_events = self._all_events
        if verbose:
            logger.info("Applying calibration to all events...")

        for i, event in enumerate(all_events):
            if verbose and i % 100 == 0:
                logger.info(f"  Calibrated {i} events...")

            for tel_id in self.source.subarray.tel_ids:
                if tel_id in event.dl1.tel:
                    raw_image = event.dl1.tel[tel_id].image
                    calibrated = calibrate_image(
                        raw_image,
                        pedestal=self.pedestals[tel_id].flatten(),
                        gains=self.gains[tel_id].flatten(),
                    )
                    event.dl1.tel[tel_id].image = calibrated

            yield event

        if verbose:
            logger.info(f"Calibration complete: {len(all_events)} events")

    def save_pedestals(self, output_dir):
        """
        Save computed pedestals to CSV files.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save pedestal files to
        """
        if not self._pedestals_computed:
            raise ValueError("Pedestals not yet computed. Call compute_pedestals() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for tel_id, ped in self.pedestals.items():
            ped_file = output_dir / f"pedestal_tel{tel_id}.csv"
            np.savetxt(ped_file, ped, delimiter=",", fmt="%.6f")
            logger.info(f"Saved pedestal to {ped_file}")

    def get_pedestals(self):
        """
        Get computed pedestals for all telescopes.

        Returns
        -------
        dict
            Dictionary mapping tel_id to 32x32 pedestal array

        Raises
        ------
        ValueError
            If pedestals have not been computed yet
        """
        if not self._pedestals_computed:
            raise ValueError("Pedestals not yet computed. Call compute_pedestals() first.")
        return self.pedestals

    def get_pedestal(self, tel_id):
        """
        Get pedestal for a specific telescope.

        Parameters
        ----------
        tel_id : int
            Telescope ID

        Returns
        -------
        np.ndarray
            32x32 pedestal array for the telescope

        Raises
        ------
        ValueError
            If pedestals have not been computed
        """
        if not self._pedestals_computed:
            raise ValueError("Pedestals not yet computed. Call compute_pedestals() first.")
        return self.pedestals.get(tel_id, None)

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
        nsig : float, optional
            Sigma threshold for outlier removal (default = 5.0)
        fit_gaussian : bool, optional
            If True, fit Gaussian to pedvar; else use std (default = True)

        Returns
        -------
        pedestal : np.ndarray
            Per-pixel mean values
        pedvar : np.ndarray
            Per-pixel variance (Gaussian sigma or std)
        """
        return calculate_pedestal_and_pedvar_robust(data, nsig=nsig, fit_gaussian=fit_gaussian)

