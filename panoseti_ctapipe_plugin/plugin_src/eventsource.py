"""
PanoEventSource: ctapipe EventSource for PANOSETI PFF files.

This module provides the main EventSource class that reads PANOSETI pulse height
data from PFF files and yields calibrated DL1 images compatible with ctapipe.

Last modified: 29 May 2026
"""

import logging
from copy import deepcopy
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
    ObservingMode,
    PointingMode,
    SchedulingBlockContainer,
    SchedulingBlockType,
    SimulationConfigContainer,
    TriggerContainer,
)
from ctapipe.io import DataLevel, EventSource

from .instrument import (
    MODULE_TO_TEL_ID,
    subarray,
)
from .functions import (
    filter_packet_loss,
    filter_rate_spikes,
    correct_gain,
    calculate_pedestal_and_pedvar_robust,
    extract_timestamps_from_metadata,
    load_pointing_offset_csv,
    match_coincident_events,
    correct_telescope_timing,
    pixel_to_skycoord,
    rotate_images_after_meridian_flip,
    subtract_pedestal,
)
from ..utils.calibration import load_gain_file

__all__ = ["PanoEventSource"]

logger = logging.getLogger(__name__)


def _count_events(telescope_data):
    """Count the total number of per-telescope events in a data mapping."""
    return sum(len(data) for data in telescope_data.values())


def _emit_progress(message, verbose=True, show_progress=False):
    """Report pipeline progress in logs and in notebooks without INFO logging."""
    if not verbose or not show_progress:
        return

    logger.info(message)
    if not logger.isEnabledFor(logging.INFO):
        print(message)


def _to_astropy_time(timestamp):
    """Convert either datetime64 or unix-float timestamps to astropy Time."""
    ts_array = np.asarray(timestamp)
    if np.issubdtype(ts_array.dtype, np.datetime64):
        return Time(timestamp, scale="utc", format="datetime64")
    return Time(float(timestamp), scale="utc", format="unix")


class PanoEventSource(EventSource):
    """
    EventSource for PANOSETI PFF pulse height data.

    Reads pulse heights from PFF module files and yields filtered, calibrated
    DL1 images.

    The source also provides helper methods for later processing stages such as
    event collection, pedestal computation, gain loading, pedestal subtraction,
    gain correction, and end-to-end calibrated iteration.

    Multi-telescope coincidence matching is applied automatically to provide
    events synchronized across telescopes.
    """

    is_simulation = False
    datalevels = (DataLevel.DL1_IMAGES,)  # Raw pulse heights as DL1 images

    def __init__(
        self,
        input_url=None,
        subarray_desc=None,
        sb_type=None,
        observing_mode=None,
        pointing_mode=None,
        pointing_offset_csv=None,
        meridian_flip_phase=None,
        reference_tel_id=2,
        coincidence_time_window=0.001,
        show_progress=False,
        **kwargs
    ):
        """
        Initialize PanoEventSource for PFF pulse height data.

        Reads pulse height data from PFF files, performs the configured
        filtering and coincidence-matching steps, then applies default
        calibration during iteration.

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
        pointing_offset_csv : str or Path, optional
            Path to CSV file with source pixel coordinates for pointing correction.
            CSV should contain: date, tel, phase, pixel_x, pixel_y.
            If None, the default packaged pointing-offset CSV is used.
        meridian_flip_phase : str, optional
            Observation phase: "pre" or "post" (relative to meridian flip).
            Default is "pre". Used to select correct pointing offset from CSV.
        reference_tel_id : int, optional
            Telescope ID to use as reference for timing correction.
            Default is 2 (Winter). All other telescopes are corrected relative to this one.
        coincidence_time_window : float, optional
            Time window in seconds for matching coincident events across telescopes.
            Default is 0.001 (1 ms). Accepts 2-tel, 3-tel, or 4-tel coincidences within this window.
        show_progress : bool, optional
            Emit one-line pipeline progress summaries during iteration.
            Default is False.
        **kwargs
            Additional arguments passed to EventSource

            Use the standard ctapipe ``allowed_tels`` EventSource trait to select
            which telescopes to process. For example, ``allowed_tels={2, 3, 4}``
            keeps Winter, Fern, and PTI.

        """
        if subarray_desc is None:
            subarray_desc = subarray

        self._subarray = subarray_desc

        # Set scheduling block parameters with defaults
        self.sb_type = (
            sb_type if sb_type is not None else SchedulingBlockType.OBSERVATION
        )
        self.observing_mode = (
            observing_mode if observing_mode is not None else ObservingMode.ON_OFF #WOBBLE
        )
        self.pointing_mode = (
            pointing_mode if pointing_mode is not None else PointingMode.TRACK
        )

        super().__init__(input_url=input_url, **kwargs)
        self._pff_files = []
        self._metadata = {}  # Store metadata for filtering use

        # Load pointing offset corrections (pixel coordinates of source)
        self._pointing_offset_df = None
        try:
            self._pointing_offset_df = load_pointing_offset_csv(pointing_offset_csv)
        except Exception as e:
            logger.warning(f"Failed to load pointing offsets: {e}. Will use housekeeping pointing.")

        # Store meridian flip phase ("pre" or "post")
        self._meridian_flip_phase = meridian_flip_phase if meridian_flip_phase is not None else "pre"
        if self._meridian_flip_phase not in ("pre", "post"):
            logger.warning(f"Invalid meridian_flip_phase '{self._meridian_flip_phase}'. Using 'pre'.")
            self._meridian_flip_phase = "pre"
        
        # Store reference telescope for timing correction
        self._reference_tel_id = reference_tel_id
        
        # Store coincidence time window
        self._coincidence_time_window = coincidence_time_window
        self._show_progress = show_progress

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

    # Data loading and timestamp extraction

    def load_raw_telescope_streams(self, verbose=True):
        """Load raw per-telescope event arrays and metadata from the observation folder."""
        obs_dir = Path(self.input_url)
        module_files_list = sorted(obs_dir.glob("start*ph1024*module_*.*.pff"))

        if not module_files_list:
            raise FileNotFoundError(f"No module_*.pff files found in {obs_dir}")

        module_files = {}
        for file_path in module_files_list:
            filename = file_path.name
            parts = filename.split("module_")
            if len(parts) > 1:
                module_num = int(parts[1].split(".")[0])
                if module_num in MODULE_TO_TEL_ID:
                    tel_id = MODULE_TO_TEL_ID[module_num]
                    module_files[tel_id] = str(file_path)

        if not module_files:
            raise FileNotFoundError(f"No recognized modules found in {obs_dir}")

        tel_ids = sorted(module_files.keys())
        if self.allowed_tels is not None:
            module_files = {
                tid: path for tid, path in module_files.items() if tid in self.allowed_tels
            }
            tel_ids = [tid for tid in tel_ids if tid in self.allowed_tels]
            if verbose:
                allowed_str = ", ".join([f"Tel{tid}" for tid in tel_ids])
                logger.info(f"Processing allowed telescopes: {allowed_str}")

        telescope_data = {}
        telescope_metadata = {}

        for tel_id, file_path in module_files.items():
            pff_file = pypff.io.datapff(file_path)
            data, metadata = pff_file.readpff(metadata=True)
            telescope_data[tel_id] = data
            telescope_metadata[tel_id] = metadata
            self._metadata[tel_id] = metadata
            self._pff_files.append(pff_file)
            if verbose:
                logger.info(f"Read {len(data)} events from telescope {tel_id}")

        return module_files, telescope_data, telescope_metadata

    def extract_telescope_timestamps(self, telescope_metadata):
        """Extract raw timestamps for each telescope from per-telescope metadata."""
        telescope_timestamps = {}
        for tel_id, metadata in telescope_metadata.items():
            timestamps, valid_mask = extract_timestamps_from_metadata(metadata)
            telescope_timestamps[tel_id] = timestamps
        return telescope_timestamps

    # Calibration helpers

    def load_gains(self, gain_file=None, verbose=True):
        """Load per-pixel gain calibration for each telescope."""
        gains = {}
        tel_ids = sorted(self.allowed_tels) if self.allowed_tels is not None else self.subarray.tel_ids
        if verbose:
            logger.info("Loading gain calibration...")

        for tel_id in tel_ids:
            gains[tel_id] = load_gain_file(tel_id, gain_file)
            if verbose:
                logger.info(f"  Tel {tel_id}: Loaded gains")

        return gains

    def compute_pedestal_pedvar(self, all_data=None, nsig=5.0, verbose=True):
        """Compute pedestal mean and pedestal-standard-deviation images for each telescope.

        Parameters
        ----------
        all_data : dict[int, np.ndarray]
            Per-telescope event arrays.
        nsig : float
            Outlier threshold in units of sigma for robust pedestal estimation.
        verbose : bool
            Emit progress logging while computing pedestal statistics.

        Returns
        -------
        tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
            Returned as ``(pedestal_means, pedestal_stds)``.
        """
        if all_data is None:
            raise ValueError("all_data must be provided to compute_pedestal_pedvar()")

        tel_ids = sorted(all_data.keys())

        pedestals = {}
        pedestal_stds = {}
        if verbose:
            logger.info("Computing pedestals from source event data...")

        for tel_id in tel_ids:
            if not all_data[tel_id].size:
                logger.warning(f"Tel {tel_id}: No data available, using zero pedestals")
                pedestals[tel_id] = np.zeros((32, 32))
                pedestal_stds[tel_id] = np.zeros((32, 32))
                continue

            ped_flat, ped_std_flat = calculate_pedestal_and_pedvar_robust(
                all_data[tel_id], nsig=nsig, fit_gaussian=True
            )
            pedestals[tel_id] = ped_flat.reshape(32, 32)
            pedestal_stds[tel_id] = ped_std_flat.reshape(32, 32)

            if verbose:
                logger.info(f"  Tel {tel_id}: Pedestal from {len(all_data[tel_id])} events")

        return pedestals, pedestal_stds

    def apply_gain_corrections(self, telescope_data, gains):
        """Apply gain correction to per-telescope image arrays."""
        calibrated_data = {}
        for tel_id, data in telescope_data.items():
            if data.size == 0:
                calibrated_data[tel_id] = data.copy()
                continue
            calibrated_data[tel_id] = correct_gain(data, gains[tel_id])
        return calibrated_data

    def apply_pedestal_subtraction(self, telescope_data, pedestals):
        """Apply pedestal subtraction to per-telescope image arrays."""
        calibrated_data = {}
        for tel_id, data in telescope_data.items():
            if data.size == 0:
                calibrated_data[tel_id] = data.copy()
                continue
            calibrated_data[tel_id] = subtract_pedestal(data, pedestals[tel_id])
        return calibrated_data

    def apply_meridian_flip_corrections(self, telescope_data):
        """Apply post-meridian-flip image rotation to per-telescope image arrays."""
        corrected_data = {}
        for tel_id, data in telescope_data.items():
            if data.size == 0:
                corrected_data[tel_id] = data.copy()
                continue

            corrected_data[tel_id] = rotate_images_after_meridian_flip(
                data,
                meridian_flip_phase=self._meridian_flip_phase,
                n_pix=32,
            )

        return corrected_data

    # Filtering and stream alignment

    def apply_metadata_masks(self, telescope_metadata, masks):
        """Apply per-telescope boolean masks to event-indexed metadata arrays."""
        filtered_metadata = {}

        for tel_id, metadata in telescope_metadata.items():
            mask = np.asarray(masks[tel_id], dtype=bool)
            filtered_metadata[tel_id] = {}

            for quabo_key, quabo_metadata in metadata.items():
                filtered_metadata[tel_id][quabo_key] = {}
                for field, values in quabo_metadata.items():
                    values_array = np.asarray(values)
                    if values_array.ndim > 0 and values_array.shape[0] == mask.shape[0]:
                        filtered_metadata[tel_id][quabo_key][field] = values_array[mask]
                    else:
                        filtered_metadata[tel_id][quabo_key][field] = deepcopy(values)

        return filtered_metadata

    def apply_metadata_sorting(self, telescope_metadata, sort_indices_by_tel):
        """Apply per-telescope sort indices to event-indexed metadata arrays."""
        sorted_metadata = {}

        for tel_id, metadata in telescope_metadata.items():
            sort_indices = np.asarray(sort_indices_by_tel[tel_id], dtype=np.int64)
            sorted_metadata[tel_id] = {}

            for quabo_key, quabo_metadata in metadata.items():
                sorted_metadata[tel_id][quabo_key] = {}
                for field, values in quabo_metadata.items():
                    values_array = np.asarray(values)
                    if values_array.ndim > 0 and values_array.shape[0] == sort_indices.shape[0]:
                        sorted_metadata[tel_id][quabo_key][field] = values_array[sort_indices]
                    else:
                        sorted_metadata[tel_id][quabo_key][field] = deepcopy(values)

        return sorted_metadata

    def apply_packet_loss_filters(self, telescope_data, telescope_timestamps, telescope_metadata=None, verbose=True):
        """Apply packet-loss filtering while preserving the provided timestamp stream."""
        if telescope_metadata is None:
            telescope_metadata = self._metadata

        filtered_data = {}
        filtered_timestamps = {}
        masks = {}

        for tel_id, data in telescope_data.items():
            loss_mask, n_removed_loss, data_filtered, ts_filtered = filter_packet_loss(
                telescope_metadata[tel_id],
                data=data,
                timestamps=telescope_timestamps[tel_id],
            )
            filtered_data[tel_id] = data_filtered
            filtered_timestamps[tel_id] = ts_filtered
            masks[tel_id] = loss_mask

            if verbose and n_removed_loss > 0:
                logger.info(f"Tel{tel_id}: Removed {n_removed_loss} events due to packet loss")

        return filtered_data, filtered_timestamps, masks

    def apply_rate_spike_filters(self, telescope_data, telescope_timestamps, bin_width=30, rate_threshold=2.0, verbose=True):
        """Apply only rate-spike filtering to per-telescope data and timestamps."""
        filtered_data = {}
        filtered_timestamps = {}
        masks = {}

        for tel_id, data in telescope_data.items():
            timestamps = telescope_timestamps[tel_id]
            spike_mask, n_removed_spike, data_filtered, ts_filtered = filter_rate_spikes(
                timestamps,
                bin_width=bin_width,
                rate_threshold=rate_threshold,
                data=data,
            )
            filtered_data[tel_id] = data_filtered
            filtered_timestamps[tel_id] = ts_filtered
            masks[tel_id] = spike_mask

            if verbose and n_removed_spike > 0:
                logger.info(f"Tel{tel_id}: Removed {n_removed_spike} events due to rate spikes")

        return filtered_data, filtered_timestamps, masks

    def apply_invalid_timestamp_filters(self, telescope_data, telescope_timestamps, min_timestamp=np.datetime64("2000-01-01", "ns"), verbose=True):
        """Drop events with invalid timestamps from per-telescope data and timestamps."""
        filtered_data = {}
        filtered_timestamps = {}
        masks = {}

        min_timestamp_float = None
        if not isinstance(min_timestamp, (int, float)):
            min_timestamp_float = (
                np.asarray(min_timestamp, dtype="datetime64[ns]").astype("int64") * 1e-9
            )

        for tel_id, data in telescope_data.items():
            timestamps = telescope_timestamps[tel_id]
            if np.issubdtype(np.asarray(timestamps).dtype, np.datetime64):
                valid_timestamp_mask = timestamps >= min_timestamp
            else:
                threshold = min_timestamp if isinstance(min_timestamp, (int, float)) else min_timestamp_float
                valid_timestamp_mask = np.asarray(timestamps, dtype=np.float64) >= threshold
            filtered_data[tel_id] = data[valid_timestamp_mask]
            filtered_timestamps[tel_id] = timestamps[valid_timestamp_mask]
            masks[tel_id] = valid_timestamp_mask

            if verbose and not np.all(valid_timestamp_mask):
                n_invalid = np.sum(~valid_timestamp_mask)
                logger.info(f"Tel{tel_id}: Removed {n_invalid} events due to invalid timestamps")

        return filtered_data, filtered_timestamps, masks

    def apply_telescope_stream_sorting(self, telescope_data, telescope_timestamps, verbose=True):
        """Sort each telescope stream by timestamp."""
        sorted_data = {}
        sorted_timestamps = {}
        sort_indices_by_tel = {}

        for tel_id, data in telescope_data.items():
            timestamps = telescope_timestamps[tel_id]
            if np.issubdtype(timestamps.dtype, np.datetime64):
                ts_float = timestamps.astype("datetime64[ns]").astype("int64") * 1e-9
            else:
                ts_float = timestamps

            sort_indices = np.argsort(ts_float)
            sorted_data[tel_id] = data[sort_indices]
            sorted_timestamps[tel_id] = timestamps[sort_indices]
            sort_indices_by_tel[tel_id] = sort_indices

            if verbose:
                logger.debug(f"Tel{tel_id}: Sorted {len(sort_indices)} events by timestamp")

        return sorted_data, sorted_timestamps, sort_indices_by_tel

    def apply_timing_corrections(self, telescope_timestamps, reference_tel_id=None, window=0.020, bin_width=120, verbose=True):
        """Apply only inter-telescope timing correction to timestamp streams."""
        corrected_timestamps = {tel_id: timestamps.copy() for tel_id, timestamps in telescope_timestamps.items()}
        timing_results = {}

        tel_ids = sorted(corrected_timestamps.keys())
        if len(tel_ids) <= 1:
            return corrected_timestamps, timing_results

        reference_tel = reference_tel_id if reference_tel_id in tel_ids else tel_ids[0]
        if verbose:
            if reference_tel_id is not None and reference_tel != reference_tel_id:
                logger.warning(f"Reference telescope {reference_tel_id} not in data. Using {reference_tel} instead.")
            logger.info(f"Using Tel{reference_tel} as timing reference")

        reference_timestamps = corrected_timestamps[reference_tel]

        for tel_id in tel_ids:
            if tel_id == reference_tel:
                if verbose:
                    logger.info(f"  Tel{reference_tel}: Reference telescope (no correction needed)")
                continue

            try:
                correction_result = correct_telescope_timing(
                    corrected_timestamps[tel_id],
                    reference_timestamps,
                    window=window,
                    bin_width=bin_width,
                )
                timestamps_corrected_float = correction_result["timestamps1_corr"]
                corrected_timestamps[tel_id] = np.array(
                    timestamps_corrected_float * 1e9, dtype="datetime64[ns]"
                )
                timing_results[tel_id] = correction_result

                if verbose:
                    rms_before = correction_result.get("rms_before", np.nan)
                    rms_after = correction_result.get("rms_after", np.nan)
                    logger.info(
                        f"  Tel{tel_id}: Timing correction applied (RMS before: {rms_before:.6f}s -> after: {rms_after:.6f}s)"
                    )
            except Exception as e:
                if verbose:
                    logger.warning(f"Timing correction failed for Tel{tel_id}: {e}. Proceeding with uncorrected timestamps.")

        return corrected_timestamps, timing_results

    def apply_coincidence_matching(self, telescope_data, telescope_timestamps, time_window=None, verbose=True):
        """Run only coincidence matching on per-telescope data and timestamps."""
        if time_window is None:
            time_window = self._coincidence_time_window

        if verbose:
            logger.info(f"Searching for coincidences within +/-{time_window*1000:.1f}ms time window...")

        coincidences = list(
            match_coincident_events(
                telescope_timestamps,
                data_dict=telescope_data,
                time_window=time_window,
            )
        )

        if verbose:
            n_2tel = sum(1 for c in coincidences if len(c["tel_ids"]) == 2)
            n_3tel = sum(1 for c in coincidences if len(c["tel_ids"]) == 3)
            n_4tel = sum(1 for c in coincidences if len(c["tel_ids"]) == 4)
            logger.info(
                f"Found {len(coincidences)} total coincidences: {n_2tel} 2-telescope, {n_3tel} 3-telescope, {n_4tel} 4-telescope"
            )

        return coincidences

    # Event-building helpers

    def build_events_from_coincidences(self, coincidences, telescope_data, telescope_timestamps, module_files=None):
        """Convert coincidence matches back into ctapipe ArrayEventContainer objects."""
        event_count = 0
        for coinc in coincidences:
            if self.max_events is not None and event_count >= self.max_events:
                break

            event_time = coinc["event_time"]
            trigger = TriggerContainer(time=_to_astropy_time(event_time))
            event = ArrayEventContainer(trigger=trigger)
            event.count = event_count
            event.trigger.tels_with_trigger = []

            for tel_id in coinc["tel_ids"]:
                event_index = coinc["indices"][tel_id]
                raw_pulse_height = np.array(telescope_data[tel_id][event_index], dtype=np.float32)
                image = raw_pulse_height.reshape((32, 32))
                event.dl1.tel[tel_id].image = image.flatten()
                event.trigger.tel[tel_id].time = _to_astropy_time(
                    telescope_timestamps[tel_id][event_index]
                )
                event.trigger.tels_with_trigger.append(tel_id)

            event.index.obs_id = list(self.obs_ids)[0] if self.obs_ids else 0
            event.meta = getattr(event, "meta", {})
            if module_files is not None:
                event.meta["source_files"] = {
                    tel_id: module_files[tel_id] for tel_id in coinc["tel_ids"]
                }
            event.meta["num_telescopes"] = len(coinc["tel_ids"])
            event.meta["time_window"] = self._coincidence_time_window

            yield event
            event_count += 1

    def _generator(self):
        """
        Generator that yields filtered, calibrated DL1 images from PFF data.

        Reads pulse heights from PFF files, applies gain correction first,
        then the default filtering steps, then pedestal subtraction, and
        finally performs multi-telescope coincidence matching based on
        corrected timestamps.

        Additional processing stages are exposed via helper methods on this source,
        such as ``compute_pedestal_pedvar()`` and ``load_gains()``.

        Yields
        ------
        event : ArrayEventContainer
            Event container with calibrated DL1 images
        """
        verbose = True
        pedestals = None
        gains = None
        nsig = 5.0
        gain_file = None

        module_files, telescope_data, telescope_metadata = self.load_raw_telescope_streams(
            verbose=verbose
        )
        _emit_progress(
            f"Raw data loading done: read {_count_events(telescope_data)} events across {len(telescope_data)} telescopes.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        telescope_timestamps = self.extract_telescope_timestamps(telescope_metadata)
        _emit_progress(
            f"Timestamp extraction done: built timestamp streams for {len(telescope_timestamps)} telescopes.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        if gains is None:
            gains = self.load_gains(gain_file=gain_file, verbose=verbose)
        _emit_progress(
            f"Gain loading done: loaded calibration for {len(gains)} telescopes.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        telescope_data = self.apply_gain_corrections(telescope_data, gains)
        _emit_progress(
            f"Gain calibration done: corrected {_count_events(telescope_data)} events across {len(telescope_data)} telescopes.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        before_packet_loss = _count_events(telescope_data)
        telescope_data, telescope_timestamps, packet_loss_masks = self.apply_packet_loss_filters(
            telescope_data,
            telescope_timestamps,
            telescope_metadata=telescope_metadata,
            verbose=verbose,
        )
        telescope_metadata = self.apply_metadata_masks(telescope_metadata, packet_loss_masks)
        after_packet_loss = _count_events(telescope_data)
        _emit_progress(
            f"Packet-loss filtering done: removed {before_packet_loss - after_packet_loss} events; {after_packet_loss} remain.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        before_rate_spikes = after_packet_loss
        telescope_data, telescope_timestamps, rate_spike_masks = self.apply_rate_spike_filters(
            telescope_data,
            telescope_timestamps,
            verbose=verbose,
        )
        telescope_metadata = self.apply_metadata_masks(telescope_metadata, rate_spike_masks)
        after_rate_spikes = _count_events(telescope_data)
        _emit_progress(
            f"Spike-rate filtering done: removed {before_rate_spikes - after_rate_spikes} events; {after_rate_spikes} remain.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        before_invalid_timestamps = after_rate_spikes
        telescope_data, telescope_timestamps, valid_timestamp_masks = self.apply_invalid_timestamp_filters(
            telescope_data,
            telescope_timestamps,
            verbose=verbose,
        )
        telescope_metadata = self.apply_metadata_masks(telescope_metadata, valid_timestamp_masks)
        after_invalid_timestamps = _count_events(telescope_data)
        _emit_progress(
            f"Invalid-timestamp filtering done: removed {before_invalid_timestamps - after_invalid_timestamps} events; {after_invalid_timestamps} remain.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        if pedestals is None:
            pedestals, _ = self.compute_pedestal_pedvar(
                all_data=telescope_data,
                nsig=nsig,
                verbose=verbose,
            )
        _emit_progress(
            f"Pedestal computation done: computed pedestals for {len(pedestals)} telescopes.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        telescope_data = self.apply_pedestal_subtraction(telescope_data, pedestals)
        _emit_progress(
            f"Pedestal subtraction done: calibrated {_count_events(telescope_data)} events.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        telescope_data = self.apply_meridian_flip_corrections(telescope_data)
        _emit_progress(
            f"Meridian-flip correction done: applied {self._meridian_flip_phase} image orientation handling.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        telescope_data, telescope_timestamps, sort_indices_by_tel = self.apply_telescope_stream_sorting(
            telescope_data,
            telescope_timestamps,
            verbose=verbose,
        )
        telescope_metadata = self.apply_metadata_sorting(telescope_metadata, sort_indices_by_tel)
        self._metadata = telescope_metadata
        _emit_progress(
            f"Timestamp sorting done: sorted {after_invalid_timestamps} surviving events across {len(telescope_data)} telescopes.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        telescope_timestamps, _ = self.apply_timing_corrections(
            telescope_timestamps,
            reference_tel_id=self._reference_tel_id,
            verbose=verbose,
        )
        _emit_progress(
            f"Timing correction done: aligned telescope timestamps using Tel{self._reference_tel_id} as the requested reference.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        telescope_data, telescope_timestamps, sort_indices_by_tel = self.apply_telescope_stream_sorting(
            telescope_data,
            telescope_timestamps,
            verbose=verbose,
        )
        if self._metadata:
            self._metadata = self.apply_metadata_sorting(self._metadata, sort_indices_by_tel)
        _emit_progress(
            "Post-correction timestamp sorting done: event streams are ready for coincidence matching.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        coincidences = self.apply_coincidence_matching(
            telescope_data,
            telescope_timestamps,
            time_window=self._coincidence_time_window,
            verbose=verbose,
        )
        _emit_progress(
            f"Coincidence matching done: found {len(coincidences)} coincident events.",
            verbose=verbose,
            show_progress=self._show_progress,
        )

        max_events_text = (
            str(self.max_events) if self.max_events is not None else "all available"
        )
        _emit_progress(
            f"Event building done: yielding up to {max_events_text} calibrated events.",
            verbose=verbose,
            show_progress=self._show_progress,
        )
        yield from self.build_events_from_coincidences(
            coincidences,
            telescope_data,
            telescope_timestamps,
            module_files=module_files,
        )

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

            mount_to_tel_id = {
                f"MOUNT_{tel_description.name.upper()}": tel_id
                for tel_id, tel_description in self.subarray.tel.items()
            }

            # Temporary fallback until housekeeping mount selection is clarified.
            # Default to the known mount key present in current housekeeping files.
            mount_key = "MOUNT_GATTINI"

            if mount_key not in hk:
                return {}

            ra_hours = float(hk[mount_key]["ra_hours"][0])
            dec_deg = float(hk[mount_key]["dec_deg"][0])
            ra_deg = ra_hours * 15

            meridian_flip = self._meridian_flip_phase
            if self._pointing_offset_df is not None:
                tel_id_from_mount = mount_to_tel_id.get(mount_key)

                if tel_id_from_mount is not None:
                    mask = (
                        (self._pointing_offset_df["date"].dt.date == obs_date)
                        & (self._pointing_offset_df["tel"] == tel_id_from_mount)
                    )
                    if mask.any():
                        phase_mask = mask & (self._pointing_offset_df["phase"] == meridian_flip)
                        if phase_mask.any():
                            offset_row = self._pointing_offset_df[phase_mask].iloc[0]
                        else:
                            logger.warning(
                                f"No matching phase '{meridian_flip}' for Tel{tel_id_from_mount} on {obs_date}. Using first available entry."
                            )
                            offset_row = self._pointing_offset_df[mask].iloc[0]

                        pixel_x = offset_row["pixel_x"]
                        pixel_y = offset_row["pixel_y"]

                        source_skycoord = pixel_to_skycoord(
                            pixel_x=pixel_x,
                            pixel_y=pixel_y,
                            tel_pointing_ra_deg=ra_deg,
                            tel_pointing_dec_deg=dec_deg,
                            obs_time=start_time,
                            focal_length_m=0.46,
                            pixel_size_mm=3.0,
                        )

                        ra_deg = source_skycoord.ra.deg
                        dec_deg = source_skycoord.dec.deg
                        logger.info(
                            f"Applied pointing correction ({meridian_flip} meridian flip): pixel ({pixel_x}, {pixel_y}) -> RA={ra_deg:.4f} deg, Dec={dec_deg:.4f} deg"
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
            obs_block.meta["meridian_flip"] = meridian_flip
            return {obs_id: obs_block}
        except Exception as e:
            logger.error(f"Error extracting observation blocks: {e}")
            return {}

    @property
    def scheduling_blocks(self):
        """Extract scheduling block metadata from observation date."""
        try:
            data_dir = Path(self.input_url)
            module_files = list(data_dir.glob("start*ph1024*module_*.*.pff"))
            if not module_files:
                return {}

            filename = module_files[0].name
            start_str = filename.split("start_")[1].split(".")[0]
            start_time = Time(start_str, format="isot")
            sb_id = np.uint64(int(start_time.strftime("%Y%m%d")))

            sb_block = SchedulingBlockContainer(
                sb_id=sb_id,
                producer_id="Panoseti",
                sb_type=self.sb_type,
                observing_mode=self.observing_mode,
                pointing_mode=self.pointing_mode,
            )
            return {sb_id: sb_block}
        except Exception:
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

