"""
Unit tests for PANOSETI EventSource plugin.

Tests the compatibility, initialization, and basic functionality of the event source.

Following ctapipe guidelines:
https://ctapipe.readthedocs.io/en/stable/developer-guide/code-guidelines.html#unit-tests

Author: Sruthi Ravikularaman
Last modified: 17 April 2026
"""

import sys
from pathlib import Path

test_folder = "/Users/ravikularaman/VScode/panoseti_ctapipe_plugin/test_data/pff/obs_Palomar.start_2026-01-15T02:26:39Z.runtype_obs-test.pffd/obs_Palomar.start_2026-01-15T02:26:39Z.runtype_obs-test.pffd"

def test_basic():
    from panoseti_ctapipe_plugin import PanoEventSource

def test_compatible():
    """Test that observation folder is recognized as compatible"""
    from panoseti_ctapipe_plugin import PanoEventSource 
    assert PanoEventSource.is_compatible(test_folder) 

def test_incompatible_path():
    """Test that non-existent paths are not compatible"""
    from panoseti_ctapipe_plugin import PanoEventSource
    assert not PanoEventSource.is_compatible("/nonexistent/path")

def test_plugin_discovery():
    """Test that PanoEventSource is discovered by ctapipe"""
    from ctapipe.io import EventSource
    assert "PanoEventSource" in EventSource.non_abstract_subclasses()

def test_init_with_folder():
    """Test initialization of event source with observation folder"""
    from panoseti_ctapipe_plugin import PanoEventSource 
    with PanoEventSource(test_folder) as source:
        assert source is not None
        assert source.is_simulation == False

def test_subarray_config():
    """Test that subarray is correctly configured"""
    from panoseti_ctapipe_plugin import PanoEventSource 
    with PanoEventSource(test_folder) as source:
        assert source.subarray is not None
        assert len(source.subarray.tel) == 4

def test_observation_blocks_metadata():
    """Test observation blocks contain correct metadata"""
    from panoseti_ctapipe_plugin import PanoEventSource
    with PanoEventSource(test_folder) as source:
        obs_blocks = source.observation_blocks
        assert 0 in obs_blocks
        obs_block = obs_blocks[0]
        assert obs_block.producer_id == "Panoseti"
        assert obs_block.actual_start_time is not None

def test_scheduling_blocks_metadata():
    """Test scheduling blocks contain correct metadata"""
    from panoseti_ctapipe_plugin import PanoEventSource
    with PanoEventSource(test_folder) as source:
        sched_blocks = source.scheduling_blocks
        assert len(sched_blocks) > 0
        sb = list(sched_blocks.values())[0]
        assert sb.producer_id == "Panoseti"

def test_obs_ids_retrieval():
    """Test that obs_ids returns correct observation IDs"""
    from panoseti_ctapipe_plugin import PanoEventSource
    with PanoEventSource(test_folder) as source:
        obs_ids = list(source.obs_ids)
        assert 0 in obs_ids

def test_event_iteration():
    """Test event iteration and basic event structure"""
    from panoseti_ctapipe_plugin import PanoEventSource
    with PanoEventSource(test_folder) as source:
        n_events = 0
        for event in source:
            n_events += 1
            assert event.count == n_events - 1
            if n_events >= 2:
                break
        assert n_events > 0

def test_allowed_tels_filter():
    """Test that allowed_tels filters telescopes correctly"""
    from panoseti_ctapipe_plugin import PanoEventSource
    with PanoEventSource(test_folder, allowed_tels=[2]) as source:
        for event in source:
            assert 2 in event.trigger.tels_with_trigger
            assert not 1 in event.trigger.tels_with_trigger
            break

def test_max_events_enforcement():
    """Test that max_events limits the number of events"""
    from panoseti_ctapipe_plugin import PanoEventSource
    with PanoEventSource(test_folder, max_events=2) as source:
        n_events = 0
        for event in source:
            n_events += 1
        assert n_events == 2

def test_source_cleanup():
    """Test that close method properly cleans up resources"""
    from panoseti_ctapipe_plugin import PanoEventSource
    source = PanoEventSource(test_folder)
    source.close()

@pytest.mark.skip(reason="Quite long")
def test_ctapipe_process(tmp_path):
    """Test integration with ctapipe ProcessorTool"""
    import pytest
    print("\n[test_ctapipe_process] Starting ctapipe ProcessorTool integration test", flush=True)
    from ctapipe.core.tool import run_tool
    from ctapipe.tools.process import ProcessorTool
    import tables
    
    input_path = Path(test_folder).absolute()
    output_path = Path(tmp_path) / "test.dl1.h5"
    
    print(f"[test_ctapipe_process] Running ProcessorTool on: {input_path}", flush=True)
    run_tool(
        tool=ProcessorTool(), 
        argv=[
            '--EventSource.input_url', str(input_path), 
            '--output', str(output_path), 
            '--log-level=WARN'
        ]
    )
    
    print(f"[test_ctapipe_process] Checking output file exists: {output_path}", flush=True)
    assert output_path.exists(), "Output file was not created"
    
    print("[test_ctapipe_process] Verifying HDF5 file contains configuration", flush=True)
    with tables.open_file(str(output_path), mode='r') as h5file:
        try:
            config = h5file.get_node('/configuration')
            assert config is not None
        except tables.NoSuchNodeError:
            raise AssertionError("No configuration group in output file")
    
    print("[test_ctapipe_process] ✓ Test passed", flush=True)