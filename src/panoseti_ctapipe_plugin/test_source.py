from pathlib import Path
test_file = "test_data/start_2025-10-22T07-18-22Z.dp_ph1024.bpp_2.module_254.seqno_0.pff"

def test_basic():
    from panoseti_ctapipe_plugin import PanoEventSource

def test_compatible():
    from panoseti_ctapipe_plugin import PanoEventSource 
    assert PanoEventSource.is_compatible(test_file) 

def test_plugin():
    from ctapipe.io import EventSource
    with EventSource(test_file) as source:
        assert source.__class__.__name__ == "PanoEventSource"

def test_eventsource_in_subclasses():
    #from panoseti_ctapipe_plugin import PanoEventSource 
    from ctapipe.io import EventSource
    assert "PanoEventSource" in EventSource.non_abstract_subclasses()

def test_init():
    from panoseti_ctapipe_plugin import PanoEventSource 
    with PanoEventSource(test_file) as source:
        pass

def test_iterate_events():
    from ctapipe.io import EventSource
    with EventSource(test_file) as source:
        n_events = 0
        for event in source:
            n_events += 1
        assert n_events == 944

def test_ctapipe_process(tmp_path):
    from ctapipe.core.tool import run_tool
    from ctapipe.tools.process import ProcessorTool
    from ctapipe.io import read_table 
    input_path=Path(test_file).absolute()
    output_path=tmp_path/"test.dl1.h5"
    result = run_tool(tool=ProcessorTool(), argv=['--input', str(input_path), '--output', str(output_path), '--log-level=INFO'])
    read_table(output_path, "/dl1/event/telescope/parameters/tel_001")
    
        


    

