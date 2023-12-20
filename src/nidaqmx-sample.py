"""
Example of doing high-throughput memory-mapped data acquisition with a NI DAQ.
TODO: Verify stability and that data is output and saved correctly.
"""

from typing import Optional, Tuple
import asyncio
from pathlib import Path
import numpy as np
import mmap
import time
import math
import nidaqmx as ni
from nidaqmx.system import System as NISystem
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogUnscaledWriter
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogUnscaledReader
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode, TerminalConfiguration, WAIT_INFINITELY

class NIDeviceError(Exception):
    pass

def niDriverVersion(system: Optional[NISystem] = None) -> str:
    if system is None:
        system = NISystem.local()
    v = system.driver_version
    return f"{v.major_version}.{v.minor_version}.{v.update_version}"

def niDevices(system: Optional[NISystem] = None) -> str:
    if system is None:
        system = NISystem.local()
    return system.devices.device_names

def niDevice(system: Optional[NISystem] = None) -> str:
    devices = niDevices(system)
    numDevices = len(devices)
    if numDevices == 0:
        raise NIDeviceError("No NI devices are detected")
    elif numDevices > 1:
        raise NIDeviceError("More than one NI device is detected (choose one)")
    return devices[0]

# TODO: How can we handle speed errors for the callback?
#       Rate issues are thrown as errors from the reading function.
# TODO: What happens to tasks and callbacks when somebody presses Ctrl-C?

def writeSamplesCallback(task, regenerations, callbacksToRun, doneBox, numberOfSamples):
    n = 1

    f = open("./testdata.bin", "wb+")
    f.seek(0, 2) # Go to end of file
    initFileLength = f.tell()
    sampleBytes = 2 # is generally np.dtype(np.uint16).itemsize

    # We want to mmap starting from the last page (really, size mmap.ALLOCATIONGRANULARITY to be cross-platform)
    # before the end of the file, and ending at the minimum size to fit the data.
    # Then create the numpy array from the mmap with the offset from the start of the mmap to the start of the data.
    # This way we mmap the minimum amount of the file we need, and the length of the file is the length of the data,
    # and does not need to be rounded up to the page size.
    fullDataLength = regenerations * numberOfSamples * sampleBytes
    initPageOffset = mmap.ALLOCATIONGRANULARITY * math.floor(initFileLength / mmap.ALLOCATIONGRANULARITY)
    dataOffset = initFileLength % mmap.ALLOCATIONGRANULARITY

    f.truncate(initFileLength + fullDataLength) # Extend by full data length
    
    mm = mmap.mmap(f.fileno(), length = dataOffset + fullDataLength, access = mmap.ACCESS_WRITE, offset = initPageOffset)
    data = np.ndarray((1, numberOfSamples), dtype = np.uint16, buffer = mm, order = 'C', offset = dataOffset)

    outdir = Path.cwd()
    aiReader = AnalogUnscaledReader(task.in_stream)

    # Why does this decrease the read time from ~65 ms to ~16 ms? Is that all really from numpy?
    aiReader.verify_array_shape = False # set True while debugging

    def callback(taskHandle, everyNSamplesEventType, numberOfSamples, callbackData):
        ta = time.perf_counter()
        nonlocal n, data, aiReader

        aiReader.read_uint16(data, numberOfSamples)

        tb = time.perf_counter()
        print(f"\t{tb - ta} s elapsed in callback {n}")

        if n >= callbacksToRun:
            ta = time.perf_counter()
            doneBox[0] = True
            del data
            mm.flush()
            mm.close()
            # For now, close f here,
            # since the test callback made f instead of getting it as a parameter
            f.close()
            tb = time.perf_counter()
            print(f"Time to flush and close: {tb - ta} s")
        else:
            n += 1

        return 0
    return callback

def firstDivisorFrom(a, b):
    """
    Returns the first divisor of b that is greater than or equal to a.
    """
    while b % a != 0:
        a += 1
    return a

async def untilTrue(box: list[bool]):
    while not box[0]:
        continue

async def main(device: str,
         samplingFrequency: float,
         outDataLength: int,
         outputChannel: str,
         inputChannel: str,
         outputRegenerations: int = 1,
         minOutputVoltage: float = -10.0,
         maxOutputVoltage: float = 10.0,
         minInputVoltage: float = -10.0,
         maxInputVoltage: float = 10.0,
         ):
    # Reading data in the every n samples callback takes a little under 65 ms on average,
    # but up to 100 ms for the first run.
    # For the callback to have enough time to run, we need n such that n > tcallback / Tsample
    # We also would like n to be a multiple of the output data length.
    # This must also be a divisor of the total number of samples.
    callbackRateMargin = 2 # Minimum callbacks worth of data to multiply the callback sample interval to keep up with data.
    # TODO: provide a way to measure the callback duration, rather than hard code it
    callbackDuration = 65e-3 # Mean value (set margin to account for initial callback and latency spikes)
    callbackMinRegenerations = math.ceil(callbackRateMargin * callbackDuration * samplingFrequency / outDataLength)
    callbackRegenerations = firstDivisorFrom(callbackMinRegenerations, outputRegenerations)
    sampleInterval = outDataLength * callbackRegenerations
    callbacksToRun = outputRegenerations // callbackRegenerations
    inputBufferMargin = 2 # Sets the size of the input buffer in units of callback data to prevent circular overwriting.
    inputBufferSize = sampleInterval * inputBufferMargin

    print(f"sampleInterval: {sampleInterval}")

    # We know that we are using an X Series device, so we do not need the logic for getting the fully qualified name of the trigger.
    aoStartTrigger = f"/{device}/ao/StartTrigger"

    regeneratedOutDataLength = outputRegenerations * outDataLength

    outData = np.zeros((outDataLength,), dtype = np.uint16)
    outData[0] = 1024 # Arbitrary values at the start and end of an output generation for debugging
    outData[-2] = 256
    outData[-1] = 4096

    coChannel = "ctr0"
    coInternalOutput = "Ctr0InternalOutput"

    coTask = ni.Task("co")
    coTask.co_channels.add_co_pulse_chan_freq(f"{device}/{coChannel}", "", freq = samplingFrequency, duty_cycle = 0.5)
    coTask.timing.cfg_implicit_timing(sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = 0)

    aoTask = ni.Task("ao")
    aoTask.ao_channels.add_ao_voltage_chan(f"{device}/{outputChannel}", "", minOutputVoltage, maxOutputVoltage)
    aoTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.FALLING, sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = regeneratedOutDataLength)
    aoTask.triggers.start_trigger.cfg_dig_edge_start_trig(coInternalOutput, Edge.RISING)
    aoTask.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
    aoWriter = AnalogUnscaledWriter(aoTask.out_stream, auto_start = False)
    aoWriter.verify_array_shape = False # set True while debugging
    aoWriter.write_uint16(np.reshape(outData, (1, outDataLength)))

    aiTask = ni.Task("ai")
    aiTask.ai_channels.add_ai_voltage_chan(f"{device}/{inputChannel}", "", TerminalConfiguration.DEFAULT, minInputVoltage, maxInputVoltage)
    aiTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.RISING, sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = regeneratedOutDataLength)
    aiTask.triggers.start_trigger.cfg_dig_edge_start_trig(aoStartTrigger, Edge.RISING)
    aiTask.in_stream.input_buf_size = inputBufferSize

    aiCallbackDone = [False]
    aiCallback = writeSamplesCallback(aiTask, regenerations = outputRegenerations, callbacksToRun = callbacksToRun, doneBox = aiCallbackDone, numberOfSamples = sampleInterval)
    aiTask.register_every_n_samples_acquired_into_buffer_event(sampleInterval, aiCallback)

    coTask.start()
    aiTask.start()
    aoTask.start()

    await untilTrue(aiCallbackDone)

    aiTask.stop()
    aiTask.close()

    aoTask.stop()
    aoTask.close()

    coTask.stop()
    coTask.close()


if __name__ == "__main__":
    niSystem = NISystem.local()
    print(f"NI driver version: {niDriverVersion(niSystem)}")
    asyncio.run(main(
        device = niDevice(niSystem),
        outputChannel = "ao3",
        inputChannel = "ai4",
        samplingFrequency = 2e6,
        outDataLength = 256 * 1024,
        outputRegenerations = 8,
        ))

