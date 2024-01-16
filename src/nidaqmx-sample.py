"""
Example of doing high-throughput memory-mapped data acquisition with a NI DAQ.
TODO: Verify stability and that data is output and saved correctly.
"""

from typing import Optional, Tuple
import asyncio
from pathlib import Path
import numpy as np
import numpy.typing as npt
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

class DataFileError(Exception):
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

def writeSamplesCallback(task, callbacksToRun, doneBox, callbackSamples):
    n = 1

    filepath = Path("./testdata.bin")
    exists = filepath.exists()
    is_file = filepath.is_file()
    if exists ^ is_file:
        raise DataFileError(f"{filepath} is already used by a non-file")
    f = open(filepath, ('r' if is_file else 'x') + "b+")
    f.seek(0, 2) # Go to end of file
    initFileLength = f.tell()
    sampleBytes = 2 # is generally np.dtype(np.int16).itemsize

    # We want to mmap starting from the last page (really, size mmap.ALLOCATIONGRANULARITY to be cross-platform)
    # before the end of the file, and ending at the minimum size to fit the data.
    # Then create the numpy array from the mmap with the offset from the start of the mmap to the start of the data.
    # This way we mmap the minimum amount of the file we need, and the length of the file is the length of the data,
    # and does not need to be rounded up to the page size.
    fullDataLength = callbacksToRun * callbackSamples * sampleBytes
    initPageOffset = mmap.ALLOCATIONGRANULARITY * math.floor(initFileLength / mmap.ALLOCATIONGRANULARITY)
    dataMmapOffset = initFileLength % mmap.ALLOCATIONGRANULARITY

    f.truncate(initFileLength + fullDataLength) # Extend by full data length
    
    mm = mmap.mmap(f.fileno(), length = dataMmapOffset + fullDataLength, access = mmap.ACCESS_WRITE, offset = initPageOffset)

    outdir = Path.cwd()
    aiReader = AnalogUnscaledReader(task.in_stream)

    # Why does this decrease the read time from ~65 ms to ~16 ms? Is that all really from numpy?
    aiReader.verify_array_shape = False # set True while debugging

    print("_" * callbacksToRun + "|")

    def callback(taskHandle, everyNSamplesEventType, callbackSamples, callbackData):
        ta = time.perf_counter()
        nonlocal n, aiReader

        if doneBox[0]:
            return 0

        dataCallbackOffset = (n - 1) * callbackSamples * sampleBytes
        data = np.ndarray((1, callbackSamples), dtype = np.int16, buffer = mm, order = 'C', offset = dataMmapOffset + dataCallbackOffset)
        aiReader.read_int16(data, callbackSamples)

        print("*", end = '', flush = True)

        if n >= callbacksToRun:
            doneBox[0] = True
            del data
            mm.flush()
            mm.close()
            # For now, close f here,
            # since the test callback made f instead of getting it as a parameter
            f.close()
            print("|")
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
         outData: npt.NDArray[np.int16],
         outputChannel: str,
         inputChannel: str,
         outputRegenerations: int = 1,
         minOutputVoltage: float = -10.0,
         maxOutputVoltage: float = 10.0,
         minInputVoltage: float = -10.0,
         maxInputVoltage: float = 10.0,
         ):
    outDataLength = outData.size
    # Reading data in the every n samples callback takes a little under 65 ms on average,
    # but up to 100 ms for the first run.
    # For the callback to have enough time to run, we need n such that n > tcallback / Tsample
    # We also would like n to be a multiple of the output data length.
    # This must also be a divisor of the total number of samples.
    callbackRateMargin = 4 # Minimum callbacks worth of data to multiply the callback sample interval to keep up with data.
    # TODO: provide a way to automatically determine the callback duration, rather than hard code it
    callbackDuration = 50e-3 # Mean value (set margin to account for initial callback and latency spikes)
    callbackMinRegenerations = math.ceil(callbackRateMargin * callbackDuration * samplingFrequency / outDataLength)
    callbackRegenerations = firstDivisorFrom(callbackMinRegenerations, outputRegenerations) if callbackMinRegenerations < outputRegenerations else outputRegenerations
    callbackSamples = outDataLength * callbackRegenerations
    callbacksToRun = outputRegenerations // callbackRegenerations
    inputBufferMargin = 4 # Sets the size of the input buffer in units of callback data to prevent circular overwriting.
    inputBufferSize = callbackSamples * inputBufferMargin

    print(f"outDataLength: {outDataLength}")
    print(f"callbackRegenerations: {callbackRegenerations}")
    print(f"callbackSamples: {callbackSamples}")

    # We know that we are using an X Series device, so we do not need the generic logic for getting the fully qualified name of the trigger, which may differ depending on the device.
    aoStartTrigger = f"/{device}/ao/StartTrigger"

    regeneratedOutDataLength = outputRegenerations * outDataLength

    coChannel = "ctr0"
    coInternalOutput = "Ctr0InternalOutput"

    coTask = ni.Task("co")
    coTask.co_channels.add_co_pulse_chan_freq(f"{device}/{coChannel}", "", freq = samplingFrequency, duty_cycle = 0.5)
    coTask.timing.cfg_implicit_timing(sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = 0)

    aoTask = ni.Task("ao")
    aoTask.ao_channels.add_ao_voltage_chan(f"{device}/{outputChannel}", "", minOutputVoltage, maxOutputVoltage)
    aoTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.FALLING, sample_mode = AcquisitionType.FINITE, samps_per_chan = regeneratedOutDataLength)
    aoTask.triggers.start_trigger.cfg_dig_edge_start_trig(coInternalOutput, Edge.RISING)
    aoTask.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
    aoWriter = AnalogUnscaledWriter(aoTask.out_stream, auto_start = False)
    aoWriter.verify_array_shape = False # set True while debugging
    aoWriter.write_int16(np.reshape(outData, (1, outDataLength)))

    aiTask = ni.Task("ai")
    aiTask.ai_channels.add_ai_voltage_chan(f"{device}/{inputChannel}", "", TerminalConfiguration.DEFAULT, minInputVoltage, maxInputVoltage)
    aiTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.RISING, sample_mode = AcquisitionType.FINITE, samps_per_chan = regeneratedOutDataLength)
    aiTask.triggers.start_trigger.cfg_dig_edge_start_trig(aoStartTrigger, Edge.RISING)
    aiTask.in_stream.input_buf_size = inputBufferSize

    aiCallbackDone = [False]
    aiCallback = writeSamplesCallback(aiTask, callbacksToRun = callbacksToRun, doneBox = aiCallbackDone, callbackSamples = callbackSamples)
    aiTask.register_every_n_samples_acquired_into_buffer_event(callbackSamples, aiCallback)

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

    outbits = 16
    # Got intermittent DaqError (-200621) Onboard device memory underflow
    # for a step of 256.
    step = 128
    peak = 2**outbits
    x = peak // step
    out = np.zeros((2 * x,), dtype = np.int16)
    for i in range(x // 2):
        out[i] = i*step
    for i in range(1, x + 1):
        out[i + (x // 2) - 1] = peak // 2 - i*step
    for i in range(x // 2):
        out[i + 3 * x // 2] = -peak // 2 + i*step

    asyncio.run(main(
        device = niDevice(niSystem),
        outputChannel = "ao3",
        inputChannel = "ai4",
        samplingFrequency = 2e6,
        outData = out,
        outputRegenerations = 1,
        minOutputVoltage = -5.0,
        maxOutputVoltage = 5.0,
        minInputVoltage = -0.2,
        maxInputVoltage = 0.2,
        ))

