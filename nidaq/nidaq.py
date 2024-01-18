"""
Example of doing high-throughput memory-mapped data acquisition with a NI DAQ.
TODO: Verify stability and that data is output and saved correctly.
"""

from typing import Optional
import dataclasses
from dataclasses import dataclass
import json
import io
import asyncio
from os import PathLike
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
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode, TerminalConfiguration, WAIT_INFINITELY, ResolutionType

@dataclass
class DAQOutputSignal:
    "Data necessary to describe a DAQ output waveform, independent of channel or range."
    samples: npt.NDArray[np.int16]
    regenerations: int
    frequency: float

@dataclass
class DAQOutputCalibration:
    "Data necessary to determine output voltage from raw DAQ samples."
    bits: int
    coefficients: list[float]
    referenceVoltage: float

@dataclass
class DAQInputCalibration:
    "Data necessary to determine output voltage from raw DAQ samples."
    bits: int
    coefficients: list[float]
    minVoltage: float
    maxVoltage: float

@dataclass
class DAQOutputTask:
    "Data necessary to describe a DAQ output waveform, independent of channel or range."
    task: ni.Task
    calibration: DAQOutputCalibration
    signal: DAQOutputSignal

@dataclass
class DAQInputTask:
    "Data necessary to describe a DAQ output waveform, independent of channel or range."
    task: ni.Task
    calibration: DAQOutputCalibration

@dataclass
class DAQSingleIO:
    input: DAQInputTask
    output: DAQOutputTask

    def __post_init__(self):
        if self.input.task.devices != self.output.task.devices:
            raise ValueError("""
Synchronized IO across different DAQ devices is possible, but not implemented.
Make sure that the input and output NI DAQ devices are the same.
            """.strip())


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

def writeSamplesCallback(
        task: ni.Task,
        dataFile: io.BufferedIOBase,
        callbacksToRun: int,
        doneBox: list[bool],
        callbackSamples: int
        ):
    n = 1

    dataFile.seek(0, 2) # Go to end of file
    initFileLength = dataFile.tell()
    sampleBytes = 2 # is generally np.dtype(np.int16).itemsize

    # We want to mmap starting from the last page (really, size mmap.ALLOCATIONGRANULARITY to be cross-platform)
    # before the end of the file, and ending at the minimum size to fit the data.
    # Then create the numpy array from the mmap with the offset from the start of the mmap to the start of the data.
    # This way we mmap the minimum amount of the file we need, and the length of the file is the length of the data,
    # and does not need to be rounded up to the page size.
    fullDataLength = callbacksToRun * callbackSamples * sampleBytes
    initPageOffset = mmap.ALLOCATIONGRANULARITY * math.floor(initFileLength / mmap.ALLOCATIONGRANULARITY)
    dataMmapOffset = initFileLength % mmap.ALLOCATIONGRANULARITY

    dataFile.truncate(initFileLength + fullDataLength) # Extend by full data length
    
    mm = mmap.mmap(dataFile.fileno(), length = dataMmapOffset + fullDataLength, access = mmap.ACCESS_WRITE, offset = initPageOffset)

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
            print("|")
        else:
            n += 1

        return 0
    return callback

def firstDivisorFrom(a: int, b: int) -> int:
    "Returns the first divisor of b that is greater than or equal to a."
    while b % a != 0:
        a += 1
    return a

async def untilTrue(box: list[bool]):
    while not box[0]:
        continue

async def daqSingleIO(
        dio: DAQSingleIO,
        dataFile: io.BufferedIOBase,
        ) -> bool:

    device = dio.input.task.devices[0].name
    samplingFrequency = dio.output.signal.frequency
    outputRegenerations = dio.output.signal.regenerations
    outData = dio.output.signal.samples
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

    coTask = ni.Task()
    coTask.co_channels.add_co_pulse_chan_freq(f"{device}/{coChannel}", "", freq = samplingFrequency, duty_cycle = 0.5)
    coTask.timing.cfg_implicit_timing(sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = 0)

    aoTask = dio.output.task
    aoTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.FALLING, sample_mode = AcquisitionType.FINITE, samps_per_chan = regeneratedOutDataLength)
    aoTask.triggers.start_trigger.cfg_dig_edge_start_trig(coInternalOutput, Edge.RISING)
    aoWriter = AnalogUnscaledWriter(aoTask.out_stream, auto_start = False)
    aoWriter.verify_array_shape = False # set True while debugging
    aoWriter.write_int16(np.reshape(outData, (1, outDataLength)))

    aiTask = dio.input.task
    aiTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.RISING, sample_mode = AcquisitionType.FINITE, samps_per_chan = regeneratedOutDataLength)
    aiTask.triggers.start_trigger.cfg_dig_edge_start_trig(aoStartTrigger, Edge.RISING)
    aiTask.in_stream.input_buf_size = inputBufferSize

    aiCallbackDone = [False]
    aiCallback = writeSamplesCallback(aiTask, dataFile = dataFile, callbacksToRun = callbacksToRun, doneBox = aiCallbackDone, callbackSamples = callbackSamples)
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

    return True


class BinaryFileError(Exception):
    pass

def openBinaryFileWithoutTruncating(filePath: PathLike) -> io.BufferedIOBase:
    filePath = Path(filePath)
    exists = filePath.exists()
    is_file = filePath.is_file()
    if exists ^ is_file:
        raise BinaryFileError(f"{filePath} is already used by a non-file")
    return open(filePath, ('r' if is_file else 'x') + "b+")

def daqInputTask(
        device: str,
        channel: str,
        minVoltage: float,
        maxVoltage: float,
        ) -> DAQInputTask:
    aiTask = ni.Task()
    aiTask.ai_channels.add_ai_voltage_chan(f"{device}/{channel}", "", TerminalConfiguration.DEFAULT, minVoltage, maxVoltage)

    actualMinVoltage = aiTask.ai_channels[0].ai_rng_low
    actualMaxVoltage = aiTask.ai_channels[0].ai_rng_high
    coefficients = aiTask.ai_channels[0].ai_dev_scaling_coeff

    resolutionUnit = aiTask.ai_channels[0].ai_resolution_units
    if resolutionUnit != ResolutionType.BITS:
        raise DAQInputError(f"I expect the resolution to be in bits, not {resolutionUnit}.")
    bits = int(aiTask.ai_channels[0].ai_resolution)

    return DAQInputTask(
            task = aiTask,
            calibration = DAQInputCalibration(
                bits = bits,
                coefficients = coefficients,
                minVoltage = actualMinVoltage,
                maxVoltage = actualMaxVoltage,
                ),
            )

class DAQOutputError(Exception):
    pass

def triangleSamplesFromZero(amplitude: int, step: int, bits: int) -> npt.NDArray[np.int16]:
    x = amplitude // step
    samples = np.zeros((2 * x,), dtype = np.int16)
    for i in range(x // 2):
        samples[i] = i*step
    for i in range(1, x + 1):
        samples[i + (x // 2) - 1] = amplitude // 2 - i*step
    for i in range(x // 2):
        samples[i + 3 * x // 2] = -amplitude // 2 + i*step
    return samples

def daqTriangleOutputFromZero(
        device: str,
        channel: str,
        amplitudeVolts: float,
        stepVolts: float,
        regenerations: int,
        maxFrequency: float,
        maxSampleFrequency: float = 2e6,
        outputRanges: list[float] = [5.0, 10.0],
        ) -> DAQOutputTask:

    aoTask = ni.Task()
    aoTask.ao_channels.add_ao_voltage_chan(f"{device}/{channel}", "", -amplitudeVolts, amplitudeVolts)
    aoTask.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

    referenceVoltage = aoTask.ao_channels[0].ao_dac_ref_val
    coefficients = aoTask.ao_channels[0].ao_dev_scaling_coeff

    resolutionUnit = aoTask.ao_channels[0].ao_resolution_units
    if resolutionUnit != ResolutionType.BITS:
        raise DAQOutputError(f"I expect the resolution to be in bits, not {resolutionUnit}.")
    bits = int(aoTask.ao_channels[0].ao_resolution)

    sampleStep = int(coefficients[1] * stepVolts)
    if sampleStep == 0:
        raise DAQOutputError(f"Step of {stepVolts} V is too small. Minimum possible step is {referenceVoltage / coefficients[1]} V.")

    sampleDesiredAmplitude = coefficients[1] * amplitudeVolts / referenceVoltage
    sampleAmplitude = int(sampleStep * math.floor(sampleDesiredAmplitude / sampleStep))
    
    samples = triangleSamplesFromZero(
        amplitude = sampleAmplitude,
        step = sampleStep,
        bits = bits,
        )

    frequency = min(maxSampleFrequency, maxFrequency * samples.size)

    print(f"Step: {sampleStep} LSB")
    print(f"Frequency: {frequency / samples.size} Hz")

    return DAQOutputTask(
            task = aoTask,
            calibration = DAQOutputCalibration(
                bits = bits,
                coefficients = coefficients,
                referenceVoltage = referenceVoltage,
                ),
            signal = DAQOutputSignal(
                samples = samples,
                regenerations = regenerations,
                frequency = frequency,
                ),
            )

def daqIVTriangleFromZero(
        totalResistanceOhms: float,
        amplitudeAmps: float,
        stepAmps: float,
        **kwargs
        ) -> DAQOutputTask:
    return daqTriangleOutputFromZero(
            amplitudeVolts = amplitudeAmps * totalResistanceOhms,
            stepVolts = stepAmps * totalResistanceOhms,
            **kwargs)

def nidaq():
    niSystem = NISystem.local()
    niVersion = niDriverVersion(niSystem)
    device = niDevice(niSystem)
    print(f"NI driver version: {niDriverVersion(niSystem)}")

    inputTaskSpec = {
            "device": device,
            "channel": "ai4",
            "minVoltage": -0.2,
            "maxVoltage": 0.2,
            }
    outputTaskSpec = {
            "device": device,
            "channel": "ao3",
            }
    daqIVTriangleFromZeroParameters = {
            "totalResistanceOhms": 15e3,
            "amplitudeAmps": 200e-6,
            "stepAmps": 100e-9,
            "regenerations": 16,
            "maxFrequency": 5e3,
            }

    input = daqInputTask(**inputTaskSpec)
    output = daqIVTriangleFromZero(**daqIVTriangleFromZeroParameters, **outputTaskSpec)

    dio = DAQSingleIO(input, output)

    # Assumes that the working directory is where we want to put data.
    dataFilePath = Path("input-samples.bin")
    samplesFile = Path("output-samples.npy")

    inputDict = {
            "calibration": dataclasses.asdict(input.calibration),
            }
    signalDict = dataclasses.asdict(output.signal)
    del signalDict["samples"]
    signalDict["samplesFile"] = samplesFile.as_posix()
    outputDict = {
            "calibration": dataclasses.asdict(output.calibration),
            "signal": signalDict,
            }
    dioDict = {
            "input": inputDict,
            "output": outputDict,
            }

    print(json.dumps(dioDict, indent = 2))
    np.save(samplesFile, output.signal.samples)

    with openBinaryFileWithoutTruncating(dataFilePath) as dataFile:
        asyncio.run(daqSingleIO(dio = dio, dataFile = dataFile))

