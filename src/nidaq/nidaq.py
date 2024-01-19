"""
Example of doing high-throughput memory-mapped data acquisition with a NI DAQ.
TODO: Verify stability and that data is output and saved correctly.
"""

from typing import Optional
import dataclasses
from dataclasses import dataclass
import json
import inspect
import logging
import io
import asyncio
import os
from os import PathLike
from pathlib import Path
import numpy as np
import numpy.typing as npt
import mmap
import time
from datetime import datetime
import math
import nidaqmx as ni
from nidaqmx.system import System as NISystem
from nidaqmx.system.device import Device as NIDevice
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogUnscaledWriter
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogUnscaledReader
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode, TerminalConfiguration, WAIT_INFINITELY, ResolutionType

def timePathComponent(dt: datetime) -> str:
    return dt.astimezone().isoformat().replace(":", "").replace(".", "d")

class RecordError(Exception):
    pass

class RecordableDefinitionError(Exception):
    pass

def recordable(cls):
    if hasattr(cls, 'recordAttribute'):
        if callable(cls.recordAttribute):
            return cls
        else:
            raise RecordableDefinitionError(f"""
A recordable class must have a recordAttribute method, but the recordAttribute of {cls} is not callable.
""")

    def recordAttribute(self, name: str, state):
        return getattr(self, name)

    setattr(cls, 'recordAttribute', recordAttribute)
    return cls

class RecordJSONEncoder(json.JSONEncoder):
    def __init__(self, state, **kwargs):
        super(RecordJSONEncoder, self).__init__(**kwargs)
        self.state = state

    def default(self, x):
        if hasattr(x, "record") and callable(x.record):
            return x.record(self.state)
        elif hasattr(x, "recordAttribute") and callable(x.recordAttribute):
            r = {}
            for k in vars(x):
                v = x.recordAttribute(k, self.state)
                if isinstance(v, RecordError):
                    raise v
                if v is not None:
                    r[k] = v
            return r
       #elif hasattr(x, "__dict__"):
       #    r = {}
       #    for k, v in vars(x).items():
       #        # is self.default necessary?
       #        r[k] = self.default(v)
       #    return r
        return super().default(x)

@recordable
@dataclass
class DAQOutputSignal:
    "Data necessary to describe a DAQ output waveform, independent of channel or range."
    samples: npt.NDArray[np.int16]
    regenerations: int
    frequency: float

    def recordAttribute(self, name: str, state):
        if name == "samples":
            path, reference = state["newPath"]("output-samples", "npy")
            try:
                np.save(path, getattr(self, name), allow_pickle = False)
                return { "path": reference }
            except Exception as e:
                return RecordError(e)
        else:
            return getattr(self, name)

@recordable
@dataclass
class DAQOutputCalibration:
    "Data necessary to determine output voltage from raw DAQ samples."
    bits: int
    coefficients: list[float]
    referenceVoltage: float
    minVoltage: float
    maxVoltage: float

@recordable
@dataclass
class DAQInputCalibration:
    "Data necessary to determine output voltage from raw DAQ samples."
    bits: int
    coefficients: list[float]
    minVoltage: float
    maxVoltage: float

@recordable
@dataclass
class DAQOutputTask:
    "Data necessary to describe a DAQ output waveform, independent of channel or range."
    task: ni.Task
    calibration: DAQOutputCalibration
    signal: DAQOutputSignal

    def recordAttribute(self, name: str, state):
        if name == "task":
            return None
        else:
            return getattr(self, name)

@recordable
@dataclass
class DAQInputTask:
    "Data necessary to describe a DAQ output waveform, independent of channel or range."
    task: ni.Task
    calibration: DAQOutputCalibration

    def recordAttribute(self, name: str, state):
        if name == "task":
            return None
        else:
            return getattr(self, name)

@recordable
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

def niDevices(system: Optional[NISystem] = None) -> list[NIDevice]:
    if system is None:
        system = NISystem.local()
    return system.devices

def niDevice(system: Optional[NISystem] = None) -> NIDevice:
    devices = niDevices(system)
    numDevices = len(devices)
    if numDevices == 0:
        raise NIDeviceError("No NI devices are detected")
    elif numDevices > 1:
        raise NIDeviceError("More than one NI device is detected (choose one)")
    return devices[0]

def hexSerialNumber(device: NIDevice) -> str:
    return hex(device.serial_num)[2:].upper()

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

    def callback(taskHandle, everyNSamplesEventType, callbackSamples, callbackData):
        ta = time.perf_counter()
        nonlocal n, aiReader

        if doneBox[0]:
            return 0

        dataCallbackOffset = (n - 1) * callbackSamples * sampleBytes
        data = np.ndarray((1, callbackSamples), dtype = np.int16, buffer = mm, order = 'C', offset = dataMmapOffset + dataCallbackOffset)
        aiReader.read_int16(data, callbackSamples)

        print(f"DAQ callback {n} / {callbacksToRun}", end = '\r', flush = True)

        if n >= callbacksToRun:
            doneBox[0] = True
            del data
            mm.flush()
            mm.close()
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
        daqio: DAQSingleIO,
        dataFile: io.BufferedIOBase,
        ) -> bool:

    device = daqio.input.task.devices[0].name
    samplingFrequency = daqio.output.signal.frequency
    outputRegenerations = daqio.output.signal.regenerations
    outData = daqio.output.signal.samples
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

    logging.info(f"outDataLength: {outDataLength}")
    logging.info(f"callbackRegenerations: {callbackRegenerations}")
    logging.info(f"callbackSamples: {callbackSamples}")

    # We know that we are using an X Series device, so we do not need the generic logic for getting the fully qualified name of the trigger, which may differ depending on the device.
    aoStartTrigger = f"/{device}/ao/StartTrigger"

    regeneratedOutDataLength = outputRegenerations * outDataLength

    coChannel = "ctr0"
    coInternalOutput = "Ctr0InternalOutput"

    coTask = ni.Task()
    coTask.co_channels.add_co_pulse_chan_freq(f"{device}/{coChannel}", "", freq = samplingFrequency, duty_cycle = 0.5)
    coTask.timing.cfg_implicit_timing(sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = 0)

    aoTask = daqio.output.task
    aoTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.FALLING, sample_mode = AcquisitionType.FINITE, samps_per_chan = regeneratedOutDataLength)
    aoTask.triggers.start_trigger.cfg_dig_edge_start_trig(coInternalOutput, Edge.RISING)
    aoWriter = AnalogUnscaledWriter(aoTask.out_stream, auto_start = False)
    aoWriter.verify_array_shape = False # set True while debugging
    aoWriter.write_int16(np.reshape(outData, (1, outDataLength)))

    aiTask = daqio.input.task
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
    return open(filePath.resolve(), ('r' if is_file else 'x') + "b+")

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

def daqTriangleVoltageFromZero(
        device: str,
        channel: str,
        amplitudeVolts: float,
        stepVolts: float,
        regenerations: int,
        maxFrequency: float,
        maxSampleFrequency: float = 2e6,
        ) -> DAQOutputTask:

    aoTask = ni.Task()
    aoTask.ao_channels.add_ao_voltage_chan(f"{device}/{channel}", "", -amplitudeVolts, amplitudeVolts)
    aoTask.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

    referenceVoltage = aoTask.ao_channels[0].ao_dac_ref_val
    minVoltage = aoTask.ao_channels[0].ao_dac_rng_low
    maxVoltage = aoTask.ao_channels[0].ao_dac_rng_high
    fullRange = maxVoltage - minVoltage
    coefficients = aoTask.ao_channels[0].ao_dev_scaling_coeff
    c = [fullRange * x for x in coefficients]

    resolutionUnit = aoTask.ao_channels[0].ao_resolution_units
    if resolutionUnit != ResolutionType.BITS:
        raise DAQOutputError(f"I expect the resolution to be in bits, not {resolutionUnit}.")
    bits = int(aoTask.ao_channels[0].ao_resolution)

    sampleStep = int(c[1] * stepVolts)
    if sampleStep == 0:
        raise DAQOutputError(f"Step of {stepVolts} V is too small. Minimum possible step is {referenceVoltage / c[1]} V.")

    sampleDesiredAmplitude = c[1] * amplitudeVolts / referenceVoltage
    sampleAmplitude = int(sampleStep * math.floor(sampleDesiredAmplitude / sampleStep))

    samples = triangleSamplesFromZero(
        amplitude = sampleAmplitude,
        step = sampleStep,
        bits = bits,
        )

    frequency = min(maxSampleFrequency, maxFrequency * samples.size)

    logging.info(f"Step: {sampleStep} LSB")
    logging.info(f"Frequency: {frequency / samples.size} Hz")

    return DAQOutputTask(
            task = aoTask,
            calibration = DAQOutputCalibration(
                bits = bits,
                coefficients = coefficients,
                referenceVoltage = referenceVoltage,
                minVoltage = minVoltage,
                maxVoltage = maxVoltage,
                ),
            signal = DAQOutputSignal(
                samples = samples,
                regenerations = regenerations,
                frequency = frequency,
                ),
            )

def daqTriangleCurrentFromZero(
        totalResistanceOhms: float,
        amplitudeAmps: float,
        stepAmps: float,
        **kwargs
        ) -> DAQOutputTask:
    return daqTriangleVoltageFromZero(
            amplitudeVolts = amplitudeAmps * totalResistanceOhms,
            stepVolts = stepAmps * totalResistanceOhms,
            **kwargs)

def newPath(rootDirectory: PathLike, relativeTo: PathLike):
    rootDirectory = Path(rootDirectory)
    relativeTo = Path(relativeTo)
    nameUses = {}
    def f(name: str, suffix: str, subDirectory: PathLike = Path(".")) -> Path:
        nonlocal rootDirectory, relativeTo, nameUses
        uniquePart = ""
        if name in nameUses:
            nameUses[name] += 1
            uniquePart = f"-{nameUses[name]}"
        else:
            nameUses[name] = 1
        path = rootDirectory / subDirectory / Path(f"{name}{uniquePart}.{suffix}")
        reference = Path(os.path.relpath(path, relativeTo)).as_posix()
        return path, reference
    return f

def nidaq():
    logging.basicConfig(level = logging.INFO)

    rootDirectory = Path("daqiv-" + timePathComponent(datetime.now()))
    dataRootDirectory = rootDirectory
    daqioDataPath = dataRootDirectory / Path("input-samples.bin")
    parametersPath = rootDirectory / Path("parameters.json")
    parametersRootDirectory = parametersPath.parent # / Path("parameter-data")

    niSystem = NISystem.local()
    device = niDevice(niSystem)
    deviceName = device.name

    p = { # Parameters
            "comment": inspect.cleandoc(f"""
            This is a test comment.
            """),
            "daqiv": {
                "daqTriangleCurrentFromZero": {
                    "device": deviceName,
                    "channel": "ao3",
                    "totalResistanceOhms": 15e3,
                    "amplitudeAmps": 100e-6,
                    "stepAmps": 1.5e-9,
                    "regenerations": 256,
                    "maxFrequency": 1e3,
                    },
                "input": {
                    "device": deviceName,
                    "channel": "ai4",
                    "minVoltage": -2.0,
                    "maxVoltage": 2.0,
                    },
                "daqioDataPath": Path(os.path.relpath(daqioDataPath, parametersPath.parent)).as_posix(),
                "daqVersionInformation": {
                    "NIDAQmxVersion": niDriverVersion(niSystem),
                    "deviceSerialNumber": hexSerialNumber(device),
                    },
                },
            }

    try:
        output = daqTriangleCurrentFromZero(**p["daqiv"]["daqTriangleCurrentFromZero"])
        input = daqInputTask(**p["daqiv"]["input"])
        daqio = DAQSingleIO(input, output)

        p["daqiv"]["daqio"] = daqio

        dataRootDirectory.mkdir(parents = True, exist_ok = True)
        parametersRootDirectory.mkdir(parents = True, exist_ok = True)
        with open(parametersPath.resolve(), "x") as f:
            json.dump(p, f, indent = 2, cls = RecordJSONEncoder, state = {
                "newPath": newPath(rootDirectory = parametersRootDirectory, relativeTo = parametersPath.parent),
                })
        with open(daqioDataPath.resolve(), "xb+") as dataFile:
            asyncio.run(daqSingleIO(daqio, dataFile = dataFile))
    except Exception as e:
        output.task.close()
        input.task.close()
        raise e

