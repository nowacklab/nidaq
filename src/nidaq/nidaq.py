"""
Example of doing high-throughput memory-mapped data acquisition with a NI DAQ.
"""

# TODO: consistent casing

from typing import Optional
import dataclasses
from dataclasses import dataclass
import json
import inspect
import logging
import io
import asyncio
import sys
import os
from os import PathLike
from pathlib import Path
import numpy as np
import numpy.typing as npt
import mmap
import time
from datetime import datetime
import math
from npy_append_array import NpyAppendArray as npaa

from zhinst.toolkit import Session as ZISession

import nidaqmx as ni
from nidaqmx.system import System as NISystem
from nidaqmx.system.device import Device as NIDevice
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogUnscaledWriter
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogUnscaledReader
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode, TerminalConfiguration, WAIT_INFINITELY, ResolutionType

from Nowack_Lab.Instruments.preamp import SR5113

import argparse
import webbrowser # for opening source files, despite the name
import subprocess
from . import code_tracking
from . import cernox

def busysleep(s):
    t0 = time.perf_counter()
    tf = t0 + s
    while time.perf_counter() < tf:
        pass

def cernoxResistanceOhms(hf2li):
    hf2taTransimpedanceVA = hf2li.zctrls[0].tamp[1].currentgain()
    demods = hf2li.demods["*"].sample()
    hf2liV = demods[hf2li.demods[0].sample]
    hf2liVx, hf2liVy = hf2liV["x"][0], hf2liV["y"][0] # Vrms
    hf2liI = demods[hf2li.demods[1].sample]
    hf2liIx, hf2liIy = hf2liI["x"][0], hf2liI["y"][0] # Vrms
    return math.sqrt((hf2liVx**2 + hf2liVy**2) / (hf2liIx**2 + hf2liIy**2)) * hf2taTransimpedanceVA

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
        return super().default(x)

@recordable
@dataclass
class DAQOutputSignal:
    "Data necessary to describe a DAQ output waveform, independent of channel or range."
    samples: npt.NDArray[np.int16]
    regenerations: int
    sampleRate: float
    frequency: float
    stepLSB: int

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
        ) -> dict:

    device = daqio.input.task.devices[0].name
    samplingFrequency = daqio.output.signal.sampleRate
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

    # TODO: include margin and callbackDuration parameters
    daqioHardwareParameters = {
            "outDataLength": outDataLength,
            "callbackRegenerations": callbackRegenerations,
            "callbackSamples": callbackSamples,
            }

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
    aiTask.register_every_n_samples_acquired_into_buffer_event(callbackSamples, None) # Unregister first for OK rerun
    aiTask.register_every_n_samples_acquired_into_buffer_event(callbackSamples, aiCallback)

    coTask.start()
    aiTask.start()
    aoTask.start()

    await untilTrue(aiCallbackDone)

    aiTask.stop()
    aoTask.stop()
    coTask.stop()

    coTask.close()

    return daqioHardwareParameters

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
    logging.info(f"step: {step}")
    logging.info(f"amplitude: {amplitude}")
    x = amplitude // step
    samples = np.zeros((4 * x,), dtype = np.int16)
    for i in range(x):
        samples[i] = i*step
    for i in range(1, 2*x + 1):
        samples[i + x - 1] = amplitude - i*step
    for i in range(x):
        samples[i + 3 * x] = -amplitude + i*step
    return samples

def daqTriangleVoltageFromZero(
        device: str,
        channel: str,
        amplitudeVolts: float,
        stepVolts: float,
        regenerations: int,
        maxFrequency: float,
        maxSampleRate: float = 2e6,
        ) -> DAQOutputTask:

    aoTask = ni.Task()
    aoTask.ao_channels.add_ao_voltage_chan(f"{device}/{channel}", "", -amplitudeVolts, amplitudeVolts)
    aoTask.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

    logging.info(f"amplitudeVolts: {amplitudeVolts}")
    logging.info(f"stepVolts: {stepVolts}")

    referenceVoltage = aoTask.ao_channels[0].ao_dac_ref_val
    minVoltage = aoTask.ao_channels[0].ao_dac_rng_low
    maxVoltage = aoTask.ao_channels[0].ao_dac_rng_high
    coefficients = aoTask.ao_channels[0].ao_dev_scaling_coeff

    logging.info(f"coefficients: {coefficients}")

    resolutionUnit = aoTask.ao_channels[0].ao_resolution_units
    if resolutionUnit != ResolutionType.BITS:
        raise DAQOutputError(f"I expect the resolution to be in bits, not {resolutionUnit}.")
    bits = int(aoTask.ao_channels[0].ao_resolution)

    sampleStep = int(coefficients[1] * stepVolts)
    if sampleStep == 0:
        raise DAQOutputError(f"Step of {stepVolts} V is too small. Minimum possible step is {referenceVoltage / coefficients[1]} V.")


    sampleDesiredAmplitude = coefficients[1] * amplitudeVolts
    sampleAmplitude = int(sampleStep * math.floor(sampleDesiredAmplitude / sampleStep))

    logging.info(f"sampleStep: {sampleStep}")
    logging.info(f"sampleDesiredAmplitude: {sampleDesiredAmplitude}")
    logging.info(f"sampleAmplitude: {sampleAmplitude}")

    samples = triangleSamplesFromZero(
        amplitude = sampleAmplitude,
        step = sampleStep,
        bits = bits,
        )

    sampleRate = min(maxSampleRate, maxFrequency * samples.size)
    frequency = sampleRate / samples.size

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
                sampleRate = sampleRate,
                frequency = frequency,
                stepLSB = sampleStep,
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
    program = Path(sys.argv[0]).name
    argumentParser = argparse.ArgumentParser(
            prog = program,
            description = "Measure an IV curve with a NI DAQ",
            )
    argumentParser.add_argument("-m", "--message", dest = "message", default = None)
    argumentSubparsers = argumentParser.add_subparsers(dest = "command")

    editSubparser = argumentSubparsers.add_parser("edit")
    editSubparser.add_argument("editor_command", nargs = argparse.REMAINDER, help = "Editor command to run on source file, like \"notepad\" or \"vim -y\"")

    dirSubparser = argumentSubparsers.add_parser("dir")
    dirSubparser.add_argument("dir", nargs = "*", help = "Print the directory where the script is located")

    gitSubparser = argumentSubparsers.add_parser("git")
    gitSubparser.add_argument("git_command", nargs = argparse.REMAINDER, help = f"Run a git command in the repo for the script")

    drySubparser = argumentSubparsers.add_parser("dry")
    drySubparser.add_argument("dry", nargs = "*", help = "Do a dry run: execute until ready to measure, but show configuration instead")

    arguments = argumentParser.parse_args()

    if arguments.command == "edit":
        sourcePath = Path(__file__).resolve()
        print(f"Editing source at {sourcePath}")
        if len(arguments.editor_command) == 0:
            # TODO: make more portable if necessary
            # See https://docs.python.org/3/library/webbrowser.html#webbrowser.open
            webbrowser.open(sourcePath)
            return 0
        else:
            return subprocess.run([*arguments.editor_command, sourcePath]).returncode

    if arguments.command == "dir":
        sourceDir = Path(__file__).parent.resolve()
        print(sourceDir) # Leave this in native format
        return 0

    if arguments.command == "git":
        completedProcess = code_tracking.runGitCommand(code_tracking.getRepo(__file__), arguments.git_command)
        return completedProcess.returncode

    logging.basicConfig(level = logging.INFO)
    startTime = datetime.now()

    rootDirectory = Path("daqiv-" + timePathComponent(startTime))
    dataRootDirectory = rootDirectory
    daqioDataPath = dataRootDirectory / Path("input-samples.bin")

    heaterVoltagePath = dataRootDirectory / Path("heater-voltages-V.npy")
    heaterTemperaturePath = dataRootDirectory / Path("heater-temperatures-K.npy")

    gainControlPath = dataRootDirectory / Path("gain.control.txt")
    gainPath = dataRootDirectory / Path("gain.npy")
    stopControlPath = dataRootDirectory / Path("stop.control.txt")
    ioCountOutputPath = dataRootDirectory / Path("io_count.output.txt")

    initialTemperaturePath = dataRootDirectory / Path("initial-temperatures-K.npy")
    finalTemperaturePath = dataRootDirectory / Path("final-temperatures-K.npy")

    parametersPath = rootDirectory / Path("parameters.json")
    parametersRootDirectory = parametersPath.parent # / Path("parameter-data")

    niSystem = NISystem.local()
    device = niDevice(niSystem)
    deviceName = device.name

    execution = code_tracking.fileExecutionData(__file__, sys.argv,
            message = f"{program}: {arguments.message}" if arguments.message is not None else f"{program} [first execution for new tree]",
            dirtyOK = arguments.command == "dry",
            )

    p = { # Parameters
            "execution": execution,
            "comment": inspect.cleandoc(f"""
            Others grounded
            Source 16, sink 14, V across 15 and 13
            Racy file-based control
            Actual cooldown
            """),
            "cooldown": 4,
            "device": {
                "id": "ns30q1",
                #"id": "Cernox X160190",
                },
            "heater": { # Sweep parameters filled in later
                "totalResistanceOhm": 1.068e3,
                "coldResistanceOhm": 0.736e3,
                },
            "preamp": {
                "gain": {
                    "path": Path(os.path.relpath(gainPath, parametersPath.parent)).as_posix(),
                    },
                "cutoffFrequencyHz": 0.0, # Filled in later
                "instrument": {
                    "port": "COM4",
                    "name": "Signal Recovery 5113",
                    "serial": "17214017",
                    "rev": "1B",
                    },
                },
            "thermometer": {
                "type": "Cernox",
                "serial": "X160190",
                },
            "magnet": { # Filled in later
                "selfResistanceOhms": 59.5,
                "totalResistanceOhms": 396.8,
            },
            "daqiv": {
                "daqTriangleCurrentFromZero": {
                    "device": deviceName,
                    "channel": "ao0",
                    "totalResistanceOhms": 7.24e3, # Estimated typical SQUID
                    "amplitudeAmps": 50e-6, # SQUID
                    #"totalResistanceOhms": 7.27e3, # Cernox
                    #"amplitudeAmps": 12e-6, # Cernox
                    #"stepAmps": 40e-9,
                    "stepAmps": 0.5e-6, # Fewer data while taking IVs over time
                    "regenerations": 1,
                    "maxFrequency": 1.0,
                    },
                "input": {
                    "device": deviceName,
                    "channel": "ai16",
                    "minVoltage": -5.0,
                    "maxVoltage": 5.0,
                    },
                "daqioDataPath": Path(os.path.relpath(daqioDataPath, parametersPath.parent)).as_posix(),
                "daqVersionInformation": {
                    "NIDAQmxVersion": niDriverVersion(niSystem),
                    "deviceSerialNumber": hexSerialNumber(device),
                    },
                },
            "version": {
                "number": 2,
                },
            }

    try:
        cernox_Ohm_to_K = cernox.temperature_for(p["thermometer"]["serial"])

        zisession = ZISession("localhost", hf2 = True)
        hf2li = zisession.connect_device("DEV131")

        mfsession = ZISession("192.168.100.78")
        mf = mfsession.connect_device("DEV3447")

        logging.info("== IV ==")
        output = daqTriangleCurrentFromZero(**p["daqiv"]["daqTriangleCurrentFromZero"])
        input = daqInputTask(**p["daqiv"]["input"])
        daqio = DAQSingleIO(input, output)
        p["daqiv"]["daqio"] = daqio

        dataRootDirectory.mkdir(parents = True, exist_ok = True)
        parametersRootDirectory.mkdir(parents = True, exist_ok = True)

#       heaterTotalResistanceOhm = p["heater"]["totalResistanceOhm"]
#       heaterThermalizationSeconds = 65.0
#       p["heater"]["thermalizationSeconds"] = heaterThermalizationSeconds
#       heaterMaxV = 7.0
#       heaterPowersW = np.linspace(0, heaterMaxV**2 / heaterTotalResistanceOhm, 6)
#       heaterVoltagesV = np.sqrt(heaterPowersW * heaterTotalResistanceOhm)
#       p["heater"]["voltagesV"] = {
#               "path": Path(os.path.relpath(heaterVoltagePath, parametersPath.parent)).as_posix(),
#       }
#       np.save(heaterVoltagePath.resolve(), heaterVoltagesV)
#       p["heater"]["temperaturesK"] = {
#               "path": Path(os.path.relpath(heaterTemperaturePath, parametersPath.parent)).as_posix(),
#       }

        if arguments.command == "dry":
            import yaml
            parametersJSON = json.dumps(p, indent = 2, cls = RecordJSONEncoder, state = {
                "newPath": newPath(rootDirectory = parametersRootDirectory, relativeTo = parametersPath.parent),
                })
            print(yaml.dump(yaml.safe_load(parametersJSON)))

            with open(parametersPath.resolve(), "x") as f:
                f.write(parametersJSON)
        else:
            print(dataRootDirectory)

            # Set up preamp and gain control
            preamp = SR5113(p["preamp"]["instrument"]["port"])
            gain = preamp.gain
            p["preamp"]["cutoffFrequencyHz"] = preamp.filter[1]
            preamp.sleep()
            with open(gainControlPath, "w") as f:
                f.write(f"{gain}")

            # Write static parameters
            with open(parametersPath.resolve(), "x") as f:
                parametersJSON = json.dumps(p, indent = 2, cls = RecordJSONEncoder, state = {
                    "newPath": newPath(rootDirectory = parametersRootDirectory, relativeTo = parametersPath.parent),
                    })
                f.write(parametersJSON)

#           heaterSamples = len(heaterVoltagesV)
#           heaterMinV, heaterMaxV = np.min(heaterVoltagesV), np.max(heaterVoltagesV)
#           with open(daqioDataPath.resolve(), "xb+") as dataFile, npaa(heaterTemperaturePath.resolve()) as heaterTemperatures:
            with open(daqioDataPath.resolve(), "xb+") as dataFile:
               #for (i, heaterVoltage) in enumerate(heaterVoltagesV):
               #    print(f"({i + 1} / {heaterSamples}) heater at {heaterVoltageV:0.2f} V in [{heaterMinV:0.2f}, {heaterMaxV:0.2f}]", flush = True)
               #    mf.auxouts[2].offset(heaterVoltageV)
               #    busysleep(heaterThermalizationSeconds)
               #    heaterTemperatures.append(np.array([cernox_Ohm_to_K(cernoxResistanceOhms(hf2li))]))
                ioCount = 0
                with open(ioCountOutputPath, "w") as f:
                    f.write(f"{ioCount}")

                stopped = False
                stopControlPath.touch()

                while not stopped:
                    # Gain control
                    try:
                        setGain = gain
                        with open(gainControlPath, "r") as f:
                            message = f.read()
                            try:
                                setGain = float(message)
                            except ValueError:
                                logging.warn(f"Could not parse gain from '{message}'. Keeping old gain: {gain}")
                        if setGain != gain: # Preamp gains are small enough to be exactly respresentable as floats
                            try:
                                logging.info(f"Setting gain to {setGain} from {gain}")
                                preamp.wake()
                                preamp.gain = setGain
                                time.sleep(0.2)
                                preamp.sleep()
                                gain = setGain
                            except Exception:
                                logging.warn(f"Could not set preamp gain to {setGain}. Keeping old gain: {gain}")
                    except OSError:
                        logging.warn(f"Could not read gain control file at '{gainControlPath.resolve()}'. Keeping old gain: {gain}")

                    with npaa(gainPath) as gains:
                        gains.append(np.array([gain]))

                    cernox_R_Ohms = cernoxResistanceOhms(hf2li)
                    with npaa(initialTemperaturePath) as initialTemperatures:
                        initialTemperatures.append(np.array([cernox_Ohm_to_K(cernox_R_Ohms)]))
                            
                    logging.info(f"Running io {ioCount}")
                    daqioHardwareParameters = asyncio.run(daqSingleIO(daqio, dataFile = dataFile))

                    cernox_R_Ohms = cernoxResistanceOhms(hf2li)
                    with npaa(finalTemperaturePath) as finalTemperatures:
                        finalTemperatures.append(np.array([cernox_Ohm_to_K(cernox_R_Ohms)]))

                    with open(ioCountOutputPath, "w") as f:
                        f.write(f"{ioCount}") # It is fast enough, so write strings for human-editability
                    ioCount += 1

                    try:
                        with open(stopControlPath, "r") as f:
                            status = f.read()
                            if status.startswith("stop"): # Test start not equality, so that trailing space is ignored
                                stopped = True
                    except OSError:
                        logging.warn("Could not read stop control file at '{stopControlPath.resolve()}'. Continuing.")
               #   heaterTemperatures.append(np.array([cernox_Ohm_to_K(cernoxResistanceOhms(hf2li))]))

    finally:
        output.task.close()
        input.task.close()

