from typing import Optional, Tuple
import asyncio
from pathlib import Path
import numpy as np
import nidaqmx as ni
from nidaqmx.system import System as NISystem
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogUnscaledWriter
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogUnscaledReader
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode, TerminalConfiguration, WAIT_INFINITELY
import time

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

def writeSamplesCallback(task, targetArray, regenerations, doneBox):
    n = 1
    i = 0
    def callback(taskHandle, everyNSamplesEventType, numberOfSamples, callbackData):
        nonlocal n, i
        data = np.zeros((numberOfSamples,), dtype = np.float64)
        aiStream = AnalogSingleChannelReader(task.in_stream)
        aiStream.read_many_sample(data, numberOfSamples)
        #inData[i:i + numberOfSamples] = data
        #i += numberOfSamples

        outdir = Path.cwd()
        np.save(outdir / f"testdata-{n}.npy", data, allow_pickle = False, fix_imports = False)

        print(f"{n}: {data}")

        if n >= regenerations:
            doneBox[0] = True
        else:
            n += 1

        return 0
    return callback

async def untilTrue(box: list[bool]):
    while not box[0]:
        pass

async def main(device: str,
         samplingFrequency: float,
         outDataLength: int,
         outputChannel: str,
         inputChannel: str,
         minOutputVoltage: float = -10.0,
         maxOutputVoltage: float = 10.0,
         minInputVoltage: float = -10.0,
         maxInputVoltage: float = 10.0,
         ):

    # We know that we are using an X Series device, so we do not need the logic for getting the fully qualified name of the trigger.
    aoStartTrigger = f"/{device}/ao/StartTrigger"

    outputRegenerations = 2
    regeneratedOutDataLength = outputRegenerations * outDataLength

    outData = np.ones((outDataLength,), dtype = np.float64)
    outData[0] = 2.0
    outData[-2] = 0.0
    outData[-1] = 3.0

    inData = np.zeros((regeneratedOutDataLength,), dtype = np.float64)


    coChannel = "ctr0"
    coInternalOutput = "Ctr0InternalOutput"

    coTask = ni.Task("co")
    coTask.co_channels.add_co_pulse_chan_freq(f"{device}/{coChannel}", "", freq = samplingFrequency, duty_cycle = 0.5)
    #coTask.timing.cfg_implicit_timing(sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = 0)
    coTask.timing.cfg_implicit_timing(sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = regeneratedOutDataLength)

    aoTask = ni.Task("ao")
    aoTask.ao_channels.add_ao_voltage_chan(f"{device}/{outputChannel}", "", minOutputVoltage, maxOutputVoltage)
    aoTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.FALLING, sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = regeneratedOutDataLength)
    aoTask.triggers.start_trigger.cfg_dig_edge_start_trig(coInternalOutput, Edge.RISING)
    aoTask.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
    aoStream = AnalogSingleChannelWriter(aoTask.out_stream, auto_start = False)
    #aoStream = AnalogUnscaledWriter(aoTask, auto_start = False)
    aoStream.write_many_sample(outData)

    aiTask = ni.Task("ai")
    aiTask.ai_channels.add_ai_voltage_chan(f"{device}/{inputChannel}", "", TerminalConfiguration.DEFAULT, minInputVoltage, maxInputVoltage)
    aiTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.RISING, sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = regeneratedOutDataLength)
    aiTask.triggers.start_trigger.cfg_dig_edge_start_trig(aoStartTrigger, Edge.RISING)
    aiTask.in_stream.input_buf_size = regeneratedOutDataLength

    aiCallbackDone = [False]
    aiCallback = writeSamplesCallback(aiTask, targetArray = inData, regenerations = outputRegenerations, doneBox = aiCallbackDone)
    sampleInterval = outDataLength
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

    print(f"Wrote: {outData}")
    print(f"Read:  {inData}")


if __name__ == "__main__":
    niSystem = NISystem.local()
    print(f"NI driver version: {niDriverVersion(niSystem)}")
    asyncio.run(main(
        device = niDevice(niSystem),
        outputChannel = "ao3",
        inputChannel = "ai4",
        samplingFrequency = 1e3,
        outDataLength = 1024,
        ))

