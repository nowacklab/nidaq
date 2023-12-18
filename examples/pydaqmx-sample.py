"""
Example of hardware-bugged reading from a X-series NI DAQ device (USB-6363).
Upon running, I see the input sample is read before the first output sample.
This uses PyDAQmx as it is easily comparable to the ANSI C API and documentation.
"""
from ctypes import byref, create_string_buffer
from PyDAQmx import *
import numpy as np
from time import perf_counter_ns

# Declaration of variable passed by reference
minVoltage = -10.0
maxVoltage = +10.0
trials = 10
data_len = 10
wrote = int32()
read = int32()
aiData = np.zeros((data_len,), dtype=np.float64)
aoData = np.ones((data_len,), dtype=np.float64)
#data = np.zeros((data_len,), dtype=np.uint16)
samplingFrequency = 1e3 # Hz
timeout = 10.0 # s

aoTaskHandle = TaskHandle()
aiTaskHandle = TaskHandle()

ttotals = []
treads = []


def getTerminalNameWithDevPrefix(taskHandle: TaskHandle, terminalName: bytes) -> bytes:
    device_len = 256
    device = bytes(device_len)
    numDevices = uInt32()
    productCategory = int32()
    DAQmxGetTaskNumDevices(taskHandle, byref(numDevices))
    i = uInt32(1)
    while i.value <= numDevices.value:
        DAQmxGetNthTaskDevice(taskHandle, i, device, device_len)
        DAQmxGetDevProductCategory(device, byref(productCategory))
        if (productCategory != DAQmx_Val_CSeriesModule and productCategory != DAQmx_Val_SCXIModule):
            return b"/" + device.split(b'\0')[0] + b"/" + terminalName
        i += 1

    return b""

try:
    ti = perf_counter_ns()

    DAQmxCreateTask(b"", byref(aiTaskHandle))
    DAQmxCreateAIVoltageChan(aiTaskHandle, b"Dev1/ai4", b"", DAQmx_Val_Cfg_Default, minVoltage, maxVoltage, DAQmx_Val_Volts, None)
    DAQmxCfgSampClkTiming(aiTaskHandle, b"", samplingFrequency, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, data_len)
    trigName = getTerminalNameWithDevPrefix(aiTaskHandle, b"ai/StartTrigger")

    DAQmxCreateTask(b"", byref(aoTaskHandle))
    DAQmxCreateAOVoltageChan(aoTaskHandle, b"Dev1/ao3", b"", minVoltage, maxVoltage, DAQmx_Val_Volts, None)
    DAQmxCfgSampClkTiming(aoTaskHandle, b"", samplingFrequency, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, data_len)
    DAQmxCfgDigEdgeStartTrig(aoTaskHandle, trigName, DAQmx_Val_Rising)

    aoData[0] = 2.0
    aoData[-2] = 0.0
    aoData[-1] = 0.0
    DAQmxWriteAnalogF64(aoTaskHandle, data_len, False, timeout, DAQmx_Val_GroupByChannel, aoData, byref(wrote), None)

    DAQmxStartTask(aoTaskHandle)
    DAQmxStartTask(aiTaskHandle)

    DAQmxReadAnalogF64(aiTaskHandle, data_len, timeout, DAQmx_Val_GroupByChannel, aiData, data_len, byref(read), None)
    print(f"ao ({wrote.value}):\t{aoData}")
    print(f"ai ({read.value}):\t{aiData}")

except DAQError as err:
    print(f"DAQmx Error: {err}")
finally:
    if aiTaskHandle:
        DAQmxStopTask(aiTaskHandle)
        DAQmxClearTask(aiTaskHandle)
    if aoTaskHandle:
        DAQmxStopTask(aoTaskHandle)
        DAQmxClearTask(aoTaskHandle)

    tf = perf_counter_ns()

