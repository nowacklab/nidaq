import nidaqmx as ni
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogUnscaledWriter
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogUnscaledReader
from nidaqmx.constants import AcquisitionType, Edge, RegenerationMode, TerminalConfiguration
import numpy as np

def main(samplingFrequency, outDataLength,
         minOutputVoltage = -10.0,
         maxOutputVoltage = 10.0,
         minInputVoltage = -10.0,
         maxInputVoltage = 10.0):

    #devices = ni.system.System.local().devices()

    outputRegenerations = 2
    regeneratedOutDataLength = outputRegenerations * outDataLength

    outData = np.ones((outDataLength,), dtype = np.float64)
    outData[0] = 2.0
    outData[-2] = 0.0
    outData[-1] = 3.0

    inData = np.zeros((regeneratedOutDataLength,), dtype = np.float64)


    coChannel = "ctr0"
    coInternalOutput = "Ctr0InternalOutput"

    coTask = ni.Task()
    coTask.co_channels.add_co_pulse_chan_freq(f"Dev1/{coChannel}", "", freq = samplingFrequency, duty_cycle = 0.5)
    coTask.timing.cfg_implicit_timing(sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = 0)

    aoTask = ni.Task()
    aoTask.ao_channels.add_ao_voltage_chan("Dev1/ao3", "", minOutputVoltage, maxOutputVoltage)
    aoTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.FALLING, sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = regeneratedOutDataLength)
    aoTask.triggers.start_trigger.cfg_dig_edge_start_trig(coInternalOutput, Edge.RISING)
    aoTask.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
    # TODO: get terminal name with dev prefix ao/StartTrigger
    aoStartTrigger = "/Dev1/ao/StartTrigger"
    aoStream = AnalogSingleChannelWriter(aoTask.out_stream, auto_start = False)
    #aoStream = AnalogUnscaledWriter(aoTask, auto_start = False)
    aoStream.write_many_sample(outData)

    aiTask = ni.Task()
    aiTask.ai_channels.add_ai_voltage_chan("Dev1/ai4", "", TerminalConfiguration.DEFAULT, minInputVoltage, maxInputVoltage)
    aiTask.timing.cfg_samp_clk_timing(samplingFrequency, source = coInternalOutput, active_edge = Edge.RISING, sample_mode = AcquisitionType.CONTINUOUS, samps_per_chan = regeneratedOutDataLength)
    aiTask.triggers.start_trigger.cfg_dig_edge_start_trig(aoStartTrigger, Edge.RISING)
    aiStream = AnalogSingleChannelReader(aiTask.in_stream)


    coTask.start()
    aiTask.start()
    aoTask.start()


    aiStream.read_many_sample(inData, regeneratedOutDataLength)


    aiTask.stop()
    aiTask.close()

    aoTask.stop()
    aoTask.close()

    coTask.stop()
    coTask.close()


    print(f"Wrote: {outData}")
    print(f"Read:  {inData}")



if __name__ == "__main__":
    main(
            samplingFrequency = 1e3,
            outDataLength = 10,
            )
