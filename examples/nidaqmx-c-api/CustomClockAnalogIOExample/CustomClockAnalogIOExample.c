/*
 * CustomClockAnalogIOExample.c
 * Alex Striff
 * 2023-12-18
 * License: MIT
 */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <NIDAQmx.h>

#define DATA_LEN 10

#define DAQmxErrChk(functionCall) if( DAQmxFailed(error=(functionCall)) ) goto Error; else

static uInt64 time_now(void)
{
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) {
        fputs("timespec_get failed!\n", stderr);
        return 0;
    }
    return 1000000000 * ts.tv_sec + ts.tv_nsec;
}

static const float64 MIN_VOLTAGE = -10.0;
static const float64 MAX_VOLTAGE = +10.0;
static const float64 SAMPLING_FREQUENCY = 1e3;
static const float64 TIMEOUT = -1; // 10.0

#define ERR_BUF_LEN 2048
#define TRIG_NAME_LEN 256

static int32 GetTerminalNameWithDevPrefix(TaskHandle taskHandle, const char* terminalName, char* triggerName)
{
    int32	error = 0;
    char	device[256];
    int32	productCategory;
    uInt32	numDevices, i = 1;

    DAQmxErrChk(DAQmxGetTaskNumDevices(taskHandle, &numDevices));
    while (i <= numDevices) {
        DAQmxErrChk(DAQmxGetNthTaskDevice(taskHandle, i++, device, TRIG_NAME_LEN));
        DAQmxErrChk(DAQmxGetDevProductCategory(device, &productCategory));
        if (productCategory != DAQmx_Val_CSeriesModule && productCategory != DAQmx_Val_SCXIModule) {
            *triggerName++ = '/';
            strcat(strcat(strcpy(triggerName, device), "/"), terminalName);
            break;
        }
    }

Error:
    return error;
}

int main(void)
{
    int32 error = 0;
    TaskHandle coTaskHandle = 0;
    TaskHandle aiTaskHandle = 0;
    TaskHandle aoTaskHandle = 0;
    int32 read;
    int32 wrote;
    float64 aiData[2 * DATA_LEN] = { 0.0 }; // or uInt16 for raw sample data
    float64 aoData[DATA_LEN] = { 1.0 };
    char errBuf[ERR_BUF_LEN] = { 0 };
    char trigName[TRIG_NAME_LEN] = { 0 };
    size_t i = 0;

    // Continuously generate a pulse train for clocking output and input
    DAQmxErrChk(DAQmxCreateTask("", &coTaskHandle));
    DAQmxErrChk(DAQmxCreateCOPulseChanFreq(coTaskHandle, "Dev1/ctr0", "", DAQmx_Val_Hz, DAQmx_Val_Low, 0.0, SAMPLING_FREQUENCY, 0.5));
    // The value of sampsPerChan is probably ignored when generating a continuous pulse train.
    DAQmxErrChk(DAQmxCfgImplicitTiming(coTaskHandle, DAQmx_Val_ContSamps, 0));

    // Initialize output data buffer
    for (i = 0; i < DATA_LEN; i++)
        aoData[i] = 1.0;
    aoData[0] = 2.0;
    aoData[DATA_LEN - 2] = 0.0;
    aoData[DATA_LEN - 1] = 3.0;

    /*
    for (i = 0; i < DATA_LEN; i++)
        aoData[i] = 0x8800;
    aoData[0] = 0xc000;
    aoData[DATA_LEN - 2] = 0x1000;
    aoData[DATA_LEN - 1] = 0xf000;
    */

    // Create analog output task
    DAQmxErrChk(DAQmxCreateTask("", &aoTaskHandle));
    DAQmxErrChk(DAQmxCreateAOVoltageChan(aoTaskHandle, "Dev1/ao3", "", MIN_VOLTAGE, MAX_VOLTAGE, DAQmx_Val_Volts, NULL));
    DAQmxErrChk(DAQmxCfgSampClkTiming(aoTaskHandle, "Ctr0InternalOutput", SAMPLING_FREQUENCY, DAQmx_Val_Falling, DAQmx_Val_ContSamps, DATA_LEN));
    DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(aoTaskHandle, "Ctr0InternalOutput", DAQmx_Val_Rising));
    DAQmxErrChk(DAQmxSetWriteRegenMode(aoTaskHandle, DAQmx_Val_AllowRegen));
    DAQmxErrChk(GetTerminalNameWithDevPrefix(aoTaskHandle, "ao/StartTrigger", trigName));
    DAQmxErrChk(DAQmxWriteAnalogF64(aoTaskHandle, DATA_LEN, 0, TIMEOUT, DAQmx_Val_GroupByChannel, aoData, &wrote, NULL));
    //DAQmxErrChk(DAQmxWriteBinaryU16(aoTaskHandle, DATA_LEN, 0, TIMEOUT, DAQmx_Val_GroupByChannel, aoData, &wrote, NULL));

    // Create analog input task
    DAQmxErrChk(DAQmxCreateTask("", &aiTaskHandle));
    DAQmxErrChk(DAQmxCreateAIVoltageChan(aiTaskHandle, "Dev1/ai4", "", DAQmx_Val_Cfg_Default, MIN_VOLTAGE, MAX_VOLTAGE, DAQmx_Val_Volts, NULL));
    DAQmxErrChk(DAQmxCfgSampClkTiming(aiTaskHandle, "Ctr0InternalOutput", SAMPLING_FREQUENCY, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 2 * DATA_LEN));
    DAQmxErrChk(DAQmxCfgDigEdgeStartTrig(aiTaskHandle, trigName, DAQmx_Val_Rising));

    // Start the pulse train
    DAQmxErrChk(DAQmxStartTask(coTaskHandle));
    // Start IO tasks
    DAQmxErrChk(DAQmxStartTask(aiTaskHandle));
    DAQmxErrChk(DAQmxStartTask(aoTaskHandle));

    // Display results
    DAQmxErrChk(DAQmxReadAnalogF64(aiTaskHandle, 2 * DATA_LEN, TIMEOUT, DAQmx_Val_GroupByChannel, aiData, 2 * DATA_LEN, &read, NULL));
    //DAQmxErrChk(DAQmxReadBinaryU16(aiTaskHandle, DATA_LEN, TIMEOUT, DAQmx_Val_GroupByChannel, aiData, DATA_LEN, &read, NULL));

    printf("Wrote %d samples and read %d samples.\n", wrote, read);
    for (i = 0; i < 2 * DATA_LEN; i++) {
        printf("%f ", aoData[i % DATA_LEN]);
        //printf("%u ", aoData[i]);
    }
    putchar('\n');
    for (i = 0; i < 2 * DATA_LEN; i++) {
        printf("%f ", aiData[i]);
        //printf("%u ", aiData[i]);
    }
    putchar('\n');

Error:
    if (DAQmxFailed(error))
        DAQmxGetExtendedErrorInfo(errBuf, ERR_BUF_LEN);

    if (coTaskHandle != 0) {
        DAQmxStopTask(coTaskHandle);
        DAQmxClearTask(coTaskHandle);
    }
    if (aoTaskHandle != 0) {
        DAQmxStopTask(aoTaskHandle);
        DAQmxClearTask(aoTaskHandle);
    }
    if (aiTaskHandle != 0) {
        DAQmxStopTask(aiTaskHandle);
        DAQmxClearTask(aiTaskHandle);
    }

    if (DAQmxFailed(error))
        printf("DAQmx Error: %s\n", errBuf);

    return 0;
}
