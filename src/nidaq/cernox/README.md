# cernox

## Quick usage

```python
import cernox
T = cernox.temperature_for("X160190")
R = T(200.0) # R in Ohms, T in K
```

## Explanation

A Cernox is defined by its `CernoxCalibration`,
which may be obtained from the serial number
```python
calibration = cernox.calibration("X160190")
```
This loads the calibration data file `X160190.dat` from `cernox.calibration_data_dir`.
To add a new Cernox, put the calibration data into the `calibration-data` directory of this package (default), or wherever you would like to store calibration data, and pass the path in the `calibration_data_dir` argument.

Once you have a `CernoxCalibration`, the temperature is determined from the resistance by linearly interpolating the calibration data:
```python
cernox.temperature(calibration, R_Ohm = 200.0) # T in K
```
The function `temperature_for` is provided for convenience to automate these steps.
