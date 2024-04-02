from functools import partial
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import numpy.typing as npt

default_cernox_id = "X160190"
calibration_data_dir = Path(__file__).resolve().parent / 'calibration-data'

@dataclass
class CernoxCalibration:
    T_K: npt.NDArray[float]
    R_Ohm: npt.NDArray[float]

def calibration(cernox_id: str = default_cernox_id,
                calibration_data_dir: Path = calibration_data_dir) -> CernoxCalibration:
    calibration_file = calibration_data_dir / f"{cernox_id}.dat"
    with open(calibration_file, 'r') as f:
        a = np.array([[float(row[0]), float(row[-1])]
                      for row in list(csv.reader(f, delimiter = ' '))[3:]])
        return CernoxCalibration(T_K = a[:, 0], R_Ohm = a[:, 1])

def temperature(calibration: CernoxCalibration, R_Ohm: float) -> float:
    # must flip so that arrays are increasing
    return np.interp(-R_Ohm, -calibration.R_Ohm, calibration.T_K)

def temperature_for(cernox_id: str = default_cernox_id,
                    calibration_data_dir: Path = calibration_data_dir):
    return partial(temperature, calibration(cernox_id))

