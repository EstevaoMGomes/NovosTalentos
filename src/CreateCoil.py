from simsopt.geo import CurveXYZFourier
import pandas as pd
from jax import jit


def CreateCoil(FourierCoefficients: list[int | float], NumberOfPoints: int, Period: float) -> CurveXYZFourier:
    # Creating a curve with "NumberOfPoints" points and "Period" periods
    curve = CurveXYZFourier(NumberOfPoints, Period)
    # Setting the Fourier coefficients
    curve.x = FourierCoefficients
   
    return curve