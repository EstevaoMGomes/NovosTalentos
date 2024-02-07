from simsopt.geo import CurveXYZFourier
import pandas as pd
from jax import jit


def CreateCoil(FourierCoefficients: list[int | float], NumberOfPoints: int, order: float) -> CurveXYZFourier:
    # Creating a curve with "NumberOfPoints" points and "order" number of Fourier coefficients
    curve = CurveXYZFourier(NumberOfPoints, order=order)
    # Setting the Fourier coefficients
    curve.x = FourierCoefficients
   
    return curve