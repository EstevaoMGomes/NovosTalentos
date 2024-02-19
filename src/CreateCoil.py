from simsopt.geo import CurveXYZFourier
import pandas as pd
from jax import jit
import jax.numpy as jnp


def oldCreateCoil(FourierCoefficients: list[int | float], NumberOfPoints: int, order: float) -> CurveXYZFourier:
    # Creating a curve with "NumberOfPoints" points and "order" number of Fourier coefficients
    curve = CurveXYZFourier(NumberOfPoints, order=order)
    # Setting the Fourier coefficients
    curve.x = FourierCoefficients
   
    return curve

#@jit
def CreateCoil(dofs: jnp.ndarray, numquadpoints: int, order: int) -> CurveXYZFourier:
    """ Creates a coil with a given set of degrees of freedom
        Attributes:
    dofs: jnp.ndarray (3*(2*order+1),): Fourier Coefficients of the coil
    intquadpoints: int: Number of points of the coil
    order: int: Order of the Fourier series
        Returns:
    data: jnp.ndarray(): Coil points
    """
    # Creating a curve with "NumberOfPoints" points and "order" number of Fourier coefficients
    dofs = jnp.reshape(dofs, (-1, 3))
    quadpoints = jnp.linspace(0, 1, numquadpoints + 1)[:-1]
    data = jnp.zeros((numquadpoints, 3))

    for i in range(3):
        data = data.at[:, i].add(dofs[i][0])
        for j in range(1, order + 1):
            data = data.at[:, i].add(dofs[i][2 * j - 1] * jnp.sin(2 * jnp.pi * j * quadpoints) + dofs[i][2 * j] * jnp.cos(2 * jnp.pi * j * quadpoints))
    
    return data
