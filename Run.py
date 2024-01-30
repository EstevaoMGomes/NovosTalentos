import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, grad
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
import numpy as np
import pandas as pd
from time import time

#from src.Dynamics import GuidingCenter
from src.VmapIntegration import B_trapezoid
from src.CreateCoil import CreateCoil

#------------------------------------------------------------------------#
# Inputs

N_coils = 1 # = len(FourierCoefficients)
N_particles = 1
FourierCoefficients = [[0, 0, 1., 0., 1., 0., 0., 0., 0.]]
N_CurvePoints = 10000
N_CurvePeriods = 1
currents = [1e4]

#------------------------------------------------------------------------#
x = [0.5, 0.5, 0.1]

curves = np.empty(N_coils, dtype=object)
curves_points = jnp.empty((N_coils, N_CurvePoints, 3))
coils = np.empty(N_coils, dtype=object)

for i in range(N_coils):
    # Creating a curve with "NCurvePoints" points and "NCurvePeriods" periods
    curves[i] = CreateCoil(FourierCoefficients[i], N_CurvePoints, N_CurvePeriods)
    # Getting the curve points  
    curves_points = curves_points.at[i].set(curves[i].gamma())
    # Creating a coil
    coils[i] = Coil(curves[i], Current(currents[i]))

    B_trapezoid(currents[i], curves_points[i], jnp.array(x))


# Results
print(f"Magnetic Field at {x}:")

time1 = time();result_trapezoid = B_trapezoid(currents[0], curves_points[0], jnp.array(x)), jnp.array(x);time2 = time()
print(f"From trapezoid: {result_trapezoid} took {(time2 - time1):.1e}s")
print(f"Magnetic Field: {jnp.linalg.norm(result_trapezoid[0])}")


field = BiotSavart(coils)
field.B()
field.set_points(np.array([x]))
time1 = time(); result = field.B(); time2 = time()



print(f"From SIMSOPT:   {np.array(field.B()[0])} took {(time2 - time1):.1e}s")
print(f"Magnetic Field: {np.linalg.norm(np.array(field.B()[0]))}")
