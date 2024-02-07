# To activate the virtual environment: source NovosTalentos/bin/activate

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
import numpy as np
from time import time

from src.Dynamics import GuidingCenter
from src.MagneticField import B
from src.CreateCoil import CreateCoil
from src.Plotter import plotter

#------------------------------------------------------------------------#
# Inputs
#------------------------------------------------------------------------#

N_particles = 1
FourierCoefficients = [[-1., 0., 0, 0., 1., 0., 0., 0., 1.], [1., 0., 0., 0., 1., 0., 0., 0., 1.]]
#FourierCoefficients = [[-1., 0., 0, 0., 1., 0., 0., 0., 1.]]
N_coils = len(FourierCoefficients)
N_CurvePoints = 1000000
FC_order = 1
currents = [1e7, -1e7]
#currents = [1e7]


#------------------------------------------------------------------------#
# Setting the initial conditions
#------------------------------------------------------------------------#

def initial_conditions(N_particles: int) -> jnp.ndarray:
    # Alpha particles energy in SI units
    E = 3.52*1.602176565e-13
    # Alpha particles mass in SI units
    m = 4*1.660538921e-27
    # Alpha particles thermal velocity in SI units
    vth = jnp.sqrt(2*E/m)

    # Initializing pitch angle
    seed = 0
    key = jax.random.PRNGKey(seed)
    pitch = jax.random.uniform(key,shape=(N_particles,), minval=-1, maxval=1)

    # Initializing velocities
    vpar = vth*pitch
    vperp = vth*jnp.sqrt(1-pitch**2)

    x = jnp.array([ 0.5])
    y = jnp.array([-0.5])
    z = jnp.array([-0.5])

    return jnp.array((x, y, z, vpar, vperp))

InitialValues = initial_conditions(N_particles)
vperp = InitialValues[4]
print(f"x: {jnp.transpose(InitialValues[:3])})")
print(f"vpar: {InitialValues[3]}")
print(f"vperp: {vperp}")
print("------------------------------------------------------------------------")


#------------------------------------------------------------------------#
# Creating the coils
#------------------------------------------------------------------------#

curves = np.empty(N_coils, dtype=object)
curves_points = jnp.empty((N_coils, N_CurvePoints, 3))
coils = np.empty(N_coils, dtype=object)

for i in range(N_coils):
    # Creating a curve with "NCurvePoints" points and "FCorder" order of the Fourier series
    curves[i] = CreateCoil(FourierCoefficients[i], N_CurvePoints, FC_order)
    # Getting the curve points  
    curves_points = curves_points.at[i].set(curves[i].gamma())
    # Creating a coil
    coils[i] = Coil(curves[i], Current(currents[i]))

currents = jnp.array(currents)

#plotter(N_coils, FourierCoefficients)

#------------------------------------------------------------------------#
# Magnetic Field Calcultions
#------------------------------------------------------------------------#

# Compiling and running the function
B(jnp.transpose(InitialValues[:3]), curves_points, currents)
time1 = time(); result_MagneticField = B(jnp.transpose(InitialValues[:3]), curves_points, currents); time2 = time()

field = BiotSavart(coils)
field.B()
field.set_points(jnp.transpose(InitialValues[:3]))
time3 = time(); result_Simsopt = field.B(); time4 = time()

print("------------------------------------------------------------------------")

print(f"Magnetic Field at {jnp.transpose(InitialValues[:3])}:")

print(f"From trapezoid: {result_MagneticField} took {(time2 - time1):.1e}s")
print(f"Magnetic Field Norm: {jnp.linalg.norm(result_MagneticField[0])}")

print(f"From SIMSOPT:   {np.array(result_Simsopt)} took {(time4 - time3):.1e}s")
print(f"Magnetic Field Norm: {np.linalg.norm(np.array(result_Simsopt))}")

#------------------------------------------------------------------------#
# Guiding Center Calculations
#------------------------------------------------------------------------#

time1 = time()
Dx, Dy, Dz, Dvpar = GuidingCenter(InitialValues[:4],0.0,currents, curves_points, vperp)
time2 = time()

print("------------------------------------------------------------------------")
print(f"Guiding Center:\n Dx: {Dx}\n Dy: {Dy}\n Dz: {Dz}\n Dvpar: {Dvpar}\nTook {(time2 - time1):.1e}s")
print("------------------------------------------------------------------------")

time1 = time()
trajectories = odeint(GuidingCenter, InitialValues[:4], jnp.linspace(0, 1e-7, 100), currents, curves_points, vperp)
time2 = time()

print(f"Trajectories took {(time2 - time1):.1e}s")
plotter(N_coils, FourierCoefficients, trajectories)
