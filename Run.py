# To activate the virtual environment: source NovosTalentos/bin/activate

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
import numpy as np
from time import time

from src.Dynamics import GuidingCenter
from src.MagneticField import B, B_Norm
from src.CreateCoil import CreateCoil
from src.Plotter import plotter

#------------------------------------------------------------------------#
# Inputs
#------------------------------------------------------------------------#

N_particles = 1
FourierCoefficients = [[-1., 0., 0, 0., 1., 0., 0., 0., 1.], [1., 0., 0., 0., 1., 0., 0., 0., 1.]]
#FourierCoefficients = [[-1., 0., 0, 0., 1., 0., 0., 0., 1.]]
N_coils = len(FourierCoefficients)
N_CurvePoints = 10000
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
    pitch = jnp.ones(N_particles)*-0.56740909
    # Initializing velocities
    vpar = vth*pitch
    vperp = vth*jnp.sqrt(1-pitch**2)

    x = jnp.array([-0.7])
    y = jnp.array([ 0.3])
    z = jnp.array([ 0.3])
    #x = jnp.array([-0.7,-0.4, -0.5,  0.5,  0.5])
    #y = jnp.array([ 0.3,-0.3,  0.5, -0.5,  0.5])
    #z = jnp.array([ 0.3,-0.3,  0.5,  0.5, -0.5])

    return jnp.array((x, y, z, vpar, vperp))

InitialValues = initial_conditions(N_particles)
vperp = InitialValues[4]
print("------------------------------------------------------------------------")
print(f"x transposed: {jnp.transpose(InitialValues[:3])})")
print(f"vpar: {InitialValues[3]}")
print(f"vperp: {vperp}")


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
B(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time1 = time()
result_MagneticField = B(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time2 = time()
normB = jnp.linalg.norm(result_MagneticField)

field = BiotSavart(coils)
field.B()
field.set_points(jnp.transpose(InitialValues[:3]))
time3 = time()
result_Simsopt = field.B()
time4 = time()

print("------------------------------------------------------------------------")

print(f"Magnetic Field at {jnp.transpose(InitialValues[:3])[0]}:")

print(f"From trapezoid: {result_MagneticField} took {(time2 - time1):.1e}s")
print(f"Magnetic Field Norm: {jnp.linalg.norm(result_MagneticField)}")

print(f"From SIMSOPT:   {np.array(result_Simsopt)[0]} took {(time4 - time3):.1e}s")
print(f"Magnetic Field Norm: {np.linalg.norm(np.array(result_Simsopt)[0])}")

print("------------------------------------------------------------------------")

#------------------------------------------------------------------------#
# Guiding Center Calculations
#------------------------------------------------------------------------#


m = 4*1.660538921e-27
# Adiabatic invariant μ
μ = m*vperp**2/(2*normB)
print(f"μ: {μ}")
print("------------------------------------------------------------------------")

for i in range(N_particles):
    time1 = time()
    Dx, Dy, Dz, Dvpar = GuidingCenter(jnp.transpose(InitialValues[:4])[i],0.0,currents, curves_points, μ[i])
    time2 = time()
    print(f"Guiding Center for particle {i+1}:\n Dx: {Dx}\n Dy: {Dy}\n Dz: {Dz}\n Dvpar: {Dvpar}\nTook {(time2 - time1):.1e}s")
    print("------------------------------------------------------------------------")

timesteps = 10
time1 = time()
trajectories = jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[0], jnp.linspace(0, 1e-7, timesteps), currents, curves_points, μ[0], atol=1e-5, rtol=1e-5)])
for i in range(1, N_particles):
    trajectories = jnp.concatenate((trajectories, jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[i], jnp.linspace(0, 1e-7, timesteps), currents, curves_points, μ[i], atol=1e-5, rtol=1e-5)])), axis=0)
time2 = time()
#print(trajectories)

print(f"Trajectories took {(time2 - time1):.1e}s")
plotter(N_coils, FourierCoefficients, trajectories)

print("------------------------------------------------------------------------")
#Plotting the magnetic field norm for each point in the particle's trajectory
import matplotlib.pyplot as plt

y = np.empty((N_particles, timesteps))
x = jnp.linspace(0, 1e-7, timesteps)
for i in range(N_particles):
    for j in range(len(x)):
        y[i][j] = B_Norm(trajectories[i][j][:3], curves_points, currents)
        plt.scatter(x[j], y[i][j], label=f"Particle {i+1}")

plt.xlabel("Time (s)")
plt.ylabel("|B| (T)")
plt.title("|B| for each point of the particle's trajectory")
plt.savefig("images/|B|_scan.png")
plt.show()
        
