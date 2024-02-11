# To activate the virtual environment: source NovosTalentos/bin/activate

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from simsopt.field import BiotSavart, Current, Coil
from simsopt import load
import numpy as np
from time import time

from src.Dynamics import GuidingCenter
from src.MagneticField import B, B_Norm, B_novo, grad_B
from src.CreateCoil import CreateCoil
from src.Plotter import plot3D, plot2D

#------------------------------------------------------------------------#
# Inputs
#------------------------------------------------------------------------#

N_particles = 1
FourierCoefficients = [[-1., 0., 0, 0., 1., 0., 0., 0., 1.], [1., 0., 0., 0., 1., 0., 0., 0., 1.]]
N_coils = len(FourierCoefficients)
N_CurvePoints = 1000
FC_order = 1
currents = [1e7, 1e7]


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
    pitch = jnp.ones(N_particles)*-0.1
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

#plot3D(N_coils, FourierCoefficients)

#------------------------------------------------------------------------#
# Magnetic Field Calcultions
#------------------------------------------------------------------------#

# Compiling and running the function
B(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time1 = time()
result_MagneticField = B(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time2 = time()
normB = jnp.linalg.norm(result_MagneticField)

Stellarator = load("biot_savart_opt.json")
#Stellarator.gamma()

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

# Compiling and running the function
B_novo(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time1 = time()
result_MagneticField = B_novo(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time2 = time()
normB = jnp.linalg.norm(result_MagneticField)
print(f"From trapezoid: {result_MagneticField} took {(time2 - time1):.1e}s")
print(f"Magnetic Field Norm: {jnp.linalg.norm(result_MagneticField)}")

print("------------------------------------------------------------------------")

#------------------------------------------------------------------------#
# Guiding Center Calculations
#------------------------------------------------------------------------#

m = 4*1.660538921e-27
q = 2*1.602176565e-19

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

timesteps = 200
maxtime = 1e-6
#maxtime = 4e-7 #for 1e8 currrent
time1 = time()
trajectories = jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[0], jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ[0], atol=1e-8, rtol=1e-8, mxstep=1000)])
for i in range(1, N_particles):
    trajectories = jnp.concatenate((trajectories, jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[i], jnp.linspace(0, 1e-6, timesteps), currents, curves_points, μ[i], atol=1e-5, rtol=1e-5)])), axis=0)
time2 = time()
print(trajectories)

print(f"Trajectories took {(time2 - time1):.1e}s")
plot3D(N_coils, FourierCoefficients, trajectories)

print("------------------------------------------------------------------------")

#------------------------------------------------------------------------#
# Plotting analysis graphs
#------------------------------------------------------------------------#


NormBScan     = np.empty((N_particles, timesteps))
NormGradBScan = np.empty((N_particles, timesteps))
x             = np.empty((N_particles, timesteps))
y             = np.empty((N_particles, timesteps))
z             = np.empty((N_particles, timesteps))
vpar          = np.empty((N_particles, timesteps))
Dx            = np.empty((N_particles, timesteps))
Dy            = np.empty((N_particles, timesteps))
Dz            = np.empty((N_particles, timesteps))
Dvpar         = np.empty((N_particles, timesteps))
t             = np.linspace(0, maxtime, timesteps)


for i in range(N_particles):
    for j in range(len(t)):
        NormBScan[i][j] = B_Norm(trajectories[i][j][:3], curves_points, currents)
        NormGradBScan[i][j] = jnp.linalg.norm(grad_B(trajectories[i][j][:3], curves_points, currents))
        x[i][j], y[i][j], z[i][j], vpar[i][j] = trajectories[i][j][:]
        Dx[i][j], Dy[i][j], Dz[i][j], Dvpar[i][j] = GuidingCenter(trajectories[i][j][:],t[j],currents, curves_points, μ[0])

plot2D("1x2", (t, t), (NormBScan, NormGradBScan), ("|B|", "|∇B|"), ("", "Time (s)"), ("|B| (T)", "|∇B|"), "B&GradB")
plot2D("1x1", t, μ[0]*NormBScan+0.5*m*vpar**2, "Energy", "Time (s)", "Energy (J)", "Energy")
plot2D("1x1", t, np.sqrt(2*m*μ[0]/NormBScan)*NormGradBScan/NormBScan/q, "⍴∇B/B", "Time (s)", "Guiding Center Validity? (⍴∇B/B <<1)", "GC_Validity")
plot2D("1x3", (t, t, t), (x, y, z), ("x", "y", "z"), ("", "", "Time (s)"), ("Distance (m)", "Distance (m)", "Distance (m)"), "Position")
plot2D("1x3", (t, t, t), (Dx, Dy, Dz), ("x, y and z derivatives", "", ""), ("", "", "Time (s)"), ("dx (m/s)", "dy (m/s)", "dz (m/s)"), "Position_Derivatives")
plot2D("1x2", (t, t), (vpar, Dvpar), ("parallel velocity and its derivative", ""), ("", "Time (s)"), ("Velocity (m/s)", "Acceleration (m/s^2)"), "Parallel_Velocity&Derivative")

