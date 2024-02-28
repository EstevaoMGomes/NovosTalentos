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
from src.CreateCoil import CreateCoil, oldCreateCoil
from src.Plotter import plot3D, plot2D

debuging = False
#------------------------------------------------------------------------#
# Inputs
#------------------------------------------------------------------------#

N_particles = 5
#FourierCoefficients = [[-1., 0., 0, 0., 1., 0., 0., 0., 1.], [1., 0., 0., 0., 1., 0., 0., 0., 1.]]
#FourierCoefficients = [[-1.39877820e+01,  9.73827622e-02, -2.04908605e+00, -1.41725830e-01,
#  4.56845054e+00, -2.32155126e-03,  1.32708122e+00,  2.58179468e-01,
#  3.05903007e+00], [-1.51871837e+00, -3.05514491e+00,  6.77326222e-01,
# -1.34030161e+01, -1.01064459e+01,  2.96167674e+00,  2.29916623e+00,
#  2.68320389e+00, -2.02021051e+00]]
#FourierCoefficients = [[-13740614718.929352, 45010001.478988245, -2546633035.205666, 96708340.3263575, 3318424872.9081783, 56425454.87037025, 394183734.8469013, 89046620.86249365, 3500784670.058241], [-2696078547.829357, -7142264944.76991, 1369023895.2295704, -7786834069.593535, -14305665912.815304, 2545483309.155122, 1266318197.132097, 2373091734.687136, -3161998497.9020004]]
#FourierCoefficients = [[-2.105436767037876, 0.16540488329413466, -0.37822677806631755, -0.07562983995169222, 1.0386110537889908, -0.002546577154512729, -0.007698645808491838, -0.0029953834138113124, 0.9946516230694082], [0.8673039028909039, 0.03449993429305001, 0.03660480002495869, -0.027350596766373882, 0.4796513442381621, -0.031708014247428744, -0.12725927105681636, -0.018898924058447703, 0.44847597843629555]]
FourierCoefficients = [[-2.147869397377754, 0.18268509104539, -0.389534664865396, 0.09630831753511991, -0.06310516552734853, -0.07635853572069998, 1.057732159083316, -0.019875193041674514, -0.05769982161346997, 0.013816961192643097, -0.003469770645142031, -0.014721939786353663, 1.0404787735329102, -0.003288229042236649, -0.06972945482360222], [0.8854020210856544, 0.007529709255217744, 0.012680877828184785, -0.03948909766063804, 0.022531239425347172, -0.033125423689350376, 0.44673917656716056, -0.059030657960613436, -0.11965553667924496, -0.07499512969218892, -0.1431894887997557, -0.04413417880890023, 0.4234527802784654, 0.03198968374874773, -0.14881502704764882 ]]
N_coils = len(FourierCoefficients)
N_CurvePoints = 1000
FC_order = 2
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
    #pitch = jnp.ones(N_particles)*-0.1
    # Initializing velocities
    vpar = vth*pitch
    vperp = vth*jnp.sqrt(1-pitch**2)

    #x = jnp.array([-0.7])
    #y = jnp.array([ 0.3])
    #z = jnp.array([ 0.3])

    x = jax.random.uniform(key,shape=(N_particles,), minval=-1, maxval=1)
    r = jax.random.uniform(key,shape=(N_particles,), minval=0, maxval=1)
    Θ = jax.random.uniform(key,shape=(N_particles,), minval=0, maxval=2*jnp.pi)
    y = r*jnp.cos(Θ)
    z = r*jnp.sin(Θ)

    return jnp.array((x, y, z, vpar, vperp))

InitialValues = initial_conditions(N_particles)
vperp = InitialValues[4]
if debuging:
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
    curves[i] = oldCreateCoil(FourierCoefficients[i], N_CurvePoints, FC_order)
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
if debuging:
    print(f"μ: {μ}")
    print("------------------------------------------------------------------------")

if debuging:
    for i in range(N_particles):
        time1 = time()
        Dx, Dy, Dz, Dvpar = GuidingCenter(jnp.transpose(InitialValues[:4])[i],0.0,currents, curves_points, μ[i])
        time2 = time()
        print(f"Guiding Center for particle {i+1}:\n Dx: {Dx}\n Dy: {Dy}\n Dz: {Dz}\n Dvpar: {Dvpar}\nTook {(time2 - time1):.1e}s")
        print("------------------------------------------------------------------------")

timesteps = 200
maxtime = 5e-7

time1 = time()
trajectories = jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[0], jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ[0], atol=1e-8, rtol=1e-8, mxstep=1000)])
for i in range(1, N_particles):
    trajectories = jnp.concatenate((trajectories, jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[i], jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ[i], atol=1e-5, rtol=1e-5)])), axis=0)
time2 = time()

if debuging:
    print(trajectories)
print(f"Trajectories took {(time2 - time1):.1e}s")
print("------------------------------------------------------------------------")

lost_particles = 0
for i in range(N_particles):
    if trajectories[i][-1][0] < -1 or trajectories[i][-1][0] > 1:
        lost_particles += 1

print(f"Lost particles fraction: {lost_particles/N_particles*100:.2f}%")
print("------------------------------------------------------------------------")

#------------------------------------------------------------------------#
# Plotting analysis graphs
#------------------------------------------------------------------------#
plot = False

if plot:
    plot3D(N_coils, FourierCoefficients, trajectories)
    if N_particles > 1:
        raise ValueError("The plotter is not ready for more than one particle")
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

