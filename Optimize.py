import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad
from jax.lax import fori_loop, select
from jax.scipy.optimize import minimize
from jax.experimental.ode import odeint

from time import time
from functools import partial

from src.Dynamics import GuidingCenter
from src.MagneticField import B_Norm
from src.CreateCoil import CreateCoil
from src.InitialConditions import initial_conditions, CreateEquallySpacedCurves

print("Current device: ", jax.devices())

@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def loss(FourierCoefficients: jnp.ndarray,
         N_particles:         int,
         N_coils:             int,
         N_CurvePoints:       int,
         maxtime:             float,
         timesteps:           int,
         R:                   float,
         loss_r:              float,        
         currents:            jnp.ndarray) -> float:
    """ Loss function to be minimized
        Attributes:
    FourierCoefficients: jnp.ndarray: Fourier Coefficients of the coils - shape (N_coils*3*(2*order+1)) - must be a 1D array
    N_particles: int: Number of particles
    N_coils: int: Number of coils
    N_CurvePoints: int: Number of points of the coil
    maxtime: float: Maximum time of the simulation
    timesteps: int: Number of timesteps
    R: float: Major radius of the loss torus
    loss_r: float: Minor radius of the loss torus
    currents: jnp.ndarray: Currents of the coils - shape (N_coils,)
        Returns:
    loss_value: float: Loss value - must be scalar
    """
    
    FourierCoefficients = jnp.reshape(FourierCoefficients, (N_coils, 3, -1))

    m = 4*1.660538921e-27

    InitialValues = initial_conditions(N_particles, "torus", R, loss_r)
    #InitialValues = initial_conditions(N_particles, "cylinder", 1.0, 0.5)
    vperp = InitialValues[4]

    curves_points = jnp.empty((N_coils, N_CurvePoints, 3))

    def fori_createcoil(coil: int, curves_points: jnp.ndarray) -> jnp.ndarray:
        return curves_points.at[coil].set(CreateCoil(FourierCoefficients[coil], N_CurvePoints))
    curves_points = fori_loop(0, N_coils, fori_createcoil, curves_points)

    normB = jnp.apply_along_axis(B_Norm, 0, InitialValues[:3], curves_points, currents)

    μ = m*vperp**2/(2*normB)
    trajectories = jnp.empty((N_particles, timesteps, 4))
    times = jnp.linspace(0, maxtime, timesteps)

    def trace_trajectory(particle: int, trajectories: jnp.ndarray) -> jnp.ndarray:
        return trajectories.at[particle,:,:].set(odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[particle], times, currents, curves_points, μ[particle], atol=1e-8, rtol=1e-8, mxstep=100))
    trajectories = fori_loop(0, N_particles, trace_trajectory, trajectories)
    
    """
    left_boundary = -1
    right_boundary = 1
    radial_boundary = 0.7
    is_lost = select(jnp.greater(trajectories[:, :, 0], right_boundary*jnp.ones((N_particles,timesteps)))|
                     jnp.less(trajectories[:, :, 0], left_boundary*jnp.ones((N_particles,timesteps))) |
                     jnp.greater(jnp.square(trajectories[:, :, 1])+jnp.square(trajectories[:, :, 2]), radial_boundary*jnp.ones((N_particles,timesteps))),
                     jnp.ones((N_particles,timesteps)), jnp.zeros((N_particles,timesteps)))

    @jit
    def loss_calc(x: jnp.ndarray) -> jnp.ndarray:
        return 3.5*jnp.exp(-2*jnp.nonzero(x, size=1, fill_value=timesteps)[0]/timesteps)
    loss_value = jnp.sum(jnp.apply_along_axis(loss_calc, 1, is_lost))
    """
    
    distances_squared = jnp.square(
        jnp.sqrt(
            trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2
        )-R
    )+(trajectories[:, :, 2])**2

    #is_lost = select(jnp.greater(distances_squared, R*jnp.ones((N_particles,timesteps))),
    #                 distances_squared, jnp.zeros((N_particles,timesteps)))
    #def loss_calc(is_lost: jnp.ndarray) -> jnp.ndarray:
    #    𝜏 = jnp.nonzero(is_lost, size=1, fill_value=timesteps)[0]
    #    return 3.5*jnp.exp(-2*𝜏/timesteps)*jnp.exp(-jnp.min(jnp.array(0,jnp.mean(is_lost)))/loss_r**2)
    #return jnp.sum(jnp.apply_along_axis(loss_calc, 1, is_lost))

    return jnp.mean(distances_squared)/loss_r**2
    #return jnp.mean(3.5*jnp.exp(-2*loss_r/jnp.sqrt(distances_squared)))

"""
print(load('TandemCoil/Optimize.json'))
currents = []
Fourier = []
BiotSavart = load('TandemCoil/Optimize.json')
for coil in BiotSavart.coils:
    dofs = coil.full_x
    currents += [dofs[0]]
    Fourier += [jnp.array(dofs[1:])]
print(currents)
print(Fourier)
for i in range(len(Fourier)):
    print(len(Fourier[i]))
FourierCoefficients = Fourier
"""

N_particles = 50
N_CurvePoints = 100
maxtime = 1e-6
timesteps = 200

ncoils = 10
order = 3
R = 5.
r = 0.5
loss_r = 0.2
FourierCoefficients = jnp.ravel(CreateEquallySpacedCurves(ncoils, order, R, r))
currents = jnp.ones(ncoils)*1e7

time1 = time()
loss_value = loss(FourierCoefficients, N_particles, ncoils, N_CurvePoints, maxtime, timesteps, R, loss_r, currents)
time2 = time()
print("-"*80)
print(f"Loss function initial value: {loss_value:.8f}")
print(f"Took: {time2-time1:.2f} seconds")

time1 = time()
loss_value = loss(FourierCoefficients, N_particles, ncoils, N_CurvePoints, maxtime, timesteps, R, loss_r, currents)
time2 = time()
print(f" Compiled took: {time2-time1:.2f} seconds")

time1 = time()
grad_loss_value = grad(loss, argnums=0)(FourierCoefficients, N_particles, ncoils, N_CurvePoints, maxtime, timesteps, R, loss_r, currents)
time2 = time()
print("-"*80)
print(f"Grad loss function initial value: {grad_loss_value}")
print(f"Took: {time2-time1:.2f} seconds")

time1 = time()
grad_loss_value = grad(loss, argnums=0)(FourierCoefficients, N_particles, ncoils, N_CurvePoints, maxtime, timesteps, R, loss_r, currents)
time2 = time()
print(f"Compiled took: {time2-time1:.2f} seconds")

start_optimize = time()
minima = minimize(loss, FourierCoefficients, args=(N_particles, ncoils, N_CurvePoints, maxtime, timesteps, R, loss_r, currents), method='BFGS', options={'maxiter': 20})
end_optimize = time()

print(f"Optimization took {end_optimize - start_optimize:.2f} s")
print(f"Loss function final value: {minima.fun:.8f}")
print(f"Optimization success: {minima.success}")
print(f"Optimization status: {minima.status}")

from datetime import datetime
now = datetime.now()
today = datetime.today()
current_time = now.strftime("%H:%M:%S")
current_day = today.strftime("%d/%m/%Y")
with open("results.txt", "a") as file:
    file.write("# optimization at " + current_time + " of " + current_day + "\n")
    file.write("# args: -N_particles=" + str(N_particles) + " -ncoils=" + str(ncoils) +
               " -N_CurvePoints=" + str(N_CurvePoints) + " -maxtime=" + str(maxtime) + 
               " -timesteps=" + str(timesteps) + " -R=" + str(R) + " -r=" + str(r) +
               " -loss_r=" + str(loss_r) + "\n")
    file.write(jnp.array_str(currents).replace("\n", "").replace(" ", ",") + "\n")
    file.write(jnp.array_str(minima.x).replace("\n", "").replace(" ", "", 1).replace("  ", ",").replace(" ", ",") + "\n")

#print(jnp.array_str(minima.x).replace("\n", "").replace(" ", "", 1).replace("  ", ",").replace(" ", ","))

print(jnp.array_str(minima.x))
print(minima.x)
print(loss(minima.x, N_particles, ncoils, N_CurvePoints, maxtime, timesteps, R, loss_r, currents))

import h5py as h5
file = h5.File("results.h5", "w")
#file.create_dataset("FourierCoefficients", data=minima.x)
file["FourierCoefficients"] = minima.x
file.close()