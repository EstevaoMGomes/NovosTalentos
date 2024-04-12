import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd
from jax.lax import fori_loop, select, while_loop
from jax.scipy.optimize import minimize
from jax.experimental.ode import odeint
from simsopt.field import Current, Coil
import numpy as np
import scipy as sp
from scipy.optimize import minimize as spminimize
from scipy.optimize import least_squares
from time import time
from functools import partial

from src.Dynamics import GuidingCenter
from src.MagneticField import B_Norm
from src.CreateCoil import CreateCoil
from src.Plotter import plot3D, plot2D
from src.InitialConditions import initial_conditions, CreateEquallySpacedCurves

print("Current device: ", jax.devices())

@partial(jit, static_argnums=(1, 2, 3))
def loss(FourierCoefficients: jnp.ndarray,
         N_particles: int,
         N_coils: int,
         N_CurvePoints: int,
         currents: jnp.ndarray) -> float:
    
    FourierCoefficients = jnp.reshape(FourierCoefficients, (N_coils, 3, -1))

    m = 4*1.660538921e-27
    q = 2*1.602176565e-19

    R = 5.5
    InitialValues = initial_conditions(N_particles, "torus", R, 0.2)
    #InitialValues = initial_conditions(N_particles, "cylinder", 1.0, 0.5)
    vperp = InitialValues[4]

    curves_points = jnp.empty((N_coils, N_CurvePoints, 3))

    def fori_createcoil(coil: jnp.int32, curves_points: jnp.ndarray) -> jnp.ndarray:
        return curves_points.at[coil].set(CreateCoil(FourierCoefficients[coil], N_CurvePoints))
    curves_points = fori_loop(0, N_coils, fori_createcoil, curves_points)

    normB = jnp.apply_along_axis(B_Norm, 0, InitialValues[:3], curves_points, currents)

    μ = m*vperp**2/(2*normB)
    timesteps = 200
    maxtime = 1e-6
    trajectories = jnp.empty((N_particles, timesteps, 4))
    times = jnp.linspace(0, maxtime, timesteps)

    def trace_trajectory(particle: jnp.int32, trajectories: jnp.ndarray) -> jnp.ndarray:
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

    loss_value = jnp.mean(jnp.sqrt(jnp.square(jnp.sqrt(jnp.square(trajectories[:][:][0]) + jnp.square(trajectories[:][:][1]))-R)+jnp.square(trajectories[:][:][2]**2)))

    return loss_value

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

N_particles = 1000
N_CurvePoints = 1000

ncoils = 50
order = 6
R = 5.5
r = 0.5
FourierCoefficients = jnp.ravel(CreateEquallySpacedCurves(ncoils, order, R, r))
currents = jnp.ones(ncoils)*1e7

time1 = time()
loss_value = loss(FourierCoefficients, N_particles, ncoils, N_CurvePoints, currents)
time2 = time()
print("-"*80)
print(f"Loss function initial value: {loss_value:.8f}")
print(f"Took: {time2-time1:.2f} seconds")

time1 = time()
grad_loss_value = grad(loss)(FourierCoefficients, N_particles, ncoils, N_CurvePoints, currents)
time2 = time()
print("-"*80)
print(f"Grad loss function initial value: {grad_loss_value}")
print(f"Took: {time2-time1:.2f} seconds")


start_optimize = time()
minima = minimize(loss, FourierCoefficients, args=(N_particles, ncoils, N_CurvePoints, currents), method='BFGS', options={'maxiter': 20})
end_optimize = time()


print("-"*80)
value = int(len(FourierCoefficients)/ncoils)
out = "[["
for i, val in enumerate(minima.x):
    j = i+1
    out += f"{val}, "
    if j % value == 0:
        out = out[:-2]
        out += "], ["
out = out[:-3]
print(out + "]")
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
with open("results.txt", "a") as file:
    file.write("# Optimization at " + current_time + "\n")
    file.write(out+ "]\n")
print(f"Optimization took {end_optimize - start_optimize:.2f} s")
print(f"Loss function final value: {minima.fun:.8f}")
print(f"Optimization success: {minima.success}")
print(f"Optimization status: {minima.status}")




