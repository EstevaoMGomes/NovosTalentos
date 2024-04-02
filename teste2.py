import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.optimize import minimize
from jax.experimental.ode import odeint
from simsopt.field import Current, Coil
import numpy as np
import scipy as sp
from scipy.optimize import minimize as spminimize
from scipy.optimize import least_squares
from time import time
from functools import partial
from simsopt import load

from src.Dynamics import GuidingCenter
from src.MagneticField import B_Norm
from src.CreateCoil import CreateCoil
from src.Plotter import plot3D

@partial(jit, static_argnums=(0,))
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

        # Initializing positions
        x = jax.random.uniform(key,shape=(N_particles,), minval=-1, maxval=1)
        r = jax.random.uniform(key,shape=(N_particles,), minval=0, maxval=0.7)
        Θ = jax.random.uniform(key,shape=(N_particles,), minval=0, maxval=2*jnp.pi)
        y = r*jnp.cos(Θ)
        z = r*jnp.sin(Θ)

        return jnp.array((x, y, z, vpar, vperp))

@partial(jit, static_argnums=(1, 2, 3))
def loss(FourierCoefficients: jnp.ndarray, N_particles: int, N_coils: int, N_CurvePoints: int, currents: jnp.ndarray) -> jnp.float64:
    FourierCoefficients = jnp.reshape(FourierCoefficients, (N_coils, -1))
    FC_order = int((len(FourierCoefficients[0])/3-1)/2)

    m = 4*1.660538921e-27
    q = 2*1.602176565e-19

    InitialValues = initial_conditions(N_particles)
    vperp = InitialValues[4]

    curves_points = jnp.empty((N_coils, N_CurvePoints, 3))
    # curves_points = jnp.empty((N_coils, 3, N_CurvePoints))

    for i in range(N_coils):
        # Creating a curve with "NCurvePoints" points and "FCorder" order of the Fourier series
        curves_points = curves_points.at[i].set(CreateCoil(FourierCoefficients[i], N_CurvePoints, FC_order))

    normB = B_Norm(jnp.transpose(InitialValues[:3])[0], curves_points, currents)

    μ = m*vperp**2/(2*normB)
    timesteps = 200
    maxtime = 1e-6

    trajectories = jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[0], jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ[0], atol=1e-8, rtol=1e-8, mxstep=1000)])
    for i in range(1, N_particles):
        trajectories = jnp.concatenate((trajectories, jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[i], jnp.linspace(0, 1e-6, timesteps), currents, curves_points, μ[i], atol=1e-5, rtol=1e-5)])), axis=0)


    loss_value = jnp.sum(jnp.linalg.norm(trajectories[:][0] - trajectories[:][-1], axis = -1), axis=0)

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

N_particles = 2
N_CurvePoints = 100
currents = jnp.array([1e7, 1e7])
FourierCoefficients = jnp.array([-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                  1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

order = 3
N_coils = 2

time1 = time()
loss_value = loss(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents)
time2 = time()
print("-"*80)
print(f"Lost particles summed distance: {loss_value:.8e} m")
print(f"Took: {time2-time1:.2f} seconds")

from jax import grad
time1 = time()
grad_loss_value = grad(loss)(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents)
time2 = time()
print("-"*80)
print(f"Loss function intial value: {grad_loss_value}")
print(f"Took: {time2-time1:.2f} seconds")
raise SystemExit

"""
loss_value = loss(FourierCoefficients*(1+1e-6), N_particles, N_coils, N_CurvePoints, currents)
time2 = time()
print(f"Lost particles summed distances: {loss_value:.8e} m")
print(f"Took: {time2-time1:.2f} seconds")
print("-"*80)
"""

#grad_loss = jax.grad(loss, argnums=(0,))

#loss2 = partial(loss, N_particles=N_particles, N_coils=N_coils, N_CurvePoints=N_CurvePoints, currents=currents)
#@jit
#def grad_loss_function(params):
#    grad_func = jax.jacrev(loss2)(params)
#    return grad_func
#import numpy as np
#from scipy.optimize import least_squares
#def loss_function_jac(params):
#    return np.array(grad_loss_function(params))
#def loss_function_np(params):
#    return np.array(loss2(params))


#print("[", end="")
#for i in grad_loss(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents)[0]:
#    print(f"{i},", end=" ")
#print("]")

#print(jax.hessian(loss, argnums=0)(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents))
#minima = spminimize(loss, FourierCoefficients, args=(N_particles, N_coils, N_CurvePoints, currents), method='BFGS', options={'maxiter': 10})


start_optimize = time()
minima = minimize(loss, FourierCoefficients, args=(N_particles, N_coils, N_CurvePoints, currents), method='BFGS', options={'maxiter': 10})
end_optimize = time()
#minima = least_squares(loss_function_np, FourierCoefficients, jac = loss_function_jac, verbose=2, x_scale='jac', max_nfev=int(3))
#print(minima.x)
print("-"*80)

value = int(3*(1+2*order))
out = "[["
for i, val in enumerate(minima.x):
    j = i+1
    out += f"{val}, "
    if j % value == 0:
        out = out[:-2]
        out += "], ["
out = out[:-3]
print(out + "]")
print("-"*80)
print(f"Optimization took {end_optimize - start_optimize:.2f} s")
print(f"Loss distance after optimization: {minima.fun:.8e} m")
print(f"Optimization status: {minima.status}")
print("-"*80)



