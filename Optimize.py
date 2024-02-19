import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.optimize import minimize
from jax.experimental.ode import odeint
from simsopt.field import Current, Coil
import numpy as np
from time import time

from src.Dynamics import GuidingCenter
from src.MagneticField import B_Norm
from src.CreateCoil import CreateCoil

#@jit # This is not working because of the random number generator
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

#@jit
def loss(FourierCoefficients: list, N_particles: int, N_coils: int, N_CurvePoints: int, currents: list) -> jnp.float64:
    FourierCoefficients = jnp.reshape(FourierCoefficients, (N_coils, -1))
    FC_order = int((len(FourierCoefficients[0])/3-1)/2)

    m = 4*1.660538921e-27
    q = 2*1.602176565e-19

    InitialValues = initial_conditions(N_particles)
    vperp = InitialValues[4]

    curves = np.empty(N_coils, dtype=object)
    curves_points = jnp.empty((N_coils, N_CurvePoints, 3))

    for i in range(N_coils):
        # Creating a curve with "NCurvePoints" points and "FCorder" order of the Fourier series
        curves_points = curves_points.at[i].set(CreateCoil(FourierCoefficients[i], N_CurvePoints, FC_order))

    currents = jnp.array(currents)

    normB = B_Norm(jnp.transpose(InitialValues[:3])[0], curves_points, currents)

    μ = m*vperp**2/(2*normB)
    timesteps = 200
    maxtime = 1e-5

    trajectories = jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[0], jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ[0], atol=1e-8, rtol=1e-8, mxstep=1000)])
    for i in range(1, N_particles):
        trajectories = jnp.concatenate((trajectories, jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[i], jnp.linspace(0, 1e-6, timesteps), currents, curves_points, μ[i], atol=1e-5, rtol=1e-5)])), axis=0)

    lost_particles = 0
    for i in range(N_particles):
        if trajectories[i][-1][0] < -1 or trajectories[i][-1][0] > 1:
            lost_particles += 1
    
    return lost_particles/N_particles

N_particles = 100
N_CurvePoints = 1000
currents = [1e7, 1e7]
FourierCoefficients = jnp.array([-1., 0., 0,   0., 1., 0.,    0., 0., 1.,   1., 0., 0., 0., 1., 0., 0., 0., 1.])
N_coils = 2
time1 = time()
loss_fraction = loss(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents)
time2 = time()
print(f"Lost particles fraction: {loss_fraction*100:.2f}%")
print(f"Took: {time2-time1:.2f} seconds")

print(jax.grad(loss, argnums=(0))(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents))
#print(jax.hessian(loss, argnums=0)(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents))
minimize(loss, FourierCoefficients, args=(N_particles, N_coils, N_CurvePoints, currents), method='BFGS', options={'maxiter': 1e3})