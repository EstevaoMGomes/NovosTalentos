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

#print("Available devices: ", jax.devices())
#jax.default_device(jax.devices('gpu')[0])
#jax.default_device(jax.devices('cuda')[0])
print("Current device: ", jax.devices())
#nvidia-smi

@partial(jit, static_argnums=(0,))
def initial_conditions(N_particles: jnp.int32) -> jnp.ndarray:
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
    x = jax.random.uniform(key,shape=(N_particles,), minval=-2, maxval=2)
    r = jax.random.uniform(key,shape=(N_particles,), minval=0, maxval=0.5)
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

    @jit
    def fori_createcoil(coil: jnp.int32, curves_points: jnp.ndarray) -> jnp.ndarray:
        return curves_points.at[coil].set(CreateCoil(FourierCoefficients[coil], N_CurvePoints, FC_order))
    curves_points = fori_loop(0, N_coils, fori_createcoil, curves_points)
    #COnfirmar se esrá bem

    normB = B_Norm(jnp.transpose(InitialValues[:3])[0], curves_points, currents)

    μ = m*vperp**2/(2*normB)
    timesteps = 200
    maxtime = 1e-6
    trajectories = jnp.empty((N_particles, timesteps, 4))
    times = jnp.linspace(0, maxtime, timesteps)

    @jit
    def trace_trajectory(particle: jnp.int32, trajectories: jnp.ndarray) -> jnp.ndarray:
        return trajectories.at[particle,:,:].set(odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[particle], times, currents, curves_points, μ[particle], atol=1e-8, rtol=1e-8, mxstep=100))
    trajectories = fori_loop(0, N_particles, trace_trajectory, trajectories)

    
    
    #left_boundary = -1
    #right_boundary = 1
    #radial_boundary = 0.7
    #is_lost = select(jnp.greater(trajectories[:, :, 0], right_boundary*jnp.ones((N_particles,timesteps)))|
    #                 jnp.less(trajectories[:, :, 0], left_boundary*jnp.ones((N_particles,timesteps))) |
    #                 jnp.greater(jnp.square(trajectories[:, :, 1])+jnp.square(trajectories[:, :, 2]), radial_boundary*jnp.ones((N_particles,timesteps))),
    #                 jnp.ones((N_particles,timesteps)), jnp.zeros((N_particles,timesteps)))
    
    #@jit
    #def loss_calc(x: jnp.ndarray) -> jnp.ndarray:
    #    return 3.5*jnp.exp(-2*jnp.nonzero(x, size=1, fill_value=timesteps)[0]/timesteps)
    #loss_value = jnp.sum(jnp.apply_along_axis(loss_calc, 1, is_lost))


    #loss_value = jnp.log(jnp.sum(jnp.linalg.norm(trajectories[:][0][:3] - trajectories[:][-1][:3], axis = -1), axis=0)/N_particles)
    loss_value = jnp.sqrt(jnp.mean(jnp.abs(trajectories[:][:][:3])))

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
currents = jnp.array([1e7, 1e7])

FourierCoefficients = jnp.array([-1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
#FourierCoefficients = jnp.array([-1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#                                  1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
#FourierCoefficients = jnp.ravel(jnp.array([[-0.90923374672835, -0.06878862239609261, -0.029756112279776925, -0.02631042513968028, 0.027830595263965283, -0.017297886473431952, 0.038929441246501134, -0.1638744802530541, 0.6020708466085564, -0.011503758095711483, -0.06610722426718924, 0.04966364451586068, -0.05893494849573262, -0.014996453778712145, -0.04885912969530354, -0.009510095817629434, 0.6506493861921251, -0.07762681881435414, 0.016207426202981528, -0.01541658713316821, 0.05617270515524379], [0.8746883002172272, 0.00919390981009342, -0.010681154782432757, -0.0019821267370478224, 0.00011889160187998762, -0.0020990015935972014, 0.001982106015646127, 0.013726948728579358, 1.125182668589405, 0.002868998546910861, 0.02079667646759333, -0.0024710250154603712, 0.025211227137632023, 0.004306320881117695, -0.00782880054131793, 0.0015650786337092727, 1.1017660778948344, 0.00830019497644951, -0.00928488444917999, 0.003781748043543047, -0.01124483287745919]]))



N_coils = 2

time1 = time()
loss_value = loss(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents)
time2 = time()
print("-"*80)
print(f"Loss function initial value: {loss_value:.8f}")
print(f"Took: {time2-time1:.2f} seconds")

time1 = time()
grad_loss_value = grad(loss)(FourierCoefficients, N_particles, N_coils, N_CurvePoints, currents)
time2 = time()
print("-"*80)
print(f"Grad loss function initial value: {grad_loss_value}")
print(f"Took: {time2-time1:.2f} seconds")

"""
indices = []
losses = []
for i in jnp.arange(-1.5, -0.5, 0.02):
    indices += [i]
    losses += [loss(FourierCoefficients.at[0].set(i), N_particles, N_coils, N_CurvePoints, currents)]
plot2D("1x1", indices, losses, "Loss function", "1st Fourier Coefficient", "Loss", "LossFunction")
raise SystemExit
"""

start_optimize = time()
minima = minimize(loss, FourierCoefficients, args=(N_particles, N_coils, N_CurvePoints, currents), method='BFGS', options={'maxiter': 10})
end_optimize = time()


print("-"*80)
value = int(len(FourierCoefficients)/N_coils)
out = "[["
for i, val in enumerate(minima.x):
    j = i+1
    out += f"{val}, "
    if j % value == 0:
        out = out[:-2]
        out += "], ["
out = out[:-3]
print(out + "]]")
print("-"*80)
print(f"Optimization took {end_optimize - start_optimize:.2f} s")
print(f"Loss function final value: {minima.fun:.8f}")
print(f"Optimization success: {minima.success}")
print(f"Optimization status: {minima.status}")
print("-"*80)




