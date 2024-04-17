import jax.numpy as jnp
from jax import random

def initial_conditions(N_particles: int,
                       type:        str,
                       s1:          float,
                       s2:          float) -> jnp.ndarray:
    """ Creates initial conditions for the particles.
        Attributes:
    N_particles: int: Number of particles
    type: str: Type of initial conditions. Choose between 'cylinder' and 'torus'
    s1: float: Half cylinder lenght or major toroidal radius
    s2: float: Cylinder radius or minor toroidal radius
        Returns:
    jnp.array: Array with the initial conditions - shape (5, N_particles)
    """
    # Alpha particles energy in SI units
    E = 3.52*1.602176565e-13
    # Alpha particles mass in SI units
    m = 4*1.660538921e-27
    # Alpha particles thermal velocity in SI units
    vth = jnp.sqrt(2*E/m)

    # Initializing pitch angle
    seed = 0
    key = random.PRNGKey(seed)
    pitch = random.uniform(key,shape=(N_particles,), minval=-1, maxval=1)

    # Initializing velocities
    vpar = vth*pitch
    vperp = vth*jnp.sqrt(1-pitch**2)

    # Initializing positions
    if type.lower() == "cylinder":
        x = random.uniform(key,shape=(N_particles,), minval=-s1, maxval=s1)
        r = random.uniform(key,shape=(N_particles,), minval=0, maxval=s2)
        Θ = random.uniform(key,shape=(N_particles,), minval=0, maxval=2*jnp.pi)
        y = r*jnp.cos(Θ)
        z = r*jnp.sin(Θ)
    elif type.lower() == "torus":
        ϕ = random.uniform(key,shape=(N_particles,), minval=0, maxval=2*jnp.pi)
        Θ = random.uniform(key,shape=(N_particles,), minval=0, maxval=2*jnp.pi)
        x = (s2*jnp.cos(Θ)+s1)*jnp.cos(ϕ)
        y = (s2*jnp.cos(Θ)+s1)*jnp.sin(ϕ)
        z = s2*jnp.sin(Θ)
    else:
        raise ValueError("Invalid type of initial conditions. Choose between 'cylinder' and 'torus'.")

    return jnp.array((x, y, z, vpar, vperp))

def CreateEquallySpacedCurves(ncurves: int,
                              order: int,
                              R: float,
                              r: float) -> jnp.ndarray:
    """ Create a toroidal set of coils equally spaced eith an outer radius R and inner radius r.
        Attributes:
    N_coils: int: Number of coils
    N_CurvePoints: int: Number of points to create the curve of the coils
    R: float: Outer radius of the coils
    r: float: Inner radius of the coils
        Returns:
    curves: jnp.ndarray: Array of curves - shape (N_coils, 3*(2*order+1))
    """
    curves = jnp.zeros((ncurves, 3, 1+2*order))
    for i in range(ncurves):
        #angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ncurves)
        angle = (i+0.5)*(2*jnp.pi)/ncurves
        curves = curves.at[i, 0, 0].set(jnp.cos(angle)*R)
        curves = curves.at[i, 0, 2].set(jnp.cos(angle)*r)
        curves = curves.at[i, 1, 0].set(jnp.sin(angle)*R)
        curves = curves.at[i, 1, 2].set(jnp.sin(angle)*r)
        curves = curves.at[i, 2, 1].set(-r)
        # In the previous line, the minus sign is for consistency with
        # Vmec.external_current(), so the coils create a toroidal field of the
        # proper sign and free-boundary equilibrium works following stage-2 optimization.
    return curves

"""
R = 1.0
r = 0.5
ncurves = 10
order = 3

curves = CreateEquallySpacedCurves(ncurves, order, R, r)

dofs = jnp.empty((ncurves, 3*(2*order+1)))
for i in range(ncurves):
    dofs = dofs.at[i,:].set(jnp.ravel(curves[i]))

################################################################################################################################

from CreateCoil import CreateCoil
from jax.lax import fori_loop

curves_points = jnp.empty((ncurves, 200, 3))
def fori_createcoil(coil: jnp.int32, curves_points: jnp.ndarray) -> jnp.ndarray:
    return curves_points.at[coil].set(CreateCoil(curves[coil], 200, order))
curves_points = fori_loop(0, ncurves, fori_createcoil, curves_points)

################################################################################################################################


import plotly.graph_objects as go
import numpy as np
from CreateCoil import CreateCoil

trace_curve = np.empty(ncurves, dtype=object)
for i in range(ncurves):
    coil = CreateCoil(dofs[i], 200, order)
    coil = curves_points[i]
    trace_curve[i] = go.Scatter3d(x=coil[:,0],y=coil[:,1],z=coil[:,2], mode='lines', name=f'Coil {i+1}', line=dict(color='rgb(179,179,179)', width=4))

# Create layout for the plot
layout = go.Layout(scene=dict(aspectmode='cube'))

# Create figure and add traces to it
fig = go.Figure(data=list(trace_curve), layout=layout)

# Show the plot
fig.show()
"""