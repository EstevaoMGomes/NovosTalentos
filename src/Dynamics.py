from jax import jit, grad
import jax.numpy as jnp
from VmapIntegration import B_trapezoid

# Initializing velocities
vpar = 1e4
vperp = 0
# Initializing position
x = jnp.array([[0.1, 0.2, 0.1]])
# Initializing current
current = 1e4

@jit
def GuidingCenter(current: jnp.ndarray[float],
                  coil: jnp.ndarray[jnp.ndarray[float]],
                  x: jnp.ndarray[jnp.ndarray[float]],
                  vpar: jnp.ndarray[float],
                  vperp: jnp.ndarray[float]) -> jnp.ndarray[jnp.ndarray[float]]:
    
    # Charge and mass for alpha particles in SI units
    q = 2*1.602176565e-19
    m = 4*1.660538921e-27
    # Reading magentic field
    B = B_trapezoid(current,coil,x)
    normB = jnp.linalg.norm(B)
    b = B/normB
    # Defining gyrofrequency
    Ω = q*B/m
    # Adiabatic invariant μ
    μ = m*vperp**2/(2*normB)

    # Gradient of the magnetic field
    gradB = grad(normB)

    # Position derivative of the particle
    Dx = vpar*b + (vpar*vpar/Ω+μ/q)*jnp.cross(b, gradB)/normB
    Dvpar = -μ/m*jnp.dot(b,gradB)

    return jnp.array([Dx,Dvpar])