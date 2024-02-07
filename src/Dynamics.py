from jax import jit
import jax.numpy as jnp
from src.MagneticField import B, grad_B

# Initializing velocities
vpar = 1e4
vperp = 0
# Initializing position
x = jnp.array([[0.1, 0.2, 0.1]])
# Initializing current
current = 1e4

@jit
def GuidingCenter(InitialValues: jnp.ndarray,
                  t:             jnp.float32,
                  currents:      jnp.ndarray,
                  curve_points:  jnp.ndarray,
                  vperp:         jnp.ndarray) -> jnp.ndarray:

    # Charge and mass for alpha particles in SI units
    q = 2*1.602176565e-19
    m = 4*1.660538921e-27
    # Calculationg the magentic field
    x, y, z, vpar = InitialValues
    r =jnp.array([x[0],y[0],z[0]])

    B_field = B(r, curve_points, currents)
    normB = jnp.linalg.norm(B_field)
    b = B_field/normB
    # Gyrofrequency
    Ω = q*B_field/m
    # Adiabatic invariant μ
    μ = m*vperp[0]**2/(2*normB)

    # Gradient of the magnetic field
    gradB = grad_B(r, curve_points, currents)

    # Position derivative of the particle
    Dx = vpar*b + (vpar*vpar/Ω+μ/q)*jnp.cross(b, gradB)/normB
    Dvpar = -μ/m*jnp.dot(b,gradB)

    return jnp.append(Dx,Dvpar)