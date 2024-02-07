import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad

@jit
def B(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.array:
    """ Attributes:
    R: jnp.array: Point where B is calculated - shape (3,)
    curve_points: jnp.array: Coil segments vectors - shape (N_coils, N_CurvePoints, 3)
    currents: jnp.array: Currents of the coils - shape (N_coils,)
        Returns:
    B: jnp.array: Magnetic field at point R - shape (3,)
    """
    directions = jnp.diff(curve_points, axis=1)
    Rprime = jsp.signal.convolve(curve_points, jnp.array([[[0.5],[0.5]]]), mode='valid')
    dB_sum = jnp.einsum("a,abc", currents*1e-7,jnp.divide(jnp.cross(directions,R-Rprime),jnp.reshape(jnp.repeat(jnp.linalg.norm(R-Rprime, axis=2)**3, 3), (len(curve_points),len(Rprime[0]),3))))
    return jsp.integrate.trapezoid(dB_sum, axis=0)

@jit
def B_Norm(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.array:
    directions = jnp.diff(curve_points, axis=1)
    Rprime = jsp.signal.convolve(curve_points, jnp.array([[[0.5],[0.5]]]), mode='valid')
    dB_sum = jnp.einsum("a,abc", currents*1e-7,jnp.divide(jnp.cross(directions,R-Rprime),jnp.reshape(jnp.repeat(jnp.linalg.norm(R-Rprime, axis=2)**3, 3), (len(curve_points),len(Rprime[0]),3))))
    return jnp.linalg.norm(jsp.integrate.trapezoid(dB_sum, axis=0))

@jit
def grad_B(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.array:
    gradB = grad(B_Norm, argnums=(0))(R, curve_points, currents)
    return jnp.array([gradB[0], gradB[1], gradB[2]])