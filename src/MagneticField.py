import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad

@jit
def B(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.array:
    """Calculates the magnetic field at a point R from linearized coils with Biot-Savart
        Attributes:
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
def B_novo(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.array:
    """Calculates the magnetic field at a point R from linearized coils with Biot-Savart
        Attributes:
    R: jnp.array: Point where B is calculated - shape (3,)
    curve_points: jnp.array: Coil segments vectors - shape (N_coils, N_CurvePoints, 3)
    currents: jnp.array: Currents of the coils - shape (N_coils,)
        Returns:
    B: jnp.array: Magnetic field at point R - shape (3,)
    """
    directions = jnp.diff(curve_points, axis=1)
    Rprime = jsp.signal.convolve(curve_points, jnp.array([[[0.5],[0.5]]]), mode='valid')
    dB_sum = jnp.einsum("a,cba", currents*1e-7,jnp.transpose(jnp.cross(directions,R-Rprime))/jnp.transpose(jnp.linalg.norm(R-Rprime, axis=2)**3))
    return jsp.integrate.trapezoid(dB_sum, axis=0)

@jit
def B_Norm(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.float32:
    """Calculates the magnetic field norm at a point R from linearized coils with Biot-Savart
        Attributes:
    R: jnp.array: Point where B is calculated - shape (3,)
    curve_points: jnp.array: Coil segments vectors - shape (N_coils, N_CurvePoints, 3)
    currents: jnp.array: Currents of the coils - shape (N_coils,)
        Returns:
    B: jnp.float32: Magnetic field Norm at point R
    """
    return jnp.linalg.norm(B(R, curve_points, currents))

@jit
def grad_B(R: jnp.array, curve_points: jnp.array, currents:jnp.array) -> jnp.array:
    """Calculates the magnetic field gradient at a point R from linearized coils with Biot-Savart
        Attributes:
    R: jnp.array: Point where B is calculated - shape (3,)
    curve_points: jnp.array: Coil segments vectors - shape (N_coils, N_CurvePoints, 3)
    currents: jnp.array: Currents of the coils - shape (N_coils,)
        Returns:
    B: jnp.array: Magnetic field gradient at point R - shape (3,)
    """
    return grad(B_Norm, argnums=(0))(R, curve_points, currents)
    #gradB = grad(B_Norm, argnums=(0))(R, curve_points, currents)
    #return jnp.array([gradB[0], gradB[1], gradB[2]])
