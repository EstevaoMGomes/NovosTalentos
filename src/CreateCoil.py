#from simsopt.geo import CurveXYZFourier
from jax import jit
from functools import partial
import jax.numpy as jnp
from jax.lax import fori_loop


#def oldCreateCoil(FourierCoefficients: list[int | float], NumberOfPoints: int, order: float) -> CurveXYZFourier:
#    # Creating a curve with "NumberOfPoints" points and "order" number of Fourier coefficients
#    curve = CurveXYZFourier(NumberOfPoints, order=order)
#    # Setting the Fourier coefficients
#    curve.x = FourierCoefficients
#    return curve

@partial(jit, static_argnums=(1, 2))
def CreateCoil(dofs: jnp.ndarray, numquadpoints: int, order: int) -> jnp.ndarray:
    """ Creates an array with nquadpoints points of the coil with a given set of Fourier Coefficients
        Attributes:
    dofs: jnp.ndarray: Fourier Coefficients of the coil - shape (3*(2*order+1),)
    intquadpoints: int: Number of points of the coil
    order: int: Order of the Fourier series
        Returns:
    data: jnp.ndarray: Coil points - shape (numquadpoints, 3)
    """
    # Creating a curve with "NumberOfPoints" points and "order" number of Fourier coefficients
    dofs = jnp.reshape(dofs, (3, -1))
    #order = int((len(dofs[0])/3-1)/2)
    quadpoints = jnp.linspace(0, 1, numquadpoints + 1)[:-1]

    data = jnp.outer(dofs[:, 0], jnp.ones(numquadpoints))
    
    @jit
    def fori_createdata(order_index: jnp.int32, data: jnp.ndarray) -> jnp.ndarray:
        return data + jnp.outer(dofs[:, 2 * order_index - 1], jnp.sin(2 * jnp.pi * order_index * quadpoints)) + jnp.outer(dofs[:, 2 * order_index], jnp.cos(2 * jnp.pi * order_index * quadpoints))
    data = fori_loop(1, order+1, fori_createdata, data) 
    
    
    #for j in range(1, order + 1):
    #    data = data + jnp.outer(dofs[:, 2 * j - 1], jnp.sin(2 * jnp.pi * j * quadpoints)) + jnp.outer(dofs[:, 2 * j], jnp.cos(2 * jnp.pi * j * quadpoints))
    return jnp.transpose(data)