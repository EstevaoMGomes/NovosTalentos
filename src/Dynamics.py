from jax import jit
import jax.numpy as jnp
from src.MagneticField import B, grad_B

@jit
def GuidingCenter(InitialValues: jnp.ndarray,
                  t:             float,
                  currents:      jnp.ndarray,
                  curve_points:  jnp.ndarray,
                  μ:             float) -> jnp.ndarray:
    
    """ Calculates the motion derivatives with the Guiding Center aproximation
        Attributes:
    InitialValues: jnp.ndarray: Point in phase space where we want to calculate the derivatives - shape (4,)
    t: float: Time when the derivative is calculated
    currents: jnp.ndarray: Currents of the coils - shape (Ncoils,)
    curve_points: jnp.ndarray: Points of the coils - shape (Ncoils, Npoints, 3)
    μ: float: Magnetic moment, the 1st adiabatic constant
        Returns:
    Dx, Dvpar: jnp.ndarray: Derivatives of position and parallel velocity at time t due to the given coils
    """

    # Charge and mass for alpha particles in SI units
    q = 2*1.602176565e-19
    m = 4*1.660538921e-27

    # Calculationg the magentic field
    x, y, z, vpar = InitialValues
    r =jnp.array([x,y,z])

    B_field = B(r, curve_points, currents)
    normB = jnp.linalg.norm(B_field)
    b = B_field/normB

    # Gyrofrequency
    Ω = q*normB/m

    # Gradient of the magnetic field
    gradB = grad_B(r, curve_points, currents)

    # Position derivative of the particle
    Dx = vpar*b + (vpar*vpar/Ω+μ/q)*jnp.cross(b, gradB)/normB
    # Parallel velocity derivative of the particle
    Dvpar = -μ/m*jnp.dot(b,gradB)

    return jnp.append(Dx,Dvpar)