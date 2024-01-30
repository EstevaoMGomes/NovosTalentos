import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from time import time
import numpy as np
import pandas as pd

# Conversion from cartesian to cylindrical coordinates
def cart2cyl(xyz: jnp.ndarray)-> jnp.ndarray:
    r = jnp.sqrt(xyz[0]**2 + xyz[1]**2)
    ϕ = jnp.arctan(xyz[1], xyz[0])
    z = xyz[2]
    return jnp.array([r, ϕ, z])

# Conversion from cylindrical to cartesian coordinates
def cyl2cart(rphiz: jnp.ndarray)-> jnp.ndarray:
    x = rphiz[0]*jnp.cos(rphiz[1])
    y = rphiz[0]*jnp.sin(rphiz[1])
    z = rphiz[2]
    return jnp.array([x, y, z])

# Y axis rotation matrix
def Ry(costheta: float, sign: int):
    return jnp.array([[costheta, 0, jnp.sqrt(1-costheta**2)*sign],
                      [0, 1, 0],
                      [-jnp.sqrt(1-costheta**2)*sign, 0, costheta]])

# Z axis rotation matrix
def Rz(costheta: float, sign: int):
    return jnp.array([[costheta, -jnp.sqrt(1-costheta**2)*sign, 0],
                      [jnp.sqrt(1-costheta**2)*sign, costheta, 0],
                      [0, 0, 1]])

# Function taking 1 point and changing coordinates
def ChangeCoords(A: jnp.ndarray, B: jnp.ndarray, xyz: jnp.ndarray)-> jnp.ndarray:
    midpoint = (A + B) / 2.0
    L = jnp.linalg.norm(B - A)
    sign = 1 if (B-midpoint).at[1]<0 else -1 
    cosθ = (B-midpoint).at[0]/L
    cosφ = (B-midpoint).at[2]/L
    xyz_rotated = Ry(cosφ, sign) @ Rz(cosθ, sign) @ xyz
    rϕz =  cart2cyl(xyz_rotated)
    return rϕz

# 
@jit
def WireMagneticField(current, wire_points, xyz): #xyz is the obsevation_poiny
    rϕz = ChangeCoords(wire_points[0], wire_points[1], xyz)
    L = jnp.linalg.norm(wire_points[1] - wire_points[0])
    B = - current * 1e-7 * ((rϕz.at[2]-L)/jnp.sqrt(rϕz.at[0]**2+(rϕz.at[2]-L)**2)-(rϕz.at[2]+L)/jnp.sqrt(rϕz.at[0]**2+(rϕz.at[2]+L)**2)) # ephi component
    return B

# JAX field
@jit
def biot_savart_law(current, wire_segment, curve_point, observation_point):
    r = observation_point - curve_point
    dl_cross_r = jnp.cross(wire_segment, r)
    r_magnitude = jnp.linalg.norm(r)
    denominator = r_magnitude ** 3
    magnetic_field = 1e-7 * current * dl_cross_r / denominator
    return magnetic_field

@jit
def B_trapezoid(current, curve_points, observation_point):
    dl = jnp.diff(curve_points, axis=0)
    mid_points = (curve_points[:-1] + curve_points[1:]) / 2.0
    magnetic_field_vectorized = vmap(lambda segment, curve_point: biot_savart_law(current, segment, curve_point, observation_point))(dl, mid_points)
    integral_result = jsp.integrate.trapezoid(magnetic_field_vectorized, axis=0)
    return integral_result

#x = [0.5, 0.5, 0.1]
## Initial compilation of B_trapezoid
#B_trapezoid(1e4, jnp.array(pd.read_csv("src/Coil.csv", skiprows=2)), jnp.array(x))
#
## Results
#print(f"Magnetic Field at {x}:")
#
#time1 = time();result_trapezoid = B_trapezoid(1e4, jnp.array(pd.read_csv("src/Coil.csv", skiprows=2)), jnp.array(x));time2 = time()
#print(f"From trapezoid: {result_trapezoid} took {(time2 - time1):.1e}s")
