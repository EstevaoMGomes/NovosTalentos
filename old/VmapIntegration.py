import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap, grad
from time import time
import numpy as np
import pandas as pd

# Conversion from cartesian to cylindrical coordinates
def cart2cyl(xyz: jnp.ndarray)-> jnp.ndarray:
    r = jnp.sqrt(jnp.square(xyz[:,0]) + jnp.square(xyz[:,1]))
    ϕ = jnp.arctan2(xyz[:,1], xyz[:,0])
    z = xyz[:,2]
    return jnp.transpose(jnp.array([r, ϕ, z]))

# Conversion from cylindrical to cartesian coordinates
def cyl2cart(rphiz: jnp.ndarray)-> jnp.ndarray:
    x = rphiz[0]*jnp.cos(rphiz[1])
    y = rphiz[0]*jnp.sin(rphiz[1])
    z = rphiz[2]
    return jnp.array([x, y, z])

# Function taking 1 point and changing coordinates

@jit
def ChangeCoords(wire_points: jnp.ndarray, xyz: jnp.ndarray)-> jnp.ndarray:
    A = wire_points[:-1]
    B = wire_points[1:]
    midpoints = (A + B) / 2.0
    # xyz[0] is only for 1 particle
    xyz_translated = jnp.array([xyz[0] - midpoints[i,:] for i in range(len(midpoints[:,0]))])
    L = jnp.linalg.norm(B[0] - A[0])
    #sign = 1 if (B-midpoint).at[1]<0 else -1
    axis_direction = jnp.subtract(B,midpoints)
    cosθ = axis_direction[:,0]/L
    sinθ = jnp.sqrt(1-jnp.square(cosθ))
    #Rz = [jnp.array([[          cosθ[i],              -jnp.sqrt(1-jnp.square(cosθ))[i],  0],
    #                 [jnp.sqrt(1-jnp.square(cosθ))[i],               cosθ[i],            0],
    #                 [             0,                                   0,               1]]) for i in range(len(cosθ))]
    cosφ =  axis_direction[:,2]/L
    sinφ = jnp.sqrt(1-jnp.square(cosφ))

    #Ry = [jnp.array([[             cosφ[i],             0, jnp.sqrt(1-jnp.square(cosφ))[i]],
    #                 [               0,                 1,              0                 ],
    #                 [-jnp.sqrt(1-jnp.square(cosφ))[i], 0,            cosφ[i]             ]]) for i in range(len(cosφ))]

    #xyz_rotated = jnp.array([jnp.matmul(Ry[i],jnp.matmul(Rz[i], xyz_translated[i,:])) for i in range(len(xyz_translated[:,0]))])
    xyz_rotated = jnp.array([jnp.array([xyz_translated[i,0]*cosθ[i]*cosφ[i] - xyz_translated[i,1]*sinθ[i]*cosφ[i] + xyz_translated[i,2]*sinφ[i],
                                        xyz_translated[i,0]*sinθ[i] + xyz_translated[i,1]*cosθ[i],
                                        -xyz_translated[i,0]*cosθ[i]*sinφ[i] + xyz_translated[i,1]*sinθ[i]*sinφ[i] + xyz_translated[i,2]*cosφ[i]]) for i in range(len(cosθ))])
    #xyz_rotated = xyz_translated
    #return xyz_rotated
    r = jnp.sqrt(jnp.square(xyz_rotated[:,0]) + jnp.square(xyz_rotated[:,1]))
    ϕ = jnp.arctan2(xyz_rotated[:,1], xyz_rotated[:,0])
    z = xyz_rotated[:,2]
    return jnp.transpose(jnp.array([r, ϕ, z]))
    #rϕz =  cart2cyl(xyz_rotated)
    #return rϕz


@jit
def WireMagneticField(current, wire_points, xyz): #xyz is the obsevation_point
    rϕz = ChangeCoords(wire_points, xyz)
    #return rϕz
    L = jnp.linalg.norm((wire_points[1,:] - wire_points[0,:]))
    B = - current * 1e-7 * jnp.sum(jnp.divide(rϕz[:,2]-L,jnp.sqrt(jnp.square(rϕz[:,0])+jnp.square(rϕz[:,2]-L)))-jnp.divide(rϕz[:,2]+L,jnp.sqrt(jnp.square(rϕz[:,0])+jnp.square(rϕz[:,2]+L)))) # ephi component
    return B

# JAX field
@jit
def biot_savart_law(current, dl, curve_point, mid_points):
    r = mid_points - curve_point
    dl_cross_r = jnp.cross(dl, r)
    r_magnitude = jnp.linalg.norm(r)
    denominator = r_magnitude ** 3
    magnetic_field = 1e-7 * current * dl_cross_r / denominator
    return magnetic_field

@jit
def B_trapezoid(current, curve_points, observation_point):
    dl = jnp.diff(curve_points, axis=0)
    mid_points = (curve_points[:-1] + curve_points[1:]) / 2.0
    #mid_points = jsp.signal.convolve(curve_points, jnp.array([[0.5],[0.5]]), mode='valid')
    magnetic_field_vectorized = vmap(lambda segment, curve_point: biot_savart_law(current, segment, curve_point, observation_point))(dl, mid_points)
    #return magnetic_field_vectorized
    integral_result = jsp.integrate.trapezoid(magnetic_field_vectorized, axis=0)
    return integral_result

@jit
def Bnorm_trapezoid(current, curve_points, x,y,z):
    observation_point = jnp.array([x,y,z])

    dl = jnp.diff(curve_points, axis=0)
    mid_points = (curve_points[:-1] + curve_points[1:]) / 2.0
    magnetic_field_vectorized = vmap(lambda segment, curve_point: biot_savart_law(current, segment, curve_point, observation_point))(dl, mid_points)
    integral_result = jsp.integrate.trapezoid(magnetic_field_vectorized, axis=0)
    return jnp.linalg.norm(integral_result)


@jit
def B(R, curve_points, currents):
    directions = jnp.diff(curve_points, axis=1)
    Rprime = jsp.signal.convolve(curve_points, jnp.array([[[0.5],[0.5]]]), mode='valid')
    dB_sum = jnp.einsum("a,abc", currents*1e-7,jnp.divide(jnp.cross(directions,R-Rprime),jnp.reshape(jnp.repeat(jnp.linalg.norm(R-Rprime, axis=2)**3, 3), (len(curve_points),len(Rprime[0]),3))))
    return jsp.integrate.trapezoid(dB_sum, axis=0)

@jit
def B_Norm(R, curve_points, currents):
    directions = jnp.diff(curve_points, axis=1)
    Rprime = jsp.signal.convolve(curve_points, jnp.array([[[0.5],[0.5]]]), mode='valid')
    dB_sum = jnp.einsum("a,abc", currents*1e-7,jnp.divide(jnp.cross(directions,R-Rprime),jnp.reshape(jnp.repeat(jnp.linalg.norm(R-Rprime, axis=2)**3, 3), (len(curve_points),len(Rprime[0]),3))))
    return jnp.linalg.norm(jsp.integrate.trapezoid(dB_sum, axis=0))

@jit
def grad_B(R, curve_points, currents):
    gradB = grad(B_Norm, argnums=(0))(R, curve_points, currents)
    return gradB


class MagneticField:
    def __init__(self, currents, curve_points):
        self.current = currents
        self.curve_points = curve_points

    def B(self, x: [float, int], y: [float, int], z: [float, int]):
        return B_trapezoid(self.current, self.curve_points, jnp.array([x,y,z]))
    
    def gradB(self, x: [float, int], y: [float, int], z: [float, int]):
        gradB = grad(Bnorm_trapezoid, argnums=(2,3,4))(self.current, self.curve_points, x,y,z)
        return jnp.array([gradB[0], gradB[1], gradB[2]])
    
