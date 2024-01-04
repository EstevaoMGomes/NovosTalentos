import jax
import jax.scipy as jsp
import jax.numpy as jnp
from jax import jit
import time

@jit
def BiotSavart(A,B,R,step,nps):
    norma = jnp.linalg.norm(B-A)
    direction = (B-A)/norma
    Rprime = jnp.array(A)
    integral = jnp.array([0,0,0])
    #jsp.integrate.trapezoid(jnp.cross(direction,R-Rprime)/(jnp.linalg.norm(R-Rprime))**3,x=Rprime,dx=step*direction)
    for n in jnp.linspace(0,nps):
        integral = integral + jnp.cross(direction,R-Rprime)/(jnp.linalg.norm(R-Rprime))**3*step
        Rprime = Rprime+direction*step
    return integral*1e-7
    
def __main__():
    A = jnp.array([0,0,-1]) # Begining of the wire
    B = jnp.array([0,0,1]) # End of the wire
    R = jnp.array([0.001,0,0]) # Position where we calculate the magnetic field
    step = 1e-3 # Step for the integration
    nps = jnp.round(jnp.linalg.norm(B-A)/step)
    starttime = time.time()
    print("Magnetic Field: ", BiotSavart(A,B,R,step,nps))
    endtime = time.time()
    print("Time elapsed: ", endtime-starttime)
    A = jnp.array([0,0,-1]) # Begining of the wire
    B = jnp.array([0,0,1]) # End of the wire
    R = jnp.array([0.001,0,0]) # Position where we calculate the magnetic field
    step = 1e-3 # Step for the integration
    nps = jnp.round(jnp.linalg.norm(B-A)/step)
    starttime = time.time()
    print("Magnetic Field: ", BiotSavart(A,B,R,step,nps))
    endtime = time.time()
    print("Time elapsed: ", endtime-starttime)
    return 0

if __name__ == "__main__":
    __main__()