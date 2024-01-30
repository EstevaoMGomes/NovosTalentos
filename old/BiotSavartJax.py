import jax.scipy as jsp
import jax.numpy as jnp
from jax import jit
import time

@jit
def BiotSavart(A,B,R,I):
    direction = (B-A)/jnp.linalg.norm(B-A)
    Rprime = jnp.linspace(A,B,num=1000+1)
    #integral = jsp.integrate.trapezoid(jnp.cross(direction,R-x)/(jnp.linalg.norm(R-x))**3,x=Rprime, axis=0)
    integral = jsp.integrate.trapezoid(jnp.cross(direction,R-Rprime)/(jnp.linalg.norm(R-Rprime))**3,x=Rprime, axis=0)
    return integral*1e-7*I
    
def __main__():
    A = jnp.array([0,0,-100]) # Begining of the wire
    B = jnp.array([0,0,100]) # End of the wire
    R = jnp.array([1,0,1]) # Position where we calculate the magnetic field
    I = 1
    
    direction = (B-A)/jnp.linalg.norm(B-A)
    Rprime = jnp.linspace(A,B,num=1000)

    print("0: ", jnp.cross(direction,R-Rprime[0,:])/(jnp.linalg.norm(R-Rprime[0,:]))**3)
    print("500: ", jnp.cross(direction,R-Rprime[500,:])/(jnp.linalg.norm(R-Rprime[500,:]))**3)
    print("999: ", jnp.cross(direction,R-Rprime[999,:])/(jnp.linalg.norm(R-Rprime[999,:]))**3)

    # Time taken with compilation
    starttime = time.time()
    print("Magnetic Field: ", BiotSavart(A,B,R,I))
    endtime = time.time()
    print("Time elapsed: ", endtime-starttime)

    # Now compiled
    starttime = time.time()
    print("Magnetic Field: ", BiotSavart(A,B,R,I))
    endtime = time.time()
    print("Time elapsed: ", endtime-starttime)
    
    return 0

if __name__ == "__main__":
    __main__()