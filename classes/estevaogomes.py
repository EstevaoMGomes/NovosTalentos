import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.ode import odeint
import numpy as np
import plotly.graph_objects as go

from CreateCoil import CreateCoil
#from Dynamics import GuidingCenter


from functools import partial

#@partial(jit, static_argnums=(0, 1))
def initial_conditions(coils, particleEnsemble) -> jnp.ndarray:
        ########### Implementing just for one type of particles ###########
        particles = particleEnsemble[0]
        energy = particles.energy
        mass = particles.mass
        number = particles.number
        # Calculating the particle's thermal velocity in SI units
        vth = jnp.sqrt(2*energy/mass)

        # Initializing pitch angle
        seed = 0
        key = jax.random.PRNGKey(seed)
        pitch = jax.random.uniform(key,shape=(number,), minval=-1, maxval=1)

        # Initializing velocities
        vpar = vth*pitch
        vperp = vth*jnp.sqrt(1-pitch**2)

        # Initializing positions
        x = jax.random.uniform(key,shape=(number,), minval=coils[0].position[0], maxval=coils[1].position[0])
        r = jax.random.uniform(key,shape=(number,), minval=0, maxval=jnp.min(jnp.array([coil.radius for coil in coils])))
        Θ = jax.random.uniform(key,shape=(number,), minval=0, maxval=2*jnp.pi)
        y = r*jnp.cos(Θ)
        z = r*jnp.sin(Θ)

        return jnp.array((x, y, z, vpar, vperp))

class Curve:
    """
        Args:
    pos: jnp.ndarray: Position of the center of the curve - shape (3,)
    R: jnp.float32: Radius of the curve
    order: jnp.int32: Order of the Fourier series
    """
    def __init__(self, pos: jnp.ndarray, R: jnp.float32, order: jnp.int32 = 2, NCurvePoints: jnp.int32 = 100):
        self.position = pos
        self.radius = R
        self.order = order
        self.coefs = jnp.concatenate((jnp.array(np.atleast_2d(pos).T), jnp.array([[0, 0], [1, 0], [0,1]])), axis=1)
        for i in range(1, order+1):
            self.coefs = jnp.concatenate(((self.coefs, jnp.array([[0, 0], [0, 0], [0, 0]]))), axis=1)
        self.NCurvePoints = NCurvePoints

    def __str__(self):
        return f"Curve(pos={self.position}, R={self.radius}, order={self.order})"
    def __repr__(self):
        return f"Curve(pos={self.position}, R={self.radius}, order={self.order})"
    def gamma(self):
        return CreateCoil(self.coefs, self.NCurvePoints, self.order)
    def plot(self):
        coil = self.gamma()
        trace_curve = go.Scatter3d(x=coil[:,0],y=coil[:,1],z=coil[:,2], mode='lines', name=f'Coil', line=dict(color='rgb(179,179,179)', width=4))
        fig = go.Figure(data=[trace_curve])
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
        fig.show()
        

class Coil(Curve):
    def __init__(self, curve: Curve, current: jnp.float32):
        super().__init__(curve.position, curve.radius, curve.order)
        self.current = current
        
    def __str__(self):
        return f"Coil(curve={super().__str__()}, current={self.current})"
    def __repr__(self):
        return f"Coil(curve={super().__str__()}, current={self.current})"
    def plot(self):
        super().plot()

class Particles:
    """
        Args:
    number: jnp.int32: Number of particles
    energy: jnp.float32: Energy of the particles in MeV
    charge: jnp.float32: Charge of the particles in e    
    mass: jnp.float32: Mass of the particles in amu
    """
    def __init__(self, number: jnp.int32, energy: jnp.float32, charge: jnp.float32, mass: jnp.float32):
        self.number = number
        self.energy = energy*1.602176634e-13
        self.charge = charge*1.602176634e-19
        self.mass = mass*1.66053906660e-27



#odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[0], jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ[0], atol=1e-8, rtol=1e-8, mxstep=1000)


class Particle_Tracer:
    """
        Args:
    coils: Coil | list[Coil]: Coils to calculate the trajectories
    particleEnsemble: Particles | list[Particles]: Particles to calculate the trajectories
    """
    def __init__(self, coils: Coil | list[Coil], particleEnsemble: Particles | list[Particles]):
        if type(coils) == Coil:
            self.coils = np.array([coils])
        else:
            try:
                self.coils = np.array(coils, dtype=Coil)
            except:
                raise ValueError("coils must be a Coil object or an array-like list of Coils objects")
        if type(particleEnsemble) == Particles:
            self.particleEnsemble = np.array([particleEnsemble])
        else:
            try:
                self.particleEnsemble = np.array(particleEnsemble, dtype=Coil)
            except:
                raise ValueError("particleEnsemble must be a Particles object or an array-like list of Particles objects")
        self.trajectories = jnp.zeros((len(self.coils), len(self.particleEnsemble), 3))
  
    def calculate_trajectories(self):
        print("Calculating trajectories...")
        print(initial_conditions(self.coils, self.particleEnsemble))

    def plot(self, time: jnp.float32 = 1e-7):
        pass

    def animation(self, time: jnp.float32 = 1e-7):
        pass

class Optimization(Particle_Tracer):
    def __init__(self, coils: Coil | list[Coil], particleEnsemble: Particles | list[Particles], maxiter: jnp.int32 = 10):
        super().__init__(coils, particleEnsemble)
        self.maxiter = maxiter

    def optimize(self):
        print("Optimizing...")

    def plot(self, time: jnp.float32 = 1e-7):
        pass

    def animation(self, time: jnp.float32 = 1e-7):
        pass


