import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.lax import fori_loop

from src.Dynamics import GuidingCenter
from src.InitialConditions import initial_conditions
from src.MagneticField import B_Norm
from src.CreateCoil import CreateCoil
from src.Plotter import plot3D

##################################################################################################################################
############################################ Load data from results.txt ##########################################################
##################################################################################################################################

file = f"results.txt"
line2 = open(file, mode="r").readlines()[-3]

begin_N_particles = line2.find("-N_particles")+len("-N_particles")+1
end_N_particles = line2.find("-ncoils")-1
N_particles = int(line2[begin_N_particles:end_N_particles])

begin_ncoils = line2.find("-ncoils")+len("-ncoils")+1
end_ncoils = line2.find("-N_CurvePoints")-1
ncoils = int(line2[begin_ncoils:end_ncoils])

begin_N_CurvePoints = line2.find("-N_CurvePoints")+len("-N_CurvePoints")+1
end_N_CurvePoints = line2.find("-maxtime")-1
N_CurvePoints = int(line2[begin_N_CurvePoints:end_N_CurvePoints])

begin_maxtime = line2.find("-maxtime")+len("-maxtime")+1
end_maxtime = line2.find("-timesteps")-1
maxtime = float(line2[begin_maxtime:end_maxtime])

begin_timesteps = line2.find("-timesteps")+len("-timesteps")+1
end_timesteps = line2.find("-R")-1
timesteps = int(line2[begin_timesteps:end_timesteps])

begin_R = line2.find("-R")+len("-R")+1
end_R = line2.find("-r") -1
R = float(line2[begin_R:end_R])

begin_r = line2.find("-r")+len("-r")+1
end_r = line2.find("-loss_r") -1
r = float(line2[begin_r:end_r])

begin_loss_r = line2.find("-loss_r")+len("-loss_r")+1
loss_r = float(line2[begin_loss_r:])

line3 = open(file, mode="r").readlines()[-2]
currents = jnp.array(eval(line3))

line4 = open(file, mode="r").readlines()[-1]
FourierCoefficients = jnp.reshape(jnp.array(eval(line4)), (ncoils, 3, -1))

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

#from src.InitialConditions import CreateEquallySpacedCurves
#FourierCoefficients = CreateEquallySpacedCurves(ncoils, 3, R, r)
plot3D(FourierCoefficients)

InitialValues = initial_conditions(N_particles, "torus", R, loss_r)
vperp = InitialValues[4]

curves_points = jnp.empty((ncoils, N_CurvePoints, 3))

def fori_createcoil(coil: int, curves_points: jnp.ndarray) -> jnp.ndarray:
    return curves_points.at[coil].set(CreateCoil(FourierCoefficients[coil], N_CurvePoints))

curves_points = fori_loop(0, ncoils, fori_createcoil, curves_points)
normB = jnp.apply_along_axis(B_Norm, 0, InitialValues[:3], curves_points, currents)

m = 4*1.660538921e-27
μ = m*vperp**2/(2*normB)

trajectories = jnp.empty((N_particles, timesteps, 4))
times = jnp.linspace(0, maxtime, timesteps)

def trace_trajectory(particle: int, trajectories: jnp.ndarray) -> jnp.ndarray:
    return trajectories.at[particle,:,:].set(odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[particle], times, currents, curves_points, μ[particle], atol=1e-8, rtol=1e-8, mxstep=100))
trajectories = fori_loop(0, N_particles, trace_trajectory, trajectories)
print(trajectories.shape)
print("loss:",
    jnp.mean(
        jnp.square(
            jnp.sqrt(
                trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2
            )-R
        )+trajectories[:, :, 2]**2
    )/loss_r**2
)
plot3D(FourierCoefficients, trajectories)