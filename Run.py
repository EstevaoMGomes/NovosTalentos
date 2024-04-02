# To activate the virtual environment: source NovosTalentos/bin/activate

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit, vmap, pmap
from jax.lax import fori_loop, scan, select
from simsopt.field import BiotSavart, Current, Coil
from simsopt import load
import numpy as np
from time import time
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.Dynamics import GuidingCenter
from src.MagneticField import B, B_Norm, B_novo, grad_B
from src.CreateCoil import CreateCoil
from src.Plotter import plot3D, plot2D, plot_animation3d
from simsopt.geo import CurveXYZFourier

def oldCreateCoil(FourierCoefficients: list[int | float], NumberOfPoints: int, order: float) -> CurveXYZFourier:
    # Creating a curve with "NumberOfPoints" points and "order" number of Fourier coefficients
    curve = CurveXYZFourier(NumberOfPoints, order=order)
    # Setting the Fourier coefficients
    curve.x = FourierCoefficients
    return curve

debuging = False
#------------------------------------------------------------------------#
# Inputs
#------------------------------------------------------------------------#

N_particles = 10
#FourierCoefficients = [[-1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,],[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,]]
FourierCoefficients = [[-2., 0., 0, 0., 1., 0., 0., 0., 1.], [2., 0., 0., 0., 1., 0., 0., 0., 1.]]
#FourierCoefficients = [[-1.39877820e+01,  9.73827622e-02, -2.04908605e+00, -1.41725830e-01,
#  4.56845054e+00, -2.32155126e-03,  1.32708122e+00,  2.58179468e-01,
#  3.05903007e+00], [-1.51871837e+00, -3.05514491e+00,  6.77326222e-01,
# -1.34030161e+01, -1.01064459e+01,  2.96167674e+00,  2.29916623e+00,
#  2.68320389e+00, -2.02021051e+00]]
#FourierCoefficients = [[-13740614718.929352, 45010001.478988245, -2546633035.205666, 96708340.3263575, 3318424872.9081783, 56425454.87037025, 394183734.8469013, 89046620.86249365, 3500784670.058241], [-2696078547.829357, -7142264944.76991, 1369023895.2295704, -7786834069.593535, -14305665912.815304, 2545483309.155122, 1266318197.132097, 2373091734.687136, -3161998497.9020004]]
#FourierCoefficients = [[-2.105436767037876, 0.16540488329413466, -0.37822677806631755, -0.07562983995169222, 1.0386110537889908, -0.002546577154512729, -0.007698645808491838, -0.0029953834138113124, 0.9946516230694082], [0.8673039028909039, 0.03449993429305001, 0.03660480002495869, -0.027350596766373882, 0.4796513442381621, -0.031708014247428744, -0.12725927105681636, -0.018898924058447703, 0.44847597843629555]]
#FourierCoefficients = [[-2.147869397377754, 0.18268509104539, -0.389534664865396, 0.09630831753511991, -0.06310516552734853, -0.07635853572069998, 1.057732159083316, -0.019875193041674514, -0.05769982161346997, 0.013816961192643097, -0.003469770645142031, -0.014721939786353663, 1.0404787735329102, -0.003288229042236649, -0.06972945482360222], [0.8854020210856544, 0.007529709255217744, 0.012680877828184785, -0.03948909766063804, 0.022531239425347172, -0.033125423689350376, 0.44673917656716056, -0.059030657960613436, -0.11965553667924496, -0.07499512969218892, -0.1431894887997557, -0.04413417880890023, 0.4234527802784654, 0.03198968374874773, -0.14881502704764882 ]]

#FourierCoefficients = [[-1.8746883002172272, 0.00919390981009342, -0.010681154782432757, -0.0019821267370478224, 0.00011889160187998762, -0.0020990015935972014, 0.001982106015646127, 0.013726948728579358, 1.125182668589405, 0.002868998546910861, 0.02079667646759333, -0.0024710250154603712, 0.025211227137632023, 0.004306320881117695, -0.00782880054131793, 0.0015650786337092727, 1.1017660778948344, 0.00830019497644951, -0.00928488444917999, 0.003781748043543047, -0.01124483287745919], [1.8746883002172272, 0.00919390981009342, -0.010681154782432757, -0.0019821267370478224, 0.00011889160187998762, -0.0020990015935972014, 0.001982106015646127, 0.013726948728579358, 1.125182668589405, 0.002868998546910861, 0.02079667646759333, -0.0024710250154603712, 0.025211227137632023, 0.004306320881117695, -0.00782880054131793, 0.0015650786337092727, 1.1017660778948344, 0.00830019497644951, -0.00928488444917999, 0.003781748043543047, -0.01124483287745919]]

#FourierCoefficients = [[-1.218925387828992, -0.002801286974587578, -0.004838642145847803, -0.008924873796238495, 0.007389075652447792, -0.0002847792442128649, 0.004470730185163476, -0.051148147716136, 0.8738088009436316, 0.01135136464901896, 0.049525538794861185, 0.02409928283357205, 0.011356459098402352, -0.005722558941984067, 0.08324409547481337, 0.01035205528344201, 0.8616590715152666, -0.025461134716604356, 0.03796286228005088, 0.00922409686862832, -0.004651945497331838], [1.4771751424204553, 0.03981571409830538, -0.16597304349841394, -0.10645120132143934, 0.0017527704832195588, -0.009571499568316983, -0.025168851046942938, 0.43331404925752054, 0.9482956559762938, -0.042735856895952225, 0.1542298352620084, -0.10603080016486552, -0.03348291498989099, -0.09303330519718032, 0.028215235585928644, 0.007570792278128409, 0.8505832125573539, 0.15394040193834002, -0.017163869793269476, -0.06843962818595095, -0.10964194059369636]]
FourierCoefficients = [[0.17098171428475298, 0.3374664716941216, -0.1254909655142695, -0.2946331032226744, -0.2622842753241998, -0.048973187270651315, 0.3497800923875128, -0.20195369438508406, 2.224162642944778, 0.07667493386512471, 0.4336633989054364, 0.10152596132329153, 0.19384256152484083, -0.09584734662997382, 0.3751829303911688, 0.08980776251682004, 2.161862082851838, -0.11547037379984784, 0.059507984463074474, 0.06412364269552938, -0.015192303015270617], [1.9838731502546585, 0.3220855891292415, -0.43852908876475594, -0.2810368529307607, -0.033568746137648375, 0.10279755055341273, 0.070932160835276, 0.0002818446146837656, 0.3713449807083593, 0.0609309044426282, -0.015151357739235734, -0.060592243757987066, -0.18420303852088024, -0.08585925882706437, 0.2629006022117243, 0.0764272877802948, 0.2651424467979431, -0.03575518750264098, 0.17674754530131961, 0.05051082210043819, -0.04370612317967714]]
N_coils = len(FourierCoefficients)
N_CurvePoints = 1000
FC_order = int((len(FourierCoefficients[0])/3-1)/2)
currents = [1e7, 1e7]


#------------------------------------------------------------------------#
# Setting the initial conditions
#------------------------------------------------------------------------#

def initial_conditions(N_particles: int) -> jnp.ndarray:
    # Alpha particles energy in SI units
    E = 3.52*1.602176565e-13
    # Alpha particles mass in SI units
    m = 4*1.660538921e-27
    # Alpha particles thermal velocity in SI units
    vth = jnp.sqrt(2*E/m)

    # Initializing pitch angle
    seed = 0
    key = jax.random.PRNGKey(seed)
    pitch = jax.random.uniform(key,shape=(N_particles,), minval=-1, maxval=1)
    #pitch = jnp.ones(N_particles)*-0.1
    # Initializing velocities
    vpar = vth*pitch
    vperp = vth*jnp.sqrt(1-pitch**2)

    #x = jnp.array([-0.7])
    #y = jnp.array([ 0.3])
    #z = jnp.array([ 0.3])

    x = jax.random.uniform(key,shape=(N_particles,), minval=-2, maxval=2)
    r = jax.random.uniform(key,shape=(N_particles,), minval=0, maxval=0.5)
    Θ = jax.random.uniform(key,shape=(N_particles,), minval=0, maxval=2*jnp.pi)
    y = r*jnp.cos(Θ)
    z = r*jnp.sin(Θ)

    return jnp.array((x, y, z, vpar, vperp))

InitialValues = initial_conditions(N_particles)
vperp = InitialValues[4]
if debuging:
    print("------------------------------------------------------------------------")
    print(f"x transposed: {jnp.transpose(InitialValues[:3])})")
    print(f"vpar: {InitialValues[3]}")
    print(f"vperp: {vperp}")


#------------------------------------------------------------------------#
# Creating the coils
#------------------------------------------------------------------------#

curves = np.empty(N_coils, dtype=object)
curves_points = jnp.empty((N_coils, N_CurvePoints, 3))
coils = np.empty(N_coils, dtype=object)

for i in range(N_coils):
    # Creating a curve with "NCurvePoints" points and "FCorder" order of the Fourier series
    curves[i] = oldCreateCoil(FourierCoefficients[i], N_CurvePoints, FC_order)
    # Getting the curve points  
    curves_points = curves_points.at[i].set(curves[i].gamma())
    # Creating a coil
    coils[i] = Coil(curves[i], Current(currents[i]))

currents = jnp.array(currents)

#plot3D(N_coils, FourierCoefficients)

#------------------------------------------------------------------------#
# Magnetic Field Calcultions
#------------------------------------------------------------------------#

# Compiling and running the function
B(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time1 = time()
result_MagneticField = B(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time2 = time()
normB = jnp.linalg.norm(result_MagneticField)

Stellarator = load("biot_savart_opt.json")
#Stellarator.gamma()

field = BiotSavart(coils)
field.B()
field.set_points(jnp.transpose(InitialValues[:3]))
time3 = time()
result_Simsopt = field.B()
time4 = time()

print("------------------------------------------------------------------------")

print(f"Magnetic Field at {jnp.transpose(InitialValues[:3])[0]}:")

print(f"From trapezoid: {result_MagneticField} took {(time2 - time1):.1e}s")
print(f"Magnetic Field Norm: {jnp.linalg.norm(result_MagneticField)}")

print(f"From SIMSOPT:   {np.array(result_Simsopt)[0]} took {(time4 - time3):.1e}s")
print(f"Magnetic Field Norm: {np.linalg.norm(np.array(result_Simsopt)[0])}")

print("------------------------------------------------------------------------")

# Compiling and running the function
B_novo(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time1 = time()
result_MagneticField = B_novo(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time2 = time()
normB = jnp.linalg.norm(result_MagneticField)
print(f"From trapezoid: {result_MagneticField} took {(time2 - time1):.1e}s")
print(f"Magnetic Field Norm: {normB}")

print("------------------------------------------------------------------------")

# Compiling and running the grad function
grad_B(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time1 = time()
result_gradMagneticField = grad_B(jnp.transpose(InitialValues[:3])[0], curves_points, currents)
time2 = time()
normgradB = jnp.linalg.norm(result_gradMagneticField)
print(f"Grad from trapezoid: {result_gradMagneticField} took {(time2 - time1):.1e}s")
print(f"Grad of Magnetic Field Norm: {normgradB}")

print("------------------------------------------------------------------------")

#------------------------------------------------------------------------#
# Guiding Center Calculations
#------------------------------------------------------------------------#

m = 4*1.660538921e-27
q = 2*1.602176565e-19

# Adiabatic invariant μ
μ = m*vperp**2/(2*normB)
if debuging:
    print(f"μ: {μ}")
    print("------------------------------------------------------------------------")

if debuging:
    for i in range(N_particles):
        time1 = time()
        Dx, Dy, Dz, Dvpar = GuidingCenter(jnp.transpose(InitialValues[:4])[i],0.0,currents, curves_points, μ[i])
        time2 = time()
        print(f"Guiding Center for particle {i+1}:\n Dx: {Dx}\n Dy: {Dy}\n Dz: {Dz}\n Dvpar: {Dvpar}\nTook {(time2 - time1):.1e}s")
        print("------------------------------------------------------------------------")

timesteps = 200
maxtime = 1e-5

print("Number of jax devices: ", jax.device_count())
print("Number of jax local devices: ", jax.local_device_count())
print("------------------------------------------------------------------------")

trajectories = jnp.empty((N_particles, timesteps, 4))
@jit
def trace_trajectory(particle: jnp.int32, InitialValues: jnp.ndarray, times: jnp.ndarray, currents: jnp.ndarray, curves_points: jnp.ndarray, μ: jnp.float32) -> jnp.ndarray:
    return odeint(GuidingCenter, InitialValues[particle], times, currents, curves_points, μ[particle], atol=1e-8, rtol=1e-8, mxstep=100)
@jit
def fori_trace_trajectory(particle: jnp.int32, trajectories: jnp.ndarray) -> jnp.ndarray:
    return trajectories.at[particle,:,:].set(trace_trajectory(particle, jnp.transpose(InitialValues[:4]), jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ))
time1 = time()
#trajectories = vmap(trace_trajectory, in_axes=(0, None, None, None, None, None))(jnp.arange(N_particles), jnp.transpose(InitialValues[:4]), jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ)
#trajectories = pmap(trace_trajectory, in_axes=(0, None, None, None, None, None))(jnp.arange(N_particles), jnp.transpose(InitialValues[:4]), jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ)
#trajectories = jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[0], jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ[0], atol=1e-8, rtol=1e-8, mxstep=1000)])
#for i in range(1, N_particles):
#    trajectories = jnp.concatenate((trajectories, jnp.array([odeint(GuidingCenter, jnp.transpose(InitialValues[:4])[i], jnp.linspace(0, maxtime, timesteps), currents, curves_points, μ[i], atol=1e-5, rtol=1e-5)])), axis=0)
trajectories = fori_loop(0, N_particles,fori_trace_trajectory, trajectories) # THIS IS SLOWER, but can be better for compilation
time2 = time()

if debuging:
    print(trajectories)
print(f"Trajectories took {(time2 - time1):.1e}s")
print(f"Trajectories shape: {trajectories.shape}")
print("------------------------------------------------------------------------")


left_boundary = -2
right_boundary = 2
radial_boundary = 0.7
is_lost = select(jnp.greater(trajectories[:, :, 0], right_boundary*jnp.ones((N_particles,timesteps))) |
                 jnp.less(trajectories[:, :, 0], left_boundary*jnp.ones((N_particles,timesteps))) |
                 jnp.greater(jnp.square(trajectories[:, :, 1])+jnp.square(trajectories[:, :, 2]), radial_boundary*jnp.ones((N_particles,timesteps))),
                 jnp.ones((N_particles,timesteps)), jnp.zeros((N_particles,timesteps)))

@jit
def loss_calc(x: jnp.ndarray) -> jnp.ndarray:
    return 3.5*jnp.exp(-2*jnp.nonzero(x, size=1, fill_value=timesteps)[0]/timesteps)
loss_value = jnp.sum(jnp.apply_along_axis(loss_calc, 1, is_lost))
print("------------------------------------------------------------------------")

print(f"Loss value: {loss_value}")

print("------------------------------------------------------------------------")

#------------------------------------------------------------------------#
# Plotting analysis graphs
#------------------------------------------------------------------------#

plot = True
only_coils = True

#surface = ["cylinder", left_boundary, right_boundary, radial_boundary]
#minor_radius = 1
#major_radius = 2
##surface = ["torus", minor_radius, major_radius]
#plot_animation3d(FourierCoefficients,surface, trajectories, timesteps, show=False, save_movie=True)

if plot:
    if only_coils:
        plot3D(FourierCoefficients)
    else:
        plot3D(FourierCoefficients, trajectories)
    if N_particles > 1:
        raise ValueError("The plotter is not ready for more than one particle")
    NormBScan     = np.empty((N_particles, timesteps))
    NormGradBScan = np.empty((N_particles, timesteps))
    x             = np.empty((N_particles, timesteps))
    y             = np.empty((N_particles, timesteps))
    z             = np.empty((N_particles, timesteps))
    vpar          = np.empty((N_particles, timesteps))
    Dx            = np.empty((N_particles, timesteps))
    Dy            = np.empty((N_particles, timesteps))
    Dz            = np.empty((N_particles, timesteps))
    Dvpar         = np.empty((N_particles, timesteps))
    t             = np.linspace(0, maxtime, timesteps)


    for i in range(N_particles):
        for j in range(len(t)):
            NormBScan[i][j] = B_Norm(trajectories[i][j][:3], curves_points, currents)
            NormGradBScan[i][j] = jnp.linalg.norm(grad_B(trajectories[i][j][:3], curves_points, currents))
            x[i][j], y[i][j], z[i][j], vpar[i][j] = trajectories[i][j][:]
            Dx[i][j], Dy[i][j], Dz[i][j], Dvpar[i][j] = GuidingCenter(trajectories[i][j][:],t[j],currents, curves_points, μ[0])

    plot2D("1x2", (t, t), (NormBScan, NormGradBScan), ("|B|", "|∇B|"), ("", "Time (s)"), ("|B| (T)", "|∇B|"), "B&GradB")
    plot2D("1x1", t, μ[0]*NormBScan+0.5*m*vpar**2, "Energy", "Time (s)", "Energy (J)", "Energy")
    plot2D("1x1", t, np.sqrt(2*m*μ[0]/NormBScan)*NormGradBScan/NormBScan/q, "⍴∇B/B", "Time (s)", "Guiding Center Validity? (⍴∇B/B <<1)", "GC_Validity")
    plot2D("1x3", (t, t, t), (x, y, z), ("x", "y", "z"), ("", "", "Time (s)"), ("Distance (m)", "Distance (m)", "Distance (m)"), "Position")
    plot2D("1x3", (t, t, t), (Dx, Dy, Dz), ("x, y and z derivatives", "", ""), ("", "", "Time (s)"), ("dx (m/s)", "dy (m/s)", "dz (m/s)"), "Position_Derivatives")
    plot2D("1x2", (t, t), (vpar, Dvpar), ("parallel velocity and its derivative", ""), ("", "Time (s)"), ("Velocity (m/s)", "Acceleration (m/s^2)"), "Parallel_Velocity&Derivative")

