# nome do código: estevaogomes
from estevaogomes import Curve, Coil, Particles, Optimization, Particle_Tracer

# Initial parameters
order = 5
curve1=Curve(pos=(0,0,0), R=1, order=order)
current1=1e5
curve2=Curve(pos=(2,0,0), R=1, order=order)
current2=1e5
###print(curve2.coefs)
###curve2.plot()
coils=[Coil(curve1,current1),Coil(curve2,current2)]
particleEnsemble=Particles(number=5, energy=3.5, charge=2, mass=4)

# Plot initial coils
###coils.plot()
# Plot initial trajectories
trajectories=Particle_Tracer(coils,particleEnsemble)
trajectories.plot()
trajectories.calculate_trajectories()
# Perform optimization
optimization=Optimization(coils,particleEnsemble,maxiter=10)

# Plot final coils
###optimization.coils.plot()
# Plot final trajectories
###optimization.particleEnsemble.plot()