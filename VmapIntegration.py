import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.integrate import trapezoid
from time import time
import numpy as np

observation_point = [0.5, 0.5, 0.1]
current = 1e4

#coil_points = ([ 1.        ,  0.        ,  0.        ],
#                 [ 0.30901699,  0.95105652,  0.        ],
#                 [-0.80901699,  0.58778525,  0.        ],
#                 [-0.80901699, -0.58778525,  0.        ],
#                 [ 0.30901699, -0.95105652,  0.        ])

coil_points = ([ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
 [ 9.98026728e-01,  6.27905195e-02,  0.00000000e+00],
 [ 9.92114701e-01,  1.25333234e-01,  0.00000000e+00],
 [ 9.82287251e-01,  1.87381315e-01,  0.00000000e+00],
 [ 9.68583161e-01,  2.48689887e-01,  0.00000000e+00],
 [ 9.51056516e-01,  3.09016994e-01,  0.00000000e+00],
 [ 9.29776486e-01,  3.68124553e-01,  0.00000000e+00],
 [ 9.04827052e-01,  4.25779292e-01,  0.00000000e+00],
 [ 8.76306680e-01,  4.81753674e-01,  0.00000000e+00],
 [ 8.44327926e-01,  5.35826795e-01,  0.00000000e+00],
 [ 8.09016994e-01,  5.87785252e-01,  0.00000000e+00],
 [ 7.70513243e-01,  6.37423990e-01,  0.00000000e+00],
 [ 7.28968627e-01,  6.84547106e-01,  0.00000000e+00],
 [ 6.84547106e-01,  7.28968627e-01,  0.00000000e+00],
 [ 6.37423990e-01,  7.70513243e-01,  0.00000000e+00],
 [ 5.87785252e-01,  8.09016994e-01,  0.00000000e+00],
 [ 5.35826795e-01,  8.44327926e-01,  0.00000000e+00],
 [ 4.81753674e-01,  8.76306680e-01,  0.00000000e+00],
 [ 4.25779292e-01,  9.04827052e-01,  0.00000000e+00],
 [ 3.68124553e-01,  9.29776486e-01,  0.00000000e+00],
 [ 3.09016994e-01,  9.51056516e-01,  0.00000000e+00],
 [ 2.48689887e-01,  9.68583161e-01,  0.00000000e+00],
 [ 1.87381315e-01,  9.82287251e-01,  0.00000000e+00],
 [ 1.25333234e-01,  9.92114701e-01,  0.00000000e+00],
 [ 6.27905195e-02,  9.98026728e-01,  0.00000000e+00],
 [ 6.12323400e-17,  1.00000000e+00,  0.00000000e+00],
 [-6.27905195e-02,  9.98026728e-01,  0.00000000e+00],
 [-1.25333234e-01,  9.92114701e-01,  0.00000000e+00],
 [-1.87381315e-01,  9.82287251e-01,  0.00000000e+00],
 [-2.48689887e-01,  9.68583161e-01,  0.00000000e+00],
 [-3.09016994e-01,  9.51056516e-01,  0.00000000e+00],
 [-3.68124553e-01,  9.29776486e-01,  0.00000000e+00],
 [-4.25779292e-01,  9.04827052e-01,  0.00000000e+00],
 [-4.81753674e-01,  8.76306680e-01,  0.00000000e+00],
 [-5.35826795e-01,  8.44327926e-01,  0.00000000e+00],
 [-5.87785252e-01,  8.09016994e-01,  0.00000000e+00],
 [-6.37423990e-01,  7.70513243e-01,  0.00000000e+00],
 [-6.84547106e-01,  7.28968627e-01,  0.00000000e+00],
 [-7.28968627e-01,  6.84547106e-01,  0.00000000e+00],
 [-7.70513243e-01,  6.37423990e-01,  0.00000000e+00],
 [-8.09016994e-01,  5.87785252e-01,  0.00000000e+00],
 [-8.44327926e-01,  5.35826795e-01,  0.00000000e+00],
 [-8.76306680e-01,  4.81753674e-01,  0.00000000e+00],
 [-9.04827052e-01,  4.25779292e-01,  0.00000000e+00],
 [-9.29776486e-01,  3.68124553e-01,  0.00000000e+00],
 [-9.51056516e-01,  3.09016994e-01,  0.00000000e+00],
 [-9.68583161e-01,  2.48689887e-01,  0.00000000e+00],
 [-9.82287251e-01,  1.87381315e-01,  0.00000000e+00],
 [-9.92114701e-01,  1.25333234e-01,  0.00000000e+00],
 [-9.98026728e-01,  6.27905195e-02,  0.00000000e+00],
 [-1.00000000e+00,  1.22464680e-16,  0.00000000e+00],
 [-9.98026728e-01, -6.27905195e-02,  0.00000000e+00],
 [-9.92114701e-01, -1.25333234e-01,  0.00000000e+00],
 [-9.82287251e-01, -1.87381315e-01,  0.00000000e+00],
 [-9.68583161e-01, -2.48689887e-01,  0.00000000e+00],
 [-9.51056516e-01, -3.09016994e-01,  0.00000000e+00],
 [-9.29776486e-01, -3.68124553e-01,  0.00000000e+00],
 [-9.04827052e-01, -4.25779292e-01,  0.00000000e+00],
 [-8.76306680e-01, -4.81753674e-01,  0.00000000e+00],
 [-8.44327926e-01, -5.35826795e-01,  0.00000000e+00],
 [-8.09016994e-01, -5.87785252e-01,  0.00000000e+00],
 [-7.70513243e-01, -6.37423990e-01,  0.00000000e+00],
 [-7.28968627e-01, -6.84547106e-01,  0.00000000e+00],
 [-6.84547106e-01, -7.28968627e-01,  0.00000000e+00],
 [-6.37423990e-01, -7.70513243e-01,  0.00000000e+00],
 [-5.87785252e-01, -8.09016994e-01,  0.00000000e+00],
 [-5.35826795e-01, -8.44327926e-01,  0.00000000e+00],
 [-4.81753674e-01, -8.76306680e-01,  0.00000000e+00],
 [-4.25779292e-01, -9.04827052e-01,  0.00000000e+00],
 [-3.68124553e-01, -9.29776486e-01,  0.00000000e+00],
 [-3.09016994e-01, -9.51056516e-01,  0.00000000e+00],
 [-2.48689887e-01, -9.68583161e-01,  0.00000000e+00],
 [-1.87381315e-01, -9.82287251e-01,  0.00000000e+00],
 [-1.25333234e-01, -9.92114701e-01,  0.00000000e+00],
 [-6.27905195e-02, -9.98026728e-01,  0.00000000e+00],
 [-1.83697020e-16, -1.00000000e+00,  0.00000000e+00],
 [ 6.27905195e-02, -9.98026728e-01,  0.00000000e+00],
 [ 1.25333234e-01, -9.92114701e-01,  0.00000000e+00],
 [ 1.87381315e-01, -9.82287251e-01,  0.00000000e+00],
 [ 2.48689887e-01, -9.68583161e-01,  0.00000000e+00],
 [ 3.09016994e-01, -9.51056516e-01,  0.00000000e+00],
 [ 3.68124553e-01, -9.29776486e-01,  0.00000000e+00],
 [ 4.25779292e-01, -9.04827052e-01,  0.00000000e+00],
 [ 4.81753674e-01, -8.76306680e-01,  0.00000000e+00],
 [ 5.35826795e-01, -8.44327926e-01,  0.00000000e+00],
 [ 5.87785252e-01, -8.09016994e-01,  0.00000000e+00],
 [ 6.37423990e-01, -7.70513243e-01,  0.00000000e+00],
 [ 6.84547106e-01, -7.28968627e-01,  0.00000000e+00],
 [ 7.28968627e-01, -6.84547106e-01,  0.00000000e+00],
 [ 7.70513243e-01, -6.37423990e-01,  0.00000000e+00],
 [ 8.09016994e-01, -5.87785252e-01,  0.00000000e+00],
 [ 8.44327926e-01, -5.35826795e-01,  0.00000000e+00],
 [ 8.76306680e-01, -4.81753674e-01,  0.00000000e+00],
 [ 9.04827052e-01, -4.25779292e-01,  0.00000000e+00],
 [ 9.29776486e-01, -3.68124553e-01,  0.00000000e+00],
 [ 9.51056516e-01, -3.09016994e-01,  0.00000000e+00],
 [ 9.68583161e-01, -2.48689887e-01,  0.00000000e+00],
 [ 9.82287251e-01, -1.87381315e-01,  0.00000000e+00],
 [ 9.92114701e-01, -1.25333234e-01,  0.00000000e+00],
 [ 9.98026728e-01, -6.27905195e-02,  0.00000000e+00])

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
    integral_result = trapezoid(magnetic_field_vectorized, axis=0)
    return integral_result

# Initial compilation of B_trapezoid
B_trapezoid(current, jnp.array(coil_points), jnp.array(observation_point))

# Results
print(f"Magnetic Field at {observation_point}:")

time1 = time();result_trapezoid = B_trapezoid(current, jnp.array(coil_points), jnp.array(observation_point));time2 = time()
print(f"From trapezoid: {result_trapezoid} took {(time2 - time1):.1e}s")

# SIMSOPT field
from simsopt.field import BiotSavart
from simsopt.field import Current, Coil
from simsopt.geo import CurveXYZFourier
curve = CurveXYZFourier(100, 1)
curve.x = [0, 0, 1., 0., 1., 0., 0., 0., 0.]
np.testing.assert_allclose(curve.gamma(),np.array(coil_points))
coil = Coil(curve, Current(current))
field = BiotSavart([coil])  # Multiple coils can be included in the list
field.set_points(np.array([observation_point]))
time1 = time();result = field.B();time2 = time()

print(f"From SIMSOPT:   {np.array(field.B()[0])} took {(time2 - time1):.1e}s")
#print(curve.gamma())