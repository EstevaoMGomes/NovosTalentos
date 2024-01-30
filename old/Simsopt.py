import simsopt
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field import BiotSavart, Coil, Current
import time

numquadpoints = 100
order = 1
curve = CurveXYZFourier(numquadpoints, order)
curve.set("xc(0)", 1)
curve.set("xc(1)", 1)
curve.set("yc(0)", 0)
curve.set("yc(1)", 0)
curve.set("zs(1)", 1)
curve.x = curve.x
#curve.plot()
starttime = time.time()
bs = BiotSavart([Coil(curve,Current(1))])
bs.set_points([[1,0,0]])
print("Magnetic fied:", bs.B())
print("Time elapsed:", (time.time() - starttime)*10**6, "microseconds")
# Coeficientes de Fourier da curva
print("Curve parameters:", curve.x)
# Valores de x, y, z ao longo da curva
print(curve.gamma())