# Solving the optimium box dimensions problem mentioned at https://towardsdatascience.com/optimization-with-scipy-and-application-ideas-to-machine-learning-81d39c7938b8)
# Given an open-top box with square bottom and rectangular sides, having volume of 256, find the dimensions that minimize the surface of the box (the 5 sides)

import numpy as np
import scipy.optimize as optimize

# Define the volume and surface functions:
# The box surface function would be: bottom + 4 * side_surface
# Having dimension of the bottom equal to x and heiht equal to y, the surface function would be: x^2 + 4*x*y
# For scipy, we encode x and y params in a vector v:

def box_surface(v):
    return v[0]**2 + 4*v[0]*v[1]

def box_volume(v):
    return v[0]**2 * v[1]

# Define the constraints of having a volume of 256:
def box_volume_constraint(v):
    return 256 - box_volume(v)
con_volume = {'type':'eq', 'fun':box_volume_constraint}
cons = (con_volume)

# Define a guess (start) and bounds
start = (0.1, 0.1)
bounds = ((0.1, None), (0.1, None))

# Optimize with two methods and show the result:
print('Applying minimize with method SLSQP:')
optimized = optimize.minimize(box_surface, x0=start, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':1000})
print(optimized)

print('Applying minimize with method trust-constr:')
optimized = optimize.minimize(box_surface, x0=start, method='trust-constr', bounds=bounds, constraints=cons, options={'maxiter':1000})
print(optimized)
