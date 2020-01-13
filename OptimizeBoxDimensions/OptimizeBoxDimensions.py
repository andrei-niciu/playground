# This python script is aiming at solving the problem mentioned at the link below (optimizing box dimensions to match given required volume) using the methods described at the link
# https://towardsdatascience.com/optimization-with-scipy-and-application-ideas-to-machine-learning-81d39c7938b8

# The problem:
# Given an open-top box with square bottom and rectangular sides, having volume of 256, find the dimensions that minimize the surface of the box (the 5 sides)

import numpy as np
import scipy.optimize as optimize

def box_surface(v):
    return v[0]**2 + 4*v[0]*v[1]

def box_volume(v):
    return v[0]**2 * v[1]

def box_volume_constraint(v):
    return 256 - box_volume(v)

con_volume = {'type':'eq', 'fun':box_volume_constraint}
cons = (con_volume)

start = (0.1, 0.1)
bounds = ((0.1, None), (0.1, None))

print('Applying minimize with method SLSQP:')
optimized = optimize.minimize(box_surface, x0=start, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':1000})
print(optimized)

print('Applying minimize with method trust-constr:')
optimized = optimize.minimize(box_surface, x0=start, method='trust-constr', bounds=bounds, constraints=cons, options={'maxiter':1000})
print(optimized)
