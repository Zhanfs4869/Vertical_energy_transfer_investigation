"""
    This is a program to resolve convection with anelastic equations
"""

import numpy as np
import dedalus.public as d3
import logging
import sys
import argparse
from dedalus.tools import post

logger = logging.getLogger(__name__)

###### Parameters
Lx, Lz = 1, 1         #The range of the box
Nx, Nz = 256, 256     #Number of grid points
Ro = 1
Rey = 1500
Nfc = 1

#Settings of solver
scale = 0.02              #the value of max_timestep
dealias = 3/2
stop_sim_time = 10
timestepper = d3.RK443
max_timestep = scale
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

###### Fields
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis)) 
v = dist.Field(name='v', bases=(xbasis,zbasis))                 
b = dist.Field(name='b', bases=(xbasis,zbasis))                
p = dist.Field(name='p', bases=(xbasis,zbasis))   

# Set of forcing
rng = np.random.default_rng(seed=42)
forcing = dist.Field(name='forcing', bases=(xbasis, zbasis))
forcing['g'] = rng.standard_normal(forcing['g'].shape)

# print(forcing['g'][15,:])

# fc = forcing['c']
# kx_local = dist.local_modes(xbasis)
# kz_local = dist.local_modes(zbasis)

# KX, KZ = np.meshgrid(kx_local, kz_local, indexing='ij')
# K = np.sqrt(KX**2 + KZ**2)

# k0 = 16
# dk = 2
# # ring_filter = np.exp(-((K-k0)**2) / (2 * dk**2))
# ring_filter = 0.5*(-np.tanh((KZ+0.95*k0)/dk)+np.tanh((KZ-0.95*k0)/dk)+2)*np.exp(-(np.abs(KZ)-k0)**2)/(2*dk**2)
# ring_filter*= np.exp(-(np.abs(KX)-k0)**2)/(2*dk**2)

# fc *= ring_filter
# forcing['c'] = fc
# fg = forcing['g']
# rms = np.sqrt(np.mean(fg**2))
# forcing['g'] *= 1 / (rms + 1e-30)
# print(ring_filter.shape)
# print(fc[15,:])
# print(KX[15,:])
# print(KZ[15,:])
# print(forcing['g'][15,:])

# the set of tau terms
tau_p = dist.Field(name='tau_p')
t = dist.Field(name='t')

###### Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)                       


sig = dist.Field(name='sig', bases=zbasis)
def sig_profile(z):
    S =  0.5*(1 + np.tanh((z-(15/16))*64))
    return S
sig['g'] = sig_profile(z)


###### Problem
# problem = d3.IVP([u, v, b, p, tau_p], time=t, namespace=locals())
# problem.add_equation("dt(u)-(2/Ro)*v*ex+grad(p)-(1/Rey)*div(grad(u))-b*ez=-u@grad(u)+forcing*np.sin(100*t)*sig*ex+forcing*np.sin(100*t)*sig*ez")
# problem.add_equation("trace(grad(u)) + tau_p = 0 ")
# problem.add_equation("dt(v)+(2/Ro)*(u@ex)-(1/Rey)*div(grad(v))=-u@grad(v)")
# problem.add_equation("dt(b)+(Nfc)*(Nfc)*(u@ez)=-u@grad(b)")
# problem.add_equation("integ(p) = 0")

# problem = d3.IVP([u, v, b, p, tau_p], time=t, namespace=locals())
# problem.add_equation("dt(u)+grad(p)-(1/Rey)*div(grad(u))-b*ez=-u@grad(u)+forcing*np.sin(100*t)*sig*ex+forcing*np.sin(100*t)*sig*ez")
# problem.add_equation("trace(grad(u)) + tau_p = 0 ")
# problem.add_equation("dt(v)-(1/Rey)*div(grad(v))=-u@grad(v)")
# problem.add_equation("dt(b)+(Nfc)*(Nfc)*(u@ez)=-u@grad(b)")
# problem.add_equation("integ(p) = 0")

problem = d3.IVP([u, v, p, tau_p], time=t, namespace=locals())
problem.add_equation("dt(u)+grad(p)-(1/Rey)*div(grad(u))=-u@grad(u)+1*ex*sig")
problem.add_equation("trace(grad(u)) + tau_p = 0 ")
problem.add_equation("dt(v)-(1/Rey)*div(grad(v))=-u@grad(v)")
problem.add_equation("integ(p) = 0")

###### Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
snapshots.add_task(u@ex, name='ux')
snapshots.add_task(u@ez, name='uz')
snapshots.add_task(v, name='uy')
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=0.01, cadence=100, safety=0.1, threshold=0.01,
             max_change=1.1, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((np.sqrt(u@u))*Rey, name='Re')

###### Main loop
startup_iter = 10
t1 = 0.01
state = 1
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()