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
Ro = 0.01
Rey = 3000
Nfc = 10
seed = 42

#Settings of solver
scale = 0.0005              #the value of max_timestep
dealias = 3/2
stop_sim_time = 1
timestepper = d3.RK443
# max_timestep = scale
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
kf = 2*np.pi*16
kfw = 2*np.pi*1
eta = 1

Fwx = dist.Field(name='Fwx', bases=(xbasis, zbasis))
Fwz = dist.Field(name='Fwz', bases=(xbasis, zbasis))
kx = xbasis.wavenumbers[dist.local_modes(xbasis)]
kz = zbasis.wavenumbers[dist.local_modes(zbasis)]
dkx = dky = 2 * np.pi / Lx
k = (kx**2 + kz**2)**0.5

# print(kx)
# print(kz)

# Forcing function
rand = np.random.RandomState(seed)

def draw_gaussian_random_field():
    """Create Gaussian random field concentrating on a ring in Fourier space with unit variance."""
    # 1D power spectrum: normalized Gaussian, no mean
    P1 = np.exp(-(k-kf)**2/2/kfw**2) / np.sqrt(kfw**2 * np.pi / 2) * (k != 0)
    # 2D power spectrum: divide by polar Jacobian
    P2 = P1 / 2 / np.pi / (k + (k==0))
    # 2D coefficient poewr spectrum: divide by mode power
    Pc = P2 / 2**((kx == 0).astype(float) + (kz == 0).astype(float) - 2)
    # Forcing amplitude, including division between sine and cosine
    f_amp = (Pc / 2 * dkx * dkx)**0.5
    # Forcing with random phase
    f = f_amp * rand.randn(*k.shape)
    return f

def set_vorticity_forcing(timestep):
    """Set vorticity forcing field from scaled Gaussian random field."""
    # Set forcing to normalized Gaussian random field
    Fwx['c'] = draw_gaussian_random_field()*(kz)/(k**2+(k==0))
    Fwz['c'] = draw_gaussian_random_field()*(-kx)/(k**2+(k==0))
    # Rescale by forcing rate, including factor for 1/2 in kinetic energy
    Fwx['c'] *= (2 * eta / timestep)**0.5
    Fwz['c'] *= (2 * eta / timestep)**0.5


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
problem = d3.IVP([u, v, b, p, tau_p], time=t, namespace=locals())
problem.add_equation("dt(u)-(2/Ro)*v*ex+grad(p)-(1/Rey)*div(grad(u))-b*ez=\
    -u@grad(u)+Fwx*sig*ex+Fwz*sig*ez")
problem.add_equation("trace(grad(u)) + tau_p = 0 ")
problem.add_equation("dt(v)+(2/Ro)*(u@ex)-(1/Rey)*div(grad(v))=-u@grad(v)")
problem.add_equation("dt(b)+(Nfc)*(Nfc)*(u@ez)=-u@grad(b)")
problem.add_equation("integ(p) = 0")

# # problem = d3.IVP([u, v, b, p, tau_p], time=t, namespace=locals())
# # problem.add_equation("dt(u)+grad(p)-(1/Rey)*div(grad(u))-b*ez=-u@grad(u)+forcing*np.sin(100*t)*sig*ex+forcing*np.sin(100*t)*sig*ez")
# # problem.add_equation("trace(grad(u)) + tau_p = 0 ")
# # problem.add_equation("dt(v)-(1/Rey)*div(grad(v))=-u@grad(v)")
# # problem.add_equation("dt(b)+(Nfc)*(Nfc)*(u@ez)=-u@grad(b)")
# # problem.add_equation("integ(p) = 0")

# # problem = d3.IVP([u, v, p, tau_p], time=t, namespace=locals())
# # problem.add_equation("dt(u)+grad(p)-(1/Rey)*div(grad(u))=-u@grad(u)+forcing*ex+forcing*ez")
# # problem.add_equation("trace(grad(u)) + tau_p = 0 ")
# # problem.add_equation("dt(v)-(1/Rey)*div(grad(v))=-u@grad(v)")
# # problem.add_equation("integ(p) = 0")

###### Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.002, max_writes=50)
snapshots.add_task(u@ex, name='ux')
snapshots.add_task(u@ez, name='uz')
snapshots.add_task(v, name='uy')
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(p, name='pressure')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=scale, cadence=100, safety=0.1, threshold=0.01,
             max_change=1.1, min_change=0.5, max_dt=scale)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((np.sqrt(u@u))*Rey, name='Re')

###### Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        set_vorticity_forcing(timestep)
        solver.step(timestep)
            
        if (solver.iteration-1) % 100 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()