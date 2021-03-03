"""
USING SCIPY TO DRIVE MULTI-REACTOR CANTERA PROBLEM

Two reactors connected with a piston, with heat loss to the environment

This script simulates the following situation. A closed cylinder with volume 2
m^3 is divided into two equal parts by a massless piston that moves with speed
proportional to the pressure difference between the two sides.  It is
initially held in place in the middle. One side is filled with 1000 K argon at
20 atm, and the other with a combustible 500 K methane/air mixture at 0.1 atm
(phi = 1.1). At t = 0 the piston is released and begins to move due to the
large pressure difference, compressing and heating the methane/air mixture,
which eventually explodes. At the same time, the argon cools as it expands.
The piston is adiabatic, but some heat is lost through the outer cylinder
walls to the environment.

Note that this simulation, being zero-dimensional, takes no account of shock
wave propagation. It is somewhat artifical, but nevertheless instructive.
"""

import numpy as np
import scipy.integrate

import cantera as ct
from reactor_system import ReactorSystemOde, ReactorSystemPyroOde

#-----------------------------------------------------------------------
# First create each gas needed, and a reactor or reservoir for each one.
#-----------------------------------------------------------------------

# create an argon gas object and set its state
ar = ct.Solution('argon.xml')
ar.TP = 1000.0, 20.0 * ct.one_atm

# create a reactor to represent the side of the cylinder filled with argon
r1 = ct.IdealGasReactor(ar)

# create a reservoir for the environment, and fill it with air.
env = ct.Reservoir(ct.Solution('air.xml'))

# use GRI-Mech 3.0 for the methane/air mixture, and set its initial state
gas = ct.Solution('gri30.xml')
gas.TP = 500.0, 0.2 * ct.one_atm
gas.set_equivalence_ratio(1.1, 'CH4:1.0', 'O2:2, N2:7.52')

# create a reactor for the methane/air side
r2 = ct.IdealGasReactor(gas)

print(gas())
print(ar())
#-----------------------------------------------------------------------------
# Now couple the reactors by defining common walls that may move (a piston) or
# conduct heat
#-----------------------------------------------------------------------------

# Given our re-implementation of the governing equations in reactor_system,
# this is currently just here for reference

# add a flexible wall (a piston) between r2 and r1
w = ct.Wall(r2, r1, A=1.0, K=0.5e-4, U=100.0)

# heat loss to the environment. Heat loss always occur through walls, so we
# create a wall separating r1 from the environment, give it a non-zero area,
# and specify the overall heat transfer coefficient through the wall.
w2 = ct.Wall(r2, env, A=1.0, U=500.0)

sim = ct.ReactorNet([r1, r2])

# Now the problem is set up, and we're ready to solve it.
print('finished setup, begin solution...')

time = 0.0
n_steps = 300

# Initialize reactor states for Leap.
state = ar.T
state = np.append(state, 1.0)
state = np.append(state, ar.Y)
state = np.append(state, gas.T)
state = np.append(state, 1.0)
state = np.append(state, gas.Y)

# Modify mass fractions in the gas so that none are zero.
sub_sum = 0
for i in range(0, 53):
    if state[5+i] > 0:
        state[5+i] -= 1e-8
        sub_sum += 1e-8

sub = sub_sum/50
for i in range(0, 53):
    if state[5+i] == 0:
        state[5+i] += sub

rtol = 1e-8
atol = 1e-8

ode = ReactorSystemOde(ar, gas, normalize=False)
#ode = ReactorSystemPyroOde(ar, gas)

atol_vec = np.zeros(gas.n_species+5)
atol_vec[0] = 1e-6
atol_vec[1] = 1e-6
atol_vec[2] = 1e-8
atol_vec[3] = 1e-6
atol_vec[4] = 1e-6
for i in range(5, gas.n_species+5):
    atol_vec[i] = 1e-8

rtol_vec = 1e-6

t = time
dt = 4.e-3
final_t = 4.e-4*n_steps
#final_t = 0.02

# New interface
solver = scipy.integrate.BDF(ode, 0, state, final_t, rtol=rtol, atol=atol)
#solver = scipy.integrate.BDF(ode, 0, state, final_t)
#solver = scipy.integrate.RK45(ode, 0, state, final_t, rtol=rtol, atol=atol)
#solver = scipy.integrate.RK23(ode, 0, state, final_t, rtol=rtol, atol=atol)
#solver = scipy.integrate.RK23(ode, 0, state, final_t, first_step=1e-5)
#solver = scipy.integrate.RK23(ode, 0, state, final_t, atol=atol_vec, rtol=rtol_vec)

# Need to obtain the pressures and temperatures for both
# the fast and slow reactors.

# Slow reactor
slow_temps = []
slow_press = []
slow_vol = []
slow_mass = []
slow_energy = []
fast_temps = []
fast_press = []
fast_vol = []
fast_mass = []
fast_energy = []
fast_species = []
fast_neg_sum = []
fast_neg_num = []
for j in range(0, gas.n_species):
    fast_species.append([])
times = []
steps = []
ar_mass = ar.density
gas_mass = gas.density

# New interface
while solver.status == 'running' and solver.t < final_t:
    solver.step()
    ar.TDY = solver.y[0], ar_mass/solver.y[1], solver.y[2]
    slow_vol.append(solver.y[1])
    slow_temps.append(ar.T)
    slow_mass.append(solver.y[1]*ar.density)
    slow_energy.append(ar.u)
    slow_press.append(ar.P/1e5)
    gas.TDY = solver.y[3], gas_mass/solver.y[4], solver.y[5:]
    frac_sum = 0
    num = 0
    for j in range(0, gas.n_species):
        fast_species[j].append(solver.y[5+j])
        if solver.y[5+j] < 0:
            frac_sum -= solver.y[5+j]
            num += 1
    fast_neg_sum.append(frac_sum)
    fast_neg_num.append(num)
    fast_vol.append(solver.y[4])
    fast_temps.append(gas.T)
    fast_press.append(gas.P/1e5)
    fast_mass.append(solver.y[4]*gas.density)
    fast_energy.append(gas.u)
    times.append(solver.t)
    steps.append(solver.t - solver.t_old)
    print("Time: ", solver.t)
    print("Timestep: ", solver.t - solver.t_old)

import matplotlib.pyplot as plt
plt.plot(times, np.log10(abs(np.array(fast_neg_sum))))
plt.title("Log of Sum of Negative Mass Fractions")
plt.xlabel("t")
plt.ylabel("log10(sum)")
plt.show()

plt.clf()
plt.plot(times, fast_neg_num)
plt.title("Number of Negative Mass Fractions")
plt.xlabel("t")
plt.ylabel("Number of Mass Fractions")
plt.show()

plt.clf()
plt.subplot(2, 2, 1)
h = plt.plot(times, slow_temps, 'g-', times, fast_temps, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')

plt.subplot(2, 2, 2)
plt.plot(times, slow_press, 'g-', times, fast_press, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Bar)')

plt.subplot(2, 2, 3)
plt.plot(times, slow_vol, 'g-', times, fast_vol, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.ylabel('Volume (m$^3$)')

plt.subplot(2, 2, 4)
plt.plot(times, slow_energy, 'g-', times, fast_energy, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.ylabel('Internal Energy (J)')

plt.figlegend(h, ['Reactor 1', 'Reactor 2'], loc='lower right')
plt.tight_layout()
plt.show()

plt.plot(times, steps)
plt.title("Timestep Sizes")
plt.xlabel("t")
plt.ylabel("dt")
plt.show()

plt.plot(times, np.log10(steps))
plt.title("Log Timestep Sizes")
plt.xlabel("t")
plt.ylabel("log10(dt)")
plt.show()

plt.clf()
plt.plot(times, np.log10(abs(np.array(fast_species[12]))))
plt.title("Mass Fraction Evolution, Species 12")
plt.xlabel('Time (s)')
plt.ylabel('Y')
plt.show()

# Plot each mass fraction in the gas mixture.
#for i in range(0, gas.n_species):
#    plt.clf()
#    plt.plot(times, fast_species[i])
#    plt.title("Mass Fraction Evolution, Species {j}".format(j=i))
#    plt.xlabel('Time (s)')
#    plt.ylabel('Y')
#    plt.savefig("MASS_FRACTIONS/species_{j}.png".format(j=i))
