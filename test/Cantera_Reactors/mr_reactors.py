"""
TRY USING LEAP TO DRIVE MULTI-REACTOR CANTERA PROBLEM

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

import cantera as ct
from reactor_system import Reactor1Ode, Reactor2Ode

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

ar_mass = ar.density
gas_mass = gas.density
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
states1 = ar.T
states1 = np.append(states1, 1.0)
states1 = np.append(states1, ar.Y)
states2 = gas.T
states2 = np.append(states2, 1.0)
states2 = np.append(states2, gas.Y)

# Now, try to set up a multi-rate adaptive Leap integrator
# for this problem.
from leap.multistep.multirate import (
        MultiRateHistory as MRHistory,
        EmbeddedMultiRateMultiStepMethodBuilder)

order = 2
step_ratio = 1
rtol = 1e-8
atol = 1e-8

method = EmbeddedMultiRateMultiStepMethodBuilder(
            order,
            (
                (
                    "dt", "fast", "=",
                    MRHistory(1, "<func>f", ("fast", "slow",)),
                    ),
                (
                    "dt", "slow", "=",
                    MRHistory(step_ratio, "<func>s", ("fast", "slow")),
                    ),
                ),
                static_dt=False,
                atol=atol,
                rtol=rtol,
                max_dt_growth=5)

code = method.generate()

from dagrt.codegen import PythonCodeGenerator
codegen = PythonCodeGenerator(class_name="Method")

stepper_cls = codegen.get_class(code)

ode1 = Reactor1Ode(ar, gas)
ode2 = Reactor2Ode(ar, gas)

t = time
#dt = 2**(-9)
dt = 4.e-4
#final_t = 4.e-4*n_steps
final_t = 0.04

# The hard part - insert functions from Cantera's kinetics objects for each reactor?
stepper = stepper_cls(
        function_map={
            "<func>f": lambda t, fast, slow: ode2(t, slow, fast),
            "<func>s": lambda t, fast, slow: ode1(t, slow, fast),
            })

stepper.set_up(
        t_start=t, dt_start=dt,
        context={
            "fast": states2,
            "slow": states1,
            })

f_times = []
f_values = []
s_times = []
s_values = []
fast_times = []
fast_values = []
slow_times = []
slow_values = []
step_sizes = []
step_factors = []
last_t = 0
istep = 0

for event in stepper.run(t_end=final_t):
    if isinstance(event, stepper_cls.StateComputed):
        if event.time_id == "slow_err":
            pass
        elif event.time_id == "fast_err":
            pass
        elif event.component_id == "fast":
            #print("Reactor 2 Volume: ", event.state_component[1])
            f_times.append(event.t)
            f_values.append(event.state_component)
        elif event.component_id == "slow":
            #print("Reactor 1 Volume: ", event.state_component[1])
            s_times.append(event.t)
            s_values.append(event.state_component)
        else:
            assert False, event.component_id
    elif isinstance(event, stepper_cls.StepCompleted):
        print("Time: ", event.t)
        if not f_times:
            continue

        # Account for bootstrapping.
        if istep < 2:
            for i in range(0, step_ratio):
                step_sizes.append((event.t - last_t)/step_ratio)
        else:
            step_sizes.append(event.t - last_t)
        print("Timestep: ", event.t - last_t)
        if istep > 1:
            print("Timestep Change Factor: ", (event.t - last_t)/step_sizes[-2])
            step_factors.append((event.t - last_t)/step_sizes[-2])
        else:
            step_factors.append(1)
        istep += 1
        last_t = event.t
        slow_times.extend(s_times)
        fast_times.extend(f_times)
        slow_values.extend(s_values)
        fast_values.extend(f_values)
        del s_times[:]
        del f_times[:]
        del s_values[:]
        del f_values[:]
    elif isinstance(event, stepper_cls.StepFailed):
        print("Step failed at t = ", event.t)
        del s_times[:]
        del f_times[:]
        del s_values[:]
        del f_values[:]

fast_times = np.array(fast_times)
slow_times = np.array(slow_times)

# Need to obtain the pressures and temperatures for both
# the fast and slow reactors.

# Slow reactor
slow_temps = []
slow_press = []
slow_vol = []
slow_energy = []
for i in range(0, len(slow_times)):
    ar.TDY = slow_values[i][0], ar_mass/slow_values[i][1], slow_values[i][2:]
    slow_temps.append(ar.T)
    slow_press.append(ar.P/1e5)
    slow_energy.append(ar.u)
    slow_vol.append(slow_values[i][1])

# Fast reactor
fast_temps = []
fast_press = []
fast_vol = []
fast_energy = []
fast_species = []
for j in range(0, gas.n_species):
    fast_species.append([])
for i in range(0, len(slow_times)):
    gas.TDY = fast_values[i][0], gas_mass/fast_values[i][1], fast_values[i][2:]
    species_temp = gas.Y
    for j in range(0, gas.n_species):
        fast_species[j].append(species_temp[j])
    fast_temps.append(gas.T)
    fast_press.append(gas.P/1e5)
    fast_energy.append(gas.u)
    fast_vol.append(fast_values[i][1])

step_sizes = np.array(step_sizes)

import matplotlib.pyplot as plt
plt.clf()
plt.subplot(2, 2, 1)
h = plt.plot(slow_times, slow_temps, 'g-', fast_times, fast_temps, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Temperature (K)')
plt.ylim([500, 3000])

plt.subplot(2, 2, 2)
plt.plot(slow_times, slow_press, 'g-', fast_times, fast_press, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Pressure (Bar)')
plt.ylim([0, 21])

plt.subplot(2, 2, 3)
plt.plot(slow_times, slow_vol, 'g-', fast_times, fast_vol, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Volume (m$^3$)')
plt.ylim([0, 2])

plt.subplot(2, 2, 4)
plt.plot(slow_times, slow_energy, 'g-', fast_times, fast_energy, 'b-')
#plt.legend(['Reactor 1','Reactor 2'],2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Internal Energy (J)')
plt.ylim([-900000, 500000])

plt.figlegend(h, ['Reactor 1', 'Reactor 2'], loc='lower right')
plt.tight_layout()
plt.show()

plt.plot(slow_times, np.log10(step_sizes))
plt.title("Log of Timestep Sizes")
plt.xlabel("t")
plt.ylabel("dt")
plt.show()

# Plot each mass fraction in the gas mixture.
#for i in range(0, gas.n_species):
#    plt.clf()
#    plt.plot(fast_times, fast_species[i])
#    plt.title("Mass Fraction Evolution, Species {j}".format(j=i))
#    plt.xlabel('Time (s)')
#    plt.ylabel('Y')
#    plt.savefig("LEAP_MASS_FRACTIONS/species_{j}.png".format(j=i))

plt.plot(slow_times, step_factors)
plt.title("Timestep Change Factor")
plt.xlabel("t")
plt.ylabel("dt_fac")
plt.show()
