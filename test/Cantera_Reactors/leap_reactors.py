"""
USING LEAP (SINGLE-RATE EXPLICIT ADAPTIVE) TO DRIVE MULTI-REACTOR CANTERA PROBLEM

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
from reactor_system import ReactorSystemOde

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

#print(gas())
#print(ar())

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
#print('finished setup, begin solution...')

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

# Test state.
#state = [ 6.50363155e+02,  1.90658730e+00,  1.00000000e+00,  1.02213305e+03,
#  9.34127017e-02,  8.20767705e-10,  1.34216995e-13,  3.34576620e-12,
#  2.18932773e-01,  1.42326772e-11,  8.45965579e-08,  7.83184265e-08,
#  2.11283921e-08, -9.64156916e-28,  7.77029520e-26,  2.58693555e-17,
#  3.30277347e-18,  6.39027289e-08,  6.03720158e-02,  8.72470858e-09,
#  3.08378478e-09,  3.14880273e-15,  1.14695040e-07,  5.98434803e-18,
#  6.07246254e-12,  1.10522776e-09,  1.78855838e-22,  6.05246550e-10,
#  8.06504739e-19,  2.03213786e-09,  4.52480733e-14,  5.65993049e-09,
#  1.83505603e-19,  6.00003267e-10,  5.99986939e-10,  1.91652267e-15,
#  7.88979494e-18,  5.99337354e-10,  6.00069955e-10,  7.60727333e-18,
#  4.46798446e-09,  7.71390197e-10,  6.08319949e-10,  2.51231024e-16,
#  2.97317637e-22,  1.16929463e-09,  9.83970037e-12,  9.41205322e-27,
#  5.99997446e-10,  5.99998176e-10,  6.00119597e-10,  2.61498780e-10,
#  7.20694813e-01,  6.00000000e-10, -2.17281060e-11,  6.01186836e-10,
#  3.36815796e-17,  5.99193637e-10]

from leap.rk import (ODE23MethodBuilder, ODE45MethodBuilder,
                    SSP43MethodBuilder, ForwardEulerMethodBuilder)
from leap.multistep import EmbeddedAdamsMethodBuilder

rtol = 1e-8
atol = 1e-8
single_order = False

method = ODE23MethodBuilder("y", rtol=rtol, atol=atol, use_high_order=True)
#method = ODE23MethodBuilder("y", rtol=rtol, atol=atol, use_high_order=True,
#                            max_dt_growth=1.01, min_dt_shrinkage=0.2)
#method = ODE23MethodBuilder("y")
#method = ODE23MethodBuilder("y", atol=atol)
#method = ODE23MethodBuilder("y", rtol=rtol, use_high_order=True)
#method = ODE45MethodBuilder("y", rtol=rtol, atol=atol, use_high_order=True)

#method = EmbeddedAdamsMethodBuilder("y", order=2, use_high_order=True,
#                                    static_dt=False, atol=atol, rtol=rtol)

#method = SSP43MethodBuilder("y", rtol=rtol, atol=atol, use_high_order=True)
#method = SSP43MethodBuilder("y", rtol=rtol, use_high_order=True)
#method = ForwardEulerMethodBuilder("y")

code = method.generate()

from dagrt.codegen import PythonCodeGenerator
codegen = PythonCodeGenerator(class_name="Method")

stepper_cls = codegen.get_class(code)

ode = ReactorSystemOde(ar, gas, normalize=False)

t = time
dt = 4.e-5
#final_t = 4.e-4*n_steps
final_t = 0.00015

# The hard part - insert functions from Cantera's kinetics objects for each reactor?
stepper = stepper_cls(
        function_map={
            "<func>y": lambda t, y: ode(t, y),
            })

stepper.set_up(
        t_start=t, dt_start=dt,
        context={
            "y": state,
            })

times = []
values = []
errs = []
rhs = []
new_times = []
new_err_times = []
new_rhs_times = []
fuzzed_rhs_times = []
unfuzzed_rhs_times = []
fuzzed_rhs_values = []
unfuzzed_rhs_values = []
cond_values = []
cond_times = []
rel_cond_values = []
new_values = []
new_rhs_values = []
new_errs = []
new_fail_times = []
new_fail = []
fail = []
step_sizes = []
last_t = 0
istep = 0

for event in stepper.run(t_end=final_t):
    if isinstance(event, stepper_cls.StateComputed):
        assert event.component_id == "y"
        if event.time_id == "y_err":
            new_err_times.append(event.t)
            new_errs.append(event.state_component)
        elif event.time_id == "rhs":
            new_rhs_times.append(event.t)
            new_rhs_values.append(event.state_component)
        elif event.component_id == "y":
            new_times.append(event.t)
            new_values.append(event.state_component)
    elif isinstance(event, stepper_cls.StepCompleted):
        if not new_times:
            continue

        print("Step completed: t = ", event.t)
        print("Step completed: dt = ", event.t - last_t)
        step_sizes.append(event.t - last_t)
        istep += 1
        last_t = event.t
        times.extend(new_times)
        values.extend(new_values)
        errs.extend(new_errs)
        rhs.extend(new_rhs_values)
        del new_times[:]
        del new_values[:]
        del new_rhs_times[:]
        del new_rhs_values[:]
        del new_err_times[:]
        del new_errs[:]
    elif isinstance(event, stepper_cls.StepFailed):
        print("Step failed")
        fail.extend(new_fail)
        del new_times[:]
        del new_values[:]
        del new_err_times[:]
        del new_errs[:]
        del new_fail[:]
        del new_rhs_times[:]
        del new_rhs_values[:]

times = np.array(times)

# Need to obtain the pressures and temperatures for both
# the fast and slow reactors.

# Slow reactor
slow_temps = []
slow_press = []
slow_vol = []
slow_energy = []
for i in range(0, len(times)):
    ar.TDY = values[i][0], ar_mass/values[i][1], values[i][2]
    slow_temps.append(ar.T)
    slow_press.append(ar.P/1e5)
    slow_vol.append(values[i][1])
    slow_energy.append(ar.u)

# Fast reactor
fast_temps = []
fast_press = []
fast_vol = []
fast_energy = []
fast_species = []
fast_neg = []
fast_num_neg = []
for j in range(0, gas.n_species):
    fast_species.append([])
for i in range(0, len(times)):
    gas.TDY = values[i][3], gas_mass/values[i][4], values[i][5:]
    fast_temps.append(gas.T)
    fast_press.append(gas.P/1e5)
    fast_vol.append(values[i][4])
    fast_energy.append(gas.u)
    species_temp = gas.Y
    fast_neg.append(0.0)
    fast_num_neg.append(0.0)
    for j in range(0, gas.n_species):
        fast_species[j].append(species_temp[j])
        # Track both magnitude and number of negative species
        if values[i][5+j] < 0:
            fast_neg[-1] -= values[i][5+j]
            fast_num_neg[-1] += 1

step_sizes = np.array(step_sizes)

import matplotlib.pyplot as plt
plt.clf()
plt.subplot(2, 2, 1)
h = plt.plot(times, slow_temps, 'g-', times, fast_temps, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Temperature (K)')
plt.ylim([500, 3000])

plt.subplot(2, 2, 2)
plt.plot(times, slow_press, 'g-', times, fast_press, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Pressure (Bar)')
plt.ylim([0, 21])

plt.subplot(2, 2, 3)
plt.plot(times, slow_vol, 'g-', times, fast_vol, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Volume (m$^3$)')
plt.ylim([0, 2])

plt.subplot(2, 2, 4)
plt.plot(times, slow_energy, 'g-', times, fast_energy, 'b-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Internal Energy (J)')
plt.ylim([-900000, 500000])

plt.figlegend(h, ['Reactor 1', 'Reactor 2'], loc='lower right')
plt.tight_layout()
plt.show()

#plt.plot(times, step_sizes)
#plt.title("Timestep Sizes")
#plt.xlabel("t")
#plt.ylabel("dt")
#plt.show()

plt.plot(times, fast_num_neg)
plt.title("Number of Negative Mass Fractions")
plt.xlabel("t")
plt.ylabel("Number of Negative Mass Fractions")
plt.show()

plt.plot(times, np.log10(fast_neg))
plt.title("Log of Sum of Negative Mass Fractions")
plt.xlabel("t")
plt.ylabel("Log of Sum of Negative Mass Fractions")
plt.show()

plt.plot(times, np.log10(step_sizes))
plt.title("Log of Timestep Sizes")
plt.xlabel("t")
plt.ylabel("dt")
plt.show()

plt.plot(times, errs)
plt.title("Weighted Error Norm Evolution")
plt.xlabel("t")
plt.ylabel("E")
plt.show()

# Plot each mass fraction in the gas mixture.
#for i in range(0, gas.n_species):
#    plt.clf()
#    plt.plot(times, np.log10(abs(np.array(fast_species[i]))))
#    plt.title("Mass Fraction Evolution, Species {j}".format(j=i))
#    plt.xlabel('Time (s)')
#    plt.ylabel('Y')
#    plt.savefig("MP_MASS_FRACTIONS/species_{j}.png".format(j=i))
