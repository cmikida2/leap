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

from leap.rk import ODE23MethodBuilder, ODE45MethodBuilder
from leap.multistep import EmbeddedAdamsMethodBuilder

rtol = 1e-8
atol = 1e-8
single_order = False

methods = []

methods.append(ODE23MethodBuilder("y", rtol=rtol, atol=atol, use_high_order=True))
methods.append(EmbeddedAdamsMethodBuilder("y", order=2, use_high_order=True,
                                    static_dt=False, atol=atol, rtol=rtol))

method_times = []
method_step_sizes = []
method_slow_temps = []
method_slow_press = []
method_slow_vol = []
method_slow_energy = []
method_fast_temps = []
method_fast_press = []
method_fast_vol = []
method_fast_energy = []
method_fast_species = []
method_errs = []

for method in methods:
    code = method.generate()

    from dagrt.codegen import PythonCodeGenerator
    codegen = PythonCodeGenerator(class_name="Method")

    stepper_cls = codegen.get_class(code)

    # Initialize reactor states for Leap.
    ar.TP = 1000.0, 20.0 * ct.one_atm
    gas.TP = 500.0, 0.2 * ct.one_atm
    gas.set_equivalence_ratio(1.1, 'CH4:1.0', 'O2:2, N2:7.52')
    state = ar.T
    state = np.append(state, 1.0)
    state = np.append(state, ar.Y)
    state = np.append(state, gas.T)
    state = np.append(state, 1.0)
    state = np.append(state, gas.Y)
    ode = ReactorSystemOde(ar, gas)

    t = time
    dt = 4.e-3
    #final_t = 4.e-4*n_steps
    final_t = 0.01

    # The hard part - insert functions from Cantera's
    # kinetics objects for each reactor?
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

    method_times.append(np.array(times))
    method_step_sizes.append(np.array(step_sizes))
    method_errs.append(np.array(errs))

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

    method_slow_temps.append(slow_temps)
    method_slow_press.append(slow_press)
    method_slow_vol.append(slow_vol)
    method_slow_energy.append(slow_energy)

    # Fast reactor
    fast_temps = []
    fast_press = []
    fast_vol = []
    fast_energy = []
    fast_species = []
    for j in range(0, gas.n_species):
        fast_species.append([])
    for i in range(0, len(times)):
        gas.TDY = values[i][3], gas_mass/values[i][4], values[i][5:]
        fast_temps.append(gas.T)
        fast_press.append(gas.P/1e5)
        fast_vol.append(values[i][4])
        fast_energy.append(gas.u)
        species_temp = gas.Y
        for j in range(0, gas.n_species):
            fast_species[j].append(species_temp[j])

    method_fast_temps.append(fast_temps)
    method_fast_press.append(fast_press)
    method_fast_vol.append(fast_vol)
    method_fast_energy.append(fast_energy)
    method_fast_species.append(fast_species)

# CONSTRUCT PLOTS THAT SHOW BOTH METHODS...
import matplotlib.pyplot as plt
plt.clf()
plt.subplot(2, 2, 1)
h = plt.plot(method_times[0], method_slow_temps[0], 'g-',
             method_times[0], method_fast_temps[0], 'b-',
             method_times[1], method_slow_temps[1], 'r-',
             method_times[1], method_fast_temps[1], 'k-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Temperature (K)')
plt.ylim([500, 3000])

plt.subplot(2, 2, 2)
h = plt.plot(method_times[0], method_slow_press[0], 'g-',
             method_times[0], method_fast_press[0], 'b-',
             method_times[1], method_slow_press[1], 'r-',
             method_times[1], method_fast_press[1], 'k-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Pressure (Bar)')
plt.ylim([0, 21])

plt.subplot(2, 2, 3)
h = plt.plot(method_times[0], method_slow_vol[0], 'g-',
             method_times[0], method_fast_vol[0], 'b-',
             method_times[1], method_slow_vol[1], 'r-',
             method_times[1], method_fast_vol[1], 'k-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Volume (m$^3$)')
plt.ylim([0, 2])

plt.subplot(2, 2, 4)
h = plt.plot(method_times[0], method_slow_energy[0], 'g-',
             method_times[0], method_fast_energy[0], 'b-',
             method_times[1], method_slow_energy[1], 'r-',
             method_times[1], method_fast_energy[1], 'k-')
#plt.legend(['Reactor 1','Reactor 2'], 2)
plt.xlabel('Time (s)')
plt.xlim([-0.01, 4.e-4*n_steps])
plt.ylabel('Internal Energy (J)')
plt.ylim([-900000, 500000])

plt.figlegend(h, ['Reactor 1 - ODE23', 'Reactor 2 - ODE23',
                  'Reactor 1 - Adams', 'Reactor 2 - Adams'],
                  loc='lower right')
plt.tight_layout()
plt.show()

plt.clf()
plt.plot(method_times[0], np.log10(method_step_sizes[0]),
         method_times[1], np.log10(method_step_sizes[1]))
plt.title("Log of Timestep Sizes")
plt.xlabel("t")
plt.ylabel("dt")
plt.legend(['ODE23', 'Adams'])
plt.tight_layout()
plt.show()

plt.clf()
plt.plot(method_times[0], method_errs[0], method_times[1], method_errs[1])
plt.title("Weighted Error Norm Evolution")
plt.xlabel("t")
plt.ylabel("E")
plt.legend(['ODE23', 'Adams'])
plt.tight_layout()
plt.show()

# Plot each mass fraction in the gas mixture.
#for i in range(0, gas.n_species):
#    plt.clf()
#    plt.plot(method_times[0], np.log10(abs(np.array(method_fast_species[0][i]))),
#             method_times[1], np.log10(abs(np.array(method_fast_species[1][i]))))
#    ax = plt.gca()
#    ax.get_yaxis().get_major_formatter().set_useOffset(False)
#    plt.title("Mass Fraction Evolution, Species {j}".format(j=i))
#    plt.xlabel('Time (s)')
#    plt.ylabel('Y')
#    plt.legend(['ODE23', 'Adams'])
#    plt.tight_layout()
#    plt.savefig("COMBO/MASS_FRAC/species_{j}.png".format(j=i))
