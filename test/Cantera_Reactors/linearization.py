import numpy as np

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
#gas = ct.Solution('gri30_mod.xml')
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

ode = ReactorSystemOde(ar, gas)
#ode = ReactorSystemPyroOde(ar, gas)

# Initialize reactor states for Leap.
ar.TP = 1000.0, 20.0 * ct.one_atm
gas.TP = 500.0, 0.2 * ct.one_atm
gas.set_equivalence_ratio(1.1, 'CH4:1.0', 'O2:2, N2:7.52')
state_0 = ar.T
state_0 = np.append(state_0, 1.0)
state_0 = np.append(state_0, ar.Y)
#state_0 = np.append(state, np.float128(gas.T))
state_0 = np.append(state_0, gas.T)
state_0 = np.append(state_0, 1.0)
#state_0 = np.append(state, np.float128(gas.Y))
state_0 = np.append(state_0, gas.Y)

# Modify mass fractions in the gas so that none are zero.
#sub_sum = 0
#for i in range(0, 53):
#    if state_0[5+i] > 0:
#        state_0[5+i] -= 1e-8
#        sub_sum += 1e-8

#sub = sub_sum/50
#for i in range(0, 53):
#    if state_0[5+i] == 0:
#        state_0[5+i] += sub

# test state.
state_0 = [6.50363155e+02,  1.90658730e+00,  1.00000000e+00,  1.02213305e+03,
  9.34127017e-02,  8.20767705e-10,  1.34216995e-13,  3.34576620e-12,
  2.18932773e-01,  1.42326772e-11,  8.45965579e-08,  7.83184265e-08,
  2.11283921e-08, -9.64156916e-28,  7.77029520e-26,  2.58693555e-17,
  3.30277347e-18,  6.39027289e-08,  6.03720158e-02,  8.72470858e-09,
  3.08378478e-09,  3.14880273e-15,  1.14695040e-07,  5.98434803e-18,
  6.07246254e-12,  1.10522776e-09,  1.78855838e-22,  6.05246550e-10,
  8.06504739e-19,  2.03213786e-09,  4.52480733e-14,  5.65993049e-09,
  1.83505603e-19,  6.00003267e-10,  5.99986939e-10,  1.91652267e-15,
  7.88979494e-18,  5.99337354e-10,  6.00069955e-10,  7.60727333e-18,
  4.46798446e-09,  7.71390197e-10,  6.08319949e-10,  2.51231024e-16,
  2.97317637e-22,  1.16929463e-09,  9.83970037e-12,  9.41205322e-27,
  5.99997446e-10,  5.99998176e-10,  6.00119597e-10,  2.61498780e-10,
  7.20694813e-01,  6.00000000e-10, -2.17281060e-11,  6.01186836e-10,
  3.36815796e-17,  5.99193637e-10]

t = time
base_rhs = ode(t, state_0)
rhs_op = np.zeros((58, 58))
#epses = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
epses = [1e-9]
max_wrs = []
from scipy.linalg.lapack import dgeev
import matplotlib.pyplot as plt
for eps in epses:
    for i in range(0, 58):
        # Calculate state perturbation.
        state = np.copy(state_0)
        state[i] += eps
        # Calculate RHS with new perturbed state.
        new_rhs = ode(t, state)
        pert_vec = (new_rhs - base_rhs)/eps
        # Obtain RHS operator element.
        for j in range(0, 58):
            rhs_op[j, i] = pert_vec[j]

    # Now, calculate the eigenvalues.
    wr, wi, vl, vr, info = dgeev(rhs_op)

    print("Largest Real Part in Eigenvalues: ", max(abs(wr)))

    max_wrs.append(wr[np.argmax(abs(wr))])
    plt.clf()
    plt.scatter(wr, wi, marker='*')
    plt.title("Normalized Cantera: Perturbed Initial Two-Reactor State")
    plt.show()

# A log-log plot of the largest real eigenvalues.
import textwrap
plt.clf()
plt.plot(np.log10(np.array(epses)), np.log10(-np.array(max_wrs)),
         linestyle='-', marker='o')
plt.title("\n".join(textwrap.wrap("Log-Log Plot w/Pyro, Clean State, "
                                   + "W/Normalization: Highest Magnitude "
                                   + "Negative Real Eigenvalue Sensitivity",
                                   80)))
plt.xlabel("log10(epsilon)")
plt.ylabel("log10(-max(wr))")
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.show()
