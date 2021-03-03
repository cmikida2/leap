import cantera as ct
import numpy as np
import pyrometheus as pyro


class ReactorSystemOde(object):
    def __init__(self, gas1, gas2, normalize=True):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas1 = gas1
        self.gas2 = gas2
        self.env = ct.Reservoir(ct.Solution('air.xml'))
        # Initial volume of each reactor is 1.0, so...
        self.gas1_mass = gas1.density
        self.gas2_mass = gas2.density
        self.normalize = normalize

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [U1, V1, Y_1, Y_2, ... Y_K, U2, V2, Y_1, .... Y_K]
        # Set gases.
        if self.normalize:
            self.gas1.TDY = y[0], self.gas1_mass * (1.0/y[1]), [y[2]]
        else:
            self.gas1.TD = y[0], self.gas1_mass/y[1]
            self.gas1.set_unnormalized_mass_fractions([y[2]])
        rho1 = self.gas1.density

        if self.normalize:
            self.gas2.TDY = y[3], self.gas2_mass * (1.0/y[4]), y[5:]
        else:
            self.gas2.TD = y[3], self.gas2_mass/y[4]
            self.gas2.set_unnormalized_mass_fractions(y[5:])
        rho2 = self.gas2.density

        # Volume rate of change (move piston based on pressure
        # difference between reactors)
        area = 1.0
        k = 0.5e-4
        u1 = 100.0
        dvdt_1 = k*area*(self.gas1.P - self.gas2.P)

        # Mass fraction rate of change (via production
        # rates as is typical)
        wdot_1 = self.gas1.net_production_rates
        dydt_1 = wdot_1 * self.gas1.molecular_weights * (1.0 / rho1)

        # Internal energy rate of change (the hard one)
        # Pressure work first.
        dtempdt_1 = -self.gas1.P*dvdt_1
        # Include heat transfer via the piston wall
        dtempdt_1 += -area*u1*(self.gas1.T - self.gas2.T)
        dtempdt_1 += -np.dot(self.gas1.partial_molar_int_energies,
                             wdot_1*self.gas1.v*self.gas1_mass)
        dtempdt_1 = dtempdt_1 * (1.0/(self.gas1_mass*self.gas1.cv_mass))

        # Volume rate of change (move piston based on pressure
        # difference between reactors)
        dvdt_2 = k*area*(self.gas2.P - self.gas1.P)

        # Mass fraction rate of change (via production
        # rates as is typical)
        wdot_2 = self.gas2.net_production_rates
        dydt_2 = wdot_2 * self.gas2.molecular_weights * (1.0 / rho2)

        # Internal energy rate of change (the hard one)
        # Pressure work first.
        dtempdt_2 = -self.gas2.P*dvdt_2
        # Include heat transfer via the piston wall
        dtempdt_2 += -area*u1*(self.gas2.T - self.gas1.T)
        # Include heat loss to the environment (air reservoir)
        # via specified wall
        area2 = 1.0
        u2 = 500.0
        dtempdt_2 += -area2*u2*(self.gas2.T - self.env.T)
        dtempdt_2 += -np.dot(self.gas2.partial_molar_int_energies,
                             wdot_2*self.gas2.v*self.gas2_mass)
        dtempdt_2 = dtempdt_2 * (1.0/(self.gas2_mass*self.gas2.cv_mass))

        return np.hstack((np.hstack((dtempdt_1, dvdt_1, dydt_1)),
                          np.hstack((dtempdt_2, dvdt_2, dydt_2))))


class ReactorSystemPyroOde(object):
    def __init__(self, gas1, gas2, normalize=True):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas1 = pyro.get_thermochem_class(gas1)()
        self.gas2 = pyro.get_thermochem_class(gas2)()
        self.gas2_ct = gas2
        self.env = ct.Reservoir(ct.Solution('air.xml'))
        # Initial volume of each reactor is 1.0, so...
        self.gas1_mass = gas1.density
        self.gas2_mass = gas2.density
        self.gas1_mw = gas1.molecular_weights
        self.gas2_mw = gas2.molecular_weights
        self.normalize = normalize

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [U1, V1, Y_1, Y_2, ... Y_K, U2, V2, Y_1, .... Y_K]
        # Set gases.
        temp1 = y[0]
        rho1 = self.gas1_mass * (1.0/y[1])
        if self.normalize:
            p1 = self.gas1.get_pressure(rho1, temp1, [y[2]])
        else:
            p1 = self.gas1.get_pressure(rho1, temp1, [y[2]])

        reac2_mf = np.copy(y[5:])

        temp2 = y[3]
        rho2 = self.gas2_mass * (1.0/y[4])
        if self.normalize:
            p2 = self.gas2.get_pressure(rho2, temp2, normalizer(reac2_mf))
        else:
            p2 = self.gas2.get_pressure(rho2, temp2, y[5:])

        # Volume rate of change (move piston based on pressure
        # difference between reactors)
        area = 1.0
        k = 0.5e-4
        u1 = 100.0
        dvdt_1 = k*area*(p1 - p2)

        # Mass fraction rate of change (via production
        # rates as is typical)
        #wdot_1 = self.gas1.get_net_production_rates(rho1, T1, [y[2]])
        wdot_1 = 0.0
        dydt_1 = wdot_1 * self.gas1_mw * (1.0 / rho1)

        # Internal energy rate of change (the hard one)
        # Pressure work first.
        dtempdt_1 = -p1*dvdt_1
        # Include heat transfer via the piston wall
        dtempdt_1 += -area*u1*(temp1 - temp2)
        # Need partial molar internal energies...
        e0_rt1 = (self.gas1.get_species_enthalpies_rt(temp1) - 1.0) * \
            self.gas1.gas_constant * temp1
        dtempdt_1 += -np.dot(e0_rt1, wdot_1*(1.0/rho1)*self.gas1_mass)
        dtempdt_1 = dtempdt_1 * (1.0
                / (self.gas1_mass
                    * self.gas1.get_mixture_specific_heat_cv_mass(temp1,
                        [y[2]])))

        # Volume rate of change (move piston based on pressure
        # difference between reactors)
        dvdt_2 = k*area*(p2 - p1)

        # Mass fraction rate of change (via production
        # rates as is typical)
        if self.normalize:
            wdot_2 = self.gas2.get_net_production_rates(rho2, temp2,
                                                        normalizer(reac2_mf))
        else:
            wdot_2 = self.gas2.get_net_production_rates(rho2, temp2, y[5:])
        dydt_2 = wdot_2 * self.gas2_mw * (1.0 / rho2)

        # Internal energy rate of change (the hard one)
        # Pressure work first.
        dtempdt_2 = -p2*dvdt_2
        # Include heat transfer via the piston wall
        dtempdt_2 += -area*u1*(temp2 - temp1)
        # Include heat loss to the environment (air reservoir)
        # via specified wall
        area2 = 1.0
        u2 = 500.0
        dtempdt_2 += -area2*u2*(temp2 - self.env.T)
        e0_rt2 = (self.gas2.get_species_enthalpies_rt(temp2) - 1.0) \
            * self.gas2.gas_constant * temp2
        dtempdt_2 += -np.dot(e0_rt2, wdot_2*(1.0/rho2)*self.gas2_mass)
        if self.normalize:
            dtempdt_2 = dtempdt_2 * (1.0
                    / (self.gas2_mass
                        * self.gas2.get_mixture_specific_heat_cv_mass(temp2,
                            normalizer(reac2_mf))))
        else:
            dtempdt_2 = dtempdt_2 * (1.0
                    / (self.gas2_mass
                        * self.gas2.get_mixture_specific_heat_cv_mass(temp2,
                            y[5:])))

        return np.hstack((np.hstack((dtempdt_1, dvdt_1, dydt_1)),
                          np.hstack((dtempdt_2, dvdt_2, dydt_2))))


class Reactor1Ode(object):
    def __init__(self, gas1, gas2, normalize=True):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas1 = gas1
        self.gas2 = gas2
        self.env = ct.Reservoir(ct.Solution('air.xml'))
        # Initial volume of each reactor is 1.0...
        self.gas1_mass = gas1.density
        self.gas2_mass = gas2.density
        self.normalize = normalize

    def __call__(self, t, y1, y2):
        """the ODE function, y' = f(t,y) """

        # State vector is [U, V, Y_1, Y_2, ... Y_K]
        # Set gases.
        if self.normalize:
            self.gas1.TDY = y1[0], self.gas1_mass/y1[1], y1[2:]
        else:
            self.gas1.TD = y1[0], self.gas1_mass/y1[1]
            self.gas1.set_unnormalized_mass_fractions([y1[2:]])
        rho1 = self.gas1.density

        if self.normalize:
            self.gas2.TDY = y2[0], self.gas2_mass/y2[1], y2[2:]
        else:
            self.gas2.TD = y2[0], self.gas2_mass/y2[1]
            self.gas2.set_unnormalized_mass_fractions([y2[2:]])

        # Volume rate of change (move piston based on pressure
        # difference between reactors)
        area = 1.0
        k = 0.5e-4
        u1 = 100.0
        dvdt = k*area*(self.gas1.P - self.gas2.P)

        # Mass fraction rate of change (via production
        # rates as is typical)
        wdot = self.gas1.net_production_rates
        dydt = wdot * self.gas1.molecular_weights / rho1

        # Internal energy rate of change (the hard one)
        # Pressure work first.
        dtempdt = -self.gas1.P*dvdt
        # Include heat transfer via the piston wall
        dtempdt += -area*u1*(self.gas1.T - self.gas2.T)
        dtempdt += -np.dot(self.gas1.partial_molar_int_energies,
                           wdot*self.gas1.v*self.gas1_mass)
        dtempdt = dtempdt/(self.gas1_mass*self.gas1.cv_mass)

        return np.hstack((dtempdt, dvdt, dydt))


class Reactor2Ode(object):
    def __init__(self, gas1, gas2, normalize=True):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas1 = gas1
        self.gas2 = gas2
        self.env = ct.Reservoir(ct.Solution('air.xml'))
        # Initial volume of each reactor is 1.0...
        self.gas1_mass = gas1.density
        self.gas2_mass = gas2.density
        self.normalize = normalize

    def __call__(self, t, y1, y2):
        """the ODE function, y' = f(t,y) """

        # State vector is [U, V, Y_1, Y_2, ... Y_K]
        # Set gases.
        if self.normalize:
            self.gas1.TDY = y1[0], self.gas1_mass/y1[1], y1[2:]
        else:
            self.gas1.TD = y1[0], self.gas1_mass/y1[1]
            self.gas1.set_unnormalized_mass_fractions([y1[2:]])

        if self.normalize:
            self.gas2.TDY = y2[0], self.gas2_mass/y2[1], y2[2:]
        else:
            self.gas2.TD = y2[0], self.gas2_mass/y2[1]
            self.gas2.set_unnormalized_mass_fractions([y2[2:]])
        rho2 = self.gas2.density

        # Volume rate of change (move piston based on pressure
        # difference between reactors)
        area = 1.0
        k = 0.5e-4
        u1 = 100.0
        dvdt = k*area*(self.gas2.P - self.gas1.P)

        # Mass fraction rate of change (via production
        # rates as is typical)
        wdot = self.gas2.net_production_rates
        dydt = wdot * self.gas2.molecular_weights / rho2

        # Internal energy rate of change (the hard one)
        # Pressure work first.
        dtempdt = -self.gas2.P*dvdt
        # Include heat transfer via the piston wall
        dtempdt += -area*u1*(self.gas2.T - self.gas1.T)
        # Include heat loss to the environment (air reservoir)
        # via specified wall
        area2 = 1.0
        u2 = 500.0
        dtempdt += -area2*u2*(self.gas2.T - self.env.T)
        dtempdt += -np.dot(self.gas2.partial_molar_int_energies,
                           wdot*self.gas2.v*self.gas2_mass)
        dtempdt = dtempdt/(self.gas2_mass*self.gas2.cv_mass)

        return np.hstack((dtempdt, dvdt, dydt))


def state_filter(y):

    # Trim the negative mass fractions, as
    # Cantera does when you provide it negative inputs.
    # For now, do this in about as hacky a way as possible.
    out_y = np.copy(y)
    count = 0
    max_mf = 0
    negative_sum = 0
    for i in range(0, 53):
        if y[5+i] < 0:
            if abs(y[5+i]) > max_mf:
                max_mf = abs(y[5+i])
            out_y[5+i] = 0.0
            negative_sum += y[5+i]
            count += 1

    # Now, we also must normalize to assume a sum to 1.
    old_sum = sum(out_y[5:])
    for i in range(0, 53):
        out_y[5+i] = out_y[5+i] * (1.0/old_sum)

    return out_y


def normalizer(y):

    # Trim the negative mass fractions, as
    # Cantera does when you provide it negative inputs.
    # For now, do this in about as hacky a way as possible.
    for i in range(0, 53):
        if y[i] < 0:
            y[i] = 0.0

    # Now, we also must normalize to assume a sum to 1.
    old_sum = sum(y)
    for i in range(0, 53):
        y[i] = y[i] * (1.0/old_sum)

    return y
