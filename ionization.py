# ionization.py

import numpy as np
from scipy.interpolate import interp1d

class IonizationTable:
    """
    Interpolates ionization fractions from ionization_table.dat
    Obtained using atomdb
    Columns: logT, ion_fraction_HII, ion_fraction_OIII
    """

    def __init__(self, filename):

        data = np.loadtxt(filename)

        logT = data[:,0]

        self.T = 10**logT

        self.HII = data[:,1]
        self.OIII = data[:,2]

        # Interpolators
        self.HII_interp = interp1d(
            self.T,
            self.HII,
            bounds_error=False,
            fill_value=(self.HII[0], self.HII[-1])
        )

        self.OIII_interp = interp1d(
            self.T,
            self.OIII,
            bounds_error=False,
            fill_value=(self.OIII[0], self.OIII[-1])
        )

    def fractions(self, T):

        ion_H = self.HII_interp(T)
        ion_O = self.OIII_interp(T)

        return ion_H, ion_O