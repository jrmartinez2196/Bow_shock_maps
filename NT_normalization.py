# radiation.py
import numpy as np
from constants import eV

'''
	This modulce calculates the normalization of the non-thermal
	electrons and protons distributions.
	Valid for power law distributions
	n(E) propto k0 * E^{-p_inj}, with p_inj > 2
	[n(E)] = erg^{-1} cm^{-3}

	U_NT = int_Emin^Emax k0 * E * E^{-p_inj} dE
	U_NT = k0 * integral approx  k0 * Emin^{-p_inj+2} / (p_inj-2) (valid for p_inj > 2)
	-> k0 = U_NT / integral

	'''


def k0_e(U_NTe, p_inj=2.5, Emine=1e6*eV):
	'''
	Parameters:
	-----------
	U_NTe : float
		Electron energy density
	p_inj : float
		spectral index
	Emine : float
		Minimum energy

	Returns:
	--------
	k0_e : float
		Distribution normalization constant
	'''

	integral = Emine**(-p_inj+2.) / (p_inj-2.)
	k0 = U_NTe/integral

	return k0


def k0_p(U_NTp, p_inj=2.5, Eminp=1e9*eV):
	''''
	Parameters:
	-----------
	U_NTp : float
		Proton energy density
	p_inj : float
		spectral index
	Eminp : float
		Minimum energy

	Returns:
	--------
	k0_p : float
		Distribution normalization constant
	'''

	integral = Eminp**(-p_inj+2.) / (p_inj-2.)
	k0 = U_NTp/integral

	return k0