import numpy as np
from scipy.optimize import minimize
from .helpers import helper_get_alpha
# Constants
from scipy.constants import Boltzmann, Avogadro
kb = Boltzmann # [J/K] Boltzman's constant
Na = Avogadro  # [mol-1] Avogadro's Number
R = Na * kb    # [J mol-1 K-1] Ideal gas constant



def Tcad_CS(ms, alpha):
    """
    Calculate the critical temperature using the Galliero's CS correlation.
    """
    # Table 1
    A0, A1, A2, A3, A4, A5, A6, A7, A8 = -4.0219, -0.6969, 3.1689, -0.5822,  -0.3909, 1.4849, 0.7195, -3.1836, 0.2520

    # Eq. 10 
    sqrt_ms = np.sqrt(ms)
    Tcad = (A0 * np.exp(A1 * sqrt_ms) + A2) * alpha**(A3 * np.exp(A4 * ms) + A5)
    Tcad += (A6 * np.exp(A7 / sqrt_ms) + A8)
    return Tcad


def rhocad_CS(ms, alpha):
    """
    Calculate the saturated liquid density using the Galliero's CS correlation.
    """
    # Table 1
    B0, B1, B2, B3, B4, B5, B6, B7, B8 = 0.3198, -0.6815, 0.2532, -1.9621, -0.4389, -2.5033, 0.1568, -0.1756, 0.5972

    # Eq. 11
    rholad = (B0 * np.exp(B1 * ms) + B2) * np.exp( (B3 * np.exp(B4 * ms) + B5) * alpha**1.75 ) 
    rholad += (B6 * np.exp(B7 * ms) + B8 )
    rholad /= ms

    return rholad


def acentric_CS(ms, alpha):

    # Table 1
    C0, C1, C2, C3 = -0.1574, -0.0008, 0.3158, -0.4292

    # Eq. 12
    alpha_inf = 1/3
    sqrt_ms = np.sqrt(ms)
    acentric = (C0 * sqrt_ms + C1) * np.log(alpha - alpha_inf)
    acentric += (C2 * sqrt_ms + C3)
    return acentric


def viscr_CS(ms, alpha):

    # Table 1
    D0, D1, D2, D3, D4, D5, D6, D7, D8 = 0.7091, -2.3960, 0.8638, 74.2270, -0.2132, -69.2268, -1.2500, 0.1229, 0.9821
    D3 /= 1e3
    D5 /= 1e3

    # Eq. 13
    alpha_inf = 1/3
    mur = ( D3 * np.exp(D4 * ms) + D5 ) * (alpha - alpha_inf)**D6
    mur += (D7 * ms + D8)
    mur *= (D0 * alpha**D1 + D2) 
    return mur


def f_obj(lambda_r, acentric_factor, reduced_viscosity, ms=1):
    alpha = helper_get_alpha(lambda_r, lambda_a=6)

    acentric_CG = acentric_CS(ms, alpha)
    viscr_CG = viscr_CS(ms, alpha)

    F = np.abs( (acentric_factor - acentric_CG) / (acentric_factor + 1.))
    F += np.abs( (reduced_viscosity - viscr_CG) / (reduced_viscosity) )
    return F


def CG_FF_galliero(Tc, rhol, acentric_factor, visc, Mw, ms=1, lambda_r0=15):

    M = (Mw/1000/Na) # kg, molecular mass 
    reduced_viscosity = visc * (M / ((rhol)**4 * (0.7 * kb * Tc)**3) )**(1/6) # reduced visc Eq. (8)

    # repulsive exponent
    sol_lr = minimize(f_obj, x0=lambda_r0, args=(acentric_factor, reduced_viscosity, ms))
    lr_galliero = float(sol_lr.x)

    alpha_galliero = helper_get_alpha(lr_galliero, lambda_a=6)
    # interaction energy
    Tcad_CG = Tcad_CS(ms, alpha_galliero)
    eps_galliero = Tc/Tcad_CG  # K
    # shape parameters
    rhol_CG = rhocad_CS(ms, alpha_galliero)
    sigma_galliero = np.cbrt((M * rhol_CG) / rhol) * 1e10 # Amstrongs

    return sigma_galliero, eps_galliero, lr_galliero
