
###### Utiities #########

from astroquery.linelists.cdms import CDMS
import jax
import jax.numpy as jnp
import numpy as np
from .constants import *
from astropy import units as u
import warnings
import numpy as np


def hankel_transform_0_jax(f, r, k, bessel):
    '''
    Perform the Hankel transform of order 0 using JAX.
    f: jnp.ndarray, function values at radial distances r (shape: [n_r])
    r: jnp.ndarray, radial distances (shape: [n_r])
    k: jnp.ndarray, spatial frequencies (shape: [n_k])
    bessel: jnp.ndarray, precomputed Bessel function values (shape: [n_k, n_r])
    Returns the Hankel transform values at spatial frequencies k (shape: [n_k]).
    '''
    
    dr = jnp.gradient(r)
    fr = f * r

    #def integrate(ki):
    #    integrand = fr * j0(k * r) 
    #    return jnp.sum(integrand * dr)
    
    return jnp.sum( 2*np.pi * fr * bessel * dr, axis=1)

def rbf_kernel(X1, X2, variance, lengthscale):
    '''
    Compute the Radial Basis Function (RBF) kernel between two sets of input points using JAX.
    X1: jnp.ndarray, first set of input points (shape: [n1, d])
    X2: jnp.ndarray, second set of input points (shape: [n2 , d])
    variance: float, variance parameter of the RBF kernel
    lengthscale: float, lengthscale parameter of the RBF kernel
    Returns the RBF kernel matrix (shape: [n1, n2]).
    '''
    
    sq_dist = jnp.sum(X1**2, 1)[:, None] + jnp.sum(X2**2, 1)[None, :] - 2 * jnp.dot(X1, X2.T)
    
    return variance**2 * jnp.exp(-0.5 / lengthscale**2 * sq_dist)

def B(nu, T):
    '''
    Calculate the Planck function B(nu, T).
    nu: jnp.ndarray or float, frequency in Hz
    T: jnp.ndarray or float, temperature in Kelvin
    Returns the Planck function values.
    ''' 

    return 2*h*nu**3/c**2 / ( jnp.exp(h*nu/k_B/T) - 1 )
    
def sigmoid_transform(x, min_val=0.0, max_val=1.0):
    '''
    Apply a sigmoid transformation to the input array x.
    The transformed values will be in the range [min_val, max_val].
    x: jnp.ndarray, input array to be transformed
    min_val: float, minimum value of the transformed output
    max_val: float, maximum value of the transformed output
    Returns the transformed array with values in the range [min_val, max_val].
    '''
    
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))   



# sierra


def F(tau, omega):
    '''
    Calculate the function F(tau, omega) used in radiative transfer.
    tau: jnp.ndarray or float, optical depth
    omega: jnp.ndarray or float, single scattering albedo
    Returns the computed values of F(tau, omega).
    Ref. Miyake & Nakagawa 1993, Icarus, 106, 20; Sierra et al. 2020, ApJ, 892, 136
    '''
    
    w = omega
    
    term1 = (jnp.sqrt(1 - w) - 1.0) * jnp.exp(-jnp.sqrt(3.0 / (1.0 - w)) * tau)
    
    A_num = 1.0 - jnp.exp(-(jnp.sqrt(3.0 * (1.0 - w)) + 1.0) * tau / (1.0 - w))
    A_den = jnp.sqrt(3.0 * (1.0 - w)) + 1.0
    A = A_num / A_den
    
    B_num = jnp.exp(-tau / (1.0 - w)) - jnp.exp(-jnp.sqrt(3.0 / (1.0 - w)) * tau)
    B_den = jnp.sqrt(3.0 * (1.0 - w)) - 1.0
    B = B_num / B_den
    
    term2 = (jnp.sqrt(1 - w) + 1.0)
    
    denom = term1 - term2
    
    return  (A + B) / denom


def f_I(nu, incl, T, Sigma_d, dust_params, f_log10_ka, f_log10_ks):
    '''
    Calculate the intensity I(nu) using radiative transfer with scattering.
    nu: jnp.ndarray or float, frequency in Hz
    incl: jnp.ndarray or float, inclination angle in radians
    T: jnp.ndarray or float, temperature in Kelvin
    Sigma_d: jnp.ndarray or float, dust surface density in g/cm^2
    dust_params: list of jnp.ndarray or float, dust parameters (e.g., maximum grain size). Assuming the order matches the interpolators.
    f_log10_ka: function, interpolator for log10 of absorption opacity
    f_log10_ks: function, interpolator for log10 of scattering opacity
    Returns the computed intensity I(nu).
    ''' 


    ka = 10**f_log10_ka( *dust_params )
    ks = 10**f_log10_ks( *dust_params )

    chi = ka + ks
    omega = ks / chi
    
    tau = ka * Sigma_d / jnp.cos(incl)

    return B(nu, T) * (  1 - jnp.exp( -tau/(1-omega) ) + omega*F(tau, omega)  )




