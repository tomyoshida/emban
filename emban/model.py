import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as cst


c = cst.c.cgs.value
k_B = cst.k_B.cgs.value
h = cst.h.cgs.value
au = cst.au.cgs.value
G = cst.G.cgs.value
M_sun = cst.M_sun.cgs.value
m_p = cst.m_p.cgs.value


import jax
import jax.numpy as jnp

import jax.random as random
import numpy as np
import warnings
import numpy as np
from scipy.special import j0


from jax.scipy.interpolate import RegularGridInterpolator

import time
import numpyro
from numpyro.distributions import MultivariateNormal, Normal, Uniform
from numpyro.infer import MCMC, NUTS, init_to_median
import matplotlib.pyplot as plt
from numpyro.infer import init_to_value, Predictive
from numpyro.infer import SVI, Trace_ELBO, init_to_median
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoDelta


from .constants import *
from .utilities import *


jax.config.update("jax_enable_x64", True) 




class model:

    def __init__(self, incl, r_in, r_out, N_GP, spacing = 'linear', userdef_vis_model = None):
        '''
        incl: inclination angle in degrees
        r_in: inner radius in arcseconds
        r_out: outer radius in arcseconds
        N_GP: number of Gaussian process points
        spacing: 'linear' or 'log' for the spacing of GP points
        userdef_vis_model: function, user-defined numpyro model to modify the visibility in Jy. Input: (V (Jy), nu (Hz)). For instance, to add free-free emission.
        '''

        self.spacing = 'linear'
        self.free_parameters = {}
        self.Nparams_forGP = 0
        self.fixed_parameters = {}

        if spacing == 'log':
            self.r_GP = jnp.logspace( jnp.log10(r_in), jnp.log10(r_out), N_GP )
        elif spacing == 'linear':
            self.r_GP = jnp.linspace( r_in, r_out, N_GP )

        self.jitter = 1e-6

        self.incl = np.deg2rad(incl)

        
        self.observations = {}
        self.s_fs = {}
        self.bands = []
        
        # do some interpolation later for better Hankel transform
        # this is the grid

        self.r_GP_rad = np.deg2rad(self.r_GP/3600)
        #_dr = np.deg2rad(dr/3600)
        self.r_rad = self.r_GP_rad #jnp.arange( jnp.min(self.r_GP_rad), jnp.max(self.r_GP_rad), _dr )


        self.dust_params = []

        self.userdef_vis_model = userdef_vis_model
                

        
    def _set_latent_params( self ):
        '''
        Set latent parameters using Gaussian processes.
        Returns a dictionary of latent parameters.
        '''

    
        if self.spacing == 'log':
            R = jnp.log10(self.r_GP)[:, None]
        elif self.spacing == 'linear':
            R = self.r_GP[:, None]
        elif self.spacing == 'log-linear':
            R = jnp.log10(self.r_GP)[:, None]
    
        f_latents = {}
    
        
        for param_name, priors in self.free_parameters.items():

            if priors['GP'] == True:
            
                _g_variance_rbf = priors['g_variance_prior']
                _g_lengthscale_rbf = priors['g_lengthscale_prior']
            
                K = rbf_kernel(R, R, _g_variance_rbf, _g_lengthscale_rbf) + jnp.eye(R.shape[0]) * self.jitter
        
                L_K = jnp.linalg.cholesky(K)
                    
                _g_latent = numpyro.sample(
                                f'g_{param_name}',
                                MultivariateNormal(loc=0.0
                                                ,scale_tril=L_K )
                            )
                    
                f_latents[param_name] = sigmoid_transform( _g_latent,
                                                                min_val=priors['f_min'], 
                                                                max_val=priors['f_max'] )
                
                
                    
            else:
                f_latents[param_name] = numpyro.sample(
                                f'{param_name}',
                                Uniform( low=priors['f_min'], high=priors['f_max'] )
                            )
        
        for param_name, profile in self.fixed_parameters.items():
            f_latents[param_name] = profile['profile']
        
    
        return f_latents


    def set_parameter(self, kind, free = True,  dust_prop = False, GP =True,
                      bounds = (10, 20),
                      g_variance_prior = 2.0, g_lengthscale_prior = 0.3,
                      profile = None):
        '''
        Set a parameter as free or fixed.
        kind: name of the parameter
        free: boolean, whether the parameter is free or fixed
        bounds: tuple, (min, max) bounds for the free parameter
        g_variance_prior: float, prior variance for the Gaussian process
        g_lengthscale_prior: float, prior lengthscale for the Gaussian process
        profile: function or array, fixed profile for the parameter if not free
        '''
    
        
        if free:

            self.free_parameters[kind] = { 'f_min' : bounds[0], 'f_max' : bounds[1], 'GP' : GP,
                                            'g_variance_prior':g_variance_prior,
                                           'g_lengthscale_prior':g_lengthscale_prior}
            
            if GP:
                self.Nparams_forGP += 1
            

        else:
            if profile is not None:
                
                if callable(profile):
                    self.fixed_parameters[kind] = { 'profile' : profile(self.r_GP), 'GP' : GP, }        
                else:
                    self.fixed_parameters[kind] = { 'profile' : profile, 'GP' : GP, }
            else:
                raise ValueError(f'Profile for {kind} is not set.')

        if dust_prop:
            self.dust_params.append( kind )
            print(f'{kind} is input {len(self.dust_params)} of the dust opacity interpolators.') 

    def _expansion_model( self, f_latents, obs, dryrun = False):
        '''
        Generate the expansion model for a given observation.
        f_latents: dictionary of latent parameters
        obs: observation object
        dryrun: boolean, if True, return intermediate results for debugging
        '''

        Sigma_d = 10**( f_latents['Sigma_d'] )
        T = 10**( f_latents['T'] )
        #log10_a_max = f_latents['a_max']
        _dust_params =  jnp.stack([f_latents[f'{dust_param}'] for dust_param in self.dust_params], axis=-1)

        _I = f_I(obs.nu, self.incl, T, Sigma_d, _dust_params, obs.f_log10_ka, obs.f_log10_ks)


        V = hankel_transform_0_jax(_I, self.r_rad, obs.q, obs._bessel_mat) / 1e-23 # Jy

        if self.userdef_vis_model is not None:
            V = self.userdef_vis_model( V, obs, f_latents )

        if dryrun:

            return V, _I

        else:

            obs.V_model = V
            

    def _generate_model( self, f_latents ):
        '''
        Generate the model for all observations.
        f_latents: dictionary of latent parameters
        '''

        for band in self.bands:
            
            obs = self.observations[band]
            
            for _obs in obs:

                self._expansion_model( f_latents, _obs )

            

    def GP_sample( self ):
        '''
        NumPyro model for Gaussian process sampling.
        '''

        f_latents = self._set_latent_params()
   
        self._generate_model( f_latents )

        self._sample_model( )


    def _sample_model( self ):
        '''
        Sample the model for all observations.
        '''
        
        for band in self.bands:
            
            obs = self.observations[band]
            
            flux_uncert = numpyro.sample(
                                f"f_uncert_{band}",
                                Normal(loc=1.0, scale= self.s_fs[band] )
                            )

            for _obs in obs:

                numpyro.sample(
                                f"Y_observed_{_obs.name}",
                                Normal(loc= flux_uncert * _obs.V_model, scale= _obs.s ),
                                obs = _obs.V
                            )

    def set_observations( self, band, q, V, s, s_f, nu, Nch, opacity_interpolator_log10ka, opacity_interpolator_log10ks ):
        '''
        Set observations for a given band.
        band: name of the band
        q: dict, spatial frequencies in wavelength units.
        V: dict, observed visibilities in Jy.
        s: dict, uncertainties in visibilities in Jy.
        s_f: float, flux calibration uncertainty factor.
        nu: dict, frequencies in Hz.
        Nch: int, number of channels.
        opacity: dict, opacity data for the observation.    
        '''

        obs_tmp = []

        for nch in range(Nch):
            
            _obs = observation( f'{band}_ch_{nch}', nu[nch], q[nch], V[nch], s[nch], opacity_interpolator_log10ka[nch], opacity_interpolator_log10ks[nch] )

            kr_matrix = _obs.q[:, jnp.newaxis] * self.r_rad[jnp.newaxis, :]
        
            _obs._bessel_mat = j0( 2*np.pi * kr_matrix )
            
            obs_tmp.append( _obs )
            
        self.bands.append(band)
            
        self.observations[band] = obs_tmp
        self.s_fs[band] = s_f


    def show_prior( self, num_samples = 20, jitter=1e-6, log =True, lw=0.1, alpha=0.5 ):
        '''
        Show prior distributions for the latent parameters.
        num_samples: number of prior samples to generate
        jitter: jitter value for numerical stability
        log: boolean, whether to plot in log scale
        lw: line width for the plots
        alpha: transparency for the plots
        '''

        self.jitter = jitter

        
        def prior_model():
            f_latents = self._set_latent_params()

            return f_latents


        rng_key = jax.random.PRNGKey(1)

        npanel = self.Nparams_forGP

        f_func_all = {}
        g_func_all = {}
       

        if npanel == 1:
            axes = [plt.gca()]
        else:
            fig, axes = plt.subplots(npanel, 1, figsize=(10, 5*npanel))

        for i, (param_name, priors) in enumerate(self.free_parameters.items()):

            if priors['GP'] == True:

                prior_predictive = Predictive(prior_model, num_samples=num_samples)

                

                rng_key, rng_key2 = jax.random.split(rng_key)

                
                prior_predictions = prior_predictive(rng_key)[f'g_{param_name}']
                
                for j, g_func in enumerate(prior_predictions):

                    f_func = sigmoid_transform(g_func, 
                                            min_val=priors['f_min'], 
                                            max_val=priors['f_max'])
                    
                    # vstack f_func
                    if j == 0:
                        _f_func_all = f_func
                    else:
                        _f_func_all = jnp.vstack((_f_func_all, f_func))
                    
                    # vstack g_func
                    if j == 0:
                        _g_func_all = g_func
                    else:
                        _g_func_all = jnp.vstack((_g_func_all, g_func))


                    axes[i].set_title(f'Prior for {param_name}')
                    if log:
                        axes[i].plot( jnp.log10(self.r_GP), f_func, color='blue', lw=lw, alpha=alpha )
                    else:
                        axes[i].plot( self.r_GP, f_func, color='blue', lw=lw, alpha=alpha )

                f_func_all[param_name] = _f_func_all
                g_func_all[param_name] = _g_func_all

        return f_func_all, g_func_all


    def plot_models( self, parameters ):
        '''
        Plot the model visibilities and intensities for given parameters.
        parameters: dictionary of latent parameters
        Returns model visibilities and intensities. 
        '''

        f_latents = parameters


        #if plot:
        #   fig, axes = plt.subplots(  N_panels, 2 , figsize=(15, 5*N_panels) )

        V_res = {}
        I_res = {}
        
        
        
        for band in self.bands:
            
            V_res[band] = {}
            I_res[band] = {}
            
            obs = self.observations[band]

            for _obs in obs:

                #DP = self.dp

                #DP.debug_time['t0'].append( time.perf_counter() )

                _V_res , _I_res= self._expansion_model( f_latents, _obs, dryrun=True )



                V_res[band][_obs.name] = jnp.array(_V_res)
                I_res[band][_obs.name] = jnp.array(_I_res)


        return V_res, I_res

        

    def run_MAP(self, rng_key, num_iterations=1000, num_particles = 1, adam_lr=0.01):
        '''
        Run Stochastic Variational Inference (SVI) to find the Maximum A Posteriori (MAP) estimate of the latent parameters.
        rng_key: random key for JAX
        num_iterations: number of SVI iterations
        num_particles: number of particles for ELBO estimation
        adam_lr: learning rate for the Adam optimizer
        '''
        
        guide = AutoDelta(self.GP_sample, init_loc_fn=init_to_median)

        optimizer = Adam(adam_lr)

        elbo = Trace_ELBO(num_particles)

        svi = SVI(self.GP_sample, 
                  guide, 
                  optimizer, 
                  elbo)
        

        rng_key, rng_key_2 = random.split(rng_key)
        
        svi_result = svi.run(
                rng_key, 
                num_iterations, 
                progress_bar=True
            )
        
        self.svi_result = svi_result
        params = svi_result.params

        medians = guide.median(params)

        loss = svi_result.losses[-1]

        self.delta_medians = {}


        for param_name, priors in self.free_parameters.items():

            if priors['GP'] == False:
                f_predictions = medians[f'{param_name}']
                self.delta_medians[param_name] = f_predictions

            else:
        
                g_predictions = medians[f'g_{param_name}']
                f_predictions = sigmoid_transform( g_predictions, 
                                                    min_val=priors['f_min'], 
                                                    max_val=priors['f_max'] )

                self.delta_medians[param_name] = f_predictions

        return medians, loss


    def run_MCMC(self, rng_key, steps, step_size, num_chains, medians, max_tree_depth=10, adapt_step_size=True):
        '''
        Run MCMC sampling to obtain posterior distributions of the latent parameters.
        rng_key: random key for JAX
        steps: number of MCMC steps
        step_size: initial step size for the NUTS sampler
        num_chains: number of MCMC chains
        medians: initial values for the latent parameters from MAP estimation
        max_tree_depth: maximum tree depth for the NUTS sampler
        adapt_step_size: boolean, whether to adapt the step size during sampling
        Returns the posterior samples of the latent parameters.
        '''

        num_warmup = steps
        num_samples = steps

        # NUTS sampler
        kernel = NUTS(self.GP_sample,
                      step_size=step_size,
                      adapt_step_size=adapt_step_size,
                      init_strategy = init_to_value(values = medians ),
                      max_tree_depth=max_tree_depth)

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
            chain_method='parallel'
        )

        mcmc.run(rng_key, 
               extra_fields=( 'diverging', 'accept_prob', 'energy') 
                )
        
        mcmc.print_summary()

        g_posterior_samples = mcmc.get_samples()


        f_posterior_samples = {}

        for param_name, priors in self.free_parameters.items():


            if priors['GP'] == False:
                f_posterior_samples[param_name] = g_posterior_samples[f'{param_name}']
               

            else:

                f_posterior_samples[param_name] = sigmoid_transform(
                    g_posterior_samples[f'g_{param_name}'], 
                    min_val=priors['f_min'], 
                    max_val=priors['f_max']
                )

        self.mcmc_results = mcmc
        
        return f_posterior_samples



class observation:

    def __init__(self, name, nu, q, V, s, opacity_interpolator_log10ka, opacity_interpolator_log10ks):

        self.name = name
        self.nu = nu
        self.q =  jax.device_put(jnp.asarray(q))
        self.V =  jax.device_put(jnp.asarray(V))
        self.s =  jax.device_put(jnp.asarray(s))

        self.f_log10_ka = opacity_interpolator_log10ka
        self.f_log10_ks = opacity_interpolator_log10ks
        

        '''
        self.f_log10_ka = RegularGridInterpolator( points = (opacity['log10_a'], ), 
                                                  values = opacity['log10_ka'] )


        self.f_log10_ks = RegularGridInterpolator( points = (opacity['log10_a'], ), 
                                                  values = opacity['log10_ks'] )

        '''
        


