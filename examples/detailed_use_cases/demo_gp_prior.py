"""
  A demo on specifying the GP prior in Dragonfly.
  -- kirthevasank
"""

# pylint: disable=invalid-name

from __future__ import print_function
from argparse import Namespace
from dragonfly import load_config_file, maximise_function, \
                      multiobjective_maximise_functions
import numpy as np
# Local imports
import moo_17d
import obj_17d


# Prior mean for conductivity
def conductivity_prior_mean_17d(x):
  """ Prior mean for conductivity in 15d example. """
  u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3 = moo_17d.get_17d_electrolyte_vars(x)
  ret0 = 1.1 * v0 * (u1 + 2*u2 + 3*u4**2)
  ret1 = v1 * (u2**1.4 + 1.4*u6 + 1.05*u7)
  ret2 = v2 * (1.05*u3*u5 + 0.95 * u1*u2 + 0.99 * u7*u6)
  ret3 = 0.99 * v3 * (u1*u2 + 1.35*u1*u2 + 0.95*u4**1.2)
  return ret0 + ret1 + ret2 + ret3

# Prior kernel --------------------------------------------------------------------------
kernel_power_hyperparam_info = ('power', 'disc', [1, 2])
kernel_scale_hyperparam_info = ('log10_scale', 'cts', [-2, 3])
kernel_bw_hyperparam_info = ('log10_bw', 'cts', [-3, 3])


def prior_kernel_17d(x1, x2, hyperparams):
  """ The prior kernel for the 15D example.
      In this example, the if there are any zeros on the "salt_present" variables,
      the corresponding value in the "salt_mol" variables do not matter.
      Hence, we will use a matern kernel after multiplying the salt_present variables
      by the salt_mol variables, which will help simplify the problem.
      We will use an exponentiated distance kernel ---------------------------------------
      Here, hyperparams are a 13D vector (power, scale, log_bandwidths_salt,
      log_bandwidths_solv) where scale (1D) is the scale parameter, bandwidths_salt are
      the bandwidths for the salt variables, and bandwidths_solv are those for the solvent
      variables.
  """
  # Hyperparams
  power = hyperparams[0]
  scale = 10 ** hyperparams[1]
  bandwidths = 10 ** np.array(hyperparams[2:])
  # Compute normalised difference
  x1_vec_ = moo_17d.get_17d_electrolyte_vars(x1)
  x2_vec_ = moo_17d.get_17d_electrolyte_vars(x2)
  x1_vec = np.array(x1_vec_)
  x2_vec = np.array(x2_vec_)
  norm_diffs = (x1_vec - x2_vec) / bandwidths
  sum_norm_diff_sq = sum(norm_diffs**power)
  return scale * np.exp(-sum_norm_diff_sq/2)


def soo_main():
  """ Main function for single objective optimisation. """
  objective = obj_17d.objective
  config = load_config_file('config_17d.json')
  max_capital = 60
  options = Namespace(
    build_new_model_every=5, # update the model every 5 iterations.
    report_results_every=4, # report progress every 4 iterations.
    report_model_on_each_build=True, # report model when you build it.
    )
  opt_method = 'bo'

  # Specify the kernel and its hyperparameters
  options.gp_prior_kernel = prior_kernel_17d
  options.gp_prior_kernel_hyperparams = \
              [kernel_power_hyperparam_info, kernel_scale_hyperparam_info] + \
              [kernel_bw_hyperparam_info] * 11
  # Uncomment the line below if you also want to specify the prior mean
  # options.gp_prior_mean = conductivity_prior_mean_17d

  # Optimise
  opt_val, opt_pt, _ = maximise_function(objective, config.domain, max_capital,
                           opt_method=opt_method, config=config, options=options)
  print('opt_pt: %s'%(str(opt_pt)))
  print('opt_val: %s'%(str(opt_val)))


def moo_main():
  """ Main function for multi objective optimisation. """
  pass


if __name__ == '__main__':
  soo_main()
  moo_main()

