"""
  Synthetic function for 17D optimisation.
  -- kirthevasank
"""

from moo_17d import synthetic_conductivity_function

def objective(x):
  """ Computes the objectives. """
  return synthetic_conductivity_function(x) # Just returns conductivity

