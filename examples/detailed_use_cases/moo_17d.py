"""
  A synthetic objective function for the electrolyte design example.
  -- celsius38
  -- kirthevasank
"""

# pylint: disable=invalid-name

def get_17d_electrolyte_vars(x):
  """ Obtain the variables. """
  u1 = float(x[0]) * x[4]
  u2 = float(x[1]) * x[5]
  u3 = float(x[2]) * x[6]
  u4 = float(x[3][0]) * x[7][0]
  u5 = float(x[3][1]) * x[7][1]
  u6 = float(x[3][2]) * x[7][2]
  u7 = float(x[3][3]) * x[7][3]
  v1 = x[8][0]
  v2 = x[8][1]
  v3 = x[8][2]
  v0 = 1 - (v1 + v2 + v3)
  return (u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3)


def synthetic_conductivity_function(x):
  """ Computes the electrolyte objective. """
  u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3 = get_17d_electrolyte_vars(x)
  ret0 = v0 * (u1 + 2*u2 + 3*u4**2)
  ret1 = v1 * (u2**1.5 + 1.3*u6 + u7)
  ret2 = v2 * (u3*u5 + u1*u2 + u7*u6)
  ret3 = v3 * (u1*u2 + 1.3*u1*u2 + u4**1.2)
  return ret0 + ret1 + ret2 + ret3


def synthetic_voltage_window_function(x):
  """ Computes the electrolyte objective. """
  u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3 = get_17d_electrolyte_vars(x)
  ret0 = (v0 + v1) * (2.4*u5 + 1.9*u4**1.5)
  ret1 = abs(v2 - v1) * (u7**1.5 + 2.3*u1)
  ret2 = (v0 + v3) * (u2*u6 + u1*u7 + u6*u3)
  ret3 = v3 * (u7*u4 + 1.3*u6*(u3**1.4) + u1**1.1)
  return ret0 + ret1 + ret2 + ret3


def compute_objective(x):
  """ Compute objective. """
  return [synthetic_conductivity_function(x), synthetic_voltage_window_function(x)]

