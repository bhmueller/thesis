'''
This module contains all tests for the functions I define for my thesis.
'''

from python.sampling import rust_model


# Test function rust_model.
def test_rust_model_simple_setup():
    method = 'exact'
    n_perms = None
    n_inputs = 2
    n_output = 1
    n_outer = 1
    n_inner = 1
    trans_probs = np.array([0.39189182, 0.59529371, 0.01281447])

    disc_fac = 0.9999
    num_states = 175

    init_dict_estimation = {
        'model_specifications': {
            'discount_factor': disc_fac,
            'number_states': num_states,
            'maint_cost_func': 'linear',
            'cost_scale': 1e-3
        },
        'optimizer': {
            'approach': 'NFXP',
            'algorithm': 'scipy_L-BFGS-B',
            'gradient': 'Yes'
        },

    }

    num_periods = 120
    num_buses = 50

    demand_dict = {
        "RC_lower_bound": 11.5,
        "RC_upper_bound": 11.5,
        "demand_evaluations": 1,
        "tolerance": 1e-10,
        "num_periods": num_periods,
        "num_buses": num_buses,
    }

    n_evaluations = n_output + np.math.factorial(n_inputs) * (n_inputs -1) * n_outer * n_inner

    x = np.array([[1, 2],
                 [2, 3],
                 [1, 4]])

    rust_model(x, method, n_perms, n_inputs, n_output, n_outer, n_inner, trans_probs, init_dict_estimation, demand_dict)