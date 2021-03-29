'''
Sample covariance matrices and mean vectors for the parameters to be estimated by the Rust model (theta vector).
'''
import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
from functools import partial
from ruspy.simulation.simulation import simulate
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation import estimate
from ruspy.model_code.demand_function import get_demand
from econsa.shapley import _r_condmvn


def approx_comp_time(time_model_eval, method, n_inputs, n_perms, n_output, n_outer, n_inner):
    '''
    Approximate time for computation in hours and minutes.
    
    Parameters
    ----------
    time_model_eval: float
        Time in seconds per 100 model evaluations (e.g. rust_model for 100 samples: approx 35 s).
    ...
    
    Returns
    -------
    
    '''
    if method == 'random':
        n_evals = n_output + n_perms * (n_inputs -1) * n_outer * n_inner
        time = (time_model_eval * (n_evals) / 100) / 3600
    elif method == 'exact':
        n_evals = n_output + np.math.factorial(n_inputs) * (n_inputs -1) * n_outer * n_inner
        time = (time_model_eval * (n_evals) / 100) / 3600

    print('', n_evals, 'model evaluations', 
          '\n', 'approx. ', time, 'hours', 
          '\n', 'approx.', time * 60, 'minutes'
         )


def simulate_cov_and_mean_rc_theta_11(num_sim,
                                      ev, 
                                      costs, 
                                      trans_mat, 
                                      init_dict_simulation, 
                                      init_dict_estimation,
                                     ):
    '''
    Calculate variance-covariance matrix (cov) and mean vector (mean) of simulated data
    for RC and theta_32 from Rust model.
    
    
    Parameters
    ----------------
    num_sim: int
        Number of simulations for the data from which cov and mean are calculated.
    ev: 
    costs:
    trans_mat:
    init_dict_simulation:
    init_dict_estimation:
        
    Returns
    ----------------
    cov: nd.array
        Variance-covariance matrix of simulated data with shape (num_sim, num_input_variables).
    mean: n.array
        Mean vector of simulated data with shape (, num_input_variables).
    
    '''
    
    parameter_estimates = np.zeros((num_sim, 2))
    
    for i in np.arange(num_sim):
        
        init_dict_simulation['simulation']['seed'] = +i
        
        df = simulate(init_dict_simulation["simulation"], ev, costs, trans_mat)
        data = df[['state', 'decision', 'usage']].copy()
        
        result_transitions_nfxp, result_fixp_nfxp = estimate(init_dict_estimation, data)
            
        parameter_estimates[i, :] = result_fixp_nfxp['x']
        
            
    cov = np.cov(parameter_estimates.T)
    mean = np.mean(parameter_estimates, axis=0)
            
    return cov, mean


# Functions for the Rust model.
def rust_model_shapley(x, method, n_perms, n_inputs, n_output, n_outer, n_inner, trans_probs, init_dict_estimation, demand_dict):
    if method == 'exact':
        n_evaluations = n_output + np.math.factorial(n_inputs) * (n_inputs -1) * n_outer * n_inner
    elif method == 'random':
        n_evaluations = n_output + n_perms * (n_inputs -1) * n_outer * n_inner
    #else: raiseerror
    
    # Adapt function to work with changed number of trans_probs as well.
    n_trans_probs = len(trans_probs)
    
    demand_inputs = np.zeros((n_evaluations, n_trans_probs +2))
    demand_inputs[:, :n_trans_probs] = trans_probs
    demand_inputs[:, n_trans_probs:] = x[:, :]
    
    demand_output = np.zeros((n_evaluations, 1))
    
    # Second, try with list comprehension (do not need to define demand_output first).
    #demand_output = [get_demand(init_dict_estimation, demand_dict, demand_inputs[sample, :]).iloc[0]['demand'] 
    #                for sample in np.arange(n_evaluations)]
    
    # First try with for loop.
    #for sample in np.arange(n_evaluations):
    #    demand_params = demand_inputs[sample, :]
    #    demand_output[sample] = get_demand(init_dict_estimation, demand_dict, demand_params).iloc[0]['demand']
    
    get_demand_partial = partial(get_demand, init_dict=init_dict_estimation, demand_dict=demand_dict)
    def _get_demand_mapping(x):
        return get_demand_partial(demand_params=x).iloc[0]['demand']
    
    demand_output = np.array(list(map(_get_demand_mapping, demand_inputs)))
    
    return demand_output


def x_all_raw(n, mean, cov):
    distribution = cp.MvNormal(mean, cov)
    return distribution.sample(n)

def x_cond_raw(n, subset_j, subsetj_conditional, xjc, mean, cov):
    if subsetj_conditional is None:
        cov_int = np.array(cov)
        cov_int = cov_int.take(subset_j, axis=1)
        cov_int = cov_int[subset_j]
        distribution = cp.MvNormal(mean[subset_j], cov_int)
        return distribution.sample(n)
    else:
        return _r_condmvn(
            n,
            mean=mean,
            cov=cov,
            dependent_ind=subset_j,
            given_ind=subsetj_conditional,
            x_given=xjc,
        )
    
    
