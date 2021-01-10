'''
Sample covariance matrices and mean vectors for the parameters to be estimated by the Rust model (theta vector).
'''
import numpy as np
import matplotlib.pyplot as plt
from ruspy.simulation.simulation import simulate
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation import estimate
#from ruspy.model_code.demand_function import get_demand


def get_cov_and_mean_four_and_five_parameters(num_sim,
                                              ev, 
                                              costs, 
                                              trans_mat, 
                                              init_dict_simulation, 
                                              init_dict_estimation
                                             ):
    '''
    Calculate variance-covariance matrix (cov) and mean vector (mean) of simulated data.
    Get cov and mean for four (without theta_32) and all five parameters in theta.
    
    
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
    cov_4: nd.array
        Variance-covariance matrix of simulated data with shape (num_sim, num_input_variables).
    mean_4: n.array
        Mean vector of simulated data with shape (, num_input_variables).
    cov_5:
    mean_5:
    
    '''
    
    parameter_estimates_4_inputs = np.zeros((num_sim, 4))
    parameter_estimates_5_inputs = np.zeros((num_sim, 5))
    
    for i in np.arange(num_sim):
        
        init_dict_simulation['simulation']['seed'] = +i
        
        df = simulate(init_dict_simulation["simulation"], ev, costs, trans_mat)
        data = df[['state', 'decision', 'usage']].copy()
        
        result_transitions_nfxp, result_fixp_nfxp = estimate(init_dict_estimation, data)
            
        # Record only two of three transition probabilities i.o.t. avoid singularity of the covariance matrix.
        parameter_estimates_4_inputs[i, :] = np.concatenate((result_transitions_nfxp['x'][:2], result_fixp_nfxp['x']))
        # All five inputs.
        parameter_estimates_5_inputs[i, :] = np.concatenate((result_transitions_nfxp['x'], result_fixp_nfxp['x']))
        
        assert_allclose(parameter_estimates_5_inputs[i, :2].sum(), 1.0, rtol=0.02)
            
    cov_4_inputs = np.cov(parameter_estimates_4_inputs.T)
    mean_4_inputs = np.mean(parameter_estimates_4_inputs, axis=0)
    
    cov_5_inputs = np.cov(parameter_estimates_5_inputs.T)
    mean_5_inputs = np.mean(parameter_estimates_5_inputs, axis=0)
            
    return cov_4_inputs, mean_4_inputs, cov_5_inputs, mean_5_inputs


def get_cov_and_mean_rc_theta_32(num_sim,
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