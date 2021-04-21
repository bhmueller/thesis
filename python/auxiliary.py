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


def setup_rust_model_001():
    """To minimise errors, set up everything needed for implementing the Rust model in
    one dictionary.

    Parameters
    ----------
    None

    Returns
    -------
    model_setup: dict
        Contains alle the required objects for implementing the Rust model.

    """
    model_setup = {}
    # Set simulating variables.
    disc_fac = 0.9999
    num_buses = 50
    num_periods = 120  # Ten years.
    gridsize = 1000  # For plotting.
    # We use the cost parameters and transition probabilities from the replication.
    params = np.array([10.07780762, 2.29417622])
    # Table X in Rust (1987): 11.7257, 2.4569
    # Old values used before February 15th: 0.39189182, 0.59529371, 0.01281447
    trans_probs = np.array([0.39189182, 0.59529371, 0.01281447])
    # trans_probs = np.array([0.0937, 0.4475, 0.4459, 0.0127])
    # Four in table X of Rust (1987): 0.0937, 0.4475, 0.4459, 0.0127
    scale = 1e-3

    init_dict_simulation = {
        "simulation": {
            "discount_factor": disc_fac,
            "periods": num_periods,
            "seed": 123,
            "buses": num_buses,
        },
        "plot": {"gridsize": gridsize},
    }

    # Calcualte objects necessary for the simulation process. See documentation for
    # details.
    # Discretization of the state space.
    # Originally 200. 90 (with a size of 5,000 miles each) and 175 (with a size of
    # 2,571) bins were used by Rust (1987).
    num_states = 175
    costs = calc_obs_costs(num_states, lin_cost, params, scale)

    trans_mat = create_transition_matrix(num_states, trans_probs)
    ev = calc_fixp(trans_mat, costs, disc_fac)[0]

    # Use one init_dict for get_demand() and estimate().
    init_dict_estimation = {
        "model_specifications": {
            "discount_factor": disc_fac,
            "number_states": num_states,
            "maint_cost_func": "linear",
            "cost_scale": 1e-3,
        },
        "optimizer": {
            "approach": "NFXP",
            "algorithm": "scipy_L-BFGS-B",
            "gradient": "Yes",
        },
    }

    # Need demand at certain value of RC only.  Note RC is scaled by 1e-03.
    demand_dict = {
        "RC_lower_bound": 11.5,
        "RC_upper_bound": 11.5,
        "demand_evaluations": 1,
        "tolerance": 1e-10,
        "num_periods": num_periods,
        "num_buses": num_buses,
    }

    model_setup["params"] = params
    model_setup["ev"] = ev
    model_setup["costs"] = costs
    model_setup["trans_probs"] = trans_probs
    model_setup["trans_mat"] = trans_mat
    model_setup["init_dict_simulation"] = init_dict_simulation
    model_setup["init_dict_estimation"] = init_dict_estimation
    model_setup["demand_dict"] = demand_dict

    return model_setup


def compute_confidence_intervals(param_estimate, variance, critical_value):
    """Compute confidence intervals (ci). Note assumptions about the distributions
    apply.

    Parameters
    ----------
    param_estimate: float
        Parameter estimate for which ci should be computed
    variance: float
        Variance of parameter estimate.
    critical_value: float
        Critical value of the t distribution, e.g. for the 95-percent-ci it's 1.96.

    Returns
    -------
    confidence_interval_dict: dict
        Lower (upper) bound of the ci can be accessed by the key 'lower_bound'
        ('upper_bound').

    """
    confidence_interval_dict = {}
    confidence_interval_dict["lower_bound"] = param_estimate - critical_value * np.sqrt(
        variance
    )
    confidence_interval_dict["upper_bound"] = param_estimate + critical_value * np.sqrt(
        variance
    )
    return confidence_interval_dict


def approx_comp_time(
    time_model_eval, method, n_inputs, n_perms, n_output, n_outer, n_inner
):
    """
    Approximate time for computation in hours and minutes.

    Parameters
    ----------
    time_model_eval: float
        Time in seconds per 100 model evaluations (e.g. rust_model for 100 samples:
        approx 35 s).
    ...

    Returns
    -------

    """
    if method == "random":
        n_evals = n_output + n_perms * (n_inputs - 1) * n_outer * n_inner
        time = (time_model_eval * (n_evals) / 100) / 3600
    elif method == "exact":
        n_evals = (
            n_output + np.math.factorial(n_inputs) * (n_inputs - 1) * n_outer * n_inner
        )
        time = (time_model_eval * (n_evals) / 100) / 3600

    print(
        "",
        n_evals,
        "model evaluations",
        "\n",
        "approx. ",
        time,
        "hours",
        "\n",
        "approx.",
        time * 60,
        "minutes",
    )


def simulate_cov_and_mean_rc_theta_11(
    num_sim,
    ev,
    costs,
    trans_mat,
    init_dict_simulation,
    init_dict_estimation,
):
    """
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
        Variance-covariance matrix of simulated data with shape (num_sim,
        num_input_variables).
    mean: n.array
        Mean vector of simulated data with shape (, num_input_variables).

    """

    def get_params(current_seed):

        init_dict_simulation["simulation"]["seed"] = current_seed

        df = simulate(init_dict_simulation["simulation"], ev, costs, trans_mat)
        data = df[["state", "decision", "usage"]].copy()

        _, result_fixp_nfxp = estimate(init_dict_estimation, data)

        return result_fixp_nfxp["x"]

    # seed_array = np.random.choice(np.arange(num_sim), num_sim, replace=False)

    parameter_estimates = np.asarray(
        list(
            map(
                get_params, np.random.choice(np.arange(num_sim), num_sim, replace=False)
            )
        )
    )

    cov = np.cov(parameter_estimates.T)
    mean = np.mean(parameter_estimates, axis=0)

    return cov, mean


# Functions for the Rust model.
def rust_model_shapley(
    x,
    method,
    n_perms,
    n_inputs,
    n_output,
    n_outer,
    n_inner,
    trans_probs,
    init_dict_estimation,
    demand_dict,
):
    if method == "exact":
        n_evaluations = (
            n_output + np.math.factorial(n_inputs) * (n_inputs - 1) * n_outer * n_inner
        )
    elif method == "random":
        n_evaluations = n_output + n_perms * (n_inputs - 1) * n_outer * n_inner
    # else: raiseerror

    # Adapt function to work with changed number of trans_probs as well.
    n_trans_probs = len(trans_probs)

    demand_inputs = np.zeros((n_evaluations, n_trans_probs + 2))
    demand_inputs[:, :n_trans_probs] = trans_probs
    demand_inputs[:, n_trans_probs:] = x[:, :]

    demand_output = np.zeros((n_evaluations, 1))

    # Second, try with list comprehension (do not need to define demand_output first).
    # demand_output = [get_demand(init_dict_estimation, demand_dict, demand_inputs[
    # sample, :]).iloc[0]['demand']
    #                for sample in np.arange(n_evaluations)]

    # First try with for loop.
    # for sample in np.arange(n_evaluations):
    #    demand_params = demand_inputs[sample, :]
    #    demand_output[sample] = get_demand(init_dict_estimation, demand_dict,
    # demand_params).iloc[0]['demand']

    get_demand_partial = partial(
        get_demand, init_dict=init_dict_estimation, demand_dict=demand_dict
    )

    def _get_demand_mapping(x):
        return get_demand_partial(demand_params=x).iloc[0]["demand"]

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
