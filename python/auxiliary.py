import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp
from functools import partial
from joblib import Parallel, delayed
from ruspy.simulation.simulation import simulate
from ruspy.model_code.fix_point_alg import calc_fixp
from ruspy.model_code.cost_functions import lin_cost
from ruspy.model_code.cost_functions import calc_obs_costs
from ruspy.estimation.estimation_transitions import create_transition_matrix
from ruspy.estimation.estimation import estimate
from ruspy.model_code.demand_function import get_demand

# from econsa.shapley import _r_condmvn
from python.shapley import get_shapley
from python.shapley import _r_condmvn


def n_o(c_s, n_v):
    """
    Number of outer samples given N_I = 3 as recommended by SNS16.

    Parameters
    ----------
    c_s : int
        Total number of model evaluations.

    n_v : int
        Number of samples desired for computing total output variance.

    Returns
    -------
    n_o : float
        Number of outer samples.
    """
    return (c_s - n_v) / 6


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

    # Calculate objects necessary for the simulation process. See documentation for
    # details.
    # Discretisation of the state space.
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


def compute_confidence_intervals(param_estimate, std_dev, critical_value):
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
    confidence_interval_dict["lower_bound"] = param_estimate - critical_value * std_dev
    confidence_interval_dict["upper_bound"] = param_estimate + critical_value * std_dev
    return confidence_interval_dict


def approx_comp_time(time_model_eval, n_inputs, n_perms, n_variance, n_outer, n_inner):
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

    n_evals = (
        n_variance + np.math.factorial(n_inputs) * (n_inputs - 1) * n_outer * n_inner
    )
    time = (time_model_eval * (n_evals) / 100) / 3600

    # print(
    #     "",
    #     n_evals,
    #     "model evaluations",
    #     "\n",
    #     "approx. ",
    #     time,
    #     "hours",
    #     "\n",
    #     "approx.",
    #     time * 60,
    #     "minutes",
    # )
    print(
        f"{n_evals} model evaulations. \n Approx. {time} hours. \n Approx. {time * 60} minutes"
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
    for RC and theta_11 from Rust model.


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

    # parameter_estimates = np.asarray(
    #     list(
    #         map(
    #             get_params, np.random.choice(np.arange(num_sim), num_sim, replace=False)
    #         )
    #     )
    # )

    parameter_estimates = np.asarray(list(map(get_params, np.arange(num_sim))))

    cov = np.cov(parameter_estimates.T)
    mean = np.mean(parameter_estimates, axis=0)

    return cov, mean


# Handle one sample at a time. Also: check that values in df are accessed in the right way.
def rust_model_morris(x, trans_probs, init_dict_estimation, demand_dict):

    # Allow fct. to be adapted to more or fewer transition steps.
    n_trans_probs = len(trans_probs)

    demand_inputs = np.zeros(n_trans_probs + 2)

    demand_inputs[:n_trans_probs] = trans_probs
    demand_inputs[n_trans_probs] = x["value"][0]
    demand_inputs[n_trans_probs + 1] = x["value"][1]

    demand_output = get_demand(init_dict_estimation, demand_dict, demand_inputs).iloc[
        0
    ]["demand"]

    return demand_output


# Functions for the Rust model.
def rust_model_shapley(
    x,
    trans_probs,
    init_dict_estimation,
    demand_dict,
):

    # Adapt function to work with changed number of trans_probs as well.
    n_trans_probs = len(trans_probs)

    demand_inputs = np.zeros(n_trans_probs + 2)
    demand_inputs[:n_trans_probs] = trans_probs

    demand_inputs[n_trans_probs] = x[:][0]
    demand_inputs[n_trans_probs + 1] = x[:][1]

    demand_output = get_demand(init_dict_estimation, demand_dict, demand_inputs).iloc[
        0
    ]["demand"]

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


def descriptives_and_data_shapley_effects(shapley_effects, n_replicates):
    """
    Parameters
    -----------

    shapley_effects : dict
        A dictionary containing the output tables for a number of n_replicates Shapley
        effects computed by using econsa's get_shapley.

    Returns
    ----------

    descriptives_shapley_effects : pd.DataFrame
        DataFrame containing the descriptives of the n_replicates Shapley effects:
        mean, standard error, and CIs.

    data_box_plots : pd.DataFrame
        DataFrame containing the data for making the boxplots.

    """
    # Get data frame suitable for plotting.
    rc_shapley_effects = [
        shapley_effects[i]["Shapley effects"]["$RC$"] for i in np.arange(n_replicates)
    ]
    theta_shapley_effects = [
        shapley_effects[i]["Shapley effects"]["$\theta_{11}$"]
        for i in np.arange(n_replicates)
    ]

    rc_shapley_effects_df = pd.DataFrame(data=rc_shapley_effects)
    rc_shapley_effects_df["input_variable"] = "rc"

    theta_shapley_effects_df = pd.DataFrame(data=theta_shapley_effects)
    theta_shapley_effects_df["input_variable"] = "theta"

    data = pd.concat(
        [rc_shapley_effects_df, theta_shapley_effects_df], ignore_index=True
    )
    data.rename(columns={0: "shapley_effect"}, inplace=True)

    # For calculation of variance use Bessel's correction (by specifying ddof=1 in np.var()).
    variance_rc_shapley_effects = np.var(rc_shapley_effects, ddof=1)
    mean_rc_shapley_effects = np.mean(rc_shapley_effects)

    variance_theta_shapley_effects = np.var(theta_shapley_effects, ddof=1)
    mean_theta_shapley_effects = np.mean(theta_shapley_effects)

    # Calc. std. errors. (std. dev. / sqrt(n)).
    se_rc_shapley_effects = np.sqrt(variance_rc_shapley_effects) / np.sqrt(n_replicates)

    se_theta_shapley_effects = np.sqrt(variance_theta_shapley_effects) / np.sqrt(
        n_replicates
    )

    # Critical value of the 95-percent confidence interval.
    crit_value = 1.96

    ci_rc = compute_confidence_intervals(
        mean_rc_shapley_effects, se_rc_shapley_effects, crit_value
    )

    ci_theta = compute_confidence_intervals(
        mean_theta_shapley_effects, se_theta_shapley_effects, crit_value
    )

    descriptives_data = np.array(
        [
            [
                mean_rc_shapley_effects,
                se_rc_shapley_effects,
                ci_rc["lower_bound"],
                ci_rc["upper_bound"],
            ],
            [
                mean_theta_shapley_effects,
                se_theta_shapley_effects,
                ci_theta["lower_bound"],
                ci_theta["upper_bound"],
            ],
        ]
    )

    descriptives_shapley_effects = pd.DataFrame(
        descriptives_data,
        columns=["Mean", "Std. errors", "CI lower bound", "CI upper bound"],
        index=["$RC$", "$\theta_{11}$"],
    )
    descriptives_shapley_effects.index.name = "Shapley Effect"

    return descriptives_shapley_effects, data


def shapley_replicate(
    # init_dict_simulation,
    model,
    x_all_partial,
    x_cond_partial,
    n_perms,
    n_inputs,
    n_variance,
    n_outer,
    n_inner,
    n_cores,
    current_seed,
):
    # init_dict_simulation["simulation"]["seed"] = current_seed

    # model = partial(rust_model_shapley, trans_probs=trans_probs, init_dict_
    # estimation=init_dict_estimation, demand_dict=demand_dict)

    # x_all_partial = partial(x_all_raw, mean=params, cov=cov)
    # x_cond_partial = partial(x_cond_raw, mean=params, cov=cov)

    np.random.seed(current_seed)

    exact_shapley = get_shapley(
        model,
        x_all_partial,
        x_cond_partial,
        n_perms,
        n_inputs,
        n_variance,
        n_outer,
        n_inner,
        n_cores,
    )
    exact_shapley.rename(index={"X1": "$RC$", "X2": "$\theta_{11}$"}, inplace=True)
    return exact_shapley
