import cvxpy as cp

import jax
import jax_verify
import functools
import jax.numpy as jnp
import torch
import numpy as np


def pytorch_model_to_jax(torch_model: torch.nn.Sequential):
    params = []
    act = None

    # Extract params (weights, biases) from torch layers, to be used in
    # jax.
    # Note: This propagator assumes a feed-forward relu NN.
    for m in torch_model.modules():
        if isinstance(m, torch.nn.Sequential):
            continue
        elif isinstance(m, torch.nn.LeakyReLU):
            if act is None or act == "leakyrelu":
                act = "leakyrelu"
        elif isinstance(m, torch.nn.Linear):
            w = m.weight.data.numpy().T
            b = m.bias.data.numpy()
            params.append((w, b))

    return functools.partial(relu_nn, params)


def relu_nn(params, inputs, alpha=1e-2):
    for W, b in params[:-1]:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(outputs, alpha * outputs)
    W, b = params[-1]
    return jnp.dot(inputs, W) + b


def jax_interval_to_np_range(interval: jax_verify.IntervalBound) -> np.ndarray:
    return np.vstack([interval.lower, interval.upper]).T


def np_range_to_jax_interval(input_range: np.ndarray) -> jax_verify.IntervalBound:
    return jax_verify.IntervalBound(input_range[:, 0], input_range[:, 1])

def nominal_and_epsilon_to_range(nominal, epsilon):
    return np.vstack([nominal-epsilon, nominal+epsilon]).T

def range_to_nominal_and_epsilon(input_range):
    nominal_input = (input_range[:, 1] + input_range[:, 0]) / 2.
    epsilon = (input_range[:, 1] - input_range[:, 0]) / 2.
    return nominal_input, epsilon


def MILP_vars(model, input_range, alpha=1e-2):
    ibp_boxes = [("input", input_range)] + verify_ibp(model, input_range, alpha=alpha)

    constraints = []
    variables = [(cp.Variable((input_range.shape[0], 1)), None)]

    for i, module in enumerate(model):
        if isinstance(module, torch.nn.Linear):
            # n is the input dimension
            # m is the number of neurons
            n = module.weight.data.shape[1]
            m = module.weight.data.shape[0]
            variables.append(cp.Variable((m, 1)))
        if isinstance(module, torch.nn.LeakyReLU):
            variables.append((cp.Variable((m, 1)), cp.Variable((m, 1), boolean=True)))

    return variables, ibp_boxes


def MILP(model, input_range, alpha=1e-2):
    variables, ibp_boxes = MILP_vars(model, input_range, alpha=alpha)
    constraints = []

    for i in range(len(variables)):

        layer_type = ibp_boxes[i][0]

        if layer_type == "input":
            lower = ibp_boxes[i][1][:, [0]];
            upper = ibp_boxes[i][1][:, [1]]
            constraints.extend([variables[i][0] <= upper, variables[i][0] >= lower])

        elif layer_type == "leakyrelu":

            lower = ibp_boxes[i - 1][1][:, [0]];
            upper = ibp_boxes[i - 1][1][:, [1]]

            for j in range(len(lower)):
                l, u = lower[j], upper[j]

                # inactive neurons
                if (l <= 0) & (u <= 0):
                    constraints.extend([
                        variables[i][1][j] == 0,  # a==0
                        variables[i][0][j] == cp.multiply(alpha, variables[i - 1][j])  # y==alpha x
                    ])

                # active neurons
                elif (l > 0) & (u > 0):
                    constraints.extend([
                        variables[i][1][j] == 1,  # a==1
                        variables[i][0][j] == variables[i - 1][j]  # y==x
                    ])

                # uncertain neurons
                else:
                    constraints.extend([
                        # y >= x
                        variables[i][0][j] >= variables[i - 1][j],

                        # y >= alpha x
                        variables[i][0][j] >= cp.multiply(alpha, variables[i - 1][j]),

                        # y <= alpha*x + (1-alpha)U*a
                        variables[i][0][j] <= cp.multiply(alpha, variables[i - 1][j]) + cp.multiply(
                            cp.multiply(1 - alpha, u), variables[i][1][j]),

                        # y <= x + (1-alpha)L(a-1)
                        variables[i][0][j] <= variables[i - 1][j] + cp.multiply(cp.multiply(1 - alpha, l),
                                                                                variables[i][1][j] - 1)

                    ])

        elif layer_type == "linear":
            W = model[i - 1].weight.data.numpy();
            b = model[i - 1].bias.data.numpy().reshape(-1, 1)
            constraints.extend([variables[i] == W @ variables[i - 1][0] + b])  # x = W@y + b

        else:
            print("That layer isn't supported.")
            assert 0

    lower_bound = []
    for i in range(variables[-1].shape[0]):
        lower_objective = cp.Minimize(variables[-1][i])
        lower_objective = cp.Problem(lower_objective, constraints)
        lower_objective.solve()
        lower_bound.append(variables[-1].value[i].item())

    upper_bound = []
    for i in range(variables[-1].shape[0]):
        upper_objective = cp.Maximize(variables[-1][i])
        upper_objective = cp.Problem(upper_objective, constraints)
        upper_objective.solve()
        upper_bound.append(variables[-1].value[i].item())

    output_range = np.vstack((lower_bound, upper_bound)).T

    return output_range