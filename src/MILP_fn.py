import cvxpy as cp

import jax
import jax_verify
import functools
import jax.numpy as jnp
import torch
import numpy as np
from tqdm import tqdm


def pytorch_model_to_jax(torch_model: torch.nn.Sequential):
    params = []
    act = None

    # Extract params (weights, biases) from torch layers, to be used in
    # jax.
    # Note: This propagator assumes a feed-forward relu NN.
    for m in torch_model.modules():
        if isinstance(m, torch.nn.Sequential):
            continue
        elif isinstance(m, torch.nn.LeakyReLU) or isinstance(m,torch.nn.ReLU):
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


def verify_ibp(model, input_range, alpha=1e-2):
    zmink, zmaxk = input_range[:, 0].reshape(-1, 1), input_range[:, 1].reshape(-1, 1)
    intermediate_range = []

    for idx, m in enumerate(model.modules()):

        if isinstance(m, torch.nn.Sequential):
            continue

        elif isinstance(m, torch.nn.LeakyReLU) or isinstance(m,torch.nn.ReLU):
            type_ = "leakyrelu"
            zmink = np.maximum(zmink, alpha * zmink)
            zmaxk = np.maximum(zmaxk, alpha * zmaxk)

        elif isinstance(m, torch.nn.Linear):
            type_ = "linear"
            W, b = m.weight.data.numpy(), m.bias.data.numpy().reshape(-1, 1)

            mk = (zmaxk + zmink) / 2
            rk = (zmaxk - zmink) / 2

            mk = W @ mk + b
            rk = np.abs(W) @ rk

            zmink = mk - rk;
            zmaxk = mk + rk

        else:
            print("That layer isn't supported.")
            assert 0
        intermediate_range.append((type_, np.hstack((zmink, zmaxk))))

    return intermediate_range

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
        if isinstance(module, torch.nn.LeakyReLU) or isinstance(module,torch.nn.ReLU):
            variables.append((cp.Variable((m, 1)), cp.Variable((m, 1), boolean=True)))

    return variables, ibp_boxes


def MILP(model, input_range, alpha=1e-2,verbose=False):
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
        lower_objective.solve(solver=cp.GUROBI,verbose=verbose)
        lower_bound.append(variables[-1].value[i].item())

    upper_bound = []
    for i in range(variables[-1].shape[0]):
        upper_objective = cp.Maximize(variables[-1][i])
        upper_objective = cp.Problem(upper_objective, constraints)
        upper_objective.solve(solver=cp.GUROBI,verbose=verbose)
        upper_bound.append(variables[-1].value[i].item())

    output_range = np.vstack((lower_bound, upper_bound)).T

    return output_range

def MILP_attack(model,X,eps, alpha=1e-2):

    lb,ub = np.zeros((X.shape[0],1)),np.zeros((X.shape[0],1))

    for (i,x) in tqdm(enumerate(X),total=X.shape[0]):

        x = x.reshape(1,-1)
        input_range = nominal_and_epsilon_to_range(x, eps)
        output_range_milp = MILP(model, input_range)
        lb[i] = output_range_milp.squeeze()[0]
        ub[i] = output_range_milp.squeeze()[1]

    return lb,ub


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle


    def convert_range_to_box(box_range):
        corner = box_range[:, 0].tolist()
        width = (box_range[:, 1] - box_range[:, 0]).tolist()

        return corner, width

    INPUT_DIM = 10
    VERBOSE = True
    ALPHA = 0.01
    EPS = 0.2

    nn_model = torch.nn.Sequential(
        torch.nn.Linear(INPUT_DIM, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 2)
    )

    jax_model = pytorch_model_to_jax(nn_model)

    nominal_input = jnp.array(np.random.rand(1, INPUT_DIM))
    # input_range = jnp.hstack((np.zeros((28*28,1)),np.ones((28*28,1))))
    input_range = nominal_and_epsilon_to_range(nominal_input, EPS)

    # Example of a fwd pass on the NN using jax
    nominal_output_jax = jax_model(jnp.array(nominal_input))

    # Example of computing bounds using IBP as implemented by jax_verify
    input_bounds = np_range_to_jax_interval(input_range)
    output_bounds_ibp_jax = jax_verify.interval_bound_propagation(
        jax_model, input_bounds)
    output_range_ibp_jax = jax_interval_to_np_range(output_bounds_ibp_jax)

    print(f"output bounds via IBP (jax_verify): \n{output_range_ibp_jax}")

    output_range_ibp = verify_ibp(nn_model, input_range)[-1][-1]
    print(f"{output_range_ibp}")

    print(f"Difference btw Jax Verify IBP versus Own IBP {(output_range_ibp_jax-output_range_ibp)}")

    from time import time

    start_time = time()
    output_range_lp_leakyrelu = MILP(nn_model, input_range,alpha=ALPHA,verbose=VERBOSE)
    end_time = time()
    print(end_time - start_time)
    print(f"{output_range_lp_leakyrelu}")

    N = 100000
    x_mc = torch.distributions.uniform.Uniform(
        low=torch.FloatTensor(input_range[:, 0]),
        high=torch.FloatTensor(input_range[:, 1])
    ).sample(sample_shape=(N,))

    output_samples = nn_model(x_mc).detach()

    mc_range = torch.stack((output_samples.min(axis=0).values, output_samples.max(axis=0).values)).T

    corner_leakyrelu, width_leakyrelu = convert_range_to_box(output_range_lp_leakyrelu)
    corner_ibp, width_ibp = convert_range_to_box(output_range_ibp_jax)

    plt.figure()
    plt.scatter(output_samples[:, 0], output_samples[:, 1], c='b', alpha=.3, s=0.8)
    plt.gca().add_patch(
        Rectangle(corner_leakyrelu, width_leakyrelu[0], width_leakyrelu[1], edgecolor="pink", fill=False, lw=3))

    plt.gca().add_patch(
        Rectangle(corner_ibp, width_ibp[0], width_ibp[1], edgecolor="green", fill=False, lw=3))


    plt.legend(["Sample Output Points", "MILP Range Box","IBP Range Box"])
    plt.xlabel("X");
    plt.ylabel("Y")
    plt.show()

    plt.figure()
    plt.scatter(output_samples[:, 0], output_samples[:, 1], c='b', alpha=.3, s=0.8)
    plt.gca().add_patch(
        Rectangle(corner_leakyrelu, width_leakyrelu[0], width_leakyrelu[1], edgecolor="pink", fill=False, lw=3))

    plt.legend(["Sample Output Points", "MILP Range Box"])
    plt.xlabel("X");
    plt.ylabel("Y")
    plt.show()