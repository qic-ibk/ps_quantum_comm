"""Evaluate results from teleportation_env and teleportation_universal_env.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os

num_agents = 100
sparsity = 10
etas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
result_path_clifford = "results/teleportation/clifford_gates/raw/"
result_path_universal = "results/teleportation/universal_gates/raw/"
plot_path_clifford = "results/teleportation/clifford_gates/plot_ready/"
plot_path_universal = "results/teleportation/universal_gates/plot_ready/"


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_label_multiplicator(eta):
    return 10**(len(str(eta)) - 2)


def get_step_curves(path, num=100, to_txt_path=None):
    res = []
    for i in range(num):
        step_curve = np.load(path + "step_curve_%d.npy" % i)
        res += [step_curve]
        if to_txt_path is not None:
            assert_dir(to_txt_path)
            np.savetxt(to_txt_path + "step_curve_%d.txt" % i, step_curve, fmt="%-d")
    return res


def count_dict(step_curve):
    a = step_curve[-400:]  # this assumes sparsity 10 of step curves
    return dict(zip(*np.unique(a, return_counts=True)))


# def has_all_solutions(step_curve, failure_number=10000):
#     a = step_curve[-400:]
#     return np.all(a != failure_number)


def cumulative_steps(count_dict, failure_number=10000):
    auxlist = [(v, int(k)) for k, v in count_dict.items()]
    if np.any([length == failure_number for count, length in auxlist]):
        return np.nan
    if len(auxlist) == 4:
        return np.sum([length for count, length in auxlist])
    elif len(auxlist) == 3:
        aux = max(auxlist, key=lambda x: x[0])
        length_list = [length for count, length in auxlist] + [aux[0]]
        return np.sum([length_list])
    else:
        raise ValueError("Could not compute cumulative_steps for " + repr(count_dict))


def evaluate(result_path, plot_path):
    average_length_list = []
    agent_fraction_list = []
    for eta in etas:
        eta_plot_path = plot_path + "eta_%d/" % (eta * get_label_multiplicator(eta))
        eta_result_path = result_path + "eta_%d/" % (eta * get_label_multiplicator(eta))
        assert_dir(eta_plot_path)
        step_curves = get_step_curves(eta_result_path, num=num_agents, to_txt_path=eta_plot_path)
        count_dicts = [count_dict(step_curve) for step_curve in step_curves]
        cumulative_steps_list = np.array([cumulative_steps(count) for count in count_dicts])
        aux = cumulative_steps_list[cumulative_steps_list != np.nan]
        print(aux)
        average_length = np.sum(aux, axis=0) / len(aux)
        agent_fraction = len(aux) / num_agents
        average_length_list += [average_length]
        agent_fraction_list += [agent_fraction]
        # first plot is the example step_curve
        my_example = step_curves[0]
        my_example[my_example == 10000] = 50
        np.savetxt(eta_plot_path + "example_agent_sparsity_%d.txt" % sparsity, my_example, fmt="%-d")
        plt.scatter(np.arange(1, len(my_example) + 1, sparsity), my_example)
        plt.title("Example agent for η=%.2f" % eta)
        plt.xlabel("Trial number")
        plt.ylabel("Number of steps")
        plt.savefig(eta_plot_path + "example_agent.png")
        plt.show()
        # second plot is the averages, this is missing the variances
        # still not convinced this is a the right way to do averages and variances
        for curve in step_curves:
            curve[curve == 10000] = 50
        average_curve = np.sum(step_curves, axis=0) / num_agents
        np.savetxt(eta_plot_path + "average_curve_sparsity_%d.txt" % sparsity, average_curve, fmt="%.6f")
        plt.plot(np.arange(1, len(average_curve) + 1, sparsity), average_curve)
        plt.title("Average of %d agents" % num_agents)
        plt.xlabel("Trial number")
        plt.ylabel("Number of steps")
        plt.savefig(eta_plot_path + "average_curve.png")
        plt.show()
    # now analysis for different etas
    np.savetxt(plot_path + "average_solution_length.txt", average_length_list, fmt="%.6f")
    np.savetxt(plot_path + "fraction_of_agents.txt", agent_fraction_list, fmt="%.6f")
    np.savetxt(plot_path + "etas.txt", fmt="%.2f")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # plt.grid()
    ax1.scatter(etas, average_length_list, c="C0")
    ax1.set_ylim(28, 36)
    ax1.set_xlabel("glow parameter η")
    ax1.set_ylabel("average number of cumulative_steps", color="C0")
    ax2.plot(etas, agent_fraction_list, "C1-")
    ax2.set_ylabel("fraction of agents with all 4 solutions", color="C1")
    ax2.set_ylim(0, 1)
    plt.savefig(plot_path + "eta_analysis.png")
    plt.show()


if __name__ == "__main__":
    evaluate(result_path_clifford, plot_path_clifford)
    evaluate(result_path_universal, plot_path_universal)
