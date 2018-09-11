"""Evaluate results from teleportation_env
"""

import numpy as np
import matplotlib.pyplot as plt


def get_step_dicts(path, num=64):
    res = []
    for i in range(num):
        print(i)
        try:
            with open(path + "/step_curve_%d.txt" % i, "r") as f:
                step_curve = np.loadtxt(f)
        except FileNotFoundError:
            print(path + "/step_curve_%d.txt" + " is missing")
            continue
        a = step_curve[-2000:]
        res += [dict(zip(*np.unique(a, return_counts=True)))]
    return res


def count_solutions(count_dict):
    count_dict = {k: v if int(k) != 0 else 0 for k, v in count_dict.items()}
    val = np.array(list(count_dict.values()))
    num_solutions = np.sum((400 < val) * (val < 600)) + 2 * np.sum((800 < val) * (val < 1200))
    return num_solutions


def solution_steps(count_dict):
    auxlist = [(v, k) for k, v in count_dict.items()]
    auxsteplist = []
    for v, k in auxlist:
        if 400 < v < 600 and int(k) != 0:
            auxsteplist += [int(k)]
        elif 800 < v < 1200 and int(k) != 0:
            auxsteplist += [int(k), int(k)]
    auxsteplist += [0] * (4 - len(auxsteplist))
    return auxsteplist


def cumulative_steps(count_dict):
    auxlist = [(v, k) for k, v in count_dict.items()]
    auxcount = 0
    auxsteps = 0
    for v, k in auxlist:
        if 400 < v < 600 and int(k) != 0:
            auxcount += 1
            auxsteps += int(k)
        elif 800 < v < 1200 and int(k) != 0:
            auxcount += 2
            auxsteps += 2 * int(k)
    if auxcount == 4:
        return auxsteps
    else:
        return 0


count_solutions_vectorized = np.vectorize(count_solutions, otypes=[np.int])
cumulative_steps_vectorized = np.vectorize(cumulative_steps, otypes=[np.int])


def solution_steps_vectorized(mylist):
    return np.array([solution_steps(dict) for dict in mylist]).flatten()


if __name__ == "__main__":
    dict_lists = [get_step_dicts("./results/128_agents_60k_trials/eta_%d" % k, num=128) for k in [1, 15, 2, 3, 4, 5]]  # [1, 15, 2, 25, 3, 35, 4, 45, 5]]
    solution_count_array = [count_solutions_vectorized(dict_list) for dict_list in dict_lists]
    # solution_steps_array = [solution_steps_vectorized(dict_list) for dict_list in dict_lists]
    cumulative_steps_array = [cumulative_steps_vectorized(dict_list) for dict_list in dict_lists]

    plt.hist(solution_count_array, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], label=["eta=0.1", "eta=0.15", "eta=0.2", "eta=0.3", "eta=0.4", "eta=0.5"])
    plt.legend()
    plt.title("128 agents, 60k trials")
    plt.xlabel("No. of solutions found")
    plt.ylabel("No. of agents")
    plt.savefig("presentation_solution_count.png")
    plt.show()
    # plt.hist(solution_steps_array, bins=np.arange(-0.5, 22.5, 1))#, label=["eta=0.1", "eta=0.2", "eta=0.3", "eta=0.4", "eta=0.5", "eta=0.15"])
    # # plt.legend()
    # plt.xticks(np.arange(0, 21))
    # plt.xlabel("Length of solution")
    # plt.ylabel("No. of solutions")
    # plt.show()
    plt.hist(cumulative_steps_array, bins=np.arange(24.5, 50.5, 1), label=["eta=0.1", "eta=0.15", "eta=0.2", "eta=0.3", "eta=0.4", "eta=0.5"])
    plt.legend()
    plt.title("128 agents, 60k trials")
    plt.xlabel("Cumulative length of agents with 4 solutions")
    plt.ylabel("No. of agents")
    plt.savefig("presentation_cumulative_steps.png")
    plt.show()

    # cumulative_steps_array = [csteps[csteps != 0] for csteps in cumulative_steps_array]
    # amounts = [len(csteps) / 128.0 for csteps in cumulative_steps_array]
    # my_array = [np.sum(csteps) / len(csteps) for csteps in cumulative_steps_array]
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # # plt.grid()
    # ax1.scatter([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], my_array)
    # ax1.set_ylim(28, 36)
    # ax1.set_xlabel("glow parameter η")
    # ax1.set_ylabel("average number of cumulative_steps", color="C0")
    # ax2.plot([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], amounts, "C1-")
    # ax2.set_ylabel("fraction of agents with all 4 solutions", color="C1")
    # ax2.set_ylim(0, 1)
    # # plt.rc('text', usetex=True)
    # # plt.rc('font', family='serif')
    # # plt.xlabel(r"$\eta$")
    # plt.show()

exit()


# if __name__ == "__main__":
#     b = np.zeros(64)
#     for i in range(64):
#         print(i)
#         with open("./results/eta_1/step_curve_%d.txt" % i, "r") as f:
#             step_curve = np.loadtxt(f)
#         a = step_curve[-2000:]
#         count = dict(zip(*np.unique(a, return_counts=True)))
#         b[i] = count_solutions(count)
#     c = np.zeros(64)
#     for i in range(64):
#         print(i)
#         with open("./results/eta_2/step_curve_%d.txt" % i, "r") as f:
#             step_curve = np.loadtxt(f)
#         a = step_curve[-2000:]
#         count = dict(zip(*np.unique(a, return_counts=True)))
#         c[i] = count_solutions(count)
#     d = np.zeros(64)
#     for i in range(64):
#         print(i)
#         try:
#             with open("./results/eta_3/step_curve_%d.txt" % i, "r") as f:
#                 step_curve = np.loadtxt(f)
#         except FileNotFoundError:
#             print(i, "is missing")
#             continue
#         a = step_curve[-2000:]
#         count = dict(zip(*np.unique(a, return_counts=True)))
#         d[i] = count_solutions(count)
#     eff = np.zeros(64)
#     for i in range(64):
#         print(i)
#         try:
#             with open("./results/eta_4/step_curve_%d.txt" % i, "r") as f:
#                 step_curve = np.loadtxt(f)
#         except FileNotFoundError:
#             print(i, "is missing")
#             continue
#         a = step_curve[-2000:]
#         count = dict(zip(*np.unique(a, return_counts=True)))
#         eff[i] = count_solutions(count)
#     plt.hist([b, c, d, eff], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
#     plt.show()
#     exit()


    # b = np.zeros(64)
    # mycount = 0
    # for i in range(64):
    #     print(i)
    #     try:
    #         with open("./results/eta_1/step_curve_%d.txt" % i, "r") as f:
    #             step_curve = np.loadtxt(f)
    #     except FileNotFoundError:
    #         print(i, "is missing")
    #         continue
    #
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     auxlist = [(v, k) for k, v in count.items()]
    #     # print(auxlist)
    #     auxcount = 0
    #     auxsteps = 0
    # b = np.zeros(64)
    # for i in range(64):
    #     print(i)
    #     with open("./results/eta_1/step_curve_%d.txt" % i, "r") as f:
    #         step_curve = np.loadtxt(f)
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     b[i] = count_solutions(count)
    # c = np.zeros(64)
    # for i in range(64):
    #     print(i)
    #     with open("./results/eta_2/step_curve_%d.txt" % i, "r") as f:
    #         step_curve = np.loadtxt(f)
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     c[i] = count_solutions(count)
    # d = np.zeros(64)
    # for i in range(64):
    #     print(i)
    #     try:
    #         with open("./results/eta_3/step_curve_%d.txt" % i, "r") as f:
    #             step_curve = np.loadtxt(f)
    #     except FileNotFoundError:
    #         print(i, "is missing")
    #         continue
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     d[i] = count_solutions(count)
    # eff = np.zeros(64)
    # for i in range(64):
    #     print(i)
    #     try:
    #         with open("./results/eta_4/step_curve_%d.txt" % i, "r") as f:
    #             step_curve = np.loadtxt(f)
    #     except FileNotFoundError:
    #         print(i, "is missing")
    #         continue
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     eff[i] = count_solutions(count)
    # plt.hist([b, c, d, eff], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    # plt.show()
    # exit()
    #
    #
    # b = np.zeros(64)
    # mycount = 0
    # for i in range(64):
    #     print(i)
    #     try:
    #         with open("./results/eta_1/step_curve_%d.txt" % i, "r") as f:
    #             step_curve = np.loadtxt(f)
    #     except FileNotFoundError:
    #         print(i, "is missing")
    #         continue
    #
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     auxlist = [(v, k) for k, v in co
    #     for v, k in auxlist:
    #         if 400 < v < 600 and int(k) != 0:
    #             auxcount += 1
    #             auxsteps += int(k)
    #         elif 800 < v < 1200 and int(k) != 0:
    #             auxcount += 2
    #             auxsteps += 2 * int(k)
    #     if auxcount == 4:
    #         b[i] = auxsteps
    #
    # c = np.zeros(64)
    # mycount = 0
    # for i in range(64):
    #     print(i)
    #     try:
    #         with open("./results/eta_2/step_curve_%d.txt" % i, "r") as f:
    #             step_curve = np.loadtxt(f)
    #     except FileNotFoundError:
    #         print(i, "is missing")
    #         continue
    #
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     auxlist = [(v, k) for k, v in count.items()]
    #     # print(auxlist)
    #     auxcount = 0
    #     auxsteps = 0
    #     for v, k in auxlist:
    #         if 400 < v < 600 and int(k) != 0:
    #             auxcount += 1
    #             auxsteps += int(k)
    #         elif 800 < v < 1200 and int(k) != 0:
    #             auxcount += 2
    #             auxsteps += 2 * int(k)
    #     if auxcount == 4:
    #         c[i] = auxsteps
    #
    # d = np.zeros(64)
    # mycount = 0
    # for i in range(64):
    #     print(i)
    #     try:
    #         with open("./results/eta_3/step_curve_%d.txt" % i, "r") as f:
    #             step_curve = np.loadtxt(f)
    #     except FileNotFoundError:
    #         print(i, "is missing")
    #         continue
    #
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     auxlist = [(v, k) for k, v in count.items()]
    #     # print(auxlist)
    #     auxcount = 0
    #     auxsteps = 0
    #     for v, k in auxlist:
    #         if 400 < v < 600 and int(k) != 0:
    #             auxcount += 1
    #             auxsteps += int(k)
    #         elif 800 < v < 1200 and int(k) != 0:
    #             auxcount += 2
    #             auxsteps += 2 * int(k)
    #     if auxcount == 4:
    #         d[i] = auxsteps
    #
    # d = np.zeros(64)
    # mycount = 0
    # for i in range(64):
    #     print(i)
    #     try:
    #         with open("./results/eta_3/step_curve_%d.txt" % i, "r") as f:
    #             step_curve = np.loadtxt(f)
    #     except FileNotFoundError:
    #         print(i, "is missing")
    #         continue
    #
    #     a = step_curve[-2000:]
    #     count = dict(zip(*np.unique(a, return_counts=True)))
    #     auxlist = [(v, k) for k, v in count.items()]
    #     # print(auxlist)
    #     auxcount = 0
    #     auxsteps = 0
    #     for v, k in auxlist:
    #         if 400 < v < 600 and int(k) != 0:
    #             auxcount += 1
    #             auxsteps += int(k)
    #         elif 800 < v < 1200 and int(k) != 0:
    #             auxcount += 2
    #             auxsteps += 2 * int(k)
    #     if auxcount == 4:
    #         d[i] = auxsteps
    #
    # plt.hist([b, c, d, eff], bins=np.arange(24.5, 50.5, 1), label=["eta=0.1", "eta=0.2", "eta=0.3", "eta=0.4"])
    # plt.xticks(np.arange(25, 51))
    # plt.xlabel("Cumulative number of steps for agents that found 4 solutions")
    # plt.ylabel("Number of agents")
    # plt.legend(loc="upper right")
    # plt.show()
