import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt


reward_constants = [0, 0, 142, 764, 1309, 4427, 7001, 9328, 11997]  # for q=0.75

for i in range(2, 9):
    resources = np.load("results/best_resources_%d.npy" % i)
    plt.scatter(np.arange(1, len(resources) + 1), resources, s=20)
    plt.yscale("log")
    plt.axhline(y=reward_constants[i], color="r")
    plt.title("repeater length: " + str(i))
    plt.show()
