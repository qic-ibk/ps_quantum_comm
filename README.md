# Projective simulation for quantum communication

This repository is an archive for the code used in:

> Machine Learning for Long-Distance Quantum Communication <br>
> J. Wallnöfer, Alexey A. Melnikov, W. Dür, H. J. Briegel <br>
> [PRX Quantum **1**, 010301 (2020)](https://doi.org/10.1103/PRXQuantum.1.010301) <br>
> Preprint: https://arxiv.org/abs/1904.10797

## Relation to qic-ibk/projectivesimulation

This project contains two parts:
1. The PS agent used in this project is based on [qic-ibk/projectivesimulation](https://github.com/qic-ibk/projectivesimulation).
The logic of the agent is the same here, but the code architecture is a bit different
and some new functionalities that were needed for our scenarios have been added. (See below for a summary)

2. The code for the scenarios, i.e. formulating them as RL environments and
the associated code to run and evaluate them, is completely new.

### Summary of changes
* Changes to interaction
  - Interactions are now a class.
  - Additional statistics can be collected with GeneralInteraction
  - A mechanism for environments to pass additional information to the agent (e.g. what actions are available).
  - MetaAnalysisInteraction is a new type of interaction that allows to simultaneously 
    explore multiple outcomes of a probabilistic environment (e.g. outcomes of quantum measurements)
    and reconcile the results of the different branches into one agent.
* Separated agent logic (agents) and the way h- and glow-values are stored (brains).
  - Alternative way to store h- and glow-values as sparse matrices or clip network with clip and edge objects
* Option to reset glow values after very trial (i.e. make the agent aware that there are multiple trials that are completely separate)
* New type of agent that can deal with a changing action sets.


## Repository Structure
* main directory: Interaction files
* agents: multiple variants of the projective simulation (PS) agent.
* environments: the considered scenarios described as reinforcement learning environments
* run: run the scenarios and save the data. These are currently set up to be run on a machine with multiple processor cores
* utils: evaluation files that take the obtained data and make some plots

The version of the code on the master branch is usable for all scenarios and contains
the up-to-date licensing information. However, the exact versions of the code that 
were used to obtain the results in the paper are available and stored on dedicated 
branches in this repository:
<details>
  <summary>Dedicated branches for old versions</summary>
  
| Scenario              | Results in paper     | Branch          |
|-----------------------|----------------------|-----------------|
| Teleportation with pre-distributed entanglement | Fig. 2a-c | results/teleportation |
| Teleportation without pre-distributed entanglement | Fig. 2d-e | teleportation_variant |
| Entanglement purification | Fig. 3 | results/epp |
| Length 2 quantum repeater | Fig. 4 | results/scaling |
| Quantum repeater with delegation and perfect memories | Fig. 5 | results/scaling |
| Quantum repeater with delegation and imperfect memories | Fig. 6 | scaling_variant |
| Choosing location of repeater stations | Table I | scaling_variant |
| Appendix: Entanglement purification with automatic depolarization | Fig. 8 | results/epp |
| Appendix: Quantum repeater with different starting fidelities | Fig. 9 | results/teleportation |
</details>

### How to use
The run and evaluation files will only work if called from the main directory,
e.g. to run the teleportation scenario with Clifford gate set, navigate to the
main directory and use
```
python run/run_teleporation_clifford.py
```

The run-files assume they are run on a cluster computer with multiple processors. If you want to run this on
different hardware, changing `num_processes` accordingly is recommended.

## Related projects
If you are interested in the Projective Simulation agent, you can find
additional information at https://projectivesimulation.org and [qic-ibk/projectivesimulation](https://github.com/qic-ibk/projectivesimulation)

One variant of the teleportation environment used in this project is also part
of SciGym, a collection of problems in science that are presented as
Reinforcement Learning environments. See https://www.scigym.net, [HendrikPN/scigym](https://github.com/hendrikpn/scigym) and [jwallnoefer/gym-teleportation](https://github.com/jwallnoefer/gym-teleportation)
