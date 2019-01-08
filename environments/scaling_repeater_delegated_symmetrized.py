"""
"""

from __future__ import print_function, division
from .scaling_repeater_delegated_env import _Pair
import numpy as np
from copy import deepcopy
from itertools import chain
try:
    from future_builtins import filter  # python 2 compatibility
except ImportError:
    pass

np.seterr(over="raise")  # can happen if resources get too high and should be counted as a failure

ACTION_PURIFY = "purify"
ACTION_SWAP = "swap"
ACTION_COMPOSITE = "composite"


class _Action(object):
    def __init__(self, type, block_size=None):
        self.type = type
        if self.type == ACTION_COMPOSITE:
            assert block_size is not None
            self.block_size = block_size

    def __eq__(self, other):
        if isinstance(other, _Action):
            if self.type == other.type:
                if self.type == ACTION_PURIFY:
                    return True
                elif self.type == ACTION_SWAP:
                    return True
                elif self.type == ACTION_COMPOSITE:
                    return self.block_size == other.block_size
        return False

    def __neq__(self, other):  # python2 compatibility
        return not self.__eq__(other)

    def __repr__(self):
        if self.type == ACTION_PURIFY:
            return "_Action(%s)" % (self.type)
        elif self.type == ACTION_SWAP:
            return "_Action(%s)" % (self.type)
        elif self.type == ACTION_COMPOSITE:
            return "_Action(%s, %s)" % (self.type, repr(self.block_size))


class TaskEnvironment(object):
    def __init__(self, length=2, available_block_lengths=[], start_fid=0.51, target_fid=0.9, reward_constant=1, reward_exponent=1, p=1.0, delegated_solutions=None):
        self.length = length
        if isinstance(start_fid, (int, float)):
            self.start_fid = [start_fid] * length
        elif isinstance(start_fid, (list, tuple)) and len(start_fid) == length and np.all(start_fid == start_fid[0]):
            self.start_fid = start_fid
        else:
            raise ValueError(repr(start_fid) + " as initial fidelities is not supported.")
        self.gate_noise = p
        self.delegated_solutions = delegated_solutions
        self.target_fid = target_fid
        self.reward_constant = reward_constant  # could simply define a function for this
        self.reward_exponent = reward_exponent
        self.state = [_Pair((i, i + 1), fid=self.start_fid[i], resources=1.0) for i in range(self.length)]
        self.base_actions = [_Action(ACTION_PURIFY), _Action(ACTION_SWAP)]
        self.n_base_actions = len(self.base_actions)
        self.available_actions = [i for i in range(len(self.base_actions))]
        self.action_list = deepcopy(self.base_actions)
        self.available_block_lengths = available_block_lengths
        for block_size in self.available_block_lengths:
            self._add_action(ACTION_COMPOSITE, block_size)

    def _add_action(self, type, info):  # inconsistent interface with _remove_action, but _remove_action should remain the way it is
        new_action = _Action(type, info)
        if new_action not in self.action_list:  # this works because of custom __eq__ method
            self.action_list += [new_action]
        my_index = self.action_list.index(new_action)  # this works because of custom __eq__ method
        if my_index not in self.available_actions:
            self.available_actions += [my_index]
            self.available_actions.sort()

    def _remove_action(self, action):
        try:
            my_index = self.action_list.index(action)
        except ValueError:
            raise ValueError("Action with type %s could not be removed because it does not exist." % action.type)
        if my_index in self.available_actions:  # so we don't get errors when we remove an unavailable action
            self.available_actions.remove(my_index)

    def _purify(self, stations):
        pair = next(filter(lambda x: x.stations == stations, self.state), None)
        if pair is None:
            raise ValueError("There is no pair between stations %s that can be purified." % str(stations))
        f = pair.fid
        if self.gate_noise != 1.0:  # imperfect cnot gates
            qq = (4 * f - 1) / 3
            f = (3 * qq * self.gate_noise**2 + 1) / 4
        p_suc = f**2 + 2 * f * (1 - f) / 3 + 5 * (1 - f)**2 / 9
        pair.fid = (f**2 + (1 - f)**2 / 9) / p_suc  # directly modifies self.state
        pair.resources *= 2 / p_suc

    def _entanglement_swapping(self, station):
        pair1 = next(filter(lambda x: x.right_station == station, self.state), None)
        pair2 = next(filter(lambda x: x.left_station == station, self.state), None)
        if pair1 is None or pair2 is None:
            raise ValueError("Entanglement swapping at station %s failed because two pairs could not be found." % str(station))
        fid1 = pair1.fid
        fid2 = pair2.fid
        q1 = (4 * fid1 - 1) / 3
        q2 = (4 * fid2 - 1) / 3
        fid_new = (3 * q1 * q2 + 1) / 4
        new_left = pair1.left_station
        new_right = pair2.right_station
        assert new_right > new_left
        self.state.remove(pair1)
        self.state.remove(pair2)
        self.state += [_Pair((new_left, new_right), fid=fid_new, resources=pair1.resources + pair2.resources)]
        self.state.sort(key=lambda x: x.stations)
        # now adjust actions appropriately

    def _remove_long_blocks(self):
        for block_size in self.available_block_lengths:
            if block_size > len(self.state):
                self._remove_action(_Action(ACTION_COMPOSITE, block_size))

    def _observation(self):
        """Get a flattened tuple representation of the current state."""
        return tuple(chain(*((pair.left_station, pair.right_station, int(pair.fid * 1000)) for pair in self.state)))

    def reset(self):
        self.state = [_Pair((i, i + 1), fid=self.start_fid[i], resources=1.0) for i in range(self.length)]
        self.available_actions = [i for i in range(len(self.base_actions))]
        for block_size in self.available_block_lengths:
            self._add_action(ACTION_COMPOSITE, block_size)
        # IMPORTANT: self.action_list is persistent between trials for consistent numbering of actions
        return self._observation(), {"available_actions": self.available_actions}

    def _check_success(self):
        if len(self.state) == 1:
            pair = self.state[0]
            if pair.fid >= self.target_fid:
                return True
        return False

    def _check_failed(self):
        for pair in self.state:
            if pair.fid <= 0.5:
                return True
        return False

    def _reward_function(self, resources):
        reward = (self.reward_constant / resources)**self.reward_exponent
        if reward > 1:
            self.reward_constant = resources
            reward = 1
        return reward

    def composite_action_from_history(self, history):
        """Return a dictionary that can be used in a new environment with higher length.

            history: list of (observation, action_index) tuples
        """
        action_sequence = []
        self.reset()
        for _, action_index in history:
            action = self.action_list[action_index]
            if action.type == ACTION_COMPOSITE:
                block_actions = self.delegated_solutions.get_block_action(self.state[0].fid, action.block_size)
                action_sequence += block_actions
                self.move(action_index)
            else:
                self.move(action_index)
                action_sequence += [action]
        return {"block_size": self.length, "actions": action_sequence}

    def move(self, action):
        if action not in self.available_actions:
            self._remove_long_blocks()
            raise ValueError("Action with number %s is not available at this time." % action)
        my_action = self.action_list[action]
        if my_action.type == ACTION_PURIFY:
            try:
                for pair in self.state:
                    self._purify(pair.stations)
            except FloatingPointError:  # happens when resources grow too large and should be counted as failure
                reward = 0
                episode_finished = 1
                observation = self._observation()
                return observation, reward, episode_finished, {"available_actions": self.available_actions}
        elif my_action.type == ACTION_SWAP:
            stations = [pair.right_station for pair in self.state[::2]]  # only take every second station
            for station in stations:
                self._entanglement_swapping(station)
            self._remove_long_blocks()
            if len(self.state) == 1:  # no more swapping if no middle station is there anymore
                # print("swap is being removed")
                self._remove_action(_Action(ACTION_SWAP))
                # print(self.available_actions)
        elif my_action.type == ACTION_COMPOSITE:
            block_actions = self.delegated_solutions.get_block_action(self.state[0].fid, my_action.block_size)
            if block_actions is None:  # abort if no action is found
                reward = 0
                episode_finished = 1
                observation = self._observation()
                return observation, reward, episode_finished, {"available_actions": self.available_actions}
            for act in block_actions:
                act_index = self.action_list.index(act)
                self.move(act_index)

        observation = self._observation()
        if self._check_success():
            reward = self._reward_function(self.state[0].resources)
            episode_finished = 1
        else:
            reward = 0
            episode_finished = int(self._check_failed())

        return observation, reward, episode_finished, {"available_actions": self.available_actions}

    def get_resources(self):
        if self._check_success():
            return self.state[0].resources
        else:
            return float("NaN")
