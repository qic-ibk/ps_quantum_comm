"""
Copyright 2020 Julius Wallnöfer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import print_function, division
from .abstract_environment import AbstractEnvironment
from copy import deepcopy
from itertools import chain
import numpy as np
from warnings import warn
try:
    from future_builtins import filter  # python 2 compatibility
except ImportError:
    pass

np.seterr(over="raise")  # can happen if resources get too high and should be counted as a failure

ACTION_PURIFY = "purify"
ACTION_SWAP = "swap"
ACTION_COMPOSITE = "composite"

c = 2 * 10**8


class _Action(object):
    def __init__(self, type, info):  # there must be a better way
        self.type = type
        if self.type == ACTION_PURIFY:
            assert isinstance(info, tuple) and len(info) == 2
            self.stations = info
        elif self.type == ACTION_SWAP:
            assert isinstance(info, int)
            self.station = info
        elif self.type == ACTION_COMPOSITE:
            assert isinstance(info, list)
            self.block_size = info[0]
            self.involved_links = info[1]
        else:
            raise ValueError("Action of type %s is not supported." % type)

    def __eq__(self, other):
        if isinstance(other, _Action):
            if self.type == other.type:
                if self.type == ACTION_PURIFY:
                    return self.stations == other.stations
                elif self.type == ACTION_SWAP:
                    return self.station == other.station
                elif self.type == ACTION_COMPOSITE:
                    return self.block_size == other.block_size and self.involved_links == other.involved_links
        return False

    def __neq__(self, other):  # python2 compatibility
        return not self.__eq__(other)

    def __repr__(self):
        if self.type == ACTION_PURIFY:
            return "_Action(%s, %s)" % (self.type, repr(self.stations))
        elif self.type == ACTION_SWAP:
            return "_Action(%s, %s)" % (self.type, repr(self.station))
        elif self.type == ACTION_COMPOSITE:
            return "_Action(%s, %s)" % (self.type, repr([self.block_size, self.involved_links]))


class _Pair(object):
    def __init__(self, stations, fid=1.0, resources=1.0, time=0.0, distance=None):
        assert len(stations) == 2
        i = stations[0]
        j = stations[1]
        if i < j:
            self.left_station = i
            self.right_station = j
            self.stations = (i, j)
        else:
            self.left_station = j
            self.right_station = i
            self.stations = (j, i)
        self.fid = fid
        self.resources = resources
        self.time = time
        self.distance = distance

    def advance_time(self, new_time, alpha=0.0):
        if alpha != 0:
            assert new_time >= self.time
            delta_t = new_time - self.time
            memory_noise = np.exp(- alpha * delta_t)
            qq = (4 * self.fid - 1) / 3
            self.fid = (3 * qq * memory_noise**2 + 1) / 4
        self.time = new_time

    def __repr__(self):
        return "_Pair(%s, fid=%s, resources=%s, time=%s, distance=%s)" % (repr(self.stations), repr(self.fid), repr(self.resources), repr(self.time), repr(self.distance))


class TaskEnvironment(AbstractEnvironment):
    """
    """
    def __init__(self, repeater_distances=None, start_fid=None, available_block_lengths=[], kappa=0.000045, target_fid=0.9, reward_constant=1, reward_exponent=1, p=1.0, alpha=0.0, delegated_solutions=None):
        self.channel_kappa = kappa
        if repeater_distances is None and start_fid is None:
            raise ValueError("One of repeater_distances OR start_fids need to be specified.")
        elif repeater_distances is None and start_fid is not None:
            self.length = len(start_fid)
            self.start_fid = start_fid
            qs = (4 * np.array(start_fid) - 1) / 3
            self.repeater_distances = -1 / self.channel_kappa * np.log(qs)
        elif repeater_distances is not None and start_fid is None:
            self.length = len(repeater_distances)
            self.repeater_distances = repeater_distances
            self.start_fid = (1 + 3 * np.exp(- self.channel_kappa * np.array(self.repeater_distances))) / 4
        elif repeater_distances is not None and start_fid is not None:
            raise ValueError("Only one of repeater_distances OR start_fids need to be specified.")
        self.gate_noise = p
        self.memory_alpha = alpha
        self.delegated_solutions = delegated_solutions
        self.target_fid = target_fid
        self.reward_constant = reward_constant  # could simply define a function for this
        self.reward_exponent = reward_exponent
        self.state = [_Pair((i, i + 1), fid=self.start_fid[i], resources=1.0, distance=self.repeater_distances[i]) for i in range(self.length)]
        self.base_actions = [_Action(ACTION_SWAP, i) for i in range(1, self.length)] + [_Action(ACTION_PURIFY, pair.stations) for pair in self.state]
        self.n_base_actions = len(self.base_actions)
        self.available_actions = [i for i in range(len(self.base_actions))]
        self.action_list = deepcopy(self.base_actions)
        self.available_block_lengths = available_block_lengths
        self._recalc_composite_actions()

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

    def _remove_composites_involving_station(self, station):
        for action in filter(lambda x: x.type == ACTION_COMPOSITE, self.action_list):
            for link in action.involved_links:
                if station in link:
                    self._remove_action(action)
                    break

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
        pair.advance_time(pair.time + pair.distance / c, self.memory_alpha)

    def _entanglement_swapping(self, station):
        pair1 = next(filter(lambda x: x.right_station == station, self.state), None)
        pair2 = next(filter(lambda x: x.left_station == station, self.state), None)
        if pair1 is None or pair2 is None:
            raise ValueError("Entanglement swapping at station %s failed because two pairs could not be found." % str(station))
        # pairs must be at the same time for entanglement swapping
        if pair1.time > pair2.time:
            pair2.advance_time(pair1.time, self.memory_alpha)
        elif pair2.time > pair1.time:
            pair1.advance_time(pair2.time, self.memory_alpha)
        fid1 = pair1.fid
        fid2 = pair2.fid
        q1 = (4 * fid1 - 1) / 3
        q2 = (4 * fid2 - 1) / 3
        fid_new = (3 * q1 * q2 * self.gate_noise**2 + 1) / 4  # add gate noise
        new_left = pair1.left_station
        new_right = pair2.right_station
        assert new_right > new_left
        self.state.remove(pair1)
        self.state.remove(pair2)
        self.state += [_Pair((new_left, new_right), fid=fid_new, resources=pair1.resources + pair2.resources, distance=pair1.distance + pair2.distance)]
        self.state.sort(key=lambda x: x.stations)
        # now adjust actions appropriately
        self._remove_action(_Action(ACTION_SWAP, station))
        self._remove_action(_Action(ACTION_PURIFY, pair1.stations))
        self._remove_action(_Action(ACTION_PURIFY, pair2.stations))
        self._remove_composites_involving_station(station)
        self._add_action(ACTION_PURIFY, (new_left, new_right))
        self._recalc_composite_actions()

    def _shift_actions(self, actions, involved_links):
        involved_stations = [involved_links[0][0]] + [link[1] for link in involved_links]
        shifted_actions = []
        for action in actions:
            if action.type == ACTION_SWAP:
                my_station = involved_stations[action.station]
                shifted_actions += [_Action(ACTION_SWAP, my_station)]
            elif action.type == ACTION_PURIFY:
                my_stations = (involved_stations[action.stations[0]], involved_stations[action.stations[1]])
                shifted_actions += [_Action(ACTION_PURIFY, my_stations)]
            else:
                raise ValueError("Disallowed action of type %s found when trying to shift actions." % action.type)
        return shifted_actions

    def _recalc_composite_actions(self):
        for block_size in self.available_block_lengths:
            num_locations = len(self.state) - (block_size - 1)
            for i in range(num_locations):
                involved_links = [pair.stations for pair in self.state[i:i + block_size]]
                self._add_action(ACTION_COMPOSITE, [block_size, involved_links])

    def _observation(self):
        """Get a flattened tuple representation of the current state."""
        return tuple(chain(*((pair.left_station, pair.right_station, int(pair.fid * 1000), int(pair.time * 10**6)) for pair in self.state)))

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

    def get_resources(self):
        if self._check_success():
            return self.state[0].resources
        else:
            return float("NaN")

    def composite_action_from_history(self, history):
        """Return a dictionary that can be used in a new environment with higher length.

            history: list of (observation, action_index) tuples
        """
        action_sequence = []
        self.reset()
        for _, action_index in history:
            action = self.action_list[action_index]
            if action.type == ACTION_COMPOSITE:
                block_actions = self._find_delegated_action(action)
                action_list = self._shift_actions(block_actions, action.involved_links)
                action_sequence += action_list
                self.move(action_index)
            else:
                self.move(action_index)
                action_sequence += [action]
        return {"block_size": self.length, "actions": action_sequence}

    def reset(self):
        self.state = [_Pair((i, i + 1), fid=self.start_fid[i], resources=1.0, distance=self.repeater_distances[i]) for i in range(self.length)]
        self.available_actions = [i for i in range(len(self.base_actions))]
        self._recalc_composite_actions()
        # IMPORTANT: self.action_list is persistent between trials for consistent numbering of actions
        return self._observation(), {"available_actions": self.available_actions}

    def _reward_function(self, resources):
        reward = (self.reward_constant / resources)**self.reward_exponent
        if reward > 1:
            self.reward_constant = resources
            reward = 1
        return reward

    def _find_delegated_action(self, action):
        links = action.involved_links
        fids = []
        for link in links:
            pair = next(filter(lambda x: x.stations == link, self.state))
            fids += [pair.fid]
        return self.delegated_solutions.get_block_action(fids)

    def move(self, action):
        if action not in self.available_actions:
            raise ValueError("Action with number %s is not available at this time." % action)
        my_action = self.action_list[action]
        if my_action.type == ACTION_PURIFY:
            try:
                self._purify(my_action.stations)
            except FloatingPointError:
                reward = 0
                episode_finished = 1
                observation = self._observation()
                return observation, reward, episode_finished, {"available_actions": self.available_actions}
        elif my_action.type == ACTION_SWAP:
            self._entanglement_swapping(my_action.station)
        elif my_action.type == ACTION_COMPOSITE:
            block_actions = self._find_delegated_action(my_action)
            if block_actions is None:  # abort if no action is found
                reward = 0
                episode_finished = 1
                observation = self._observation()
                return observation, reward, episode_finished, {"available_actions": self.available_actions}
            action_list = self._shift_actions(block_actions, my_action.involved_links)
            for act in action_list:
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
