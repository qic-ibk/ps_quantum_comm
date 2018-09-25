"""
"""

from __future__ import print_function, division
from .abstract_environment import AbstractEnvironment
import numpy as np
from copy import deepcopy
from itertools import chain
try:
    from future_builtins import filter
except ImportError:
    pass

ACTION_PURIFY = "purify"
ACTION_SWAP = "swap"
ACTION_COMPOSITE = "composite"


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
            self.composite_id = info[0]
            self.involved_links = info[1]
            self.constituent_actions = info[2]
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
                    return self.composite_id == other.composite_id and self.involved_pairs == other.involved_pairs
        return False

    def __neq__(self, other):  # python2 compatibility
        return not self.__eq__(other)


class _Pair(object):
    def __init__(self, stations, fid=1.0):
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



class TaskEnvironment(AbstractEnvironment):
    """
    """
    def __init__(self, length=2, additional_actions=[], q=0.57):
        self.length = length
        self.additional_actions = additional_actions
        self.start_fid = (3 * q + 1) / 4
        self.base_actions = [_Action(ACTION_SWAP, i) for i in range(1, self.length)] + [_Action(ACTION_PURIFY, pair.stations) for pair in self.state]
        self.action_list = deepcopy(self.base_actions)  # not sure if deepcopy is necessary
        return self.reset()

    def _add_action(self, type, info):
        new_action = _Action(type, info)
        if new_action not in self.action_list:  # this works because of custom __eq__ method
            self.action_list += [new_action]
        self.available_actions += [self.action_list.index(new_action)]  # this works because of custom __eq__ method

    def _remove_action(self, type, info):
        my_action = _Action(type, info)
        if my_action in self.action_list:  # this works because of custom __eq__ method
            my_index = self.action_list.index(my_action)
            self.available_actions.remove(my_index)
        else:
            raise ValueError("Action with type %s and info %s cannot be removed because it does not exist." % (type, str(info)))

    def _purify(self, stations):
        pair = next(filter(lambda x: x.stations == stations, self.state), None)
        if pair is None:
            raise ValueError("There is no pair between stations %s that can be purified." % str(stations))
        f = pair.fid
        pair.fid = (f**2 + (1 - f)**2 / 9) / (f**2 + 2 * f * (1 - f) / 3 + 5 * (1 - f)**2 / 9)  # directly modifies self.state

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
        self.state += [_Pair((new_left, new_right), fid_new)]
        self.state.sort(key=lambda x: x.stations)
        self._remove_action(ACTION_SWAP, station)
        self._remove_action(ACTION_PURIFY, pair1.stations)
        self._remove_action(ACTION_PURIFY, pair2.stations)
        self._add_action(ACTION_PURIFY, (new_left, new_right))
        # self._recalc_actions()

    def _add_block_action(self, block_action):
        """
        """
        pass

    # def _recalc_actions(self):
    #     for pair in self.state:
    #         if not _action_exists(ACTION_PURIFY, pair.stations):
    #             self.action_list += [_Action(ACTION_PURIFY, pair.stations)]

    # def _decode_action(self, action):
    #     # actions are structured as follows:
    #     # Action 0 is entanglement purification on all pairs that currently exist
    #     # actions 1 to self.length are entanglement swapping on that station
    #     # actions after that are the special composite actions specified by self.additional_actions
    #     pass

    def _observation(self):
        """returns a flattened tuple of the current state"""
        return tuple(chain(*((pair.left_station, pair.right_station, int(pair.fid * 100)) for pair in self.state)))

    def _check_success(self):
        return False

    def _check_finished(self):
        return False

    def reset(self):
        self.state = [_Pair((i, i + 1), fid=self.start_fid) for i in range(self.length)]
        self.available_actions = [i for i in range(len(self.base_actions))]
        return self._observation(), {"available_actions": self.available_actions}

    def move(self, action):
        if action not in self.available_actions:
            raise ValueError("Action with number %s is not available at this time." % action)
        my_action = self.action_list[action]
        if my_action.type == ACTION_PURIFY:
            self._purify(my_action.stations)
        elif my_action.type == ACTION_SWAP:
            self._entanglement_swapping(my_action.station)
        elif my_action.type == ACTION_COMPOSITE:
            pass

        observation = self._observation()
        if self._check_success():
            reward = 1
            episode_finished = 1
        else:
            reward = 0
            episode_finished = int(self._check_finished())

        return observation, reward, episode_finished, {"available_actions": self.available_actions}
