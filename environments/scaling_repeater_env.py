"""
"""

from __future__ import print_function, division
from .abstract_environment import AbstractEnvironment
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
                    return self.composite_id == other.composite_id and self.involved_links == other.involved_links
        return False

    def __neq__(self, other):  # python2 compatibility
        return not self.__eq__(other)

    def __repr__(self):
        if self.type == ACTION_PURIFY:
            return "_Action(%s, %s)" % (self.type, repr(self.stations))
        elif self.type == ACTION_SWAP:
            return "_Action(%s, %s)" % (self.type, repr(self.station))
        elif self.type == ACTION_COMPOSITE:
            return "_Action(%s, %s)" % (self.type, repr([self.composite_id, self.involved_links, self.constituent_actions]))


class _Pair(object):
    def __init__(self, stations, fid=1.0, resources=1.0):
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

    def __repr__(self):
        return "_Pair(%s, fid=%s, resources=%s)" % (repr(self.stations), repr(self.fid), repr(self.resources))


class TaskEnvironment(AbstractEnvironment):
    """
    """
    def __init__(self, length=2, composite_actions=[], q=0.57, target_fid=0.9, reward_constant=1, reward_exponent=1, p=1.0):
        self.length = length
        if isinstance(q, (int, float)):
            self.start_fid = [(3 * q + 1) / 4] * length
        elif isinstance(q, (list, tuple)) and len(q) == length:
            self.start_fid = [(3 * qq + 1) / 4 for qq in q]
        else:
            raise ValueError(repr(q) + " as initial noise is not supported.")
        self.gate_noise = p
        self.target_fid = target_fid
        self.reward_constant = reward_constant  # could simply define a function for this
        self.reward_exponent = reward_exponent
        self.state = [_Pair((i, i + 1), fid=self.start_fid[i], resources=1.0) for i in range(self.length)]
        self.base_actions = [_Action(ACTION_SWAP, i) for i in range(1, self.length)] + [_Action(ACTION_PURIFY, pair.stations) for pair in self.state]
        self.n_base_actions = len(self.base_actions)
        self.available_actions = [i for i in range(len(self.base_actions))]
        self.action_list = deepcopy(self.base_actions)
        self.composite_actions = composite_actions
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
        for composite_id, composite_action in enumerate(self.composite_actions):
            block_size = composite_action["block_size"]
            actions = composite_action["actions"]
            num_locations = len(self.state) - (block_size - 1)
            for i in range(num_locations):
                involved_links = [pair.stations for pair in self.state[i:i + block_size]]
                constituent_actions = self._shift_actions(actions, involved_links)  # often unnecessary but I don't know how to properly do it without
                self._add_action(ACTION_COMPOSITE, [composite_id, involved_links, constituent_actions])

    def _observation(self):
        """Get a flattened tuple representation of the current state."""
        return tuple(chain(*((pair.left_station, pair.right_station, int(pair.fid * 1000)) for pair in self.state)))

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
        for _, action_index in history:
            action = self.action_list[action_index]
            if action.type == ACTION_COMPOSITE:
                for act in action.constituent_actions:
                    action_sequence += [act]
            else:
                action_sequence += [action]
        return {"block_size": self.length, "actions": action_sequence}

    def reset(self):
        self.state = [_Pair((i, i + 1), fid=self.start_fid[i], resources=1.0) for i in range(self.length)]
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

    def move(self, action):
        if action not in self.available_actions:
            raise ValueError("Action with number %s is not available at this time." % action)
        my_action = self.action_list[action]
        if my_action.type == ACTION_PURIFY:
            self._purify(my_action.stations)
        elif my_action.type == ACTION_SWAP:
            self._entanglement_swapping(my_action.station)
        elif my_action.type == ACTION_COMPOSITE:
            for act in my_action.constituent_actions:
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
