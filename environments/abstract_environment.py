"""
Copyright 2020 Julius Wallnöfer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import sys
import abc

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class AbstractEnvironment(ABC):

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def move(self, action):
        pass
        # return observation, reward, episode_finished, info  # where info might be {"available_actions", self.available_actions}
