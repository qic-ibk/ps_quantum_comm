import numpy as np


class edge(object):
    def __init__(self, origin, destination, weight=0.0, glow_destination=0.0, flag=False, previousEdge=None, split=False):
        self.w_destination = weight
        self.w_origin = None
        self.weights = [self.w_destination, self.w_origin]
        self.glow_destination = glow_destination
        self.destination = destination
        self.origin = origin
        self.flag = flag
        self.previousEdge = previousEdge
        self.split = split

    # accumulative glow
    def mark_path(self):
        self.glow_destination += 1.0

    def setting_flag(self, reward):
        if reward != 0:
            self.flag = True


class clip(object):
    def __init__(self, type='clip', edges=None, parents=None, children=None, name='Noname', connect=True, split=False):
        self.connect = connect
        self.name = name
        if not edges:
            edges = []
        if not parents:
            parents = []
        if not children:
            children = []
        self.parents = parents
        self.children = children
        self.edges = edges
        self.type = type
        self.split = split

    @property  # for looping through the list
    def clips(self):
        for edge in self.edges:
            yield edge.destination  # generator

    def add_action_clip(self, weight=1.0, name="Noname", type='Action'):
        child = ActionClip(name=name, type=type)
        newedge = edge(self, child, weight)
        self.edges.append(newedge)
        return child

    def add_percept_clip(self, weight=1.0, name="Noname", type='Percept'):
        child = PerceptClip(name=name, type=type)
        newedge = edge(self, child, weight)
        self.edges.append(newedge)
        return child

    # between action and top-layer
    def add_edge(self, destination):
            newedge = edge(self, destination, weight=1.0, glow_destination=0.0, flag=False)
            self.edges.append(newedge)
            destination.parents.append(newedge)
            return newedge

    def remove_edge(self, destination):
            for edge in self.edges:
                if edge.destination == destination:
                    self.edges.remove(edge)
                    self.parents.remove(edge)

    def __str__(self):
        return self.name


class ActionClip(clip):
    clip.type = 'Action'


class PerceptClip(clip):
    clip.type = 'Percept'


class TopClip(clip):
    parent = []
    pass


def print_clip(clip, depth=0, leaf=False):
    # print('_' * depth + str(clip) + ('*=' + (str(clip.parents[-1].w_destination)) if leaf else ' '))
    print('_' * depth + str(clip))


def print_weight(clip, depth=0, leaf=False):
    print('_' * depth + str(clip) + ('*=' + (str(clip.parents[-1].w_destination)) if leaf else ' ' + str(clip.type)))


def print_glow(clip, depth=0, leaf=False):
    print('_' * depth + str(clip) + ('*=' + (str(clip.parents[-1].w_destination)) if leaf else ' ' + str(clip.type)))


def walknetwork(iter_clip, depth=0, pfunc=print_clip):
    if iter_clip.children == []:
        pfunc(iter_clip, depth, leaf=True)
    else:
        pfunc(iter_clip, depth, leaf=False)
        for current_clip in iter_clip.children:
            walknetwork(current_clip, depth=depth + 1, pfunc=pfunc)


def update_glow(clip, eta):
    for i in range(len(clip.edges)):
        clip.edges[i].glow_destination *= (1 - eta)
        # print(clip.edges[i].glow_destination)


def update_weights(clip, eta):
    # if clip.type == 'Percept':
        for i in range(0, len(clip.edges)):
            # print(clip.edges[i].glow_destination)
            clip.edges[i].w_destination += clip.edges[i].glow_destination
            clip.edges[i].glow_destination *= (1 - eta)


def update(iter_clip, eta=0.5, depth=0, pfunc=update_weights):
    if iter_clip.children == []:
        pfunc(iter_clip, eta)
    else:
        pfunc(iter_clip, eta)
        for current_clip in iter_clip.children:
            update(current_clip, eta=eta, depth=depth + 1, pfunc=pfunc)


def check_clip(clipA, percept):
    check = False
    for edge in clipA.edges:
        if edge.destination.name == str(percept):
            check = True
            break
    return check


def find_clip(clipA, percept):
    foundClip = None
    for edge in clipA.edges:
        if edge.destination.name == str(percept):
            foundClip = edge.destination
            break
    return foundClip


def find_edge(clipA, destination_name):
    foundEdge = None
    for edge in clipA.edges:
        if edge.destination.name == destination_name:
            foundEdge = edge
            break
    return foundEdge


class ClipBrain(object):
    """
    """
    def __init__(self, n_actions=0, n_percepts=0):
        self.all_clips = []
        self.all_edges = []
        self.action_clips = []
        self.empty_action = ActionClip(name='emptyAction')
        self.current_clip = self.empty_action
        self.all_clips += [self.empty_action]
        for i in range(n_actions):
            new_clip = ActionClip(name=str(i))
            self.all_clips += [new_clip]
            self.action_clips += [new_clip]
        self.n_percepts = 0
        for i in range(n_percepts):
            self.add_percept()

    def add_percept(self):
        new_clip = self.empty_action.add_percept_clip(name=str(self.n_percepts))
        new_clip.parents += [self.empty_action]
        self.empty_action.children += [new_clip]
        self.n_percepts += 1
        self.all_clips += [new_clip]
        for action_clip in self.action_clips:
            new_edge = new_clip.add_edge(action_clip)  # should be safe to do this
            new_clip.children += [action_clip]
            self.all_edges += [new_edge]

    def decay(self, gamma):
        for edge in self.all_edges:
            edge.w_destination = (1 - gamma) * edge.w_destination + gamma * 1

    def update_g_matrix(self, eta, history_since_last_reward):  # rethink history_since_last_reward in this context
        if not isinstance(history_since_last_reward[0], tuple):
            history_since_last_reward = [history_since_last_reward]  # if it is only a tuple, make it a list of tuples anyway
        n = len(history_since_last_reward)
        for edge in self.all_edges:
            edge.glow_destination = (1 - eta)**n * edge.glow_destination
        for i, [action, percept] in enumerate(history_since_last_reward):
            percept_clip = find_clip(self.empty_action, percept)
            edge_to_update = find_edge(percept_clip, str(action))
            edge_to_update.glow_destination = (1 - eta)**(n - 1 - i)

    def update_h_matrix(self, reward):
        for edge in self.all_edges:
            edge.w_destination += edge.glow_destination * reward

    def get_h_vector(self, percept):
        percept_clip = find_clip(self.empty_action, percept)
        h_vector = np.array([edge.w_destination for edge in percept_clip.edges])
        return h_vector
