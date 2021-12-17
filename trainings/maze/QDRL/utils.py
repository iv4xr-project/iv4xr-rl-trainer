# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #
#                                                                              #
# Copyright © 2021-2021 Thales SIX GTS FRANCE                                  #
#                                                                              #
# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #


import copy
import math
from itertools import compress
import numpy as np
import pickle
import torch


def transs(actor, device):
    """
    Converts an actor NN into an actor function
    Args:
        actor: an actor
        device: the device

    Returns:
        (func):the function that is an actor
    """

    def new_actor(state):
        model_state = torch.FloatTensor(state).to(device)
        action = actor['net'](model_state).detach().cpu().numpy()
        return action

    return new_actor

class KNN:
    """
    KNN object to compute nearest neighbors

    Attributes:
        k(int): number of nearest neighbors
        distance(function): function to compute the distance
    """
    def __init__(self, k, distance_func):
        """
        Builds the KNN object

        Args:
            k(int): number of nearest neighbors
            distance(function): function to compute the distance
        """
        self.k = k
        self.distance = distance_func

    def compute_knn(self, archive, ref_element):
        """
        computes the knn of an element in the archive
        Args:
            archive: an archive
            ref_element: an element for which we compute the KNN
        Returns:
            the k nearest neighbors
        """
        list_to_sort = []
        for element in archive.elements:
            dist = self.distance(ref_element.outcome, element.outcome)

            list_to_sort.append({'distance': dist, 'element': element})
        new_list = sorted(list_to_sort, key=lambda d: d['distance'])
        return new_list[0:self.k]


class Archive:
    """
        Archive object collecting actors

        Attributes:
            elements(float): a list of actors
    """
    def __init__(self):
        """
        Constructing the archive
        """
        self.elements = []

    def add_element(self, element):
        """
        Adding the element to the archive
        Args:
            element: element to add to the archive

        Returns:
            (int): the index associated to the actor
        """
        idx = len(self.elements)
        element.id = idx
        self.elements.append(element)
        return idx

    def replace_element(self, idx, element):
        """
        Replacing the element in the archive at the index idx
        Args:
            idx: index at which to put the actor
            element: element to add to the archive
        """
        self.elements[idx] = element
        element.id = idx

    def distance(self, element, distance_func):
        """
        Computes the distance from the element to the archive as the minimum distance to any element within the archive
        Args:
            element: element for which to compute the distance
            distance_func: the distance function to use

        Returns:
            (float,element): the distance and the closest element
        """
        min_distance = 100000
        min_el = None
        for el in self.elements:
            dist = distance_func(el.outcome, element.outcome)
            if min_distance > dist:
                min_el = el
            min_distance = min(min_distance, dist)
        return min_distance, min_el

    def print_info(self, my_logger):
        """
        Print the information of the archive
        Args:
            my_logger: a logger
        """
        my_logger.info('printing the archive with {} elements'.format(len(self.elements)))
        for el in self.elements:
            el.print_info(my_logger)


def create_element(return_ep, success, rollout, agent):
    """
        Create an archive element

    Args:
        return_ep: performance
        success: boolean of success
        rollout: the representative outcome
        agent: the actor

    Returns:
        (ArchiveElement) the created archive element
    """
    agent_copy = copy.deepcopy(agent)
    element = ArchiveElement(return_ep, success, rollout, agent_copy, idx=None)
    return element


class ArchiveElement:
    """
        Archive Element for the Archive which is an actor with some characteristics

        Attributes:
            quality(float): the performance of the element
            success(bool): whether or not the actor is successful
            outcome(object): an object describing the element
            actor(actor): the actor
            diversity(float): a diversity score
            id(integer): an id tag for the actor
    """

    def __init__(self, quality, success, outcome, actor, idx):
        """
            Creating the Archive Element

            Args:
                quality(float): the performance of the element
                success(bool): whether or not the actor is successful
                outcome(object): an object describing the element
                actor(actor): the actor
                diversity(float): a diversity score
                id(integer): an id tag for the actor
        """
        self.quality = quality
        self.success = success
        self.outcome = outcome
        self.actor = actor
        self.diversity = None
        self.id = idx

    def print_info(self, my_logger):
        """
        print the information of the element
        Args:
            my_logger: a logger
        """
        my_logger.info(
            'id {} quality {}, success {} diversity {} outcome {} actor {}'.format(self.id, self.quality, self.success,
                                                                                   self.diversity,
                                                                                   # 'outcome not printed',
                                                                                   self.outcome,
                                                                                   'actor not printed'))


class NoveltyScore:
    """
    Computation unit for the novelty score of an agent in the archive

    Attributes:
        knn(KNN): the k nearest neighbors
    """

    def __init__(self, knn):
        """
        Construct the novelty score unit

        Args:
            knn(KNN): the k nearest neighbors
        """
        self.knn = knn

    def compute_ns_actor(self, archive, element_archive):
        """
        Computes the novelty score
        Args:
            archive (Archive): the archive
            element_archive (ArchiveElement): the archive element for which to compute the NS

        Returns:
            (float) : the novelty score as a mean distance
        """
        nearest_neighbours = self.knn.compute_knn(archive, element_archive)
        sum_distance = 0
        for neighbour in nearest_neighbours:
            sum_distance += neighbour['distance']
        distance = sum_distance / self.knn.k
        assert (0 <= distance <= 1)
        return distance




def euc_dist(a, b):
    """
    Computes the Euclidian distance between two lists
    Args:
        a(list): first list
        b(list): second list

    Returns:
        (float): the distance
    """
    assert (len(a) == len(b))
    distance = 0
    for i in range(len(a)):
        distance += (a[i] - b[i]) ** 2
    return math.sqrt(distance)


def distance_test(rollout1_, rollout2_):
    """
    Computes the distance between two rollouts
    Args:
        rollout1_(list): first rollout
        rollout2_(list): second rollout

    Returns:
        (float): the distance
    """
    # make sure the distance is in (0,1) to normalise behavior of the algo
    default_max_distance = math.sqrt(2)
    sampling_rate = 1
    rollout1 = rollout1_[:]
    rollout2 = rollout2_[:]

    # removing the end  of a rollout if it is not moving
    rev_rollout1 = list(reversed(rollout1))
    cut = len(rollout1)
    for i, el in enumerate(rev_rollout1[:-1]):
        a = rev_rollout1[i] < rev_rollout1[i + 1] - .00000001
        b = rev_rollout1[i] > rev_rollout1[i + 1] + .00000001
        if a.any() or b.any():
            cut = i
            break
    rollout1 = rollout1[:len(rollout1) - cut]

    rev_rollout2 = list(reversed(rollout2))
    cut = len(rollout2)
    for i, el in enumerate(rev_rollout2[:-1]):
        a = rev_rollout2[i] < rev_rollout2[i + 1] - .00000001
        b = rev_rollout2[i] > rev_rollout2[i + 1] + .00000001
        if a.any() or b.any():
            cut = i
            break
    rollout2 = rollout2[:len(rollout2) - cut]

    max_len = max(len(rollout2), len(rollout1))
    if len(rollout1) > len(rollout2):
        rollout2 += [rollout2[-1]] * (len(rollout1) - len(rollout2))
    if len(rollout2) > len(rollout1):
        rollout1 += [rollout1[-1]] * (len(rollout2) - len(rollout1))

    distance = 0
    for i in range(0, max_len, sampling_rate):
        distance += euc_dist(rollout1[i], rollout2[i])

    distance_norm = distance / default_max_distance / 400 * sampling_rate  # 400 is the max step !

    return distance_norm


def keep_efficient(pts):
    """
    computes and returns Pareto efficient row subset of pts
    Args:
        pts (list): list of actors with their values

    Returns:
        (list): Pareto efficient row subset of pts
    """

    # sort points by decreasing sum of coordinates
    pts = [pts[i] for i in np.array([p['values'] for p in pts]).sum(1).argsort()[::-1]]
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    undominated = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        # process each point in turn
        n = len(pts)
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i + 1:n] = (
                np.array([p['values'] for p in pts])[i + 1:] >= np.array([p['values'] for p in pts])[i]
        ).any(1)

        # keep points undominated so far
        pts = list(compress(pts, undominated[:n]))
    return pts


# TODO: merge with TISS ReplayBuffer
class QDRLReplayBuffer(object):
    """Replay buffer implementation with an underlying numpy array container.

    Args:
        capacity (int): Capacity of the buffer.
        seed (int): Random seed for reproducible experiments
    """

    def __init__(self, capacity, seed=0):
        super().__init__()
        self.capacity = capacity
        self.length = 0
        self.start = 0
        self.random_state = np.random.RandomState(seed)
        self._transition_type = None
        self._buffer = None

    def __len__(self):
        """Returns int: current length of the buffer."""
        return self.length

    def initialize_buffer(self, state, action):
        """
        Infer the state and action types to initialize the transition type and
        internal buffer attributes accordingly.

        Args:
            state (np.ndarray): Preceding state of the environment.
            action (Union[int,np.ndarray]): Action taken.
        """
        state_type, state_shape = state.dtype, state.shape
        if type(action) in (int, np.int8, np.int16, np.int32, np.int64):
            action_type = np.int8
            action_shape = ()
        elif type(action) is np.ndarray:
            action_type, action_shape = action.dtype, action.shape
        else:
            raise ValueError(
                "Unsupported action input: {}".format(type(action))
            )
        self._transition_type = np.dtype([
            ('state', state_type, state_shape),
            ('action', action_type, action_shape),
            ('reward', np.float32),
            ('next_state', state_type, state_shape),
            ('done', np.int8),
            ('actor_id', np.longlong)
        ])
        self._buffer = np.zeros((self.capacity,), dtype=self._transition_type)

    def push(self, state, action, reward, next_state, done, actor_id):
        """Push a step representation into the buffer.

        Args:
            state (np.ndarray): Preceding state of the environment.
            action (Union[int,np.ndarray]): Action taken.
            reward (float): Reward of the action taken.
            next_state (np.ndarray): Resulting state of the environment.
            done (bool): Whether the episode is done or not.
            actor_id (int): id of the actor generating the sample.
        """
        if self._transition_type is None and self._buffer is None:
            self.initialize_buffer(state, action)
        if self.length < self.capacity:
            self.length += 1
        elif self.length == self.capacity:
            self.start = (self.start + 1) % self.capacity
        else:
            raise RuntimeError("Buffer length exceeds capacity")
        self._buffer[(self.start + self.length - 1) % self.capacity] = (
            np.array((state, action, reward, next_state, done, actor_id), dtype=self._transition_type)
        )

    def sample(self, batch_size):
        """Pull a step representation from the buffer.

        Args:
            batch_size (int): Size of the batch.
        Returns:
            np.array * 5: state, action, reward, next_state, done.
        """
        subset = self.random_state.choice(
            self._buffer[:self.length], batch_size, replace=False
        )
        return (
            subset['state'].copy(),
            subset['action'].copy(),
            subset['reward'].copy(),
            subset['next_state'].copy(),
            subset['done'].copy(),
            subset['actor_id'].copy()
        )

    def subset(self, indices):
        """Pull a step representation from the buffer from already sampled
        subset indices.

        Args:
            indices (np.ndarray): Indices of the subset.
        Returns:
            np.array * 5: state, action, reward, next_state, done.
        """
        subset = self._buffer[indices]
        return (
            subset['state'].copy(),
            subset['action'].copy(),
            subset['reward'].copy(),
            subset['next_state'].copy(),
            subset['done'].copy(),
            subset['actor_id'].copy()
        )

    @staticmethod
    def dump_to_file(replay_buffer, file_name):
        """store a replay buffer in a file

        Args:
            file_name (str): name of the file.
            replay_buffer (ReplayBuffer): the replay buffer to store.
        """
        with open(file_name, 'wb') as file:
            pickle.dump(replay_buffer, file)

    @staticmethod
    def load_from_file(file_name):
        """Load a replay buffer from a file.

        Args:
            file_name (str): name of the file.
        Returns:
            ReplayBuffer: the replay buffer read.
        """
        with open(file_name, 'rb') as file:
            return pickle.load(file)