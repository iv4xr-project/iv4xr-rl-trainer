import threading
import logging
import time
import signal
import zmq

import gym
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.envs.registration import EnvSpec


logger = logging.getLogger(__name__)


def parse_gym_space(json_space):
    if json_space['name'] == 'Box':
        low = np.array(json_space['minBoundaries'])
        high = np.array(json_space['maxBoundaries'])
        return Box(low=low, high=high)
    elif json_space['name'] == 'Discrete':
        return Discrete(n=json_space['numberOfActions'])
    elif json_space['name'] == 'Dict':
        raise NotImplementedError("Dict space not yet implemented")
    else:
        raise ValueError("Unhandled action/observation space: " + json_space['name'])


class EnvProperties:
    def __init__(self, action_space, observation_space, spec):
        """
        Container for an environment properties, gives the necessary information
        to initiate a reinforcement learning agent
        Args:
            action_space (`gym.Space`):
            observation_space (`gym.Space`):
            spec (`gym.EnvSpec`):
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.spec = spec

    def __str__(self):
        return "{}\n{}\n{}".format(
            f"Name: {self.spec.id}",
            f"ActionSpace: {str(self.action_space)}",
            f"Observation Space: {str(self.observation_space)}"
        )


class DisconnectEnvException(Exception):
    pass


class PolicyServer(threading.Thread):
    def __init__(self, port, timeout=None):
        super().__init__()
        self.context = zmq.Context()
        if timeout:
            self.context.setsockopt(zmq.RCVTIMEO, timeout)
            self.context.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket = self.context.socket(zmq.REP)
        url = f"tcp://*:{port}"
        self.socket.bind(url)
        self._running = True
        self._connected = False

        self._env_properties = None

        self._returns_cv = threading.Condition()
        self._returns = None

        self._state_cv = threading.Condition()
        self._state = None

        self._action_cv = threading.Condition()
        self._action_to_send = None

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGABRT, self.stop)

    def run(self):
        while self._running:
            #  Wait for next request from client
            try:
                message = self.socket.recv_json()
            except zmq.error.ContextTerminated:
                # Forcefully closing the connection
                return
            self._connected = True
            logger.debug("[Server] Received request: %s" % message)
            #  Send reply back to client
            if message["cmd"] == "ENV_SPEC":
                content = message['arg']
                try:
                    env_spec = EnvSpec(content['envName'])
                except gym.error.Error:
                    env_spec = EnvSpec(content['envName']+"-v0")
                action_space = parse_gym_space(content['actionSpace'])
                observation_space = parse_gym_space(content['observationSpace'])
                self._env_properties = EnvProperties(
                    action_space, observation_space, env_spec
                )
                logger.info(f"Connected to environment: {self.env_spec.id}")
                self.socket.send_json(True)
            elif message["cmd"] == "GET_ACTION":
                content = message['arg']
                with self._state_cv:
                    self._state = content["rawObservation"]
                    self._state_cv.notify()
                with self._action_cv:
                    self._action_cv.wait()
                    self.socket.send_json({"rawAction": self._action_to_send})
            elif message["cmd"] == "LOG_RETURNS":
                content = message['arg']
                with self._returns_cv:
                    self._returns = (
                        content["nextObservation"],
                        content["reward"],
                        content["done"],
                        content["info"]
                    )
                    self._returns_cv.notify()
                self.socket.send_json(True)
            elif message["cmd"] == "DISCONNECT":
                self._env_properties = None
                self.socket.send_json(True)
                self._connected = False
                with self._state_cv:
                    self._state_cv.notify()
            else:
                raise ValueError(f"[Server] Unexpected command: {message['cmd']}")

    def stop(self):
        print("[Server] Closing...")
        self._running = False
        self.socket.close()
        self.context.destroy()

    def has_env(self):
        return self._env_properties is not None

    def poll_state(self):
        with self._state_cv:
            self._state_cv.wait()
            if self._connected:
                return self._state
            else:
                raise DisconnectEnvException

    def poll_returns(self):
        with self._returns_cv:
            self._returns_cv.wait()
            return self._returns

    def send_action(self, action):
        with self._action_cv:
            self._action_to_send = action
            self._action_cv.notify()

    @property
    def action_space(self):
        assert(self.has_env())
        return self._env_properties.action_space

    @property
    def observation_space(self):
        assert(self.has_env())
        return self._env_properties.observation_space

    @property
    def env_spec(self):
        assert(self.has_env())
        return self._env_properties.spec


class PolicyClient:
    """
    Helper class to test basic interactions with the PolicyServer.
    An usual client is a RLEnvironment from the iv4XR framework
    """
    def __init__(self, port):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{port}")

    def request(self, req):
        self.socket.send_json(req)
        #  Get the reply.
        message = self.socket.recv_json()
        print("[Client] Received reply %s" % message)
