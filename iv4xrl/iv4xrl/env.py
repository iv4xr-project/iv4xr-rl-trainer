# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #
#                                                                              #
# Copyright © 2021-2021 Thales SIX GTS FRANCE                                  #
#                                                                              #
# **************************************************************************** #
#                                   Copyright                                  #
# **************************************************************************** #


from .policy_server import PolicyServer, PolicyClient, DisconnectEnvException
import gym


class Iv4xrServerEnv(gym.Env):
    def __init__(self, port, timeout=None):
        """Gym environment wrapper over a PolicyServer. Ease integration with
        existing RL code.

        Args:
            port (int): listening port for this server.
            timeout (int|None): if None, the socket will be blocking. Otherwise,
                it defines the timeout in seconds for a non-blocking socket.
        """
        self.policy_server = PolicyServer(port, timeout)
        self.policy_server.start()
        while not self.policy_server.has_env():
            pass
        self.action_space = self.policy_server.action_space
        self.observation_space = self.policy_server.observation_space
        self.spec = self.policy_server.env_spec
        self._done = True
        self._must_poll_state = False

    def step(self, action):
        """Play a step of the iv4XR RL environment.

        Args:
            action: RL agent action.

        Returns:
            (tuple): next_state, reward, done, info
        """
        try:
            if self._must_poll_state:
                _ = self.policy_server.poll_state()
            self._must_poll_state = True
            self.policy_server.send_action(action)
            next_state, reward, self._done, info = self.policy_server.poll_returns()
            return next_state, reward, self._done, info
        except DisconnectEnvException as e:
            self.policy_server.stop()
            self.policy_server.join()
            raise e

    def reset(self):
        """Reset the iv4XR RL environment.

        Returns:
            initial state.
        """
        if not self._done:
            raise ValueError(
                "Cannot force the remote iv4XR RL Environment to reset."
            )
        self._done = False
        try:
            state = self.policy_server.poll_state()
            self._must_poll_state = False
            return state
        except DisconnectEnvException as e:
            self.policy_server.stop()
            self.policy_server.join()
            raise e

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        """Close and cleanup the policy server."""
        self.policy_server.stop()
        self.policy_server.join()


class Iv4xrClientEnv(gym.Env):
    def __init__(self, port, timeout=None):
        """Gym environment wrapper over a PolicyClient. Ease integration with
        existing RL code.

        Args:
            port (int): port of the iv4XR RL iv4xrl in server mode.
            timeout (int|None): if None, the socket will be blocking. Otherwise,
                it defines the timeout in seconds for a non-blocking socket.
        """
        self.policy_client = PolicyClient(port, timeout)
        self.policy_client.get_spec()
        self.action_space = self.policy_client.action_space
        self.observation_space = self.policy_client.observation_space
        self.spec = self.policy_client.env_spec

    def step(self, action):
        """Play a step of the iv4XR RL environment.

        Args:
            action: RL agent action.

        Returns:
            (tuple): next_state, reward, done, info
        """
        return self.policy_client.step(action)

    def reset(self):
        """Reset the iv4XR RL environment.

        Returns:
            initial state.
        """
        return self.policy_client.reset()

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        self.policy_client.stop()
