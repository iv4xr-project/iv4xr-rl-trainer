from iv4xrl.env import Iv4xrServerEnv, Iv4xrClientEnv
from iv4xrl.policy_server import DisconnectEnvException
import coloredlogs
import logging
import numpy as np


logger = logging.getLogger('policy_server')
coloredlogs.install(level='INFO')

PORT = 5555


def gym_server_env():
    """Implement a Iv4XR Gym environment in server mode. Must be used with the
    JAVA RLAgentSocketConnector in client mode.

    Associated test in iv4xr-rl-env: RLAgentConnectTest.clientModeTest
    """
    env = Iv4xrServerEnv(PORT)
    print(f"Environment initialized: {env.spec.id}")
    print(f"Action space: {env.action_space}")
    print(f"State space: {env.observation_space}")

    try:
        while True:
            # Because this env is based on a policy server, we wait for the remote
            # env to declare its reset with the 'done' attribute
            # Any forceful call to env.reset() will fail
            state = env.reset()
            done = False
            while not done:
                action = [-25.45, 266.85]
                state, reward, done, info = env.step(action)
                print(f"Current state: {state}")
                assert env.observation_space.contains(state)
    except DisconnectEnvException:
        env.close()


def gym_client_env():
    """Implement a Iv4XR Gym environment in server mode. Must be used with the
    JAVA RLAgentSocketConnector in server mode.

    Associated test in iv4xr-rl-env: RLAgentConnectTest.serverModeTest
    """
    env = Iv4xrClientEnv(PORT)
    print(f"Environment initialized: {env.spec.id}")
    print(f"Action space: {env.action_space}")
    print(f"State space: {env.observation_space}")

    try:
        num_episodes = 10
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = [-25.45, 266.85]
                state, reward, done, info = env.step(action)
                print(f"Current state: {state}")
                assert env.observation_space.contains(state)
        # This lets us notify the JAVA test that the last episode is done
        env.reset()
    except DisconnectEnvException:
        env.close()


if __name__ == "__main__":
    gym_client_env()
