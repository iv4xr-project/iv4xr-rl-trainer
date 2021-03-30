from connector.policy_server import PolicyServer, DisconnectEnvException
import coloredlogs
import logging
import numpy as np


logger = logging.getLogger('policy_server')
coloredlogs.install(level='INFO')


def normalize_obs(obs, obs_space):
    return (np.array(obs) - obs_space.low) / (obs_space.high - obs_space.low)

def main():
    PORT = 5555
    server = PolicyServer(PORT)
    server.start()

    # Wait for an environment connection to the server
    while not server.has_env():
        pass

    print("Environment initialized")
    print(server.env_spec.id)
    print(f"Action space: {server.action_space}")
    print(f"State space: {server.observation_space}")

    try:
        while True:
            state = server.poll_state()
            action = server.action_space.sample().tolist()
            #action = [138500, 441747]
            print(f"Taking action: {action}")
            # action = agent.get_action(state)
            server.send_action(action)
            next_state, reward, done, info = server.poll_returns()
            print(f"Current state: {normalize_obs(next_state['rawObservation'], server.observation_space)}")
            # replay_buffer.push((state, action, next_state, reward, done))
    except DisconnectEnvException:
        server.stop()
        server.join()


if __name__ == "__main__":
    main()