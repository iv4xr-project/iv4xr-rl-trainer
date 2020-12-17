from connector.policy_server import PolicyServer, DisconnectEnvException
import coloredlogs
import logging


logger = logging.getLogger('policy_server')
coloredlogs.install(level='INFO')


def main():
    PORT = 5555
    server = PolicyServer(PORT)
    server.start()

    # Wait for an environment connection to the server
    while not server.has_env():
        pass

    print("Environment initialized")
    print(server.env_spec.id)
    print(server.action_space)
    print(server.observation_space)

    try:
        while True:
            state = server.poll_state()
            # action = agent.get_action(state)
            action = server.action_space.sample().tolist()
            server.send_action(action)
            next_state, reward, done, info = server.poll_returns()
            # replay_buffer.push((state, action, next_state, reward, done))
    except DisconnectEnvException:
        server.stop()
        server.join()


if __name__ == "__main__":
    main()