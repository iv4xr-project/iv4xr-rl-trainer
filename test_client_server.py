from connector.policy_server import PolicyServer, PolicyClient
import json


def main():
    PORT = 5555

    server = PolicyServer(PORT)
    server.start()

    client = PolicyClient(PORT)
    for _ in range(10):
        json_object = {"content": "Hello"}
        client.request(json_object)

    server.stop()
    print("stopping server", server.is_alive(), server._running)
    server.join()


if __name__ == "__main__":
    main()