from connector.policy_server import PolicyServer

def main():
    PORT = 5555
    server = PolicyServer(PORT)
    server.start()
    server.join()

if __name__ == "__main__":
    main()