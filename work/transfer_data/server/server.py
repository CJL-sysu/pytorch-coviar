# server.py

import os
import socket

# config begin

# ip address of server
ip = '127.0.0.1'
# server should listen on this port
port = 6001
receive_path = "receive"

# config end

def receive_file(server_socket, filename):
    client_socket, addr = server_socket.accept()
    print(f"Connection from {addr[0]}:{addr[1]}")
    try:
        with open(filename, 'wb') as file:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                file.write(data)
        print(f"Received {filename} successfully.")
        client_socket.close()
    except Exception as e:
        print(f"Error receiving {filename}: {e}")


def main():
    host = ip

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr[0]}:{addr[1]}")

        filename = client_socket.recv(1024).decode()
        client_socket.close()
        receive_file(server_socket, f"{receive_path}/{filename}")


if __name__ == '__main__':
    if not os.path.exists(receive_path):
        os.makedirs(receive_path)
    main()
