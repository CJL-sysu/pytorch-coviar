# server.py

import os
import socket
import argparse
import server


def receive_file(server_socket, filepath):
    client_socket, addr = server_socket.accept()
    print(f"Connection from {addr[0]}:{addr[1]}")
    try:
        with open(filepath, 'wb') as file:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                file.write(data)
        print(f"Received {filepath} successfully.")
        client_socket.close()
    except Exception as e:
        print(f"Error receiving {filepath}: {e}")


def main(args):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args.ip, args.port))
    server_socket.listen(1)

    print(f"Server listening on {args.ip}:{args.port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr[0]}:{addr[1]}")

        filename = client_socket.recv(1024).decode()
        client_socket.close()
        filepath = f"{args.receive_path}/{filename}"
        receive_file(server_socket, filepath)
        args.file_path = filepath
        classify_num, classify = server.main(args)
        print("the classify result is {}#{}".format(classify_num, classify))


def parse_args():
    # parse args
    # test_segments和representation必须和client端保持一致
    parser = argparse.ArgumentParser(description="classify video")
    parser.add_argument("--data_name", type=str, choices=["ucf101", "hmdb51"])
    parser.add_argument("--representation", type=str, choices=["iframe", "residual", "mv"])
    parser.add_argument("--test_segments", type=int, default=25)
    parser.add_argument("--arch", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--test-crops", type=int, default=10)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument(
        "-j",
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of workers.",
    )
    parser.add_argument('--ip', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--receive_path', type=str, default='receive')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    if not os.path.exists(args.receive_path):
        os.makedirs(args.receive_path)
    main(args)
