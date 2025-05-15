#!/usr/bin/env python3
import socket

def find_available_port(start_port: int = 8888, max_tries: int = 10) -> int:
    for port in range(start_port, start_port + max_tries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except Exception:
            continue
    return start_port + max_tries

if __name__ == "__main__":
    print(find_available_port())
