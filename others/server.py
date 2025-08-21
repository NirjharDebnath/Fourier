import socket
import threading

# Graph: allowed communication links
graph = {
    1: {2, 3},
    2: {1, 4},
    3: {1},
    4: {2}
}

clients = {}  # client_id -> socket object
lock = threading.Lock()

def client_handler(conn, addr):
    try:
        # First message is the client ID
        cid = int(conn.recv(1024).decode().strip())
        with lock:
            clients[cid] = conn
        print(f"Client {cid} connected from {addr}")

        while True:
            data = conn.recv(1024)
            if not data:
                break
            msg = data.decode().strip()
            print(f"From {cid}: {msg}")

            # Send only to friends in the graph
            with lock:
                for friend in graph.get(cid, []):
                    if friend in clients:
                        try:
                            clients[friend].sendall(f"{cid}: {msg}\n".encode())
                        except:
                            pass
    except:
        pass
    finally:
        with lock:
            if cid in clients:
                del clients[cid]
        conn.close()
        print(f"Client {cid} disconnected.")

def main():
    host = "127.0.0.1"
    port = 8000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()

    print(f"Server listening on {host}:{port}")
    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=client_handler, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()
