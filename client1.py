import socket
import threading

def receive_messages(sock):
    while True:
        try:
            data = sock.recv(1024)
            if not data:
                break
            print(data.decode(), end="")
        except:
            break

def main():
    host = "127.0.0.1"
    port = 8000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    cid = input("Enter your client ID: ")
    sock.sendall((cid + "\n").encode())

    threading.Thread(target=receive_messages, args=(sock,), daemon=True).start()

    while True:
        msg = input()
        sock.sendall((msg + "\n").encode())

if __name__ == "__main__":
    main()
