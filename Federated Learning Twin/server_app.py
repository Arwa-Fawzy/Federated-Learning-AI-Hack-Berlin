# server_app.py
"""
Starts Flower server and spawns 5 local client threads so running one command runs the whole FDDT demo.
Outputs per-round aggregated info plus client evaluations.
"""

import threading, time, socket
import flwr as fl
from flwr.server.strategy import FedAvg
from client_app import make_client_instance

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"
NUM_CLIENTS = 5
NUM_ROUNDS = 6

def start_server():
    # FedAvg default: averages weights. We also accept client-reported threshold as a metric (not used by aggregator)
    strategy = FedProx(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        proximal_mu=0.01  # <-- required argument, you can tune this
    )
    print("[server] starting Flower server...")
    fl.server.start_server(server_address=SERVER_ADDRESS, config={"num_rounds": NUM_ROUNDS}, strategy=strategy)

def start_client_thread(cid):
    import flwr as fl
    from client_app import make_client_instance
    client = make_client_instance(cid, epochs=3, batch_size=32, seq_len=50)
    print(f"[launcher] client {cid} connecting...")
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)

def wait_for_port(host, port, timeout=8.0):
    t0 = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except Exception:
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.2)

def main():
    # start server thread
    s_thread = threading.Thread(target=start_server, daemon=True)
    s_thread.start()

    ok = wait_for_port(SERVER_HOST, SERVER_PORT, timeout=10.0)
    if not ok:
        print("[launcher] server did not open port; aborting.")
        return
    print("[launcher] server up — starting clients...")

    client_threads = []
    for cid in range(NUM_CLIENTS):
        t = threading.Thread(target=start_client_thread, args=(cid,), daemon=True)
        t.start()
        client_threads.append(t)
        time.sleep(0.3)

    # wait for server to finish
    s_thread.join()
    print("[launcher] federated training finished.")

if __name__ == "__main__":
    main()

