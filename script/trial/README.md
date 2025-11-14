# Federated Learning Trial - Pump Sensor Anomaly Detection

Welcome to the federated learning trial for pump sensor anomaly detection. This guide will walk you through setting up and running a distributed machine learning experiment using three laptops. One laptop will act as the central server that coordinates the training, while two other laptops will act as clients that train models on their local data without sharing the raw sensor readings.

## Understanding What We're Building

In this trial, we're implementing a privacy-preserving machine learning system for detecting anomalies in industrial pump sensors. The key innovation is that each facility (represented by a client) keeps its sensitive operational data local while still contributing to a shared machine learning model. We use a Convolutional Autoencoder, which is a type of neural network that learns to compress and reconstruct sensor data. When the model encounters unusual patterns (anomalies), it will struggle to reconstruct them accurately, giving us a way to detect potential pump failures or malfunctions.

The Federated Averaging (FedAvg) strategy works by having each client train the model on their local data for a few iterations, then sending only the updated model parameters (not the raw data) to the central server. The server averages these parameters from all clients to create an improved global model, which is then sent back to clients for another round of training. This process repeats for several rounds until the model converges to a solution that benefits from everyone's data without anyone having to share their sensitive information.

## System Requirements and Preparation

Before we begin, each of the three participants needs to ensure their laptop meets the basic requirements. You'll need Python 3.8 or higher installed on your system, along with at least 4GB of available RAM and about 2GB of free disk space. The laptops should be connected to the same WiFi network so they can communicate with each other during the federated learning process.

All three participants should start by cloning the repository to their laptops. Open a terminal or command prompt and navigate to a directory where you want to store the project. Then run the command to clone the repository: `git clone https://github.com/ramdhiwakar1/fl-dist-hack-sensors.git`. After cloning, navigate into the project directory with `cd fl-dist-hack-sensors` and switch to the MVP-v1 branch using `git checkout MVP-v1`. This branch contains all the trial scripts we'll be using.

## Installing Dependencies

Now that you have the code, you need to install the required Python libraries. These libraries provide the machine learning framework (PyTorch), the federated learning infrastructure (Flower), and data processing utilities (pandas, numpy, scikit-learn). From the project root directory, run the command: `pip install -r requirements.txt`. This installation might take a few minutes as it downloads and installs all the necessary packages. If you encounter any permission errors, try adding the `--user` flag to the command.

Once the installation completes, you can verify it worked by trying to import the libraries in Python. Open a Python interpreter by typing `python` in your terminal, then try running `import flwr`, `import torch`, and `import pandas`. If none of these commands produce errors, you're ready to proceed. Type `exit()` to close the Python interpreter.

## Understanding Your Role

Now it's time to determine who will play which role in this federated learning experiment. The three of you need to decide who will be the server and who will be the two clients. The server person has a slightly more complex setup because they need to find and share their IP address, but once the system is running, all three roles are equally important.

The person designated as the server will coordinate the entire training process. Their laptop will host the central aggregation point where model updates from both clients are combined using the FedAvg strategy. The server doesn't train any models itself; it just orchestrates the process and performs the averaging of parameters.

The two people designated as clients will each work with different subsets of the pump sensor data. Client 0 will use the data from `federated_data/hybrid/client_0.csv`, and Client 1 will use the data from `federated_data/hybrid/client_1.csv`. These datasets represent two different industrial facilities with slightly different operating conditions and failure patterns. Each client will train a Convolutional Autoencoder on their local data and periodically share the learned model parameters with the server.

## Setting Up Network Communication

For the three laptops to communicate, the two clients need to know the IP address of the server laptop. The server person should find their local IP address first. If you're on Windows, open Command Prompt and type `ipconfig`, then look for the "IPv4 Address" under your active network connection (usually something like WiFi or Ethernet). It will look something like `192.168.1.100` or `10.0.0.50`. If you're on Mac or Linux, open Terminal and type `ifconfig` or `ip addr`, and look for an address starting with `192.168` or `10.0`. Write this IP address down and share it with the other two participants.

Make absolutely sure all three laptops are connected to the same WiFi network. If you're on different networks, the clients won't be able to connect to the server. You can verify you're on the same network by checking your WiFi settings and confirming everyone sees the same network name.

It's a good idea to test the connection before running the federated learning scripts. From one of the client laptops, try pinging the server's IP address. On Windows, open Command Prompt and type `ping 192.168.1.100` (replacing with the actual server IP). On Mac or Linux, open Terminal and use the same command. If you see replies coming back, the connection is working. Press Ctrl+C to stop the ping test.

## Remote Setup: If You're NOT on the Same WiFi Network

If the three of you are working from different locations—different homes, different cities, or different networks—you cannot use local IP addresses to connect. In this case, you need to create a tunnel that exposes the server to the internet so clients can reach it from anywhere. The simplest solution is to use ngrok, which creates a secure tunnel to your localhost.

The server person needs to download and install ngrok first. Visit https://ngrok.com and create a free account. Download the ngrok executable for your operating system. On Windows, you'll get a zip file containing ngrok.exe. Extract it to a convenient location. On Mac or Linux, you can install it via package managers or download the binary and make it executable with `chmod +x ngrok`.

Once ngrok is installed, the server person needs to authenticate it with their account token. After signing up on the ngrok website, you'll find your auth token in the dashboard. Run the command `ngrok authtoken <your_token_here>` to link ngrok to your account. This is a one-time setup step.

Now here's how the process works when you're remote. The server person will run two terminal windows simultaneously. In the first terminal, navigate to `script/trial` and start the server as usual with `python server.py 10 2`. The server will start on localhost port 8080. In a second terminal window (keep the first one running), run the command `ngrok tcp 8080`. This tells ngrok to create a public TCP tunnel to your local port 8080.

When ngrok starts, it will display connection information that looks something like this: "Forwarding tcp://2.tcp.ngrok.io:15432 -> localhost:8080". The important part is the public address—in this example, `2.tcp.ngrok.io:15432`. This is what you'll share with the two client people. Every time you restart ngrok, you'll get a different address, so make sure to share the current one.

The two client people can now connect from anywhere in the world using this ngrok address. When they run their client scripts, they'll use the ngrok hostname and port instead of a local IP. For example, Client 0 would run `python client.py 0 2.tcp.ngrok.io` and the script needs a small modification to handle the port. Actually, let me provide you with a modified command that works with ngrok.

Since the ngrok address includes a non-standard port, you need to modify how clients connect. Open the `client.py` file and find the line near the end that says `fl.client.start_numpy_client(server_address=f"{server_ip}:8080", client=client)`. Change it to `fl.client.start_numpy_client(server_address=f"{server_ip}", client=client)` (removing the hardcoded :8080). Then when running the client, include the full ngrok address with port: `python client.py 0 2.tcp.ngrok.io:15432`.

One important limitation of the free ngrok account is that it only allows one simultaneous tunnel and the connection URLs change every time you restart ngrok. If you need more stable connections or multiple tunnels, you'd need to upgrade to a paid plan. However, for this trial with 2 clients connecting to 1 server, the free tier works fine since both clients can connect to the same tunnel.

An alternative to ngrok is Tailscale, which might actually be easier for your scenario. Tailscale creates a virtual private network that makes all your devices appear to be on the same local network, even when they're physically remote. All three of you would install Tailscale, sign in with the same account or join the same network, and then you can use the Tailscale IP addresses just like local IPs. The advantage is that it's more stable and doesn't require running a separate tunnel process. Visit https://tailscale.com to try it if ngrok gives you trouble.

Another simple alternative is for the server person to run the server on a cloud VM like AWS EC2, Google Cloud, or Azure. You'd rent a small virtual machine (costs about five dollars per month), install Python and the dependencies there, copy the code and data to it, and run the server. Then clients connect to the public IP address of the VM. This is the most robust solution for serious federated learning experiments but requires a bit more setup and costs money.

## Running the Federated Learning Experiment

Now we're ready to start the actual federated learning process. The order of operations is important: the server must start first and be waiting for clients before the clients try to connect. If clients try to connect before the server is ready, they'll get connection errors.

### Server Person Instructions

The person running the server should navigate to the trial scripts directory. From the project root, use the command: `cd script/trial`. Now start the server by running: `python server.py 10 2`. Let me explain what these numbers mean. The first number (10) is the number of training rounds. Each round consists of clients training locally, sending updates to the server, and the server aggregating them. Ten rounds is enough to see meaningful learning without taking too long. The second number (2) tells the server to wait for two clients before starting. If you wanted to run with more clients, you'd increase this number.

After running the command, you should see output indicating the server is starting and waiting for clients to connect. The server will print its configuration settings and then display a message saying it's waiting for 2 clients to connect. At this point, don't do anything else on the server laptop—just let it wait. The server will automatically start the training process once both clients have connected.

During the training rounds, you'll see progress messages showing when clients are selected for training, when their updates are received, and when the aggregation happens. You'll also see the aggregated loss metric, which indicates how well the global model is performing. Lower loss values are better and indicate the autoencoder is learning to reconstruct the sensor data more accurately.

### Client 0 Person Instructions

The first client person should also navigate to the trial scripts directory with `cd script/trial` from the project root. Before running the client script, make sure you have the server's IP address that was shared earlier. Start your client by running: `python client.py 0 192.168.1.100`, replacing `192.168.1.100` with the actual server IP address. If the server is running on the same laptop (for testing purposes), you can use `localhost` instead of the IP address.

The number 0 in the command identifies you as Client 0, which means the script will load the data from `federated_data/hybrid/client_0.csv`. This dataset represents a facility with about 11,000 pump sensor readings. When you run the command, you'll first see the client loading and preprocessing this data. It will create sequences of sensor readings, normalize them, and prepare them for training the Convolutional Autoencoder.

After the data is loaded, the client will create the autoencoder model and connect to the server. Once connected, it will wait for the server to initiate training rounds. When a round begins, you'll see your client training the model locally for 3 epochs, which should take a minute or two depending on your laptop's speed. After training, the client sends its updated model parameters to the server and waits for the next round. This process repeats for all 10 rounds.

### Client 1 Person Instructions

The second client person follows almost the same process as Client 0, but with a different client ID number. Navigate to the trial directory with `cd script/trial`, and then run: `python client.py 1 192.168.1.100`, again replacing the IP address with your server's actual IP. The key difference is the number 1, which tells the script to load `federated_data/hybrid/client_1.csv` instead. This dataset is much larger, with about 77,000 sensor readings, representing a bigger industrial facility.

Just like Client 0, you'll see data loading and preprocessing messages first. Your client will take a bit longer to create the data sequences because you have more samples to process. Once the model is created and connected to the server, the training process is identical to Client 0. You'll train locally for 3 epochs per round and send updates to the server.

Both clients must remain connected for the entire duration of the 10 training rounds. If one client disconnects, the training process will pause or fail, depending on the round. Each round typically takes 2-5 minutes depending on your hardware, so the complete training process should take about 20-40 minutes total.

## Understanding the Output and Results

As the federated learning process runs, all three participants will see different but related information. The server displays high-level coordination information: which clients are participating in each round, when aggregation happens, and the overall loss metric. Lower loss values indicate better model performance.

The clients display more detailed training information. You'll see the training loss decrease over the 3 local epochs in each round, which shows the model is learning from your local data. After local training, you'll see evaluation metrics showing how well the model reconstructs your test data. The reconstruction error (also called test loss) is the key metric for anomaly detection—normal pump operations should have low reconstruction error, while anomalies will have high error.

After all 10 rounds complete, each laptop will print a completion message. The server will confirm that training is done and all clients participated successfully. The clients will show their final evaluation metrics. At this point, you've successfully completed a federated learning experiment where two facilities collaborated to build a shared anomaly detection model without sharing their raw sensor data.

## What This Accomplishes

Through this trial, you've demonstrated several important concepts in privacy-preserving machine learning. First, you've shown that multiple parties can collaborate on training a machine learning model without centralizing their data. Each client's sensitive pump sensor readings stayed on their local laptop throughout the entire process. Only the model parameters (weights and biases) were shared, not the actual data.

Second, you've implemented the FedAvg algorithm, which is the foundational strategy for most federated learning systems. The server computed weighted averages of the client model updates, giving more weight to updates from the client with more data (Client 1 with 77,000 samples versus Client 0 with 11,000 samples). This ensures the global model benefits fairly from all participants.

Third, you've built a Convolutional Autoencoder for anomaly detection, which is a practical tool for industrial IoT applications. The model learns the normal patterns in pump sensor data and can detect anomalies by measuring reconstruction error. High reconstruction error on new data suggests the pump is operating in an unusual way that might indicate impending failure.

## Troubleshooting Common Issues

If clients can't connect to the server, first verify all three laptops are on the same WiFi network. Then double-check the IP address you're using is correct and matches what `ipconfig` or `ifconfig` shows on the server laptop. Some WiFi networks, especially corporate or university networks, may block direct connections between devices for security reasons. If you suspect this is the case, try using a mobile hotspot or home network instead.

If you see errors about port 8080 already being in use, it means another program is using that port or you have a previous server instance still running. On Windows, you can find and kill the process using `netstat -ano | findstr :8080` to find the process ID, then `taskkill /PID <number> /F` to kill it. On Mac or Linux, use `lsof -i :8080` to find the process and `kill <PID>` to stop it.

If the training seems to hang or clients timeout, the server might not be receiving updates from clients. This could be due to firewall settings blocking the connection. On Windows, you may need to allow Python through Windows Defender Firewall. On Mac, check System Preferences > Security & Privacy > Firewall. Temporarily disabling the firewall can help diagnose if it's the issue, but remember to re-enable it afterward.

If you encounter CUDA or GPU-related errors, the script is trying to use a GPU that isn't available or isn't configured properly. The good news is the script will automatically fall back to CPU if GPU isn't available. If you're seeing errors, you can force CPU usage by modifying the `device` line in `client.py` to always use CPU: `device = torch.device("cpu")`.

## Next Steps and Experimentation

Now that you've successfully run a basic federated learning trial, there are many ways to extend and experiment with the system. You could try increasing the number of training rounds to see if the model continues to improve beyond 10 rounds. You could modify the autoencoder architecture in `client.py` to use more layers or different kernel sizes. You could also experiment with the learning rate (currently 0.001) or the number of local epochs (currently 3) to see how they affect convergence speed.

For a more challenging extension, you could try running with all 5 clients instead of just 2. You'd need 6 people (or 6 laptops) total—one server and 5 clients. Change the server command to `python server.py 10 5` and have each client use their respective client ID (0, 1, 2, 3, or 4). This would give you a more realistic federated learning scenario with more heterogeneity in the data.

You could also add visualization of the results. After training completes, you could modify the client script to save the model and create plots showing the reconstruction error distribution, which would help you understand what threshold to use for classifying anomalies. Samples with reconstruction error above the threshold would be flagged as potential pump failures.

## Conclusion

Congratulations on completing your first federated learning experiment! You've built a practical privacy-preserving machine learning system for industrial IoT data. The techniques you've learned—federated averaging, convolutional autoencoders, and distributed training—are used in production systems at companies developing privacy-conscious AI solutions. This hands-on experience gives you foundational knowledge in an increasingly important area of machine learning, especially as data privacy regulations become more strict worldwide.

Remember that federated learning is not just about privacy—it's also about enabling machine learning in scenarios where centralizing data is impractical or impossible. In the industrial context, this could mean training models across competing companies that each want to benefit from shared knowledge without revealing their operational data. In healthcare, it enables training on sensitive patient data across hospitals without violating privacy regulations. The principles you've learned here apply broadly across these domains.

Thank you for participating in this trial. If you have questions or want to learn more about federated learning, the Flower documentation (https://flower.dev/) is an excellent resource, as are recent research papers on federated learning strategies beyond FedAvg, such as FedProx, FedAdam, and Differential Privacy in federated settings.

