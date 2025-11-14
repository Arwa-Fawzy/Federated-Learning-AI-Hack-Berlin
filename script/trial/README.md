# Federated Learning Trial - Pump Sensor Anomaly Detection

Welcome to the federated learning trial for pump sensor anomaly detection. This guide will walk you through setting up and running a distributed machine learning experiment using three laptops. One laptop will act as the central server that coordinates the training, while two other laptops will act as clients that train models on their local data without sharing the raw sensor readings.

## Understanding What We're Building

In this trial, we're implementing a privacy-preserving machine learning system for detecting anomalies in industrial pump sensors. The key innovation is that each facility (represented by a client) keeps its sensitive operational data local while still contributing to a shared machine learning model. We use a Convolutional Autoencoder, which is a type of neural network that learns to compress and reconstruct sensor data. When the model encounters unusual patterns (anomalies), it will struggle to reconstruct them accurately, giving us a way to detect potential pump failures or malfunctions.

The Federated Averaging (FedAvg) strategy works by having each client train the model on their local data for a few iterations, then sending only the updated model parameters (not the raw data) to the central server. The server averages these parameters from all clients to create an improved global model, which is then sent back to clients for another round of training. This process repeats for several rounds until the model converges to a solution that benefits from everyone's data without anyone having to share their sensitive information.

## System Requirements and Preparation

Before we begin, make sure both participants have their laptops ready. You'll need Python 3.8 or higher installed on your system, along with at least 4GB of available RAM and about 2GB of free disk space. **Most importantly, both laptops must be connected to the same WiFi network.** This is crucial - verify you're both connected to the exact same network name before proceeding.

Both participants should start by cloning the repository to their laptops. Open a terminal or command prompt and navigate to a directory where you want to store the project. Then run the command to clone the repository: `git clone https://github.com/ramdhiwakar1/fl-dist-hack-sensors.git`. After cloning, navigate into the project directory with `cd fl-dist-hack-sensors` and switch to the MVP-v1 branch using `git checkout MVP-v1`. This branch contains all the trial scripts we'll be using.

## Installing Dependencies

Now that you have the code, you need to install the required Python libraries. These libraries provide the machine learning framework (PyTorch), the federated learning infrastructure (Flower), and data processing utilities (pandas, numpy, scikit-learn). From the project root directory, run the command: `pip install -r requirements.txt`. This installation might take a few minutes as it downloads and installs all the necessary packages. If you encounter any permission errors, try adding the `--user` flag to the command.

Once the installation completes, you can verify it worked by trying to import the libraries in Python. Open a Python interpreter by typing `python` in your terminal, then try running `import flwr`, `import torch`, and `import pandas`. If none of these commands produce errors, you're ready to proceed. Type `exit()` to close the Python interpreter.

## Understanding Your Role

Now it's time to determine who will play which role in this federated learning experiment. The two of you need to decide who will be the server and who will be the client. The server person needs to find and share their IP address, but once the system is running, both roles are equally important.

**Person 1 - Server**: You will coordinate the entire training process. Your laptop will host the central aggregation point where model updates from the client are received. The server doesn't train any models itself; it just orchestrates the process and performs the coordination of the federated learning rounds. You'll see high-level information about training progress, client connections, and aggregated metrics.

**Person 2 - Client**: You will work with pump sensor data from `federated_data/hybrid/client_0.csv`. This dataset represents an industrial facility with about 11,000 pump sensor readings. You'll train a Convolutional Autoencoder on this local data and periodically share the learned model parameters with the server. Your laptop does all the actual model training work.

## Setting Up Network Communication

For the two laptops to communicate, the client needs to know the IP address of the server laptop. The server person should find their local IP address first. If you're on Windows, open Command Prompt and type `ipconfig`, then look for the "IPv4 Address" under your active WiFi connection. It will look something like `192.168.1.100` or `10.0.0.50`. If you're on Mac or Linux, open Terminal and type `ifconfig` or `ip addr show`, and look for an address starting with `192.168` or `10.0`. Write this IP address down and share it with the client person.

Double-check that both laptops are connected to the same WiFi network. You can verify this by checking your WiFi settings and confirming you both see the same network name. This is absolutely essential - if you're on different networks, the connection won't work.

It's a good idea to test the connection before running the federated learning scripts. From the client laptop, try pinging the server's IP address. On Windows, open Command Prompt and type `ping 192.168.1.100` (replacing with the actual server IP). On Mac or Linux, open Terminal and use the same command. If you see replies coming back, the connection is working perfectly. Press Ctrl+C to stop the ping test. If the ping fails, check that you're both on the same WiFi network and that any firewall on the server laptop isn't blocking incoming connections.

## Running the Federated Learning Experiment

Now we're ready to start the actual federated learning process. The order of operations is important: the server must start first and be waiting for clients before the clients try to connect. If clients try to connect before the server is ready, they'll get connection errors.

### Server Person Instructions (Person 1)

The person running the server should navigate to the trial scripts directory. From the project root, use the command: `cd script/trial`. Now start the server by running: `python server.py 10 1`. Let me explain what these numbers mean. The first number (10) is the number of training rounds. Each round consists of the client training locally, sending updates to the server, and the server receiving them. Ten rounds is enough to see meaningful learning without taking too long. The second number (1) tells the server to wait for one client before starting.

After running the command, you should see output indicating the server is starting and waiting for a client to connect. The server will print its configuration settings and then display a message saying it's waiting for 1 client to connect. Keep this terminal window open and running - don't close it. The server will automatically start the training process once the client has connected.

During the training rounds, you'll see progress messages showing when the client is selected for training, when updates are received, and when processing happens. You'll also see the loss metric, which indicates how well the model is performing. Lower loss values are better and indicate the autoencoder is learning to reconstruct the sensor data more accurately. Your role is primarily to monitor and coordinate - the actual model training happens on the client laptop.

### Client Person Instructions (Person 2)

Navigate to the trial scripts directory with `cd script/trial` from the project root. Before running the client script, make sure you have the server's IP address that Person 1 shared with you earlier. Start your client by running: `python client.py 0 192.168.1.100`, replacing `192.168.1.100` with the actual server IP address that was shared with you.

The number 0 in the command identifies you as Client 0, which means the script will load the data from `federated_data/hybrid/client_0.csv`. This dataset contains about 11,000 sensor readings representing an industrial facility. When you run the command, you'll first see the client loading and preprocessing this data. It will create sequences of sensor readings, normalize them, and prepare them for training the Convolutional Autoencoder.

After the data is loaded, the client will create the autoencoder model and connect to the server. Once connected, it will wait for the server to initiate training rounds. When a round begins, you'll see your client training the model locally for 3 epochs, which should take a minute or two depending on your laptop's speed. After training, the client sends its updated model parameters to the server and waits for the next round. This process repeats for all 10 rounds.

Keep your terminal open and connected for the entire duration of the 10 training rounds. If you disconnect, the training process will stop. Each round typically takes 2-4 minutes depending on your hardware, so the complete training process should take about 20-40 minutes total. You can watch the progress in your terminal as your local model trains and communicates with the server.

## Understanding the Output and Results

As the federated learning process runs, both participants will see different but related information on their screens. Person 1 (server) will see high-level coordination information in their terminal: when the client connects, when each training round starts, when updates are received, and the overall loss metric. Lower loss values indicate better model performance. The server acts as the coordinator but doesn't do the actual model training.

Person 2 (client) will see detailed training information in their terminal. You'll see the training loss decrease over the 3 local epochs in each round, which shows the model is learning from your local data. After local training, you'll see evaluation metrics showing how well the model reconstructs your test data. The reconstruction error (also called test loss) is the key metric for anomaly detection—normal pump operations should have low reconstruction error, while anomalies will have high error.

After all 10 rounds complete, both terminals will print completion messages. The server will confirm that training is done and the client participated successfully. The client terminal will show the final evaluation metrics. At this point, you've successfully completed a federated learning experiment where you built a shared anomaly detection model without the client sharing their raw sensor data with the server.

## What This Accomplishes

Through this trial, you've demonstrated several important concepts in privacy-preserving machine learning. First, you've shown how a machine learning model can be trained without centralizing the data. The client's sensitive pump sensor readings stayed on their local laptop throughout the entire process. Only the model parameters (weights and biases) were shared with the server, not the actual data.

Second, you've implemented the FedAvg algorithm, which is the foundational strategy for most federated learning systems. The server received model updates from the client and coordinated the training rounds. In a real-world scenario with multiple clients, the server would compute weighted averages of all client model updates, ensuring the global model benefits fairly from all participants.

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

