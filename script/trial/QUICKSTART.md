# Quick Start - Same WiFi Setup

## Prerequisites
- 2 laptops on the **same WiFi network**
- Python 3.8+ installed
- All dependencies installed: `pip install -r requirements.txt`

---

## Step 1: Find Server IP

**Person 1 (Server)** finds their IP:

**Windows:**
```bash
ipconfig
# Look for "IPv4 Address" like 192.168.1.100
```

**Mac/Linux:**
```bash
ifconfig
# Or: ip addr show
# Look for address starting with 192.168 or 10.0
```

Share this IP with Person 2.

---

## Step 2: Start Server

**Person 1 (Server) - Terminal 1:**
```bash
cd script/trial
python server.py 10 2
```

Wait for: `"Server is starting and waiting for 2 clients..."`

---

## Step 3: Start Client 0

**Person 1 (Server) - Terminal 2 (same laptop):**
```bash
cd script/trial
python client.py 0 localhost
```

You should see: `"Client 0: Loading data..."`

---

## Step 4: Start Client 1

**Person 2 (Client):**
```bash
cd script/trial
python client.py 1 192.168.1.100
```

Replace `192.168.1.100` with the actual server IP from Step 1.

---

## What You'll See

**Server Terminal**: 
```
Round 1/10
Client 0 selected, Client 1 selected
Aggregating 2 results...
→ Aggregated loss: 0.0234
```

**Client Terminals**:
```
Client X: Starting local training
  Epoch 1/3: Training Loss = 0.0456
  Epoch 2/3: Training Loss = 0.0312
  Epoch 3/3: Training Loss = 0.0267
✓ Local training complete
→ Test Loss: 0.0289
```

---

## Timeline

- **10 rounds** × **3-5 min/round** = **30-50 minutes total**
- Keep all terminals open until completion

---

## Troubleshooting

**Problem**: Client can't connect  
**Fix**: 
1. Check both on same WiFi
2. Ping server IP from client laptop
3. Check firewall isn't blocking port 8080

**Problem**: "Port already in use"  
**Fix**: 
```bash
# Windows
taskkill /F /IM python.exe

# Mac/Linux  
pkill python
```

---

## Summary

✅ 2 people, 2 laptops, same WiFi  
✅ Person 1 runs server + Client 0  
✅ Person 2 runs Client 1  
✅ 10 rounds of federated learning  
✅ Privacy-preserving: no raw data shared  

That's it! 🚀

