# Quick Guide: Remote Setup with ngrok

## If You're NOT on the Same WiFi

Use this guide if the three of you are in different locations.

---

## Setup Summary

**Server person** runs TWO terminals:
1. Terminal 1: Python server
2. Terminal 2: ngrok tunnel

**Client people** connect using the ngrok address.

---

## Step-by-Step Instructions

### Server Person (Terminal 1)

```bash
cd script/trial
python server.py 10 2
```

Leave this running.

### Server Person (Terminal 2)

Install ngrok first:
- Go to https://ngrok.com and sign up
- Download ngrok for your OS
- Run: `ngrok authtoken <your_token>` (one-time setup)

Then start the tunnel:
```bash
ngrok tcp 8080
```

You'll see output like:
```
Forwarding: tcp://2.tcp.ngrok.io:15432 -> localhost:8080
```

**Share this address** with both clients: `2.tcp.ngrok.io:15432`

---

### Client 0 Person

```bash
cd script/trial
python client.py 0 2.tcp.ngrok.io:15432
```

(Use the actual ngrok address shared by server)

---

### Client 1 Person

```bash
cd script/trial
python client.py 1 2.tcp.ngrok.io:15432
```

(Use the actual ngrok address shared by server)

---

## What You'll See

**Server Terminal 1**: Training rounds, aggregation, loss metrics  
**Server Terminal 2**: ngrok connection status  
**Clients**: Local training progress, evaluation metrics

---

## Important Notes

- **ngrok address changes** every time you restart ngrok
- **Free ngrok** works fine for 2 clients connecting to 1 server
- Keep **both server terminals running** throughout the experiment
- If connection fails, check firewall settings

---

## Alternative: Tailscale (Easier)

Instead of ngrok, all three people can install Tailscale:
1. Install from https://tailscale.com
2. Sign in with same account or join same network
3. Use Tailscale IPs like regular local IPs

**Advantage**: More stable, no tunnel needed

---

## Cloud Alternative (Most Robust)

Server person can run server on cloud VM:
- AWS EC2 / Google Cloud / Azure
- ~$5/month for small VM
- Install Python, clone repo, run server
- Clients connect to public IP

**Best for**: Serious experiments, multiple sessions

---

## Troubleshooting

**Problem**: "Connection refused"  
**Fix**: Make sure server is running BEFORE starting ngrok

**Problem**: "Port already in use"  
**Fix**: Kill existing server/ngrok: `taskkill /F /IM python.exe`

**Problem**: ngrok connection drops  
**Fix**: Free ngrok has connection limits; upgrade or use Tailscale

---

## Ready to Run?

1. ✅ Server starts Python server (Terminal 1)
2. ✅ Server starts ngrok (Terminal 2)
3. ✅ Server shares ngrok address
4. ✅ Both clients connect using that address
5. ✅ Watch 10 rounds complete!

**Total time**: ~30-50 minutes

