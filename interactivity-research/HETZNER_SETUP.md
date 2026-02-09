# Hetzner Cloud Setup for LED Project

## Why Hetzner?
- ✅ **70% cheaper than AWS** ($5 vs $17/month)
- ✅ EU-based, German company (strong privacy)
- ✅ No big tech political baggage
- ✅ Simple, transparent pricing
- ✅ Perfect for Gastown LED development

## 💰 Cost Breakdown

**Recommended:** CPX11 (ARM-based, best value)
- 2 vCPU (ARM)
- 2GB RAM
- 40GB SSD
- 20TB traffic
- **€4.51/month (~$5)** 🎉

**Alternative:** CX22 (x86 if you need it)
- 2 vCPU (Intel/AMD)
- 4GB RAM
- 40GB SSD
- **€5.83/month (~$6.50)**

vs AWS t3.small: $17/month

**Annual savings: ~$144** 💸

---

## 🚀 Quick Start (Web UI Method - Easiest)

### Step 1: Create Hetzner Account
1. Go to https://console.hetzner.cloud/
2. Sign up (requires payment method)
3. Verify email

### Step 2: Create Project
1. Click "New Project"
2. Name: "led-audio-reactive" or similar

### Step 3: Create Server
1. Click "Add Server"
2. **Location:** Ashburn, VA (closest to you) or Falkenstein, Germany
3. **Image:** Ubuntu 24.04
4. **Type:** Shared vCPU → **CPX11** (ARM)
5. **SSH Key:**
   - Click "Add SSH Key"
   - Paste your public key: `cat ~/.ssh/id_ed25519.pub` (or create new one)
6. **Name:** gastown-led
7. Click "Create & Buy"

**Done!** You'll get a public IP immediately.

### Step 4: Connect
```bash
ssh root@<your-server-ip>

# First login, update system
apt update && apt upgrade -y

# Create non-root user
adduser ubuntu
usermod -aG sudo ubuntu

# Copy SSH key to new user
rsync --archive --chown=ubuntu:ubuntu ~/.ssh /home/ubuntu
```

### Step 5: Install Gastown
```bash
# Switch to ubuntu user
su - ubuntu

# Install dependencies
sudo apt install -y git curl

# Install Gastown (follow their docs)
# https://github.com/steveyegge/gastown

# Clone your LED repo
git clone https://github.com/yourusername/led.git
cd led
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🛠️ CLI Method (Optional - for automation)

### Install Hetzner CLI
```bash
brew install hcloud
```

### Setup API Token
1. Hetzner Console → Project → Security → API Tokens
2. Generate token (Read & Write permissions)
3. Save token

```bash
# Configure CLI
hcloud context create led-project
# Paste your API token when prompted

# Verify
hcloud server list
```

### Create Server via CLI
```bash
# Create SSH key first (if needed)
ssh-keygen -t ed25519 -f ~/.ssh/hetzner-led -N ""

# Upload SSH key to Hetzner
hcloud ssh-key create \
  --name led-project \
  --public-key-from-file ~/.ssh/hetzner-led.pub

# Create server
hcloud server create \
  --type cpx11 \
  --image ubuntu-24.04 \
  --ssh-key led-project \
  --name gastown-led \
  --location ash

# Get IP
hcloud server list
```

---

## 🔧 Post-Setup: SSH Config

Add to `~/.ssh/config`:
```
Host gastown
  HostName <your-server-ip>
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60
```

Now connect with just: `ssh gastown`

---

## 📊 Management

### Web Console
https://console.hetzner.cloud/

**Common tasks:**
- Power on/off server
- View metrics (CPU, network)
- Create snapshots (backups)
- Resize server

### CLI Commands
```bash
# List servers
hcloud server list

# Power off (saves money! ~€0.01/month when off)
hcloud server poweroff gastown-led

# Power on
hcloud server poweron gastown-led

# Delete server (careful!)
hcloud server delete gastown-led

# Create snapshot (backup)
hcloud server create-image gastown-led --type snapshot
```

---

## 💡 Cost Optimization Tips

**1. Power off when not in use**
```bash
# Powered off: ~€0.01/month (only storage cost)
# Powered on: €4.51/month
```

**2. Use snapshots before experiments**
```bash
hcloud server create-image gastown-led --type snapshot --description "before-gastown-install"
# Snapshots: €0.01 per GB/month
```

**3. Monitor billing**
- Console → Billing → Usage
- Set up billing alerts

---

## 🔒 Security Checklist

- ✅ Disable root SSH login (use ubuntu user)
- ✅ Use SSH keys (no passwords)
- ✅ Enable firewall (ufw)
- ✅ Keep system updated
- ✅ Use non-default SSH port (optional)

```bash
# Basic firewall
sudo ufw allow 22/tcp
sudo ufw enable
```

---

## 🎯 Next Steps

1. ✅ Create Hetzner account
2. ✅ Launch CPX11 server (~5 min)
3. ✅ SSH in and set up
4. 🔄 Install Gastown
5. 🔄 Clone LED repo
6. 🎉 Start developing!

---

## 📞 Support

- **Docs:** https://docs.hetzner.com/
- **Community:** https://community.hetzner.com/
- **Status:** https://status.hetzner.com/

---

## 🌍 Locations (pick closest)

- **Ashburn, VA (USA)** - ash - Closest to you
- Hillsboro, OR (USA) - hil
- Falkenstein (Germany) - fsn1
- Nuremberg (Germany) - nbg1
- Helsinki (Finland) - hel1

**Latency from your Mac:**
```bash
# Test before choosing
ping ash.icmp.hetzner.com
ping hil.icmp.hetzner.com
```

---

**Ready to launch?** Let me know and I can walk you through it! 🚀
