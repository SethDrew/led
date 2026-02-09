#!/bin/bash
# Hetzner Setup Helper for LED Project

echo "🇩🇪 Hetzner Cloud Setup for LED Project"
echo "========================================"
echo ""

# Check if SSH key exists
echo "1. Checking SSH keys..."
if [ -f ~/.ssh/id_ed25519.pub ]; then
    echo "   ✓ SSH key found: ~/.ssh/id_ed25519.pub"
    echo ""
    echo "   Your public key (copy this to Hetzner):"
    echo "   ----------------------------------------"
    cat ~/.ssh/id_ed25519.pub
    echo "   ----------------------------------------"
else
    echo "   ⚠️  No SSH key found. Creating one..."
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "led-project-$(whoami)"
    echo "   ✓ SSH key created!"
    echo ""
    echo "   Your public key (copy this to Hetzner):"
    echo "   ----------------------------------------"
    cat ~/.ssh/id_ed25519.pub
    echo "   ----------------------------------------"
fi

echo ""
echo "2. Next steps:"
echo ""
echo "   📋 Copy the public key above (entire line)"
echo ""
echo "   🌐 Go to: https://console.hetzner.cloud/"
echo ""
echo "   ➕ Create server:"
echo "      • Location: Ashburn, VA (closest to you)"
echo "      • Image: Ubuntu 24.04"
echo "      • Type: CPX11 (€4.51/month)"
echo "      • SSH Key: Paste the key above"
echo "      • Name: gastown-led"
echo ""
echo "   ⏱️  Server will be ready in ~30 seconds"
echo ""
echo "3. After server is created:"
echo ""
echo "   Get your server's IP from Hetzner console, then:"
echo ""
echo "   ssh root@YOUR_SERVER_IP"
echo ""
echo "4. Optional: Install Hetzner CLI for automation"
echo ""
echo "   brew install hcloud"
echo ""
echo "========================================"
echo ""
echo "💰 Cost: €4.51/month (~$5) vs AWS $17/month"
echo "💾 Power off when not using: ~€0.01/month"
echo ""
echo "📚 Full guide: See HETZNER_SETUP.md"
echo ""

# Check if hcloud is installed
if command -v hcloud &> /dev/null; then
    echo "✓ Hetzner CLI (hcloud) is installed"
    echo ""
    if hcloud context active 2>/dev/null | grep -q "led"; then
        echo "✓ CLI is configured for LED project"
        echo ""
        echo "Quick launch command:"
        echo "  hcloud server create --type cpx11 --image ubuntu-24.04 --name gastown-led --location ash"
    else
        echo "⚠️  CLI not configured yet. Run:"
        echo "  hcloud context create led-project"
        echo "  # Then paste your API token from Hetzner console"
    fi
else
    echo "ℹ️  Install Hetzner CLI for automation: brew install hcloud"
fi

echo ""
