# Toradex Apalis iMX6Q Reclamation Project

## Goal
Reclaim retired Heila v4 hardware as a field art / LED orchestrator device.
Replace vendor Ubuntu 16.04 + Heila stack with mainline Linux (Debian armhf).

## Hardware Identification

- **SoM**: Toradex Apalis iMX6 Quad (product ID 0028)
- **Carrier board**: Ixora V1.1
- **SoC**: NXP i.MX6 Quad — 4x Cortex-A9 @ ~1.2GHz, ARMv7 (32-bit)
- **RAM**: 2GB DDR3
- **Storage**: 3.5GB eMMC (92% full with Heila stuff), 59GB SD card at /mnt/sdcard
- **MAC (enp1s0)**: c4:00:ad:83:a4:3e
- **Serial**: 10566180
- **Hostname**: heila-e-gtl-12
- **Mainline DT**: `imx6q-apalis-ixora-v1.1.dts` (already in mainline kernel tree)
- **Mainline U-Boot**: `apalis_imx6_defconfig`

## Interfaces Available

- 2x Gigabit Ethernet (eth0 + enp1s0)
- 2x CAN bus (can0, can1)
- USB 2.0 hub (onboard)
- Micro-USB (bottom center of Ixora — possibly USB OTG for recovery mode)
- Micro-SD card slot (bottom center of Ixora)
- UART debug pins (bottom-left of Ixora board, labeled TX/RX on silkscreen)
- No Bluetooth, no WiFi, no Zigbee (would need USB dongles)

## Current Software State

- **OS**: Ubuntu 16.04.7 LTS (Xenial), kernel 4.9.67-dirty
- **OpenSSH**: 7.2p2
- **Heila services running on boot** (should be disabled):
  - greengrass.service — AWS IoT Greengrass v2 (Java, CPU hog)
  - heilaiq-admin-tunnel.service
  - openvpn@client.service — VPN to dead Heila infra
  - fluent-bit-sXk.service — log shipping
  - nginx.service — reverse proxy
- **Greengrass deploying**: `com.riot.assetsim` (an asset simulator, probably user-installed)
- **Disk nearly full**: root partition 3.5GB, 308MB free. Greengrass + pip installs filling it.

## Network Access

- Connected via Ethernet to Mac on interface `en9`
- Device port: `enp1s0` (the one labeled "dev" on the enclosure exterior)
- Other port `eth0` had no carrier
- Device has no IPv4 configured on this port — IPv6 link-local only
- Discovery method: `ping6 -I en9 ff02::1` then look for non-self responses
- **IPv6 address changes on each reboot** — must re-discover each time
- SSH: `ssh user@"fe80::<discovered-addr>%en9"` — password auth, BatchMode works
- **Only one SSH session at a time** — device rejects concurrent connections

## Credentials

- SSH user: `user`
- Sudo password: `sudo` (note: piping via `echo 'sudo' | sudo -S` did NOT work reliably over SSH)

## Stabilization TODO (not yet done)

From an interactive SSH session, run:
```bash
sudo systemctl stop greengrass && sudo systemctl disable greengrass
sudo systemctl stop heilaiq-admin-tunnel openvpn@client fluent-bit-sXk nginx
sudo systemctl disable heilaiq-admin-tunnel openvpn@client fluent-bit-sXk nginx
```

## Plan: Flash Mainline Linux

### What we decided
Go fully mainline — no Toradex BSP dependency. The i.MX6Q has excellent upstream support.

### Steps
1. **Get serial console working** — UART debug pins on bottom-left of Ixora board (TX/RX labeled on silkscreen). Need a USB-to-UART adapter (FTDI/CP2102/CH340). Connect TX->RX, RX->TX, GND->GND. Baud 115200.
2. **Cross-compile U-Boot** — mainline U-Boot, `apalis_imx6_defconfig`. Build on Mac with ARM cross-compiler.
3. **Cross-compile kernel** — mainline Linux, `imx_v6_v7_defconfig`. Device tree `imx6q-apalis-ixora-v1.1.dts` is already upstream.
4. **Build root filesystem** — Debian armhf (bookworm) minimal, or Alpine.
5. **Flash** — via USB OTG recovery mode (micro-USB on Ixora) using `imx_usb_loader`, or write to SD card and boot from there first.
6. **Configure** — networking, SSH, systemd services for LED orchestration.

### Hardware needed
- USB-to-UART adapter (for serial console / boot debugging)
- Possibly: USB Bluetooth dongle, USB WiFi adapter (if wireless is wanted)

## Physical Notes

- Black metal enclosure, Heila branding
- Back panel: two Ethernet ports labeled "cntrl" (eth0) and "dev" (enp1s0), USB ports, power
- Internal: blue Ixora carrier board on top, green Apalis SoM plugged in below
- Ribbon cable (blue/white/red) connects Ixora to a Heila custom daughter board (dark PCB, top of enclosure) — likely CAN bus / Heila I/O
- Yellow wire on right side to green connector (power or relay)
- Photo: `~/Downloads/20260311_162501.jpg`
