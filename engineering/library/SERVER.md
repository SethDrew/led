# Server (Hetzner)

Remote server for the LED web viewer.

## Access

```
ssh hetzner
```

Host configured in `~/.ssh/config`.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| `led-viewer.service` | 5555 | Web viewer backend |
| `webhook.service` | — | Auto-deploy on GitHub push |
| nginx | 80/443 | Reverse proxy to led-viewer |

## Deploy

Automatic on `git push` via webhook. Manual restart if needed:

```
ssh hetzner sudo systemctl restart led-viewer
```
