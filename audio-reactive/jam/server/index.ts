import { WebSocketServer, WebSocket } from 'ws';

// --- Types (mirror client protocol) ---
interface RoomState {
  trackUrl: string;
  playing: boolean;
  position: number;        // position at last known reference point
  positionTimestamp: number; // Date.now() when position was set
  effect: string;
  users: Map<WebSocket, string>;
}

const room: RoomState = {
  trackUrl: '',
  playing: false,
  position: 0,
  positionTimestamp: Date.now(),
  effect: '',
  users: new Map(),
};

// Compute current playback position accounting for elapsed time
function currentPosition(): number {
  if (!room.playing) return room.position;
  const elapsed = (Date.now() - room.positionTimestamp) / 1000;
  return room.position + elapsed;
}

function setPosition(pos: number): void {
  room.position = pos;
  room.positionTimestamp = Date.now();
}

function userList(): string[] {
  return [...room.users.values()];
}

function broadcast(data: object, exclude?: WebSocket): void {
  const msg = JSON.stringify(data);
  for (const client of wss.clients) {
    if (client !== exclude && client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  }
}

// --- Server ---
const PORT = Number(process.env.PORT) || 8080;
const wss = new WebSocketServer({ port: PORT });

console.log(`Jam server listening on ws://localhost:${PORT}`);

wss.on('connection', (ws) => {
  // Send current room state to new connection
  ws.send(JSON.stringify({
    type: 'state',
    trackUrl: room.trackUrl,
    playing: room.playing,
    position: currentPosition(),
    effect: room.effect,
    users: userList(),
    serverTime: Date.now(),
  }));

  ws.on('message', (raw) => {
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      return; // ignore malformed messages
    }

    switch (msg.type) {
      case 'join':
        room.users.set(ws, String(msg.name ?? 'anon'));
        broadcast({ type: 'presence', users: userList() });
        break;

      case 'play':
        room.trackUrl = String(msg.trackUrl ?? '');
        setPosition(Number(msg.position ?? 0));
        room.playing = true;
        broadcast(
          { type: 'play', trackUrl: room.trackUrl, position: room.position, serverTime: Date.now() },
          ws
        );
        break;

      case 'pause':
        setPosition(Number(msg.position ?? currentPosition()));
        room.playing = false;
        broadcast({ type: 'pause', position: room.position }, ws);
        break;

      case 'seek':
        setPosition(Number(msg.position ?? 0));
        broadcast({ type: 'seek', position: room.position, serverTime: Date.now() }, ws);
        break;

      case 'select_effect':
        room.effect = String(msg.effect ?? '');
        broadcast({ type: 'select_effect', effect: room.effect }, ws);
        break;
    }
  });

  ws.on('close', () => {
    room.users.delete(ws);
    broadcast({ type: 'presence', users: userList() });
  });
});

// Periodic sync heartbeat every 5s
setInterval(() => {
  if (room.playing) {
    broadcast({ type: 'sync', position: currentPosition(), serverTime: Date.now() });
  }
}, 5_000);
