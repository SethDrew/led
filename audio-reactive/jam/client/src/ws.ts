import type { ClientMessage, ServerMessage } from './protocol';

export type MessageHandler = (msg: ServerMessage) => void;
export type StatusHandler = (status: 'connecting' | 'connected' | 'disconnected') => void;

export class JamSocket {
  private ws: WebSocket | null = null;
  private handlers: MessageHandler[] = [];
  private statusHandlers: StatusHandler[] = [];
  private openCallbacks: (() => void)[] = [];
  private url: string;

  constructor(url = `ws://${location.hostname}:8080`) {
    this.url = url;
  }

  connect(): void {
    this.setStatus('connecting');
    this.ws = new WebSocket(this.url);

    this.ws.addEventListener('open', () => {
      this.setStatus('connected');
      for (const cb of this.openCallbacks) cb();
      this.openCallbacks = [];
    });

    this.ws.addEventListener('message', (event) => {
      const msg: ServerMessage = JSON.parse(event.data as string);
      for (const handler of this.handlers) {
        handler(msg);
      }
    });

    this.ws.addEventListener('close', () => {
      this.setStatus('disconnected');
      // Reconnect after 2 seconds
      setTimeout(() => this.connect(), 2000);
    });
  }

  send(msg: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  onMessage(handler: MessageHandler): void {
    this.handlers.push(handler);
  }

  onStatus(handler: StatusHandler): void {
    this.statusHandlers.push(handler);
  }

  // Run callback once the socket is open (immediately if already open)
  onceOpen(cb: () => void): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      cb();
    } else {
      this.openCallbacks.push(cb);
    }
  }

  private setStatus(status: 'connecting' | 'connected' | 'disconnected'): void {
    for (const handler of this.statusHandlers) {
      handler(status);
    }
  }
}
