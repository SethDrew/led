// Client → Server
export type ClientMessage =
  | { type: 'join'; name: string }
  | { type: 'play'; trackUrl: string; position: number }
  | { type: 'pause'; position: number }
  | { type: 'seek'; position: number }
  | { type: 'select_effect'; effect: string };

// Server → Client
export type ServerMessage =
  | { type: 'state'; trackUrl: string; playing: boolean; position: number; effect: string; users: string[]; serverTime: number }
  | { type: 'play'; trackUrl: string; position: number; serverTime: number }
  | { type: 'pause'; position: number }
  | { type: 'seek'; position: number; serverTime: number }
  | { type: 'select_effect'; effect: string }
  | { type: 'presence'; users: string[] }
  | { type: 'sync'; position: number; serverTime: number };
