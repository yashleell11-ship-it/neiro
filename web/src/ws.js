/**
 * The socket to the daemon.
 *
 * The token is injected into the page by the server, never put in the
 * URL: URLs end up in history, logs and Referer headers — uvicorn logs
 * every handshake's path, query string included. It rides the
 * Sec-WebSocket-Protocol header instead, the one header a page can set
 * on a WebSocket. The prefix is TOKEN_SUBPROTOCOL_PREFIX in
 * src/neiro/server.py; the server selects this same subprotocol back,
 * which is what lets the browser complete the handshake.
 *
 * It reconnects on its own. `neiro talk --browser` binds the port before
 * the models warm, so the tab may open seconds before anything listens,
 * and a restarted daemon should find the tab still there.
 */

export const TOKEN_SUBPROTOCOL_PREFIX = 'neiro.token.';

// Between attempts. A second is slow enough not to spam a daemon that is
// still warming and fast enough that a restart is not noticed.
const RECONNECT_MS = 1000;

export class Transport {
  /**
   * `handlers`: onOpen(), onClose(event), onMessage(parsedJson),
   * onBinary(arrayBuffer).
   */
  constructor(token, handlers) {
    this.token = token;
    this.handlers = handlers;
    this.socket = null;
    this.closed = false;
  }

  connect() {
    const socket = new WebSocket(`ws://${location.host}/neiro`, [TOKEN_SUBPROTOCOL_PREFIX + this.token]);
    socket.binaryType = 'arraybuffer';
    socket.onopen = () => this.handlers.onOpen?.();
    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) this.handlers.onBinary?.(event.data);
      else this.handlers.onMessage?.(JSON.parse(event.data));
    };
    socket.onclose = (event) => {
      this.socket = null;
      this.handlers.onClose?.(event);
      if (!this.closed) setTimeout(() => this.connect(), RECONNECT_MS);
    };
    this.socket = socket;
  }

  get open() { return this.socket?.readyState === WebSocket.OPEN; }

  /** Send one client message. False if there is no open socket. */
  send(message) {
    if (!this.open) return false;
    this.socket.send(JSON.stringify(message));
    return true;
  }

  close() {
    this.closed = true;
    this.socket?.close();
  }
}
