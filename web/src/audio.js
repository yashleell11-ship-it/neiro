/**
 * Audio playback, and the only honest "first audio" timestamp.
 *
 * Python could play this audio itself. It must not: whoever owns the
 * audio clock owns the animation clock, and if the mouth is scheduled
 * against a different clock than the sound, every viseme drifts.
 *
 * So one AudioContext holds everything. Chunks are appended to a running
 * `nextStartTime` rather than played on arrival, which is what makes
 * consecutive sentences sound continuous instead of gapped. And the
 * `onstarted` callback fires from the context's own clock at the moment
 * sequence 0 actually begins — that timestamp is the end of the one
 * metric this project measures. Everything earlier is "we sent it".
 *
 * A suspended context is the one way that report could lie. The browser
 * refuses to start audio until the page has been clicked, and while the
 * context is suspended its clock stands still: a chunk "scheduled" then
 * makes no sound until the click. So a start report is held until the
 * context is actually running, and then carries the time it really
 * started — a slow number is a true number; an early one is a bug.
 */

// A small lead so the first chunk is scheduled rather than racing the
// clock. Too large and it is dead latency; 20 ms is under one PipeWire
// quantum at 48 kHz.
const SCHEDULE_LEAD_S = 0.02;
// After an underrun the cursor is put just ahead of "now" again. Smaller
// than the lead above: the queue is already late, add as little as
// possible.
const CATCH_UP_S = 0.01;

export class AudioSink {
  constructor(sampleRate) {
    this.ctx = new AudioContext({ sampleRate, latencyHint: 'interactive' });
    this.sampleRate = sampleRate;
    this.nextStartTime = 0;
    this.sources = [];
    this.audioId = 0;
    this.onstarted = null;   // (seq, ctxTime, audioId) => void
    this.onended = null;     // (seq) => void, when nothing is left playing
    this.onunderrun = null;  // (seq) => void
    this.onstate = null;     // (ctx.state) => void
    this._held = [];         // start reports waiting for the context to run
    this.ctx.addEventListener('statechange', () => {
      if (this.ctx.state === 'running') this._release();
      this.onstate?.(this.ctx.state);
    });
  }

  /** The browser blocks audio until a gesture. Call from a click. */
  async unlock() {
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    return this.ctx.state;
  }

  get running() { return this.ctx.state === 'running'; }

  beginUtterance(audioId) {
    this.audioId = audioId;
    this.nextStartTime = this.ctx.currentTime + SCHEDULE_LEAD_S;
  }

  /**
   * Queue one chunk. Returns the context time it will start at, or null
   * when it was refused: `audioId` lets a late chunk from a cancelled
   * reply be dropped — otherwise it arrives after the next turn has
   * started and she talks over herself.
   */
  play(audioId, pcm, seq) {
    if (audioId !== this.audioId) return null;

    const buffer = this.ctx.createBuffer(1, pcm.length, this.sampleRate);
    buffer.copyToChannel(pcm, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);

    // If the queue ran dry the clock has moved past our cursor; starting
    // in the past silently truncates, so catch up and say so.
    if (this.nextStartTime < this.ctx.currentTime) {
      if (seq > 0) this.onunderrun?.(seq);
      this.nextStartTime = this.ctx.currentTime + CATCH_UP_S;
    }

    const startAt = this.nextStartTime;
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;
    this.sources.push(source);

    // setTimeout would report when the TIMER fired, not when the sound
    // did; the value reported is the audio clock's, the timer only
    // decides when to send it.
    const delayMs = Math.max(0, (startAt - this.ctx.currentTime) * 1000);
    setTimeout(() => this._started(seq, startAt, audioId), delayMs);

    source.onended = () => {
      this.sources = this.sources.filter((s) => s !== source);
      if (!this.sources.length) this.onended?.(seq);
    };
    return startAt;
  }

  _started(seq, startAt, audioId) {
    if (this.running) this.onstarted?.(seq, startAt, audioId);
    else this._held.push({ seq, audioId });
  }

  _release() {
    // Everything held started the moment the context resumed: sources
    // scheduled for a time that has passed begin immediately.
    const held = this._held;
    this._held = [];
    for (const { seq, audioId } of held) {
      if (audioId === this.audioId) this.onstarted?.(seq, this.ctx.currentTime, audioId);
    }
  }

  /** Barge-in: stop now and drop everything queued. */
  cancel() {
    for (const source of this.sources) {
      try { source.stop(); } catch { /* already finished */ }
    }
    this.sources = [];
    this._held = [];
    this.audioId = -1;                       // reject late chunks
    this.nextStartTime = this.ctx.currentTime;
  }

  /** Seconds of audio still queued — the HUD's "is she behind" number. */
  get queuedSeconds() {
    return Math.max(0, this.nextStartTime - this.ctx.currentTime);
  }
}
