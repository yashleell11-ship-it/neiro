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
 */

export class AudioSink {
  constructor(sampleRate = 24000) {
    this.ctx = new AudioContext({ sampleRate, latencyHint: 'interactive' });
    this.sampleRate = sampleRate;
    this.nextStartTime = 0;
    this.sources = [];
    this.audioId = 0;
    this.onstarted = null;   // (seq, ctxTime) => void
    this.onended = null;
    this.onunderrun = null;
  }

  /** The browser blocks audio until a gesture. Call from a click. */
  async unlock() {
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    return this.ctx.state;
  }

  beginUtterance(audioId) {
    this.audioId = audioId;
    // A small lead so the first chunk is scheduled rather than racing
    // the clock. Too large and it is dead latency; 20 ms is under one
    // PipeWire quantum at 48 kHz.
    this.nextStartTime = this.ctx.currentTime + 0.02;
  }

  /**
   * Queue one chunk. `audioId` lets a late chunk from a cancelled reply
   * be dropped — otherwise it arrives after the next turn has started
   * and she talks over herself.
   */
  play(audioId, pcm, seq) {
    if (audioId !== this.audioId) return false;

    const buffer = this.ctx.createBuffer(1, pcm.length, this.sampleRate);
    buffer.copyToChannel(pcm, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);

    // If the queue ran dry the clock has moved past our cursor; starting
    // in the past silently truncates, so catch up and say so.
    if (this.nextStartTime < this.ctx.currentTime) {
      if (seq > 0 && this.onunderrun) this.onunderrun(seq);
      this.nextStartTime = this.ctx.currentTime + 0.01;
    }

    const startAt = this.nextStartTime;
    source.start(startAt);
    this.nextStartTime = startAt + buffer.duration;
    this.sources.push(source);

    if (this.onstarted) {
      // setTimeout would report when the TIMER fired, not when the
      // sound did. This measures the audio clock.
      const delayMs = Math.max(0, (startAt - this.ctx.currentTime) * 1000);
      setTimeout(() => this.onstarted(seq, startAt), delayMs);
    }
    source.onended = () => {
      this.sources = this.sources.filter((s) => s !== source);
      if (!this.sources.length && this.onended) this.onended(seq);
    };
    return true;
  }

  /** Barge-in: stop now and drop everything queued. */
  cancel() {
    for (const source of this.sources) {
      try { source.stop(); } catch { /* already finished */ }
    }
    this.sources = [];
    this.audioId = -1;                       // reject late chunks
    this.nextStartTime = this.ctx.currentTime;
  }

  /** Seconds of audio still queued — the HUD's "is she behind" number. */
  get queuedSeconds() {
    return Math.max(0, this.nextStartTime - this.ctx.currentTime);
  }
}
