/**
 * The face: VRM when there is one, a drawn fallback when there is not.
 *
 * The fallback is not a placeholder to be deleted. Without it nothing
 * about expression blending or viseme timing can be looked at until an
 * avatar has been downloaded, licensed and rigged — and those are the
 * parts most likely to be wrong. It draws the same five expression
 * weights and the same five mouth shapes the VRM does, from the same
 * numbers, so what you see here is what the avatar will do.
 *
 * VRM 1.0 expression presets are `happy angry sad relaxed surprised`
 * and mouth shapes are `aa ih ou ee oh`. VRM 0.x used `A I U E O`.
 * Using the wrong spelling makes setValue() silently do nothing, which
 * is the commonest reason an avatar's mouth never opens.
 */

export const EXPRESSIONS = ['happy', 'angry', 'sad', 'relaxed', 'surprised'];
export const VISEMES = ['aa', 'ih', 'ou', 'ee', 'oh'];

export class FallbackFace {
  constructor(canvas) {
    this.canvas = canvas;
    this.g = canvas.getContext('2d');
    this.weights = Object.fromEntries(EXPRESSIONS.map((e) => [e, 0]));
    this.mouth = { shape: '', amount: 0 };
    this.blink = 0;
    this.available = new Set([...EXPRESSIONS, 'neutral']);
  }

  setExpression(weights) { Object.assign(this.weights, weights); }
  setMouth(shape, amount) { this.mouth = { shape, amount }; }

  draw(t) {
    const { g, canvas: c } = this;
    const w = c.width, h = c.height;
    g.clearRect(0, 0, w, h);

    const { happy = 0, angry = 0, sad = 0, relaxed = 0, surprised = 0 } = this.weights;
    const cx = w / 2, cy = h / 2;
    const unit = Math.min(w, h) / 6;

    // Auto-blink: 2-6 s, and the single cheapest thing that stops a face
    // reading as dead.
    const phase = (t % 4.2) / 4.2;
    this.blink = phase > 0.97 ? 1 : 0;

    g.strokeStyle = '#e8e4df';
    g.fillStyle = '#e8e4df';
    g.lineWidth = Math.max(2, unit * 0.12);
    g.lineCap = 'round';

    // Eyes. Surprise widens them, sadness and relaxation narrow them.
    const openness = this.blink ? 0.05 : Math.max(0.12, 1 + surprised * 0.7 - sad * 0.35 - relaxed * 0.4);
    for (const side of [-1, 1]) {
      const ex = cx + side * unit * 1.1;
      const ey = cy - unit * 0.5;
      g.beginPath();
      g.ellipse(ex, ey, unit * 0.34, unit * 0.34 * openness, 0, 0, Math.PI * 2);
      g.fill();
    }

    // Brows carry most of what reads as emotion.
    for (const side of [-1, 1]) {
      const bx = cx + side * unit * 1.1;
      const by = cy - unit * 1.35 - surprised * unit * 0.3 + sad * unit * 0.1;
      const inner = angry * unit * 0.45 - sad * unit * 0.4;
      g.beginPath();
      g.moveTo(bx - side * unit * 0.42, by - inner * 0.3);
      g.lineTo(bx + side * unit * 0.42, by + inner);
      g.stroke();
    }

    // Mouth: viseme shape when speaking, expression curve when not.
    const my = cy + unit * 1.15;
    const open = this.mouth.amount;
    const shapeWidth = { aa: 1.0, ih: 1.25, ou: 0.55, ee: 1.35, oh: 0.75, '': 1.0 }[this.mouth.shape] ?? 1.0;
    const shapeHeight = { aa: 1.0, ih: 0.35, ou: 0.85, ee: 0.3, oh: 1.1, '': 0 }[this.mouth.shape] ?? 0;

    g.beginPath();
    if (open > 0.02) {
      g.ellipse(cx, my, unit * 0.5 * shapeWidth, unit * 0.42 * shapeHeight * open, 0, 0, Math.PI * 2);
      g.fill();
    } else {
      const curve = (happy - sad - angry * 0.4 + relaxed * 0.4) * unit * 0.45;
      g.moveTo(cx - unit * 0.55, my);
      g.quadraticCurveTo(cx, my + curve, cx + unit * 0.55, my);
      g.stroke();
    }
  }
}

/**
 * Schedules mouth shapes against the AUDIO clock, not a timer.
 *
 * A viseme timeline is relative to the start of its chunk, and the chunk
 * starts at a time the AudioContext chose. Driving the mouth from
 * setTimeout would drift against the sound within a sentence.
 */
export class VisemeScheduler {
  constructor() { this.events = []; }

  /** `timeline` is [[startSeconds, viseme, durationSeconds], …]. */
  schedule(timeline, startAtCtxTime) {
    if (!timeline) return;
    for (const [offset, viseme, duration] of timeline) {
      this.events.push({ at: startAtCtxTime + offset, until: startAtCtxTime + offset + duration, viseme });
    }
    this.events.sort((a, b) => a.at - b.at);
  }

  clear() { this.events = []; }

  /** The shape and openness for this instant. */
  at(ctxTime) {
    while (this.events.length && this.events[0].until < ctxTime) this.events.shift();
    const active = this.events.find((e) => e.at <= ctxTime && ctxTime < e.until);
    if (!active || !active.viseme) return { shape: '', amount: 0 };
    // Ease in and out across the phoneme so the jaw does not snap.
    const span = active.until - active.at;
    const p = span > 0 ? (ctxTime - active.at) / span : 0;
    return { shape: active.viseme, amount: Math.sin(Math.PI * Math.min(1, Math.max(0, p))) };
  }
}
