# web/

The face, and the audio clock. Served by `elizabeth talk --browser` at
http://127.0.0.1:8760 — the page is `index.html`, the daemon injects the
session token into it, and everything else here is fetched relative to it.

## Why the browser owns playback

Python could play the audio itself. It must not. **Whoever owns the audio
clock owns the animation clock** — if the mouth is scheduled against a
different clock than the sound, every viseme drifts within a sentence.

So one `AudioContext` holds everything: chunks are appended to a running
`nextStartTime` rather than played on arrival (which is what makes
consecutive sentences sound continuous instead of gapped), and the
`played` message fires from the context's own clock at the moment
sequence 0 actually begins. **That timestamp is the end of the one metric
this project measures.** Everything earlier is "we sent it".

The browser will not make a sound until the page has been clicked. Until
then the context is suspended, its clock does not move, and `played` is
held back rather than sent early — the page shows a button saying so.

## What the daemon sends, and what the page does with it

`src/elizabeth/server.py` freezes the message set; `src/elizabeth/audio/sink_ws.py`
sends it. Per reply:

| message | the page |
|---|---|
| `hello{samplerate, face, avatar}` | makes the `AudioContext` at that rate, the expression blender with those time constants, loads `avatar` if there is one, then answers `ready{expressions}` with what the face can actually do |
| `utt.begin{audio_id, emotion, intensity, blend_ms, weights}` | starts an utterance on the audio clock; the blender's target becomes `weights` |
| `utt.chunk{seq, audio_id, dur_ms, text, visemes}` + a binary frame | schedules the PCM after the previous chunk and the visemes against that start time; reports `played{seq, at, audio_id}` when it begins |
| `emotion{…}` | a new blender target, mid-reply |
| `utt.end` | phase back to idle once the queue has drained |
| `cancel{audio_id}` | stops every queued source now; late frames with that id are dropped |

A chunk's header and its binary frame are one entry in the daemon's
outgoing queue, sent back to back. The queue is bounded and a tab that
stops draining it loses whole chunks (logged on the daemon side), never
half of one — so the page's rule that each binary frame belongs to the
header before it always holds, and a frame arriving with no header
pending is refused and reported as `error` rather than played as the
wrong chunk.

## Why there is a drawn face and not only a VRM

`FallbackFace` is not a placeholder to be deleted. Without it, nothing
about expression blending or viseme timing can be *looked at* until an
avatar has been downloaded, licensed and rigged — and those are the parts
most likely to be wrong. It draws the same five expression weights and the
same five mouth shapes from the same numbers the VRM will use, so what you
see here is what the avatar will do.

Served with no token (any static server on this directory, e.g.
`bun x serve web`) it runs a demo cycling the expressions, so the face is
inspectable on its own.

## The avatar

Drop a `.vrm` under `web/public/avatar/` (gitignored; it is a separate
download with its own terms — see `ASSETS.md`) and the next connection
loads it. There is no bundled avatar and `modelspec.py` names none: nothing
is invented, the drawn face stays until one is chosen.

## The libraries, and why they are vendored

`three` and `@pixiv/three-vrm` (both MIT) are loaded through the import map
in `index.html` from `vendor/` — **never from a CDN**. The demo has to work
with Wi-Fi off, and the network this project is built on blocks GitHub
release assets (CLAUDE.md rule 3), so the files come from npm, once:

```sh
cd web
bun add three@0.180.0 @pixiv/three-vrm@3.5.5
cp node_modules/three/build/three.module.min.js  vendor/
cp node_modules/three/build/three.core.min.js    vendor/      # three.module imports it
cp node_modules/three/examples/jsm/loaders/GLTFLoader.js      vendor/addons/loaders/
cp node_modules/three/examples/jsm/utils/BufferGeometryUtils.js vendor/addons/utils/   # GLTFLoader imports it
cp node_modules/@pixiv/three-vrm/lib/three-vrm.module.min.js vendor/
```

About 1 MB in total, committed, with both licences beside them
(`vendor/LICENSE.three`, `vendor/LICENSE.three-vrm`). `node_modules/` is
not. Nothing imports any of it unless `hello` names an avatar, so the
drawn face works with `vendor/` deleted.

## The spelling that silently breaks everything

VRM **1.0** expression presets are `happy angry sad relaxed surprised`,
and mouth shapes are `aa ih ou ee oh`.

VRM **0.x** used `A I U E O`.

`expressionManager.setValue()` with the wrong spelling does nothing and
raises nothing. It is the commonest reason an avatar's mouth never opens.
three-vrm maps a 0.x model's names onto the 1.0 ones on load, so the page
only ever speaks 1.0.

## Where the blending happens

The *targets* are decided in Python (`src/elizabeth/emotion/blend.py`): one
preset at the tag's intensity, redistributed to what the avatar reported
it has. The *easing* toward them runs here, on the frame clock, with the
same curve — asymmetric time constants, surprise releasing itself, the
total clamped, tiny weights snapped to zero — and every one of those
numbers arrives in `hello.face`, straight from `config.py`. The page
carries no tuning of its own.
