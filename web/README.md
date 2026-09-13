# web/

The face, and the audio clock.

## Why the browser owns playback

Python could play the audio itself. It must not. **Whoever owns the audio
clock owns the animation clock** — if the mouth is scheduled against a
different clock than the sound, every viseme drifts within a sentence.

So one `AudioContext` holds everything: chunks are appended to a running
`nextStartTime` rather than played on arrival (which is what makes
consecutive sentences sound continuous instead of gapped), and the
`played` callback fires from the context's own clock at the moment
sequence 0 actually begins. **That timestamp is the end of the one metric
this project measures.** Everything earlier is "we sent it".

## Why there is a drawn face and not only a VRM

`FallbackFace` is not a placeholder to be deleted. Without it, nothing
about expression blending or viseme timing can be *looked at* until an
avatar has been downloaded, licensed and rigged — and those are the parts
most likely to be wrong. It draws the same five expression weights and the
same five mouth shapes from the same numbers the VRM will use, so what you
see here is what the avatar will do.

Open `index.html` with no server and it runs a demo cycling the
expressions, so the face is inspectable on its own.

## The spelling that silently breaks everything

VRM **1.0** expression presets are `happy angry sad relaxed surprised`,
and mouth shapes are `aa ih ou ee oh`.

VRM **0.x** used `A I U E O`.

`expressionManager.setValue()` with the wrong spelling does nothing and
raises nothing. It is the commonest reason an avatar's mouth never opens.

## Where the blending happens

In Python (`src/neiro/emotion/blend.py`), not here — so the curves are
unit tested and the browser only renders. The small tail in `index.html`
is visual smoothing between `state` frames arriving at roughly 10 Hz,
nothing more.
