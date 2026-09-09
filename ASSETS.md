# Asset licences

Non-code assets in this repository (avatars, audio, images) are **not**
covered by the project's Apache-2.0 `LICENSE` — code and assets have
different rules.

## Avatar

- **Default avatar** (ships in this repo): a CC0 model, to be sourced from
  opensourceavatars.com or a similar CC0 registry. Fill in the source URL
  and CC0 confirmation here once T13/Stage 1 picks one.
- **Original "Neiro" avatar** (optional, separate download): if authored in
  VRoid Studio, its base body mesh, clothing meshes, and preset items are
  pixiv content under the VRoid Studio Guidelines
  (https://vroid.com/en/studio/guidelines) and are explicitly **not CC0**.
  pixiv's terms permit selling and redistributing the exported model but do
  not transfer ownership of the base content. If this avatar is ever
  shipped, it is distributed separately from this repo (not in git), with
  its `VRMC_vrm.meta` fields filled in (avatarPermission, commercialUsage,
  allowRedistribution, modification, licenseUrl) rather than left at
  VRoid's defaults.

## Voice recordings

Yash's own recorded utterances (WER and emotion evaluation sets) never
enter this repository. `data/voice/` is gitignored; only derived,
non-biometric artifacts are committed — reference transcripts, self-labels,
prosody feature vectors, and a sha256 per file. See `docs/PRIVACY.md`.
