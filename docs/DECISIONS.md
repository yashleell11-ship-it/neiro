# Decisions

One dated line per gate verdict or one-way-door choice. Format:
`YYYY-MM-DD — <what> — <verdict/number> — <what happens as a result>`.

- 2026-09-10 — Character name — **Neiro** (音色, "timbre") — repo, package, and console command all named after it.
- 2026-09-10 — Repo licence — **Apache-2.0** — matches every model licence in the plan; patent grant matters for a tool registry that executes local actions.
- 2026-09-10 — Framework — **from scratch, asyncio** — the emotion state (`Turn.user_affect`/`neiro_state`) has nowhere clean to live inside pipecat's frame taxonomy or LiveKit's `AgentSession`; both were also disqualified outright (pipecat's interruption API churned across 1.0.0; LiveKit needs a server + API keys, contradicting "no API keys and no network").
- 2026-09-10 — Audio device profiles — **earbuds primary, speakers best-effort** — default audio on this machine is Bluetooth (Mivi SuperPods), built-ins are muted; see Task 0.
