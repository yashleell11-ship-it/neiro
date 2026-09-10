# Working on Neiro

This is currently a solo project, built by directing Claude Code through
`docs/superpowers/specs/2026-09-10-lilt-design.md`'s plan, task by task.
The workflow rule below applies regardless of who or what is committing.

## Commit and push discipline

**Commit every small, working step as its own commit. `git push` immediately
after every single commit — not once per sitting, not batched at the end
of a session.**

- One commit per task or sub-step that leaves the tree in a working
  state: a scaffold, a contract file, a passing test, a gate verdict
  recorded in `docs/DECISIONS.md`. Never fold an evening's work into one
  commit.
- Push right after committing, every time. A commit that only exists
  locally does nothing — only commits pushed to the **default branch**
  (`master`) show up anywhere that matters, including the GitHub
  contribution graph, which is part of why this repo is public.
- Remote: `origin` → [github.com/yashleell11-ship-it/neiro](https://github.com/yashleell11-ship-it/neiro).
- Commit messages: plain, present tense, describe what actually changed
  and what was verified. No AI attribution of any kind (no
  `Co-Authored-By`, no "Generated with" line, no `@anthropic.com`
  author) — this is a standing rule, not a per-session choice.
- Every task should end with something observable — a test that passes,
  a command that prints a real result, a recording you can listen to —
  and, where the result can't be verified mechanically (audio you have
  to listen to, a UI you have to look at), say so plainly rather than
  claiming it's done.

See `CLAUDE.md` for the full set of project-specific rules an agent
working in this repo must follow (Hyprland's Lua config API, the
typed-tool-registry safety rule, the emotion-state invariants, licensing
constraints, and more).
