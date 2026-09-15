"""The one tool that reaches the open internet, not just Yash's own
two machines.

Every other tool, including `apps.open_app("pc", ...)`, stays inside
the pair of machines Yash owns — the same Tailscale link Tier already
uses for the LLM/STT provider (state.py). A web search leaves that
boundary entirely, which is why it exists as its own module with its
own docstring rather than folding into apps.py: it is the one place
this "fully local" project deliberately opts out, and that decision
should be legible from the file, not buried in a shared one.

See docs/DECISIONS.md (2026-09-15, "web_search opts out of local-only")
for the full reasoning. Short version: RED tier already carves out
"sending anything off-machine the user didn't name" as the thing to
forbid — a search query Yash spoke this turn is the opposite of that,
so YELLOW (confirm-gated, like everything else with a real effect)
fits without changing what RED means.

No API key: DuckDuckGo's HTML endpoint (`html.duckduckgo.com`) returns
plain search-result markup with no auth, which is what makes this
buildable at all without a "the assistant now needs a provisioned key"
step folded into gate G-anything. Parsed with a plain regex rather than
adding a new HTML-parsing dependency for five lines of extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

SEARCH_URL = "https://html.duckduckgo.com/html/"
REQUEST_TIMEOUT_S = 8.0
MAX_RESULTS = 5

# DuckDuckGo's lite HTML wraps each result title in a `result__a` link
# and a `result__snippet` span. This is scraping a page, not an API —
# it breaks if DuckDuckGo changes their markup, and `search()` raises
# rather than returning something wrong if the pattern stops matching.
_RESULT_RE = re.compile(
    r'class="result__a"[^>]*>(?P<title>.*?)</a>.*?'
    r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _TAG_RE.sub("", html).strip()


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str


class SearchUnavailable(RuntimeError):
    """The request failed, timed out, or the page didn't parse. Spoken
    to Yash as "couldn't search right now", not a traceback."""


def search(query: str) -> list[SearchResult]:
    """A handful of result snippets for `query`. Raises rather than
    returning an empty list on failure, so a network hiccup is spoken
    as "couldn't search" rather than silently "found nothing"."""
    try:
        response = httpx.post(
            SEARCH_URL,
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Neiro voice assistant)"},
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SearchUnavailable(f"the search request failed: {exc}") from exc

    results = [
        SearchResult(title=_strip_tags(m["title"]), snippet=_strip_tags(m["snippet"]))
        for m in _RESULT_RE.finditer(response.text)
    ]
    return results[:MAX_RESULTS]


def search_and_describe(query: str) -> str:
    """What Neiro says about a search — the tool handler's return
    value, which is what reaches the model as the tool result."""
    results = search(query)
    if not results:
        return f"No results for {query!r}."
    lines = [f"{r.title} — {r.snippet}" for r in results if r.title]
    return "\n".join(lines) or f"No results for {query!r}."
