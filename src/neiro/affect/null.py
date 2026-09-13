"""The affect provider that hears nothing.

Ships in Stage 0 and stays the default until gate G3b passes. The
orchestrator cannot tell it apart from the real one, which is the point:
turning affect on is a config flip, not a code path.
"""

from __future__ import annotations

import numpy as np

from neiro.state import Locality, UserAffect


class NullAffectProvider:
    """protocols.AffectProvider. Always `UserAffect.NONE`."""

    locality = Locality.LOCAL_PINNED

    async def observe(self, pcm_window_16k: np.ndarray) -> UserAffect:
        return UserAffect.NONE

    def commit_utterance(self) -> bool:
        """No baseline to update."""
        return False

    def discard_utterance(self) -> None:
        """Nothing was staged."""
