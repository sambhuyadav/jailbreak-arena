"""
OpenEnv-native `Environment` subclass that wraps `JailbreakArena`.

This is what `HTTPEnvServer` instantiates for each WebSocket session — the
framework treats every `/ws` connection as its own arena episode. The HTTP
session-aware `/reset` and `/step` routes (in `server.py`) reuse the same
`JailbreakArena` instance via a process-shared singleton so existing clients
that pass `X-Session-Id` keep working unchanged.

Why a thin wrapper instead of porting `JailbreakArena` wholesale: the arena's
state machine, multi-turn rollouts, defender lifecycle, and reward shaping are
all already covered by the existing pytest suite. Wrapping preserves that
test surface while exposing the framework-native interfaces.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from openenv.core.env_server import Environment

from config import MAX_TURNS
from environment import JailbreakArena
from models import AttackAction
from openenv_models import JailbreakAction, JailbreakObservation, JailbreakState


_SHARED_ARENA: Optional[JailbreakArena] = None


def get_shared_arena() -> JailbreakArena:
    """Lazy singleton — keeps a single `Defender` (and its HTTP client) for the
    whole process. WebSocket sessions and the legacy HTTP routes both go
    through it, so metrics and idempotent stub state stay coherent."""
    global _SHARED_ARENA
    if _SHARED_ARENA is None:
        _SHARED_ARENA = JailbreakArena()
    return _SHARED_ARENA


class JailbreakerEnv(Environment[JailbreakAction, JailbreakObservation, JailbreakState]):
    """OpenEnv-native facade over a single arena episode.

    Each instance owns one session in the shared `JailbreakArena`. The
    framework calls `reset()` then `step()` repeatedly until `done=True` and
    finally `close()` to free the session.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        curriculum_level: Optional[int] = None,
        topic_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._arena = get_shared_arena()
        self._default_level = curriculum_level or int(os.getenv("CURRICULUM_LEVEL", "1"))
        self._default_topic_id = topic_id
        self._session_id: Optional[str] = None
        self._state = JailbreakState()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        curriculum_level: Optional[int] = None,
        **kwargs: Any,
    ) -> JailbreakObservation:
        # Free any previous session before starting a new episode so the
        # shared arena's session map doesn't leak across resets on the same
        # WebSocket connection.
        self.close()

        cl = curriculum_level or self._default_level
        tid = topic_id or self._default_topic_id
        observation, sid = self._arena.reset(
            session_id=episode_id,
            topic_id=tid,
            curriculum_level=cl,
        )
        self._session_id = sid
        self._state = JailbreakState(
            episode_id=sid,
            step_count=0,
            topic_id=observation.topic_id,
            curriculum_level=observation.curriculum_level,
            turn_count=0,
            done=False,
            total_reward=0.0,
            jailbreak_succeeded=False,
            episode_history=[],
        )
        return JailbreakObservation(
            done=False,
            reward=0.0,
            topic_id=observation.topic_id,
            topic_description=observation.topic_description,
            topic_category=observation.topic_category,
            curriculum_level=observation.curriculum_level,
            defender_context=observation.defender_context,
            turn_count=observation.turn_count,
            max_turns=observation.max_turns,
            last_attack_prompt=observation.last_attack_prompt,
            last_defender_response=observation.last_defender_response,
            last_detector_result=observation.last_detector_result,
            available_strategies=observation.available_strategies,
            episode_history=observation.episode_history,
            metadata={"session_id": sid},
        )

    def step(self, action: JailbreakAction, **kwargs: Any) -> JailbreakObservation:
        if self._session_id is None:
            # Auto-reset for callers that step without a prior reset (matches
            # the framework's stateless HTTP `/step` path semantics).
            self.reset()

        internal = AttackAction(
            strategy=action.strategy or "roleplay_injection",
            payload=action.payload or " ",
            raw_dsl=action.raw_dsl,
            turn_context=action.turn_context,
            is_legitimate=action.is_legitimate,
        )
        result = self._arena.step(session_id=self._session_id, action=internal)
        obs = result.observation

        self._state.step_count += 1
        self._state.turn_count = obs.turn_count
        self._state.done = result.done
        self._state.total_reward += result.reward.attacker_value
        self._state.episode_history = obs.episode_history
        self._state.jailbreak_succeeded = bool(result.info.get("jailbreak_succeeded"))

        return JailbreakObservation(
            done=result.done,
            reward=result.reward.attacker_value,
            topic_id=obs.topic_id,
            topic_description=obs.topic_description,
            topic_category=obs.topic_category,
            curriculum_level=obs.curriculum_level,
            defender_context=obs.defender_context,
            turn_count=obs.turn_count,
            max_turns=obs.max_turns,
            last_attack_prompt=obs.last_attack_prompt,
            last_defender_response=obs.last_defender_response,
            last_detector_result=obs.last_detector_result,
            available_strategies=obs.available_strategies,
            episode_history=obs.episode_history,
            metadata={
                "session_id": self._session_id,
                "defender_reward": result.reward.defender_value,
                "attacker_breakdown": result.reward.attacker_breakdown,
                "defender_breakdown": result.reward.defender_breakdown,
                "detector_result": result.reward.detector_result,
                "jailbreak_succeeded": bool(result.info.get("jailbreak_succeeded")),
            },
        )

    @property
    def state(self) -> JailbreakState:
        return self._state

    def close(self) -> None:
        if self._session_id is not None:
            self._arena.sessions.pop(self._session_id, None)
            self._session_id = None


# Re-export so callers can `from jailbreaker_env import *` cleanly.
__all__ = [
    "JailbreakerEnv",
    "JailbreakAction",
    "JailbreakObservation",
    "JailbreakState",
    "get_shared_arena",
    "MAX_TURNS",
]
