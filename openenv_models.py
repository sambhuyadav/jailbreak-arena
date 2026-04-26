"""
OpenEnv-typed Action / Observation / State for Jailbreak Arena.

These subclass the framework's base types (`openenv.core.env_server.Action`,
`Observation`, `State`) so the environment plugs into OpenEnv-native clients
(`EnvClient`, WebSocket `/ws`, `/schema`, Gradio `/web`) without bespoke
serialization on either side.

`models.AttackAction` / `AttackObservation` are kept as the internal Pydantic
types used by `JailbreakArena` and the existing FastAPI session layer; the
classes below are the framework-facing wrappers.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class JailbreakAction(Action):
    """Attacker action — strategy + payload, plus optional pre-formed DSL."""

    strategy: str = Field(default="", max_length=64)
    payload: str = Field(default="", max_length=8000)
    raw_dsl: Optional[str] = Field(default=None, max_length=10000)
    turn_context: Optional[str] = Field(default=None, max_length=4000)
    is_legitimate: bool = False


class JailbreakObservation(Observation):
    """Attacker observation — current topic, defender state, reward signal.

    Inherits `done`, `reward`, `metadata` from `Observation`.
    """

    topic_id: str = ""
    topic_description: str = ""
    topic_category: str = ""
    curriculum_level: int = 0
    defender_context: str = ""
    turn_count: int = 0
    max_turns: int = 0
    last_attack_prompt: Optional[str] = None
    last_defender_response: Optional[str] = None
    last_detector_result: Optional[str] = None
    available_strategies: List[str] = Field(default_factory=list)
    episode_history: List[Dict] = Field(default_factory=list)


class JailbreakState(State):
    """Per-session state. Inherits `episode_id`, `step_count` from `State`."""

    topic_id: str = ""
    curriculum_level: int = 0
    turn_count: int = 0
    done: bool = False
    total_reward: float = 0.0
    jailbreak_succeeded: bool = False
    episode_history: List[Dict] = Field(default_factory=list)
