from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List


class AttackAction(BaseModel):
    strategy: str = Field(..., min_length=1, max_length=64)
    payload: str = Field(..., min_length=1, max_length=8000)
    raw_dsl: Optional[str] = Field(default=None, max_length=10000)
    turn_context: Optional[str] = Field(default=None, max_length=4000)
    # When True, the env sends `payload` to the defender verbatim (no DSL / no
    # strategy template) and grades the defender against the legitimate-prompt
    # reward path. Used to train the defender's false-positive rate.
    is_legitimate: bool = False

    @field_validator("strategy")
    @classmethod
    def _strategy_is_identifier(cls, v: str) -> str:
        # Strategies are pinned identifiers (snake_case). Reject random text so
        # callers fail loudly instead of silently falling back inside the env.
        if not v.replace("_", "").isalnum():
            raise ValueError(f"strategy must be snake_case identifier, got {v!r}")
        return v


class AttackObservation(BaseModel):
    topic_id: str
    topic_description: str
    topic_category: str
    curriculum_level: int
    defender_context: str
    turn_count: int
    max_turns: int
    done: bool
    last_attack_prompt: Optional[str] = None
    last_defender_response: Optional[str] = None
    last_detector_result: Optional[str] = None
    last_reward: Optional[float] = None
    available_strategies: List[str] = Field(default_factory=list)
    episode_history: List[Dict] = Field(default_factory=list)


class ArenaRewardModel(BaseModel):
    attacker_value: float
    defender_value: float
    attacker_breakdown: Dict[str, float] = Field(default_factory=dict)
    defender_breakdown: Dict[str, float] = Field(default_factory=dict)
    attacker_message: str = ""
    defender_message: str = ""
    detector_result: str = ""


class StepRequest(BaseModel):
    action: AttackAction


class StepResponse(BaseModel):
    observation: AttackObservation
    reward: ArenaRewardModel
    done: bool
    info: Dict = Field(default_factory=dict)


class ResetRequest(BaseModel):
    topic_id: Optional[str] = Field(default=None, max_length=128)
    curriculum_level: int = Field(default=1, ge=1)


class SessionState(BaseModel):
    session_id: str
    topic_id: str
    turn_count: int
    done: bool
    total_reward: float
    episode_history: List[Dict] = Field(default_factory=list)
    jailbreak_succeeded: bool = False
