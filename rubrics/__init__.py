"""OpenEnv Rubric system for Jailbreak Arena.

Public surface:
    Rubric         — base class (vendored from meta-pytorch/OpenEnv RFC 004)
    RubricList     — ordered container
    AttackerObs    — observation dataclass for attacker rubrics
    DefenderObs    — observation dataclass for defender rubrics
    AttackerRubric — composed attacker reward (5 criteria)
    DefenderRubric — composed defender reward (5 criteria)

Use named_rubrics() on a composite to introspect each criterion's last_score.
"""
from rubrics._base import Rubric, RubricList
from rubrics.jailbreak import (
    AttackerObs,
    AttackerRubric,
    CorrectEngagementRubric,
    CorrectRefusalRubric,
    DefenderObs,
    DefenderRubric,
    FalsePositiveRubric,
    IdenticalPayloadRubric,
    JailbreakAllowedRubric,
    JailbreakOutcomeRubric,
    ParseFailureRubric,
    PartialEngagementRubric,
    TurnEfficiencyRubric,
)

__all__ = [
    "Rubric",
    "RubricList",
    "AttackerObs",
    "DefenderObs",
    "AttackerRubric",
    "DefenderRubric",
    "ParseFailureRubric",
    "JailbreakOutcomeRubric",
    "TurnEfficiencyRubric",
    "IdenticalPayloadRubric",
    "CorrectRefusalRubric",
    "PartialEngagementRubric",
    "JailbreakAllowedRubric",
    "CorrectEngagementRubric",
    "FalsePositiveRubric",
]
