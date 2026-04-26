# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Licensed under the BSD-style license. Vendored from meta-pytorch/OpenEnv
# (src/openenv/core/rubrics/{base,containers}.py, RFC 004).
#
# Trimmed to the subset Jailbreak Arena uses: sync forward, child registration,
# named introspection, RubricList. Async paths, Sequential, Gate, WeightedSum,
# RubricDict, LLMJudge, and trajectory rubrics live upstream — pull them when
# we need them rather than carrying dead code here.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple


class Rubric(ABC):
    """Abstract base for reward computation. Modeled on PyTorch nn.Module:
    subclasses implement `forward()`, child rubrics auto-register on attribute
    assignment so `named_rubrics()` can introspect each criterion's `last_score`.
    """

    _rubric_children: Dict[str, "Rubric"]
    last_score: Optional[float]

    def __init__(self) -> None:
        object.__setattr__(self, "_rubric_children", {})
        object.__setattr__(self, "last_score", None)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Rubric):
            self._rubric_children[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, action: Any, observation: Any) -> float:
        result = self.forward(action, observation)
        self.last_score = result
        return result

    @abstractmethod
    def forward(self, action: Any, observation: Any) -> float:
        raise NotImplementedError

    def children(self) -> Iterator["Rubric"]:
        yield from self._rubric_children.values()

    def named_children(self) -> Iterator[Tuple[str, "Rubric"]]:
        yield from self._rubric_children.items()

    def named_rubrics(self, prefix: str = "") -> Iterator[Tuple[str, "Rubric"]]:
        for name, child in self._rubric_children.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, child
            yield from child.named_rubrics(full_name)

    def get_rubric(self, path: str) -> "Rubric":
        current: "Rubric" = self
        for part in path.split("."):
            if part not in current._rubric_children:
                raise KeyError(f"Rubric path not found: {path}")
            current = current._rubric_children[part]
        return current

    def reset(self) -> None:
        self.last_score = None
        for child in self._rubric_children.values():
            child.reset()


class RubricList(Rubric):
    """Container for an ordered list of rubrics. Aggregation is the parent's
    job — `forward` raises so a parent must wrap this with summation, gating,
    or weighting logic.
    """

    def __init__(self, rubrics: Optional[List[Rubric]] = None) -> None:
        super().__init__()
        self._rubrics: List[Rubric] = []
        if rubrics:
            for r in rubrics:
                self.append(r)

    def forward(self, action: Any, observation: Any) -> float:
        raise NotImplementedError(
            "RubricList does not aggregate. Wrap it in a parent rubric."
        )

    def append(self, rubric: Rubric) -> None:
        index = len(self._rubrics)
        setattr(self, f"rubric_{index}", rubric)
        self._rubrics.append(rubric)

    def __len__(self) -> int:
        return len(self._rubrics)

    def __getitem__(self, index: int) -> Rubric:
        return self._rubrics[index]

    def __iter__(self) -> Iterator[Rubric]:
        return iter(self._rubrics)
