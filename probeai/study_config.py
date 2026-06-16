"""Load + validate the study (discussion guide) and the moderation policy.

In Great Question's model the researcher authors a *discussion guide*: a research
goal plus prioritized *objectives* with seed questions. The moderator treats it as
a framework, not a script. This module just loads and validates that config — no
LLM, no I/O beyond reading YAML — so it is trivially testable.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, field_validator

# Repo-relative config locations (overridable by callers / env).
CONFIG_DIR = Path(__file__).parent / "config"
DEFAULT_STUDY_PATH = CONFIG_DIR / "study_checkout.yaml"
DEFAULT_POLICY_PATH = CONFIG_DIR / "policy.yaml"


class Priority(str, Enum):
    """Objective priority — drives probe depth and skip behavior."""

    high = "high"
    medium = "medium"
    low = "low"


class Objective(BaseModel):
    """One thing the study must learn."""

    id: str
    priority: Priority
    label: str
    description: str
    seed_questions: List[str] = Field(default_factory=list)

    @field_validator("seed_questions")
    @classmethod
    def _need_a_seed(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("each objective needs at least one seed question")
        return v


class Study(BaseModel):
    """A complete discussion guide."""

    study_id: str
    title: str
    research_goal: str
    intro: str = ""
    outro: str = ""
    objectives: List[Objective]

    @field_validator("objectives")
    @classmethod
    def _need_objectives_with_unique_ids(cls, v: List[Objective]) -> List[Objective]:
        if not v:
            raise ValueError("a study needs at least one objective")
        ids = [o.id for o in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"objective ids must be unique, got {ids}")
        return v

    def objective(self, objective_id: str) -> Objective:
        for o in self.objectives:
            if o.id == objective_id:
                return o
        raise KeyError(objective_id)


class ProbeTrigger(BaseModel):
    """How we decide an answer needs a follow-up.

    Specificity dominates: the LLM's verdict is primary. ``short_answer_word_count``
    is only a weak secondary nudge — never a standalone reason to probe.
    """

    specificity_is_primary: bool = True
    short_answer_word_count: int = 6


class Policy(BaseModel):
    """Tunable moderation thresholds (see config/policy.yaml)."""

    turn_budget_per_objective: float = 3.0
    turn_budget_floor: int = 8
    probe_caps: dict[Priority, int]
    low_priority_skip_threshold: int = 4
    probe_trigger: ProbeTrigger = Field(default_factory=ProbeTrigger)
    require_all_objectives_to_close: bool = False

    def turn_budget(self, num_objectives: int) -> int:
        """Total participant turns allowed before a forced wrap-up."""
        scaled = math.ceil(self.turn_budget_per_objective * num_objectives)
        return max(scaled, self.turn_budget_floor)

    def probe_cap(self, priority: Priority) -> int:
        """Max follow-ups allowed for an objective of this priority."""
        return self.probe_caps[priority]


def load_study(path: str | Path | None = None) -> Study:
    """Load + validate a study config. Defaults to the shipped sample study."""
    p = Path(path) if path else DEFAULT_STUDY_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Study.model_validate(data)


def load_policy(path: str | Path | None = None) -> Policy:
    """Load + validate the moderation policy. Defaults to config/policy.yaml."""
    p = Path(path) if path else DEFAULT_POLICY_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Policy.model_validate(data)
