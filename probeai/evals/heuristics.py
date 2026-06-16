"""Deterministic leading-question + stacked-question detectors.

Encodes the chosen leading-question definition (a DESIGN DECISION):
flag (a) PRESUPPOSITION (assuming the answer), (b) LOADED/suggestive phrasing,
and (c) DOUBLE-BARRELED (two asks in one). Neutral closed yes/no questions are
allowed. These rules are the cheap, fully-testable backstop; the isolated LLM
judge (judge.py) catches subtler presupposition the regexes miss.

Everything here is pure string logic -> directly unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# (a) LOADED / suggestive phrasing — the classic "you're nudging the answer" tells.
_LOADED_PATTERNS: list[tuple[str, str]] = [
    (r"\bdon'?t you (?:think|agree|feel|find)\b", "loaded: 'don't you think/agree'"),
    (r"\bwouldn'?t you (?:say|agree|think)\b", "loaded: 'wouldn't you agree'"),
    (r"\bdo you not (?:think|agree)\b", "loaded: 'do you not think'"),
    (r"\bisn'?t it\b", "loaded: 'isn't it'"),
    (r"\baren'?t you\b", "loaded: 'aren't you'"),
    (r"\bwasn'?t it\b", "loaded: 'wasn't it'"),
    (r"\bdidn'?t you\b", "loaded: 'didn't you'"),
    (r"\byou (?:must|surely) (?:have|felt|found)\b", "loaded: assumes experience"),
    (r"\b(?:surely|obviously|clearly)\b", "loaded: presumptive adverb"),
    (r"[a-z],?\s*(?:right|correct)\?\s*$", "loaded: tag question ('..., right?')"),
]

# (b) PRESUPPOSITION — asking *about* a reaction the participant never reported,
# which smuggles in the answer. "What frustrated you about X" assumes frustration.
_PRESUPPOSITION_PATTERNS: list[tuple[str, str]] = [
    (
        r"\bwhat (?:did you (?:like|love|hate|dislike|enjoy)|frustrated|annoyed|"
        r"bothered|confused|excited|impressed|disappointed)\b",
        "presupposition: assumes a specific reaction",
    ),
    (
        r"\bhow (?:frustrating|confusing|annoying|difficult|hard|easy|great|"
        r"good|bad|painful|smooth) (?:was|were|did)\b",
        "presupposition: adjective presumes the verdict",
    ),
    (r"\bwhy (?:was|were) it so\b", "presupposition: 'why was it so <X>'"),
]

# (c) DOUBLE-BARRELED — two questions stacked, or two asks joined by and/or.
_INTERROGATIVE = r"(?:what|why|how|when|where|who|which|did|do|does|are|is|was|were|can|could|would|will|have)"


@dataclass
class QuestionFlags:
    question: str
    leading_reasons: List[str] = field(default_factory=list)
    stacked: bool = False

    @property
    def is_leading(self) -> bool:
        return bool(self.leading_reasons)

    @property
    def is_clean(self) -> bool:
        return not self.is_leading and not self.stacked


def find_leading_violations(question: str) -> List[str]:
    """Return reasons this question is leading/loaded (empty == clean)."""
    q = question.lower()
    reasons: List[str] = []
    for pattern, label in _LOADED_PATTERNS + _PRESUPPOSITION_PATTERNS:
        if re.search(pattern, q):
            reasons.append(label)
    return reasons


def is_stacked(question: str) -> bool:
    """True if the question stacks two asks (double-barreled / multi-question)."""
    # More than one question mark => clearly multiple questions.
    if question.count("?") > 1:
        return True
    q = question.lower()
    # "<interrogative> ... and/or <interrogative> ...?" => two asks joined.
    joined = re.search(
        rf"\b{_INTERROGATIVE}\b.+\b(?:and|or)\b.+\b{_INTERROGATIVE}\b.*\?", q
    )
    return bool(joined)


def analyze_question(question: str) -> QuestionFlags:
    return QuestionFlags(
        question=question,
        leading_reasons=find_leading_violations(question),
        stacked=is_stacked(question),
    )
