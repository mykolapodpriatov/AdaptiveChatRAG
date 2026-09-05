"""Demoting documents that produced a wrong answer.

A thumbs-down used to add a correction document and leave the passages that
produced the wrong answer exactly where they were, at the top of the retrieval,
ready to produce it again. Whether the correction won next time came down to
whether it happened to embed closer to the query. That is not a self-learning
loop, it is a hope.

This is the arithmetic half: penalties in, re-ranked candidates out. No
database, no LangChain, no vector store, so the shape of the penalty can be
tested directly rather than inferred from what a model did.

Two properties the penalty has to have, and the reasons they are not optional:

* **It saturates.** Without a ceiling, one determined user can bury a correct
  document permanently. The ceiling is also deliberately well below what would
  exclude a document, because the UI cannot tell "this document is wrong" from
  "the answer was wrong even though the documents were right". Until it can,
  a thumbs-down is evidence for a demotion, not for a veto.
* **It decays.** Otherwise a document stays punished for a problem that was
  fixed months ago, and the store only ever gets more pessimistic.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = [
    "DEMOTION_CEILING",
    "OVERFETCH_FACTOR",
    "PENALTY_HALF_LIFE_DAYS",
    "PenaltyRecord",
    "demote",
    "penalty_for",
]

#: Most a document's score can be reduced, on the retriever's own score scale
#: (relevance in ``[0, 1]``). Deliberately a demotion rather than a veto: see
#: the module docstring.
DEMOTION_CEILING = float(os.getenv("DEMOTION_CEILING", "0.35"))

#: Days after which an unrepeated penalty has halved.
PENALTY_HALF_LIFE_DAYS = float(os.getenv("PENALTY_HALF_LIFE_DAYS", "30"))

#: How many candidates to pull per requested result before re-ranking.
#: Demoting is not excluding: a document pushed down has to have somewhere to
#: go, and a document that should rise has to have been fetched at all.
OVERFETCH_FACTOR = int(os.getenv("OVERFETCH_FACTOR", "3"))


@dataclass(frozen=True)
class PenaltyRecord:
    """What is known about one document's negative feedback."""

    document_id: str
    negative_count: int
    last_negative_at: datetime | None = None


def penalty_for(
    record: PenaltyRecord | None,
    now: datetime,
    *,
    ceiling: float = DEMOTION_CEILING,
    half_life_days: float = PENALTY_HALF_LIFE_DAYS,
) -> float:
    """How much to subtract from this document's score, in ``[0, ceiling]``.

    The count contributes on a saturating curve, ``1 - 2**-count``, so the first
    thumbs-down carries half the available penalty and the tenth carries almost
    nothing. Then the whole thing decays by half every ``half_life_days``.

    ``None``, a non-positive count, or a non-positive ceiling all yield 0.0.

    Raises:
        ValueError: If ``half_life_days`` is not positive. A zero half-life
            would divide by zero; a negative one would make a penalty grow with
            age, which is the opposite of the point.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    if record is None or record.negative_count <= 0 or ceiling <= 0:
        return 0.0

    magnitude = ceiling * (1.0 - 2.0 ** -float(record.negative_count))

    if record.last_negative_at is None:
        # No timestamp means the age is unknown. Applying no decay is the
        # cautious reading: it keeps a real penalty rather than discarding it
        # because a row predates the column.
        return magnitude

    age = now - record.last_negative_at
    if age <= timedelta(0):
        return magnitude
    return magnitude * 2.0 ** (-(age.total_seconds() / 86400.0) / half_life_days)


def demote(
    candidates: Sequence[tuple[str, float]],
    penalties: Mapping[str, PenaltyRecord],
    *,
    k: int,
    now: datetime,
    ceiling: float = DEMOTION_CEILING,
    half_life_days: float = PENALTY_HALF_LIFE_DAYS,
) -> list[tuple[str, float]]:
    """Re-rank ``(document_id, score)`` candidates and return the best ``k``.

    Order is by adjusted score, ties broken by the original position so the
    retriever's own ranking decides when the penalties do not, and two runs over
    the same input agree.

    Args:
        candidates: Best first, as the retriever returned them.
        penalties: Known negative feedback, keyed by document id.
        k: How many to return. ``<= 0`` returns nothing.
        now: The clock, injected so the decay is testable.
        ceiling: Maximum demotion.
        half_life_days: Decay half-life.

    Returns:
        The best ``k`` as ``(document_id, adjusted_score)``.
    """
    if k <= 0:
        return []
    adjusted = [
        (
            position,
            document_id,
            score
            - penalty_for(
                penalties.get(document_id),
                now,
                ceiling=ceiling,
                half_life_days=half_life_days,
            ),
        )
        for position, (document_id, score) in enumerate(candidates)
    ]
    adjusted.sort(key=lambda row: (-row[2], row[0]))
    return [(document_id, score) for _position, document_id, score in adjusted[:k]]


def overfetch_size(k: int, factor: int = OVERFETCH_FACTOR) -> int:
    """How many candidates to ask the retriever for, given a requested ``k``."""
    return max(k, k * max(factor, 1))


def penalties_by_id(records: Iterable[PenaltyRecord]) -> dict[str, PenaltyRecord]:
    """Index penalty records by document id."""
    return {record.document_id: record for record in records}
