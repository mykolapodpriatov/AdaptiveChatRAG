"""Tests for the feedback demotion arithmetic.

No database, no LangChain, no vector store: penalties in, ranking out, so the
shape of the penalty is asserted directly rather than inferred from what a model
happened to do.
"""

from datetime import datetime, timedelta

import pytest

from ranking import (
    DEMOTION_CEILING,
    PenaltyRecord,
    demote,
    overfetch_size,
    penalties_by_id,
    penalty_for,
)

NOW = datetime(2026, 6, 1, 12, 0)


def _record(count: int, days_ago: float = 0.0, document_id: str = "d") -> PenaltyRecord:
    return PenaltyRecord(
        document_id=document_id,
        negative_count=count,
        last_negative_at=NOW - timedelta(days=days_ago),
    )


# ---------------------------------------------------------------------------
# penalty_for
# ---------------------------------------------------------------------------


def test_no_feedback_means_no_penalty():
    assert penalty_for(None, NOW) == 0.0
    assert penalty_for(_record(0), NOW) == 0.0


def test_the_first_thumbs_down_carries_half_the_available_penalty():
    assert penalty_for(_record(1), NOW, ceiling=0.4) == pytest.approx(0.2)


def test_the_penalty_saturates():
    # Without a ceiling, one determined user could bury a correct document.
    ten = penalty_for(_record(10), NOW, ceiling=0.4)
    hundred = penalty_for(_record(100), NOW, ceiling=0.4)

    assert ten <= 0.4
    assert hundred <= 0.4
    assert hundred - ten < 0.001


def test_the_penalty_never_reaches_the_ceiling():
    # A demotion, not a veto: the UI cannot tell "this document is wrong" from
    # "the answer was wrong even though the documents were right".
    assert penalty_for(_record(50), NOW, ceiling=0.4) < 0.4


def test_the_penalty_halves_over_a_half_life():
    fresh = penalty_for(_record(3, days_ago=0), NOW, half_life_days=30)
    aged = penalty_for(_record(3, days_ago=30), NOW, half_life_days=30)

    assert aged == pytest.approx(fresh / 2)


def test_an_old_penalty_decays_towards_nothing():
    ancient = penalty_for(_record(3, days_ago=365), NOW, half_life_days=30)

    assert ancient < 0.001


def test_a_missing_timestamp_keeps_the_penalty_rather_than_discarding_it():
    # A row that predates the column is still evidence.
    record = PenaltyRecord(document_id="d", negative_count=2, last_negative_at=None)

    assert penalty_for(record, NOW) > 0.0


def test_a_future_timestamp_does_not_amplify_the_penalty():
    future = PenaltyRecord(
        document_id="d", negative_count=2, last_negative_at=NOW + timedelta(days=5)
    )

    assert penalty_for(future, NOW) == pytest.approx(penalty_for(_record(2, 0), NOW))


def test_a_zero_ceiling_disables_demotion():
    assert penalty_for(_record(5), NOW, ceiling=0.0) == 0.0


def test_a_nonpositive_half_life_is_rejected():
    # Zero divides; negative would make a penalty grow with age.
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="half_life_days"):
            penalty_for(_record(1), NOW, half_life_days=bad)


# ---------------------------------------------------------------------------
# demote
# ---------------------------------------------------------------------------


def test_a_penalised_document_ranks_below_an_equal_one():
    candidates = [("bad", 0.9), ("good", 0.9)]
    penalties = penalties_by_id([_record(3, document_id="bad")])

    ranked = demote(candidates, penalties, k=2, now=NOW)

    assert [doc for doc, _ in ranked] == ["good", "bad"]


def test_a_demoted_document_still_wins_when_it_is_clearly_the_best():
    # Over-fetching exists so a demotion does not become a deletion: a document
    # with one thumbs-down that is still the right answer must survive.
    candidates = [("bad", 0.95), ("other", 0.20)]
    penalties = penalties_by_id([_record(1, document_id="bad")])

    ranked = demote(candidates, penalties, k=1, now=NOW)

    assert [doc for doc, _ in ranked] == ["bad"]


def test_the_retrievers_own_order_breaks_ties():
    candidates = [("first", 0.5), ("second", 0.5)]

    ranked = demote(candidates, {}, k=2, now=NOW)

    assert [doc for doc, _ in ranked] == ["first", "second"]


def test_ranking_is_deterministic():
    candidates = [("a", 0.7), ("b", 0.7), ("c", 0.7)]
    penalties = penalties_by_id([_record(2, document_id="b")])

    first = demote(candidates, penalties, k=3, now=NOW)
    second = demote(candidates, penalties, k=3, now=NOW)

    assert first == second


def test_it_truncates_to_k():
    candidates = [(f"d{i}", 1.0 - i / 10) for i in range(9)]

    assert len(demote(candidates, {}, k=4, now=NOW)) == 4


def test_a_nonpositive_k_returns_nothing():
    assert demote([("a", 1.0)], {}, k=0, now=NOW) == []


def test_no_candidates_is_not_an_error():
    assert demote([], {}, k=3, now=NOW) == []


def test_decay_actually_changes_the_ranking():
    # Against an injected clock, not the wall clock.
    candidates = [("bad", 0.90), ("good", 0.80)]
    penalties = penalties_by_id([_record(4, days_ago=0, document_id="bad")])

    fresh = demote(candidates, penalties, k=2, now=NOW, ceiling=0.35, half_life_days=30)
    later = demote(
        candidates,
        penalties_by_id([_record(4, days_ago=0, document_id="bad")]),
        k=2,
        now=NOW + timedelta(days=365),
        ceiling=0.35,
        half_life_days=30,
    )

    assert [doc for doc, _ in fresh] == ["good", "bad"]
    assert [doc for doc, _ in later] == ["bad", "good"]


# ---------------------------------------------------------------------------
# over-fetching
# ---------------------------------------------------------------------------


def test_overfetch_asks_for_more_than_k():
    assert overfetch_size(4, factor=3) == 12
    assert overfetch_size(4, factor=1) == 4


def test_a_nonsense_factor_never_shrinks_the_pool():
    # Fetching fewer than k would be a bug that looks like a config value.
    assert overfetch_size(5, factor=0) == 5
    assert overfetch_size(5, factor=-2) == 5


def test_the_shipped_ceiling_is_a_demotion_not_a_veto():
    assert 0.0 < DEMOTION_CEILING < 1.0
